# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import string
from collections import abc
from contextlib import contextmanager
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.utils import dtypes as dtypeutils
from cudf.utils.temporal import unit_to_nanoseconds_conversion

if TYPE_CHECKING:
    import pylibcudf as plc

    from cudf.core.column.column import ColumnBase

supported_numpy_dtypes = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "datetime64[ms]",
    "datetime64[us]",
]

SIGNED_INTEGER_TYPES = sorted(list(dtypeutils.SIGNED_INTEGER_TYPES))
UNSIGNED_TYPES = sorted(list(dtypeutils.UNSIGNED_TYPES))
INTEGER_TYPES = sorted(list(dtypeutils.INTEGER_TYPES))
FLOAT_TYPES = sorted(list(dtypeutils.FLOAT_TYPES))
SIGNED_TYPES = sorted(list(dtypeutils.SIGNED_TYPES))
NUMERIC_TYPES = sorted(list(dtypeutils.NUMERIC_TYPES))
DATETIME_TYPES = sorted(list(dtypeutils.DATETIME_TYPES))
TIMEDELTA_TYPES = sorted(list(dtypeutils.TIMEDELTA_TYPES))
OTHER_TYPES = sorted(list(dtypeutils.OTHER_TYPES))
ALL_TYPES = sorted(list(dtypeutils.ALL_TYPES))

SERIES_OR_INDEX_NAMES = [
    None,
    pd.NA,
    cudf.NA,
    np.nan,
    float("NaN"),
    "abc",
    1,
    pd.NaT,
    np.datetime64("nat"),
    np.timedelta64("NaT"),
    np.timedelta64(10, "D"),
    np.timedelta64(5, "D"),
    np.datetime64("1970-01-01 00:00:00.000000001"),
    np.datetime64("1970-01-01 00:00:00.000000002"),
    pd.Timestamp(1),
    pd.Timestamp(2),
    pd.Timedelta(1),
    pd.Timedelta(2),
    Decimal("NaN"),
    Decimal("1.2"),
    np.int64(1),
    np.int32(1),
    np.float32(1),
    pd.Timestamp(1),
]


def set_random_null_mask_inplace(series, null_probability=0.5):
    """Randomly nullify elements in series with the provided probability."""
    probs = [null_probability, 1 - null_probability]
    rng = np.random.default_rng(seed=0)
    mask = rng.choice([False, True], size=len(series), p=probs)
    series.iloc[mask] = None


def gen_rand_ufunc_input(ufunc, rng, size, low=1, high=10):
    """Generate random input appropriate for a ufunc's valid domain."""
    name = ufunc.__name__
    if name.startswith("bitwise") or name in (
        "invert",
        "left_shift",
        "right_shift",
        "gcd",
        "lcm",
        "ldexp",
        "ceil",
        "floor",
        "trunc",
        "fmod",
        "remainder",
    ):
        return rng.integers(low=low, high=high, size=size)
    if name == "arccosh":
        return rng.uniform(low=0.9, high=10.0, size=size)
    if name in ("log", "log2", "log10", "log1p", "sqrt", "float_power"):
        return rng.uniform(low=0.0, high=10.0, size=size)
    return rng.uniform(low=-1.1, high=1.1, size=size)


def assert_exceptions_equal(
    lfunc,
    rfunc,
    lfunc_args_and_kwargs=None,
    rfunc_args_and_kwargs=None,
    check_exception_type=True,
):
    """Compares if two functions ``lfunc`` and ``rfunc`` raise
    same exception or not.

    Parameters
    ----------
    lfunc : callable
        A callable function to obtain the Exception.
    rfunc : callable
        A callable function to compare the Exception
        obtained by calling ``rfunc``.
    lfunc_args_and_kwargs : tuple, default None
        Tuple containing positional arguments at first position,
        and key-word arguments at second position that need to be passed into
        ``lfunc``. If the tuple is of length 1, it must either contain
        positional arguments(as a Sequence) or key-word arguments(as a Mapping
        dict).
    rfunc_args_and_kwargs : tuple, default None
        Tuple containing positional arguments at first position,
        and key-word arguments at second position that need to be passed into
        ``rfunc``. If the tuple is of length 1, it must either contain
        positional arguments(as a Sequence) or key-word arguments(as a Mapping
        dict).
    check_exception_type : boolean, default True
        Whether to compare the exception types raised by ``lfunc``
        with ``rfunc`` exception type or not. If False, ``rfunc``
        is simply evaluated against `Exception` type.

    Returns
    -------
    None
        If exceptions raised by ``lfunc`` and
        ``rfunc`` match.

    Raises
    ------
    AssertionError
        If call to ``lfunc`` doesn't raise any Exception.
    """

    lfunc_args, lfunc_kwargs = _get_args_kwars_for_assert_exceptions(
        lfunc_args_and_kwargs
    )
    rfunc_args, rfunc_kwargs = _get_args_kwars_for_assert_exceptions(
        rfunc_args_and_kwargs
    )

    try:
        lfunc(*lfunc_args, **lfunc_kwargs)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        with pytest.raises(type(e) if check_exception_type else Exception):
            rfunc(*rfunc_args, **rfunc_kwargs)
    else:
        raise AssertionError("Expected to fail with an Exception.")


def _get_args_kwars_for_assert_exceptions(func_args_and_kwargs):
    if func_args_and_kwargs is None:
        return [], {}
    else:
        if len(func_args_and_kwargs) == 1:
            func_args, func_kwargs = [], {}
            if isinstance(func_args_and_kwargs[0], abc.Sequence):
                func_args = func_args_and_kwargs[0]
            elif isinstance(func_args_and_kwargs[0], abc.Mapping):
                func_kwargs = func_args_and_kwargs[0]
            else:
                raise ValueError(
                    "length 1 func_args_and_kwargs must be "
                    "either a Sequence or a Mapping"
                )
        elif len(func_args_and_kwargs) == 2:
            if not isinstance(func_args_and_kwargs[0], abc.Sequence):
                raise ValueError(
                    "Positional argument at 1st position of "
                    "func_args_and_kwargs should be a sequence."
                )
            if not isinstance(func_args_and_kwargs[1], abc.Mapping):
                raise ValueError(
                    "Key-word argument at 2nd position of "
                    "func_args_and_kwargs should be a dictionary mapping."
                )

            func_args, func_kwargs = func_args_and_kwargs
        else:
            raise ValueError("func_args_and_kwargs must be of length 1 or 2")
        return func_args, func_kwargs


def gen_rand(dtype, size, **kwargs):
    rng = np.random.default_rng(seed=kwargs.get("seed", 0))
    dtype = cudf.dtype(dtype)
    if dtype.kind == "f":
        res = rng.random(size=size).astype(dtype)
        if kwargs.get("positive_only", False):
            return res
        else:
            return res * 2 - 1
    elif dtype == np.int8 or dtype == np.int16:
        low = kwargs.get("low", -32)
        high = kwargs.get("high", 32)
        return rng.integers(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == "i":
        low = kwargs.get("low", -10000)
        high = kwargs.get("high", 10000)
        return rng.integers(low=low, high=high, size=size).astype(dtype)
    elif dtype == np.uint8 or dtype == np.uint16:
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 32)
        return rng.integers(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == "u":
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 128)
        return rng.integers(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == "b":
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 2)
        return rng.integers(low=low, high=high, size=size).astype(np.bool_)
    elif dtype.kind == "M":
        low = kwargs.get("low", 0)
        time_unit, _ = np.datetime_data(dtype)
        high = kwargs.get(
            "high",
            int(1e18) / unit_to_nanoseconds_conversion[time_unit],
        )
        return pd.to_datetime(
            rng.integers(low=low, high=high, size=size), unit=time_unit
        )
    elif dtype.kind in ("O", "U"):
        low = kwargs.get("low", 10)
        high = kwargs.get("high", 11)
        nchars = rng.integers(low=low, high=high, size=1)[0]
        char_options = np.array(list(string.ascii_letters + string.digits))
        all_chars = "".join(rng.choice(char_options, nchars * size))
        return np.array(
            [all_chars[nchars * i : nchars * (i + 1)] for i in range(size)]
        )

    raise NotImplementedError(f"dtype.kind={dtype.kind}")


def gen_rand_series(dtype, size, **kwargs):
    values = gen_rand(dtype, size, **kwargs)
    ser = cudf.Series(values)
    if kwargs.get("has_nulls", False):
        rng = np.random.default_rng(0)
        boolmask = rng.choice([True, False], size=size)
        ser.loc[boolmask] = None
    return ser


def _decimal_series(input, dtype):
    return cudf.Series(
        [x if x is None else Decimal(x) for x in input],
        dtype=dtype,
    )


def _assert_column_memory_eq(lhs: plc.Column, rhs: plc.Column):
    """Assert the memory location and size of `lhs` and `rhs` are equivalent.

    Both data pointer and mask pointer are checked. Also recursively check for
    children to the same constraints. Also fails check if the number of
    children mismatches at any level.
    """

    def get_ptr(x) -> int:
        return x.ptr if x else 0

    assert get_ptr(lhs.data()) == get_ptr(rhs.data())
    assert get_ptr(lhs.null_mask()) == get_ptr(rhs.null_mask())
    assert lhs.size() == rhs.size()
    assert lhs.offset() == rhs.offset()
    assert lhs.num_children() == rhs.num_children()
    for lhs_child, rhs_child in zip(
        lhs.children(), rhs.children(), strict=True
    ):
        _assert_column_memory_eq(lhs_child, rhs_child)


def assert_column_memory_eq(lhs: ColumnBase, rhs: ColumnBase):
    """Assert the memory location and size of `lhs` and `rhs` are equivalent.

    Both data pointer and mask pointer are checked. Also recursively check for
    children to the same constraints. Also fails check if the number of
    children mismatches at any level.
    """

    _assert_column_memory_eq(lhs.plc_column, rhs.plc_column)
    if isinstance(lhs, cudf.core.column.CategoricalColumn) and isinstance(
        rhs, cudf.core.column.CategoricalColumn
    ):
        assert_column_memory_eq(lhs.categories, rhs.categories)


def assert_column_memory_ne(lhs: ColumnBase, rhs: ColumnBase):
    try:
        assert_column_memory_eq(lhs, rhs)
    except AssertionError:
        return
    raise AssertionError("lhs and rhs holds the same memory.")


def assert_asserters_equal(
    pandas_asserter,
    cudf_asserter,
    pandas_left,
    pandas_right,
    cudf_left,
    cudf_right,
    *args,
    **kwargs,
):
    """
    Assert that a pandas and cudf asserter have equivalent behavior.

    Parameters
    ----------
    pandas_asserter : callable
        A pandas asserter function.
    cudf_asserter : callable
        A cudf asserter function.
    pandas_left : object
        A pandas object as the left argument to the pandas asserter.
    pandas_right : object
        A pandas object as the right argument to the pandas asserter.
    cudf_left : object
        A cudf object as the left argument to the cudf asserter.
    cudf_right : object
        A cudf object as the right argument to the pandas asserter.
    *args : tuple
        Additional arguments to pass to both asserters.
    **kwargs : dict
        Additional keyword arguments to both asserters.
    """
    # TypeError is raised (erroneously from pandas) when comparing
    # categorical indices with different categories.
    exceptions = (AssertionError, TypeError)
    try:
        pandas_asserter(pandas_left, pandas_right, *args, **kwargs)
    except exceptions:
        with pytest.raises(exceptions):
            cudf_asserter(cudf_left, cudf_right, *args, **kwargs)
    else:
        cudf_asserter(cudf_left, cudf_right, *args, **kwargs)


@contextmanager
def expect_warning_if(condition, warning=FutureWarning, *args, **kwargs):
    """Catch a warning using pytest.warns if the expect_warning is True.

    All arguments are forwarded to pytest.warns if expect_warning is True.
    """
    if condition:
        with pytest.warns(warning, *args, **kwargs):
            yield
    else:
        yield
