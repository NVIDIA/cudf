# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator
import re

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_series_where_mixed_dtypes_error():
    s = cudf.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "cudf does not support mixed types, please type-cast "
            "the column of dataframe/series and other "
            "to same dtypes."
        ),
    ):
        s.where([True, False, True], [1, 2, 3])


def test_series_where_mixed_bool_dtype():
    s = cudf.Series([True, False, True])
    with pytest.raises(TypeError):
        s.where(~s, 10)


@pytest.mark.parametrize("fill_value", [100, 100.0, 128.5])
@pytest.mark.parametrize("op", [operator.gt, operator.eq, operator.lt])
def test_series_where(numeric_types_as_str, fill_value, op):
    psr = pd.Series(list(range(10)), dtype=numeric_types_as_str)
    sr = cudf.from_pandas(psr)

    try:
        scalar_fits = sr.dtype.type(fill_value) == fill_value
    except OverflowError:
        scalar_fits = False

    if not scalar_fits:
        with pytest.raises(TypeError):
            sr.where(op(sr, 0), fill_value)
    else:
        # Cast back to original dtype as pandas automatically upcasts
        expect = psr.where(op(psr, 0), fill_value)
        got = sr.where(op(sr, 0), fill_value)
        # pandas returns 'float16' dtype, which is not supported in cudf
        assert_eq(
            expect,
            got,
            check_dtype=expect.dtype.kind != "f",
        )


@pytest.mark.parametrize("fill_value", [100, 100.0, 100.5])
def test_series_with_nulls_where(fill_value):
    psr = pd.Series([None] * 3 + list(range(5)))
    sr = cudf.from_pandas(psr)

    expect = psr.where(psr > 0, fill_value)
    got = sr.where(sr > 0, fill_value)
    assert_eq(expect, got)

    expect = psr.where(psr < 0, fill_value)
    got = sr.where(sr < 0, fill_value)
    assert_eq(expect, got)

    expect = psr.where(psr == 0, fill_value)
    got = sr.where(sr == 0, fill_value)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtype", ["datetime64[ns]", "datetime64[us]", "timedelta64[ns]"]
)
def test_series_mask_inplace_int_into_temporal_pandas_compat(dtype):
    # Series.mask with inplace=True must reject incompatible dtype values
    # (e.g. int into datetime/timedelta) under pandas-compatible mode.
    from cudf.testing._utils import assert_exceptions_equal

    psr = pd.Series([1, 2, 3], dtype=dtype)
    gsr = cudf.from_pandas(psr)
    pmask = pd.Series([True, False, True])
    gmask = cudf.from_pandas(pmask)

    def pmask_call():
        psr_copy = psr.copy()
        psr_copy.mask(pmask, 4, inplace=True)

    def gmask_call():
        gsr_copy = gsr.copy()
        gsr_copy.mask(gmask, 4, inplace=True)

    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=pmask_call,
            rfunc=gmask_call,
            lfunc_args_and_kwargs=([], {}),
            rfunc_args_and_kwargs=([], {}),
        )


@pytest.mark.parametrize("dtype", ["UInt8", "Int64", "Float64"])
@pytest.mark.parametrize(
    "nat",
    [
        pd.NaT,
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ],
)
def test_series_where_nat_invalid_for_masked_dtypes(dtype, nat):
    # NaT is only a valid missing value for datetime/timedelta dtypes;
    # pandas raises TypeError for masked (nullable extension) dtypes.
    gsr = cudf.Series(pd.array([1, 2, 3], dtype=dtype))
    with pytest.raises(
        TypeError, match=rf"Invalid value '.*' for dtype '{dtype}'"
    ):
        gsr.where(cudf.Series([True, True, False]), nat)


@pytest.mark.parametrize("nat", [pd.NaT, np.datetime64("NaT")])
def test_series_where_nat_valid_for_datetime(nat):
    gsr = cudf.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))
    result = gsr.where(cudf.Series([True, False]), nat)
    assert result.isnull().sum() == 1
