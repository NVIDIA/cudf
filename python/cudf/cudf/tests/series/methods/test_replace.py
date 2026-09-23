# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.mark.parametrize(
    "gsr_data, dtype",
    [
        [[5, 1, 2, 3, None, 243, None, 4], None],
        [["one", "two", "three", None, "one"], "category"],
        [[*list(range(400)), None], None],
    ],
)
@pytest.mark.parametrize(
    "to_replace,value",
    [
        (0, 5),
        ("one", "two"),
        ("one", "five"),
        ("abc", "hello"),
        ([0, 1], [5, 6]),
        ([22, 323, 27, 0], -1),
        ([1, 2, 3], cudf.Series([10, 11, 12])),
        (cudf.Series([1, 2, 3]), None),
        ({1: 10, 2: 22}, None),
        (np.inf, 4),
    ],
)
def test_series_replace_all(request, gsr_data, dtype, to_replace, value):
    if dtype == "category" and isinstance(value, str) and value == "five":
        pytest.skip("Tested in test_series_replace_new_string_category_raises")

    gsr = cudf.Series(gsr_data, dtype=dtype)
    psr = gsr.to_pandas()

    gd_to_replace = to_replace
    if isinstance(to_replace, cudf.Series):
        pd_to_replace = to_replace.to_pandas()
    else:
        pd_to_replace = to_replace

    gd_value = value
    if isinstance(value, cudf.Series):
        pd_value = value.to_pandas()
    else:
        pd_value = value

    request.applymarker(
        pytest.mark.xfail(
            isinstance(gsr.dtype, cudf.CategoricalDtype)
            and isinstance(value, str)
            and value == "two",
            reason="cuDF missing 'one' from the result categories",
        )
    )
    actual = gsr.replace(to_replace=gd_to_replace, value=gd_value)
    if pd_value is None:
        # TODO: Remove this workaround once cudf
        # introduces `no_default` values
        expected = psr.replace(to_replace=pd_to_replace)
    else:
        expected = psr.replace(to_replace=pd_to_replace, value=pd_value)

    assert_eq(expected, actual)


def test_series_replace_new_string_category_raises():
    psr = pd.Series(["one", "two", "three", None, "one"], dtype="category")
    gsr = cudf.Series(["one", "two", "three", None, "one"], dtype="category")
    with pytest.raises(TypeError):
        psr.replace("one", "five")
    with pytest.raises(TypeError):
        gsr.replace("one", "five")


def test_series_replace():
    a1 = np.array([0, 1, 2, 3, 4])

    # Numerical
    a2 = np.array([5, 1, 2, 3, 4])
    sr1 = cudf.Series(a1)
    sr2 = sr1.replace(0, 5)
    assert_eq(a2, sr2.to_numpy())

    # List input
    a6 = np.array([5, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [5, 6])
    assert_eq(a6, sr6.to_numpy())

    assert_eq(
        sr1.replace([0, 1], [5.5, 6.5]),
        sr1.to_pandas().replace([0, 1], [5.5, 6.5]),
    )

    # Series input
    a8 = np.array([5, 5, 5, 3, 4])
    sr8 = sr1.replace(sr1[:3].to_numpy(), 5)
    assert_eq(a8, sr8.to_numpy())

    # large input containing null
    sr9 = cudf.Series([*list(range(400)), None])
    sr10 = sr9.replace([22, 323, 27, 0], None)
    assert sr10.null_count == 5
    assert len(sr10.dropna().to_numpy()) == (401 - 5)

    sr11 = sr9.replace([22, 323, 27, 0], -1)
    assert sr11.null_count == 1
    assert len(sr11.dropna().to_numpy()) == (401 - 1)

    # large input not containing nulls
    sr9 = sr9.fillna(-11)
    sr12 = sr9.replace([22, 323, 27, 0], None)
    assert sr12.null_count == 4
    assert len(sr12.dropna().to_numpy()) == (401 - 4)

    sr13 = sr9.replace([22, 323, 27, 0], -1)
    assert sr13.null_count == 0
    assert len(sr13.to_numpy()) == 401


def test_series_replace_with_nulls():
    a1 = np.array([0, 1, 2, 3, 4])

    # Numerical
    a2 = np.array([-10, 1, 2, 3, 4])
    sr1 = cudf.Series(a1)
    sr2 = sr1.replace(0, None).fillna(-10)
    assert_eq(a2, sr2.to_numpy())

    # List input
    a6 = np.array([-10, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [None, 6]).fillna(-10)
    assert_eq(a6, sr6.to_numpy())

    sr1 = cudf.Series([0, 1, 2, 3, 4, None])
    assert_eq(
        sr1.replace([0, 1], [5.5, 6.5]).fillna(-10),
        sr1.to_pandas().replace([0, 1], [5.5, 6.5]).fillna(-10),
    )

    # Series input
    a8 = np.array([-10, -10, -10, 3, 4, -10])
    sr8 = sr1.replace(cudf.Series([-10] * 3, index=sr1[:3]), None).fillna(-10)
    assert_eq(a8, sr8.to_numpy())

    a9 = np.array([-10, 6, 2, 3, 4, -10])
    sr9 = sr1.replace([0, 1], [None, 6]).fillna(-10)
    assert_eq(a9, sr9.to_numpy())


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([0, 1, None, 2, None], dtype=pd.Int8Dtype()),
        pd.Series([0, 1, np.nan, 2, np.nan]),
    ],
)
@pytest.mark.parametrize("fill_value", [10, pd.Series([10, 20, 30, 40, 50])])
def test_series_fillna_numerical(
    psr, numeric_types_as_str, fill_value, inplace
):
    if inplace:
        psr = psr.copy(deep=True)
    # TODO: These tests should use Pandas' nullable int type
    # when we support a recent enough version of Pandas
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
    if np.dtype(numeric_types_as_str).kind != "f" and psr.dtype.kind == "i":
        psr = psr.astype(
            cudf.utils.dtypes.np_dtypes_to_pandas_dtypes[
                np.dtype(numeric_types_as_str)
            ]
        )

    gsr = cudf.from_pandas(psr)

    if isinstance(fill_value, pd.Series):
        fill_value_cudf = cudf.from_pandas(fill_value)
    else:
        fill_value_cudf = fill_value

    expected = psr.fillna(fill_value, inplace=inplace)
    actual = gsr.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        expected = psr
        actual = gsr

    # TODO: Remove check_dtype when we have support
    # to compare with pandas nullable dtypes
    assert_eq(expected, actual, check_dtype=False)


def test_series_multiple_times_with_nulls():
    sr = cudf.Series([1, 2, 3, None])
    expected = cudf.Series([None, None, None, None], dtype=np.int64)

    for i in range(3):
        got = sr.replace([1, 2, 3], None)
        assert_eq(expected, got)
        # BUG: #2695
        # The following series will acquire a chunk of memory and update with
        # values, but these values may still linger even after the memory
        # gets released. This memory space might get used for replace in
        # subsequent calls and the memory used for mask may have junk values.
        # So, if it is not updated properly, the result would be wrong.
        # So, this will help verify that scenario.
        cudf.Series([1, 1, 1, None])


@pytest.mark.parametrize(
    "replacement", [128, 128.0, 128.5, 32769, 32769.0, 32769.5]
)
def test_numeric_series_replace_dtype(
    request, numeric_types_as_str, replacement
):
    request.applymarker(
        pytest.mark.xfail(
            (
                numeric_types_as_str == "int8"
                and replacement in {128, 128.0, 32769, 32769.0}
            )
            or (
                numeric_types_as_str == "int16"
                and replacement in {32769, 32769.0}
            ),
            reason="Something has gone wrong, please report a bug at https://github.com/pandas-dev/pandas/issues",
        )
    )
    psr = pd.Series([0, 1, 2, 3, 4, 5], dtype=numeric_types_as_str)
    sr = cudf.from_pandas(psr)

    expect = psr.replace(1, replacement)
    got = sr.replace(1, replacement)

    assert_eq(expect, got)

    # to_replace is a list, replacement is a scalar
    expect = psr.replace([2, 3], replacement)
    got = sr.replace([2, 3], replacement)

    assert_eq(expect, got)

    # If to_replace is a scalar and replacement is a list
    with pytest.raises(TypeError):
        sr.replace(0, [replacement, 2])

    # Both list of unequal length
    with pytest.raises(ValueError):
        sr.replace([0, 1], [replacement])

    # Both lists of equal length
    expect = psr.replace([2, 3], [replacement, replacement])
    got = sr.replace([2, 3], [replacement, replacement])
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "pframe, replace_args",
    [
        (
            pd.Series([5, 1, 2, 3, 4]),
            {"to_replace": 5, "value": 0, "inplace": True},
        ),
        (
            pd.Series([5, 1, 2, 3, 4]),
            {"to_replace": {5: 0, 3: -5}, "inplace": True},
        ),
        pytest.param(
            pd.Series(["one", "two", "three"], dtype="category"),
            {"to_replace": "one", "value": "two", "inplace": True},
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/43232"
                "https://github.com/pandas-dev/pandas/issues/53358",
            ),
        ),
        (
            pd.DataFrame({"A": [0, 1, 2, 3, 4], "B": [5, 6, 7, 8, 9]}),
            {"to_replace": 5, "value": 0, "inplace": True},
        ),
        (
            pd.Series([1, 2, 3, 45]),
            {
                "to_replace": np.array([]).astype(int),
                "value": 77,
                "inplace": True,
            },
        ),
        (
            pd.Series([1, 2, 3, 45]),
            {
                "to_replace": np.array([]).astype(int),
                "value": 77,
                "inplace": False,
            },
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]}),
            {"to_replace": {"a": 2}, "value": {"a": -33}, "inplace": True},
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]}),
            {
                "to_replace": {"a": [2, 5]},
                "value": {"a": [9, 10]},
                "inplace": True,
            },
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]}),
            {"to_replace": [], "value": [], "inplace": True},
        ),
    ],
)
def test_replace_inplace(pframe, replace_args):
    gpu_frame = cudf.from_pandas(pframe)
    pandas_frame = pframe.copy()

    gpu_copy = gpu_frame.copy()
    cpu_copy = pandas_frame.copy()

    assert_eq(gpu_frame, pandas_frame)
    assert_eq(gpu_copy, cpu_copy)
    gpu_frame.replace(**replace_args)
    pandas_frame.replace(**replace_args)
    assert_eq(gpu_frame, pandas_frame)
    assert_eq(gpu_copy, cpu_copy)


def test_series_replace_value_no_args():
    psr = pd.Series([5, 1, 2, 3, 4])
    gsr = cudf.Series([5, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        psr.replace()
    with pytest.raises(ValueError):
        gsr.replace()


def test_series_replace_errors():
    gsr = cudf.Series([1, 2, None, 3, None])
    psr = gsr.to_pandas()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "to_replace and value should be of same types,"
            "got to_replace dtype: int64 and "
            "value dtype: str"
        ),
    ):
        gsr.replace(1, "a")

    gsr = cudf.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "to_replace and value should be of same types,"
            "got to_replace dtype: int64 and "
            "value dtype: str"
        ),
    ):
        gsr.replace([1, 2], ["a", "b"])

    assert_exceptions_equal(
        psr.replace,
        gsr.replace,
        ([{"a": 1}, 1],),
        ([{"a": 1}, 1],),
    )

    assert_exceptions_equal(
        lfunc=psr.replace,
        rfunc=gsr.replace,
        lfunc_args_and_kwargs=([[1, 2], [1]],),
        rfunc_args_and_kwargs=([[1, 2], [1]],),
    )

    assert_exceptions_equal(
        lfunc=psr.replace,
        rfunc=gsr.replace,
        lfunc_args_and_kwargs=([object(), [1]],),
        rfunc_args_and_kwargs=([object(), [1]],),
    )

    assert_exceptions_equal(
        lfunc=psr.replace,
        rfunc=gsr.replace,
        lfunc_args_and_kwargs=([{"a": 1}, object()],),
        rfunc_args_and_kwargs=([{"a": 1}, object()],),
    )


@pytest.mark.parametrize(
    "gsr,old,new",
    [
        (
            lambda: cudf.Series(["a", "b", "c", None]),
            pd.NA,
            "a",
        ),
        (
            lambda: cudf.Series(["a", "b", "c", None]),
            [pd.NA, "a", "a"],
            ["c", "b", "d"],
        ),
        (
            lambda: cudf.Series(["a", "b", "c", None]),
            [pd.NA, "a"],
            ["b", pd.NA],
        ),
        (
            lambda: cudf.Series(["a", "b", "c", None]),
            [pd.NA, pd.NA],
            [pd.NA, pd.NA],
        ),
        (
            lambda: cudf.Series([1, 2, None, 3], dtype="Int64"),
            pd.NA,
            10,
        ),
        (
            lambda: cudf.Series([1, 2, None, 3], dtype="Int64"),
            [None, 1, 1],
            [3, 2, 4],
        ),
        (
            lambda: cudf.Series([1, 2, None, 3], dtype="Int64"),
            [pd.NA, 1],
            [2, pd.NA],
        ),
        (
            lambda: cudf.Series(["a", "q", "t", None], dtype="category"),
            None,
            "z",
        ),
        (
            lambda: cudf.Series(["a", "q", "t", None], dtype="category"),
            [None, "a", "q"],
            ["z", None, None],
        ),
        (
            lambda: cudf.Series(["a", None, "t", None], dtype="category"),
            [None, "t"],
            ["p", None],
        ),
    ],
)
def test_replace_nulls(gsr, old, new):
    gsr = gsr()
    psr = gsr.to_pandas()

    try:
        expected = psr.replace(old, new)
    except Exception as e:
        with pytest.raises(type(e)):
            gsr.replace(old, new)
    else:
        actual = gsr.replace(old, new)
        assert_eq(expected, actual)


def test_replace_with_index_objects():
    result = cudf.Series([1, 2]).replace(cudf.Index([1]), cudf.Index([2]))
    expected = pd.Series([1, 2]).replace(pd.Index([1]), pd.Index([2]))
    assert_eq(result, expected)


def test_replace_datetime_series():
    pd_series = pd.Series(pd.date_range("20210101", periods=5))
    pd_result = pd_series.replace(
        pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-10")
    )

    cudf_series = cudf.Series(pd.date_range("20210101", periods=5))
    cudf_result = cudf_series.replace(
        pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-10")
    )

    assert_eq(pd_result, cudf_result)


def test_replace_timedelta_series():
    pd_series = pd.Series(pd.timedelta_range("1 days", periods=5))
    pd_result = pd_series.replace(
        pd.Timedelta("2 days"), pd.Timedelta("10 days")
    )

    cudf_series = cudf.Series(pd.timedelta_range("1 days", periods=5))
    cudf_result = cudf_series.replace(
        pd.Timedelta("2 days"), pd.Timedelta("10 days")
    )

    assert_eq(pd_result, cudf_result)


def test_replace_datetime_sub_resolution_value_is_noop():
    # The to_replace value has sub-second precision that cannot exist in
    # a seconds-resolution column; it must not be truncated into a match.
    data = np.array(["2000-01-01"], dtype="datetime64[s]")
    pd_result = pd.Series(data).replace(
        pd.Timestamp("2000-01-01 00:00:00.5"), pd.Timestamp("1999-01-01")
    )
    cudf_result = cudf.Series(data).replace(
        pd.Timestamp("2000-01-01 00:00:00.5"), pd.Timestamp("1999-01-01")
    )
    assert_eq(pd_result, cudf_result)
