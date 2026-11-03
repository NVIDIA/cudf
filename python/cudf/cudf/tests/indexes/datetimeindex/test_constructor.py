# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.core.index import DatetimeIndex
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.mark.parametrize(
    "data,dtype,freq",
    [
        ([10], "datetime64[ns]", "2ns"),
        ([10, 12, 14, 16], "datetime64[ns]", "2ns"),
        ([10, 11, 12, 13], "datetime64[ns]", "1ns"),
        ([100, 200, 300, 400], "datetime64[s]", "100s"),
        ([101, 201, 301, 401], "datetime64[ms]", "100ms"),
    ],
)
def test_datetime_index_with_freq(data, dtype, freq):
    actual = cudf.DatetimeIndex(data, dtype=dtype, freq=freq)
    expected = pd.DatetimeIndex(data, dtype=dtype, freq=freq)
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data,dtype,freq",
    [
        ([10, 1232, 13244, 13426], "datetime64[ns]", "2ns"),
        ([10, 11, 12, 13], "datetime64[ns]", "1s"),
        ([10000, 200, 300, 400], "datetime64[s]", "100s"),
        ([107871, 201, 301, 401], "datetime64[ms]", "100ns"),
    ],
)
def test_datetime_index_freq_error(data, dtype, freq):
    assert_exceptions_equal(
        pd.DatetimeIndex,
        cudf.DatetimeIndex,
        ([data], {"dtype": dtype, "freq": freq}),
        ([data], {"dtype": dtype, "freq": freq}),
    )


def test_strings_with_utc_offset_not_implemented():
    with pytest.raises(NotImplementedError):
        DatetimeIndex(["2022-07-22 00:00:00+02:00"])


def test_dateimeindex_from_noniso_string():
    data = ["20160920", "20160925"]
    gdti = cudf.DatetimeIndex(data)
    pdti = pd.DatetimeIndex(data)

    assert_eq(gdti, pdti)


@pytest.mark.parametrize(
    "data",
    [
        [
            "2020-01-31",
            "2020-02-15",
            "2020-02-29",
            "2020-03-15",
            "2020-03-31",
            "2020-04-15",
            "2020-04-30",
        ],
        [43534, 43543, 37897, 2000],
    ],
)
@pytest.mark.parametrize("dtype", [None, "datetime64[ns]"])
def test_datetime_constructor(data, dtype):
    expected = pd.DatetimeIndex(data=data, dtype=dtype)
    actual = cudf.DatetimeIndex(data=data, dtype=dtype)

    assert_eq(expected, actual)

    expected = pd.DatetimeIndex(data=pd.Series(data), dtype=dtype)
    actual = cudf.DatetimeIndex(data=cudf.Series(data), dtype=dtype)

    assert_eq(expected, actual)


def test_from_pandas_datetimeindex_freq():
    expected = pd.date_range(start="1990-01-01", periods=10, freq="h")
    actual = cudf.from_pandas(expected)

    assert_eq(expected, actual)
    assert actual.freq is not None

    actual = cudf.Index(expected)
    assert_eq(expected, actual)
    assert actual.freq is not None


@pytest.mark.parametrize(
    "data",
    [
        [],
        (),
        [None],
        [None, None],
        pd.Series([], dtype="object"),
    ],
)
def test_datetime_index_empty_object_default_dtype(data):
    # Pandas's array_to_datetime falls back to "s" precision when no
    # concrete (non-NaT) datetime is observed (pandas-dev/pandas#55901).
    # cuDF should match so cudf.pandas-wrapped pd.DatetimeIndex([]) doesn't
    # diverge from plain pandas. Inputs to this test deliberately carry no
    # explicit dtype so cuDF's default-unit logic is what's exercised.
    pd_data = data
    gd_data = cudf.from_pandas(data) if isinstance(data, pd.Series) else data
    expected = pd.DatetimeIndex(pd_data)
    actual = cudf.DatetimeIndex(gd_data)
    assert actual.dtype == expected.dtype
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "values",
    [
        ["2020-01-01"],
        ["2020-01-01", "2020-01-02"],
        ["1970-01-01 00:00:00.000000"],
    ],
)
def test_datetime_index_string_input_us_default(values):
    # Non-empty string parsing should still resolve to [us] (pandas 3 default
    # when no nanosecond-precision component is present).
    expected = pd.DatetimeIndex(values)
    actual = cudf.DatetimeIndex(values)
    assert actual.dtype == expected.dtype
    assert_eq(actual, expected)
