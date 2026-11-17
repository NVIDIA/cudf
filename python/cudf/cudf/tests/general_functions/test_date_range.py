# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_date_range_freq_default():
    result = pd.date_range("2020-01-01", periods=2, name="foo")
    expected = cudf.date_range("2020-01-01", periods=2, name="foo")
    assert_eq(result, expected)


def test_date_range_tz():
    result = pd.date_range("2020-01-01", periods=2, tz="UTC")
    expected = cudf.date_range("2020-01-01", periods=2, tz="UTC")
    assert_eq(result, expected)

    result = pd.date_range("2020-01-01", "2020-01-02", periods=2, tz="UTC")
    expected = cudf.date_range("2020-01-01", "2020-01-02", periods=2, tz="UTC")
    assert_eq(result, expected)


def test_date_range_start_end_divisible_by_freq():
    result = cudf.date_range("2011-01-01", "2011-01-02", freq="h")
    expected = pd.date_range("2011-01-01", "2011-01-02", freq="h")
    assert_eq(result, expected)


@pytest.fixture(
    params=[
        "2000-02-13 08:41:06",
        "1996-11-21 04:05:30",
        "1970-01-01 00:00:00",
        "1831-05-08 15:23:21",
    ],
    ids=["leap_year", "non_leap_year", "unix_epoch_time_0", "random_date"],
)
def start(request):
    return request.param


@pytest.fixture(
    params=[
        "2000-02-13 08:41:06",
        "1996-11-21 04:05:30",
        "1970-01-01 00:00:00",
        "1831-05-08 15:23:21",
    ],
    ids=["leap_year", "non_leap_year", "unix_epoch_time_0", "random_date"],
)
def end(request):
    return request.param


@pytest.fixture(params=[1, 10])
def periods(request):
    return request.param


@pytest.fixture(
    params=[
        {"months": 3, "years": 1},
        {"hours": 10, "days": 57, "nanoseconds": 3},
        "83D",
        "17h",
        "-680min",
        "110546s",
        "110546789ms",
        "110546789248us",
    ]
)
def freq(request):
    return request.param


def test_date_range_start_end_periods(start, end, periods):
    expect = pd.date_range(start=start, end=end, periods=periods, name="a")
    got = cudf.date_range(start=start, end=end, periods=periods, name="a")

    np.testing.assert_allclose(
        expect.to_numpy().astype("int64"),
        got.to_pandas().to_numpy().astype("int64"),
    )


def test_date_range_end_freq_periods(end, freq, periods):
    if isinstance(freq, str):
        _gfreq = _pfreq = freq
    else:
        _gfreq = cudf.DateOffset(**freq)
        _pfreq = pd.DateOffset(**freq)

    expect = pd.date_range(end=end, periods=periods, freq=_pfreq, name="a")
    got = cudf.date_range(end=end, periods=periods, freq=_gfreq, name="a")

    np.testing.assert_allclose(
        expect.to_numpy().astype("int64"),
        got.to_pandas().to_numpy().astype("int64"),
    )


def test_date_range_freq_does_not_divide_range():
    expect = pd.date_range(
        "2001-01-01 00:00:00.000000", "2001-01-01 00:00:00.000010", freq="3us"
    )
    got = cudf.date_range(
        "2001-01-01 00:00:00.000000", "2001-01-01 00:00:00.000010", freq="3us"
    )
    np.testing.assert_allclose(
        expect.to_numpy().astype("int64"),
        got.to_pandas().to_numpy().astype("int64"),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nanoseconds": 1},
        {"months": 1},
    ],
)
def test_date_range_raise_overflow(kwargs):
    start = np.datetime64(np.iinfo("int64").max, "ns")
    periods = 2
    freq = cudf.DateOffset(**kwargs)
    with pytest.raises(pd.errors.OutOfBoundsDatetime):
        cudf.date_range(start=start, periods=periods, freq=freq)


@pytest.mark.parametrize(
    "freqstr_unsupported",
    [
        "1ME",
        "2SME",
        "3MS",
        "4BME",
        "5CBME",
        "6SMS",
        "7BMS",
        "8CBMS",
        "QE",
        "2BQE",
        "3BQS",
        "10YE",
        "9BYE",
        "8YS",
        "7BYS",
        "bh",
        "B",
    ],
)
@pytest.mark.parametrize("case", ["upper", "lower"])
def test_date_range_raise_unsupported(freqstr_unsupported, case):
    if case == "lower" and freqstr_unsupported == "3MS":
        pytest.skip(
            "3ms parsed as 3 milliseconds, generates too large of range to test"
        )
    if case == "lower" and freqstr_unsupported == "B":
        pytest.skip("b is deprecated in pandas 3.0 - no need to test.")
    s, e = "2001-01-01", "2008-01-31"
    freq = getattr(freqstr_unsupported, case)()
    with pytest.raises(ValueError):
        cudf.date_range(start=s, end=e, freq=freq)


def test_date_range_start_freq_periods(start, freq, periods):
    if isinstance(freq, str):
        _gfreq = _pfreq = freq
    else:
        _gfreq = cudf.DateOffset(**freq)
        _pfreq = pd.DateOffset(**freq)

    expect = pd.date_range(start=start, periods=periods, freq=_pfreq, name="a")
    got = cudf.date_range(start=start, periods=periods, freq=_gfreq, name="a")

    np.testing.assert_allclose(
        expect.to_numpy().astype("int64"),
        got.to_pandas().to_numpy().astype("int64"),
    )


def test_daterange_pandas_compatibility():
    with cudf.option_context("mode.pandas_compatible", True):
        expected = pd.date_range(
            "2010-01-01", "2010-02-01", periods=10, name="times"
        )
        actual = cudf.date_range(
            "2010-01-01", "2010-02-01", periods=10, name="times"
        )
    assert_eq(expected, actual)


def test_date_range_start_end_freq(start, end, freq):
    if isinstance(freq, str):
        _gfreq = _pfreq = freq
    else:
        _gfreq = cudf.DateOffset(**freq)
        _pfreq = pd.DateOffset(**freq)

    expect = pd.date_range(start=start, end=end, freq=_pfreq, name="a")
    got = cudf.date_range(start=start, end=end, freq=_gfreq, name="a")

    np.testing.assert_allclose(
        expect.to_numpy().astype("int64"),
        got.to_pandas().to_numpy().astype("int64"),
    )


def test_date_range_noniso_start_end_string():
    result = cudf.date_range("20161101", "20161130", freq="4h")
    expected = pd.date_range("20161101", "20161130", freq="4h")
    assert_eq(result, expected)
