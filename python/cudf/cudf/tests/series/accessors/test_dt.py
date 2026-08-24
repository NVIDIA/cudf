# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import zoneinfo

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq

# Parametrize between numpy datetime64 and pandas ArrowDtype timestamp so
# every dt accessor test exercises both code paths.
_DT_DTYPES = [
    "datetime64[ns]",
    pd.ArrowDtype(pa.timestamp("ns")),
]


@pytest.fixture(
    params=[
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [
            136457654736252,
            134736784364431,
            245345345545332,
            223432411,
            2343241,
            3634548734,
            23234,
        ],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ]
)
def timedelta_data(request):
    return request.param


def test_timedelta_dt_components(timedelta_data, timedelta_types_as_str):
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    expected = psr.dt.components
    actual = gsr.dt.components

    if gsr.isnull().any():
        assert_eq(expected, actual.astype("float"))
    else:
        assert_eq(expected, actual)


def test_timedelta_dt_properties(timedelta_data, timedelta_types_as_str):
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    def local_assert(expected, actual, **kwargs):
        if gsr.isnull().any():
            assert_eq(expected, actual.astype("float"), **kwargs)
        else:
            assert_eq(expected, actual, **kwargs)

    expected_days = psr.dt.days
    actual_days = gsr.dt.days

    local_assert(expected_days, actual_days, check_dtype=False)

    expected_seconds = psr.dt.seconds
    actual_seconds = gsr.dt.seconds

    local_assert(expected_seconds, actual_seconds, check_dtype=False)

    expected_microseconds = psr.dt.microseconds
    actual_microseconds = gsr.dt.microseconds

    local_assert(expected_microseconds, actual_microseconds, check_dtype=False)

    expected_nanoseconds = psr.dt.nanoseconds
    actual_nanoseconds = gsr.dt.nanoseconds

    local_assert(expected_nanoseconds, actual_nanoseconds, check_dtype=False)


def test_timedelta_series_total_seconds(
    timedelta_data, timedelta_types_as_str
):
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    expected = psr.dt.total_seconds()
    actual = gsr.dt.total_seconds()
    assert_eq(expected, actual)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_timedelta_total_seconds_arrow_dtype(unit):
    # ArrowDtype duration: result must be float64[pyarrow] to match pandas.
    ps = pd.Series(
        [
            pd.Timedelta(days=1, seconds=2, microseconds=3, nanoseconds=4),
            None,
            pd.Timedelta(seconds=86400.5),
        ],
        dtype=pd.ArrowDtype(pa.duration(unit)),
    )
    gs = cudf.from_pandas(ps)

    expected = ps.dt.total_seconds()
    actual = gs.dt.total_seconds()
    assert_eq(expected, actual)


@pytest.mark.parametrize("meth", ["day_name", "month_name"])
@pytest.mark.parametrize("klass", [cudf.Series, cudf.DatetimeIndex])
def test_day_month_name_locale_not_implemented(meth, klass):
    obj = klass(cudf.date_range("2020-01-01", periods=7))
    if klass is cudf.Series:
        obj = obj.dt
    with pytest.raises(NotImplementedError):
        getattr(obj, meth)(locale="pt_BR.utf8")


@pytest.mark.parametrize("meth", ["day_name", "month_name"])
@pytest.mark.parametrize("klass", [pd.Series, pd.DatetimeIndex])
def test_day_month_name(meth, klass):
    data = [
        "2020-05-31 08:00:00",
        None,
        "1999-12-31 18:40:00",
        "2000-12-31 04:00:00",
        None,
        "1900-02-28 07:00:00",
        "1800-03-14 07:30:00",
        "2100-03-14 07:30:00",
        "1970-01-01 00:00:00",
        "1969-12-31 12:59:00",
    ]

    p_obj = klass(data, dtype="datetime64[s]")
    g_obj = cudf.from_pandas(p_obj)

    if klass is pd.Series:
        p_obj = p_obj.dt
        g_obj = g_obj.dt

    expect = getattr(p_obj, meth)()
    got = getattr(g_obj, meth)()

    assert_eq(expect, got)


@pytest.mark.parametrize("meth", ["day_name", "month_name"])
def test_day_month_name_arrow_dtype(meth):
    data = [
        "2020-05-31 08:00:00",
        None,
        "1999-12-31 18:40:00",
        "2000-12-31 04:00:00",
        None,
        "1900-02-28 07:00:00",
        "1800-03-14 07:30:00",
        "2100-03-14 07:30:00",
        "1970-01-01 00:00:00",
        "1969-12-31 12:59:00",
    ]

    ps = pd.Series(data, dtype=pd.ArrowDtype(pa.timestamp("s")))
    gs = cudf.from_pandas(ps)

    expect = getattr(ps.dt, meth)()
    got = getattr(gs.dt, meth)()

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "freqstr",
    [
        "H",
        "N",
        "T",
        "L",
        "U",
        "S",
    ],
)
def test_datetime_ceil_invalid_freq_raises(freqstr):
    t = cudf.Series(
        ["2001-01-01 00:04:45", "2001-01-01 00:04:58", "2001-01-01 00:05:04"],
        dtype="datetime64[ns]",
    )
    with pytest.raises(ValueError):
        t.dt.ceil(freqstr)


@pytest.mark.parametrize(
    "resolution", ["D", "h", "min", "min", "s", "ms", "us", "ns"]
)
def test_round(datetime_types_as_str, resolution):
    data = [
        "2020-05-31 08:00:00",
        "1999-12-31 18:40:10",
        "2000-12-31 04:00:05",
        "1900-02-28 07:00:06",
        "1800-03-14 07:30:20",
        "2100-03-14 07:30:20",
        "1970-01-01 00:00:09",
        "1969-12-31 12:59:10",
    ]
    gs = cudf.Series(data, dtype=datetime_types_as_str)
    ps = gs.to_pandas()

    expect = ps.dt.round(resolution)
    got = gs.dt.round(resolution)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "resolution", ["D", "h", "min", "min", "s", "ms", "us", "ns"]
)
def test_floor(datetime_types_as_str, resolution):
    data = [
        "2020-05-31 08:00:00",
        "1999-12-31 18:40:10",
        "2000-12-31 04:00:05",
        "1900-02-28 07:00:06",
        "1800-03-14 07:30:20",
        "2100-03-14 07:30:20",
        "1970-01-01 00:00:09",
        "1969-12-31 12:59:10",
    ]
    gs = cudf.Series(data, dtype=datetime_types_as_str)
    ps = gs.to_pandas()

    expect = ps.dt.floor(resolution)
    got = gs.dt.floor(resolution)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "resolution", ["D", "h", "min", "min", "s", "ms", "us", "ns"]
)
def test_ceil(datetime_types_as_str, resolution):
    data = [
        "2020-05-31 08:00:00",
        "1999-12-31 18:40:10",
        "2000-12-31 04:00:05",
        "1900-02-28 07:00:06",
        "1800-03-14 07:30:20",
        "2100-03-14 07:30:20",
        "1970-01-01 00:00:09",
        "1969-12-31 12:59:10",
    ]
    gs = cudf.Series(data, dtype=datetime_types_as_str)
    ps = gs.to_pandas()

    expect = ps.dt.ceil(resolution)
    got = gs.dt.ceil(resolution)
    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", _DT_DTYPES)
def test_days_in_months(dtype):
    data = [
        "2020-05-31",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-02-28",
        "1800-03-14",
        "2100-03-10",
        "1970-01-01",
        "1969-12-11",
    ]
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    assert_eq(ps.dt.days_in_month, gs.dt.days_in_month)
    assert_eq(ps.dt.daysinmonth, gs.dt.daysinmonth)


@pytest.mark.parametrize("dtype", _DT_DTYPES)
def test_is_month_start(dtype):
    data = [
        "2020-05-31",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-02-28",
        "1800-03-14",
        "2100-03-10",
        "1970-01-01",
        "1969-12-11",
    ]
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_month_start
    got = gs.dt.is_month_start

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", _DT_DTYPES)
def test_is_month_end(dtype):
    data = [
        "2020-05-31",
        "2020-02-29",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-02-28",
        "1800-03-14",
        "2100-03-10",
        "1970-01-01",
        "1969-12-11",
    ]
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_month_end
    got = gs.dt.is_month_end

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", _DT_DTYPES)
def test_is_year_start(dtype):
    data = [
        "2020-05-31",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-01-01",
        "1800-03-14",
        "2100-03-10",
        "1970-01-01",
        "1969-12-11",
        "2017-12-30",
        "2017-12-31",
        "2018-01-01",
    ]
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_year_start
    got = gs.dt.is_year_start

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", _DT_DTYPES)
def test_is_year_end(dtype):
    data = [
        "2020-05-31",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-12-31",
        "1800-03-14",
        "2017-12-30",
        "2017-12-31",
        "2020-12-31 08:00:00",
        None,
        "1999-12-31 18:40:00",
        "2000-12-31 04:00:00",
        None,
        "1800-12-14 07:30:00",
        "2100-12-14 07:30:00",
        "2020-05-31",
    ]
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_year_end
    got = gs.dt.is_year_end

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", _DT_DTYPES)
def test_is_quarter_start(dtype):
    data = [
        "2020-05-01",
        "2020-05-31",
        "2020-02-29",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-02-28",
        "1800-03-14",
        "2100-03-10",
        "1970-04-1",
        "1970-01-01",
        "1969-12-11",
        "2020-12-31",
    ]
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_quarter_start
    got = gs.dt.is_quarter_start

    assert_eq(expect, got)


@pytest.mark.parametrize("dtype", _DT_DTYPES)
def test_is_quarter_end(dtype):
    data = [
        "2020-05-01",
        "2020-05-31",
        "2020-02-29",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-02-28",
        "1800-03-14",
        "2100-03-10",
        "1970-04-1",
        "1970-01-01",
        "1969-12-11",
        "2020-12-31",
    ]
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_quarter_end
    got = gs.dt.is_quarter_end

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtype",
    ["datetime64[s]", pd.ArrowDtype(pa.timestamp("s"))],
)
def test_is_leap_year(dtype):
    data = [
        "2020-05-31 08:00:00",
        None,
        "1999-12-31 18:40:00",
        "2000-12-31 04:00:00",
        None,
        "1900-02-28 07:00:00",
        "1800-03-14 07:30:00",
        "2100-03-14 07:30:00",
        "1970-01-01 00:00:00",
        "1969-12-31 12:59:00",
    ]

    # Series
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_leap_year
    got = gs.dt.is_leap_year

    assert_eq(expect, got)

    # DatetimeIndex (ArrowDtype isn't accepted by DatetimeIndex; only
    # exercise the index code path for the numpy variant.)
    if isinstance(dtype, str):
        pIndex = pd.DatetimeIndex(data)
        gIndex = cudf.from_pandas(pIndex)

        expect2 = pIndex.is_leap_year
        got2 = gIndex.is_leap_year

        assert_eq(expect2, got2)


def test_quarter():
    data = [
        "2020-05-31 08:00:00",
        "1999-12-31 18:40:00",
        "2000-12-31 04:00:00",
        "1900-02-28 07:00:00",
        "1800-03-14 07:30:00",
        "2100-03-14 07:30:00",
        "1970-01-01 00:00:00",
        "1969-12-31 12:59:00",
    ]
    dtype = "datetime64[s]"

    # Series
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.quarter
    got = gs.dt.quarter

    assert_eq(expect, got, check_dtype=False)

    # DatetimeIndex
    pIndex = pd.DatetimeIndex(data)
    gIndex = cudf.from_pandas(pIndex)

    expect2 = pIndex.quarter
    got2 = gIndex.quarter

    assert_eq(expect2.values, got2.values)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
        pd.Series("2020-05-31 08:00:00", dtype="datetime64[s]"),
        pd.Series(
            pd.date_range(start="2021-07-25", end="2021-07-30"),
            index=["a", "b", "c", "d", "e", "f"],
        ),
        pd.Series(
            ["2010-01-01", "2010-12-31", None],
            dtype=pd.ArrowDtype(pa.timestamp("ns")),
        ),
        pd.Series([None, None], dtype=pd.ArrowDtype(pa.timestamp("us"))),
    ],
)
def test_isocalendar_series(data):
    ps = data.copy()
    gs = cudf.from_pandas(ps)

    expect = ps.dt.isocalendar()
    got = gs.dt.isocalendar()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("dtype", _DT_DTYPES)
def test_isocalendar_dtype(dtype):
    # isocalendar's year/week/day columns must match pandas' backing dtype:
    # UInt32 for numpy datetime, int64[pyarrow] for ArrowDtype timestamps.
    ps = pd.Series(["2010-01-01", "2010-12-31", None], dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.isocalendar()
    got = gs.dt.isocalendar()

    # Exercise both value equality and the exact per-column dtype.
    assert_eq(expect, got)
    assert dict(got.dtypes) == dict(expect.dtypes)
    if isinstance(dtype, pd.ArrowDtype):
        assert all(dt == pd.ArrowDtype(pa.int64()) for dt in got.dtypes)
    else:
        assert all(dt == pd.UInt32Dtype() for dt in got.dtypes)


@pytest.mark.parametrize("dtype", _DT_DTYPES)
@pytest.mark.parametrize("attr", ["dayofweek", "day_of_week", "weekday"])
def test_dayofweek_dtype(dtype, attr):
    # dayofweek/day_of_week/weekday are aliases. pandas returns int32 for the
    # numpy-backed variant but int64[pyarrow] for the ArrowDtype variant.
    ps = pd.Series(["2010-01-01", "2010-12-31", "2011-06-15"], dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = getattr(ps.dt, attr)
    got = getattr(gs.dt, attr)

    assert_eq(expect, got)
    if isinstance(dtype, pd.ArrowDtype):
        assert got.dtype == pd.ArrowDtype(pa.int64())
        assert expect.dtype == pd.ArrowDtype(pa.int64())
    else:
        assert got.dtype == np.dtype("int32")
        assert expect.dtype == np.dtype("int32")


@pytest.mark.parametrize(
    "attr, expected_dtype",
    [
        ("days", np.dtype("int64")),
        ("seconds", np.dtype("int32")),
        ("microseconds", np.dtype("int32")),
        ("nanoseconds", np.dtype("int32")),
    ],
)
def test_timedelta_components_dtype(attr, expected_dtype):
    # pandas returns int64 for .dt.days but int32 for the sub-day components
    # (seconds/microseconds/nanoseconds); cudf must match.
    ps = pd.Series([1000000, 200000, 3000000], dtype="timedelta64[ns]")
    gs = cudf.from_pandas(ps)

    expect = getattr(ps.dt, attr)
    got = getattr(gs.dt, attr)
    assert_eq(expect, got)
    assert got.dtype == expected_dtype
    # Sanity: our expectation tracks pandas' actual dtype.
    assert expect.dtype == expected_dtype


@pytest.mark.parametrize(
    "data", [[1, 2, 3, None], [], [100121, 1221312, 321312321, 1232131223]]
)
@pytest.mark.parametrize(
    "date_format",
    [
        "%d - %m",
        "%y/%H",
        "%Y",
        "%I - %M / %S",
        "%f",
        "%j",
        "%p",
        "%w",
        "%U",
        "%W",
        "%G",
        "%u",
        "%V",
        "%b",
        "%B",
        "%a",
        "%A",
        "%U_",
        "_%b",
        "%B*",
        "%a ",
        "%A1",
    ],
)
def test_datetime_strftime(data, datetime_types_as_str, date_format):
    gsr = cudf.Series(data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    expected = psr.dt.strftime(date_format=date_format)
    actual = gsr.dt.strftime(date_format=date_format)

    assert_eq(expected, actual)


@pytest.mark.parametrize("date_format", ["%c", "%x", "%X"])
def test_datetime_strftime_not_implemented_formats(date_format):
    gsr = cudf.Series([1, 2, 3], dtype="datetime64[ms]")

    with pytest.raises(NotImplementedError):
        gsr.dt.strftime(date_format=date_format)


@pytest.mark.parametrize(
    "unit",
    ["s", "ms", "us", "ns"],
)
@pytest.mark.parametrize(
    "date_format",
    [
        # %S in pyarrow includes the subsecond fraction at the timestamp's
        # resolution -- cudf must translate this so the output matches.
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%H:%M:%S no-subsec-after",
    ],
)
def test_datetime_strftime_arrow_dtype(unit, date_format):
    # ArrowDtype timestamps: pyarrow's "%S" carries the subsecond fraction
    # at the timestamp's resolution; cudf must produce the same output.
    # Use second-resolution input strings so each unit can hold them
    # losslessly.
    data = [
        "2023-01-02 03:04:05",
        None,
        "2024-12-31 23:59:59",
    ]
    ps = pd.Series(data, dtype=pd.ArrowDtype(pa.timestamp(unit)))
    gs = cudf.from_pandas(ps)

    # pandas' ArrowTemporalProperties.strftime takes the format positionally.
    expected = ps.dt.strftime(date_format)
    actual = gs.dt.strftime(date_format)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        pd.date_range("20010101", "20020215", freq="400h", name="times"),
        pd.date_range(
            "20010101", freq="243434324423423234ns", name="times", periods=10
        ),
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
        "weekday",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "unit",
        "tz",
    ],
)
def test_dt_series_datetime_fields(data, field):
    pd_data = pd.Series(data)
    gdf_data = cudf.Series(pd_data)
    base = getattr(pd_data.dt, field)
    test = getattr(gdf_data.dt, field)
    assert_eq(base, test, check_dtype=False)


@pytest.mark.parametrize("fmt", ["%Y-%m-%dT%H:%M%z", "%Y-%m-%dT%H:%M"])
def test_strftime_tz_aware(fmt):
    data = [datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)]
    cudf_pacific = cudf.Series(data).dt.tz_convert("US/Pacific")
    pd_pacific = pd.Series(data).dt.tz_convert("US/Pacific")
    result = cudf_pacific.dt.strftime(fmt)
    expected = pd_pacific.dt.strftime(fmt)
    assert_eq(result, expected)


def test_tz_localize(datetime_types_as_str, all_timezones):
    s = cudf.Series(cudf.date_range("2001-01-01", "2001-01-02", freq="1s"))
    s = s.astype(datetime_types_as_str)
    s = s.dt.tz_localize(all_timezones)
    assert isinstance(s.dtype, pd.DatetimeTZDtype)
    assert s.dtype.unit == datetime_types_as_str.removeprefix(
        "datetime64["
    ).removesuffix("]")
    assert str(s.dtype.tz) == all_timezones


def test_localize_ambiguous(request, datetime_types_as_str, all_timezones):
    request.applymarker(
        pytest.mark.xfail(
            condition=(all_timezones == "America/Metlakatla"),
            reason="https://www.timeanddate.com/news/time/metlakatla-quits-dst.html",
        )
    )
    s = cudf.Series(
        [
            "2018-11-04 00:30:00",
            "2018-11-04 01:00:00",
            "2018-11-04 01:30:00",
            "2018-11-04 02:00:00",
            None,
            "2018-11-04 02:30:00",
        ],
        dtype=datetime_types_as_str,
    )
    expect = s.to_pandas().dt.tz_localize(
        zoneinfo.ZoneInfo(all_timezones), ambiguous="NaT", nonexistent="NaT"
    )
    got = s.dt.tz_localize(all_timezones)
    assert_eq(expect, got)


def test_localize_nonexistent(request, datetime_types_as_str, all_timezones):
    request.applymarker(
        pytest.mark.xfail(
            condition=all_timezones == "America/Grand_Turk",
            reason="https://www.worldtimezone.com/dst_news/dst_news_turkscaicos03.html",
        )
    )
    s = cudf.Series(
        [
            "2018-03-11 01:30:00",
            "2018-03-11 02:00:00",
            "2018-03-11 02:30:00",
            "2018-03-11 03:00:00",
            None,
            "2018-03-11 03:30:00",
        ],
        dtype=datetime_types_as_str,
    )
    expect = s.to_pandas().dt.tz_localize(
        zoneinfo.ZoneInfo(all_timezones), ambiguous="NaT", nonexistent="NaT"
    )
    got = s.dt.tz_localize(all_timezones)
    assert_eq(expect, got)


def test_delocalize(datetime_types_as_str, limited_timezones):
    psr = pd.Series(
        pd.date_range("2001-01-01", "2001-01-02", freq="1s")
    ).astype(datetime_types_as_str)
    sr = cudf.from_pandas(psr)

    expect = psr.dt.tz_localize(limited_timezones).dt.tz_localize(None)
    got = sr.dt.tz_localize(limited_timezones).dt.tz_localize(None)
    assert_eq(expect, got)


def test_delocalize_naive():
    # delocalizing naive datetimes should be a no-op
    psr = pd.Series(["2001-01-01"], dtype="datetime64[ns]")
    sr = cudf.from_pandas(psr)

    expect = psr.dt.tz_localize(None)
    got = sr.dt.tz_localize(None)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "from_tz", ["Europe/London", "America/Chicago", "UTC"]
)
@pytest.mark.parametrize(
    "to_tz", ["Europe/London", "America/Chicago", "UTC", None]
)
def test_convert(from_tz, to_tz):
    from_tz = zoneinfo.ZoneInfo(from_tz)
    if to_tz is not None:
        to_tz = zoneinfo.ZoneInfo(to_tz)
    ps = pd.Series(pd.date_range("2023-01-01", periods=3, freq="h"))
    gs = cudf.from_pandas(ps)
    ps = ps.dt.tz_localize(from_tz)
    gs = gs.dt.tz_localize(from_tz)
    expect = ps.dt.tz_convert(to_tz)
    got = gs.dt.tz_convert(to_tz)
    assert_eq(expect, got)


@pytest.mark.parametrize("unit", ["us", "ns"])
@pytest.mark.parametrize("to_tz", ["US/Eastern", "Europe/London", "UTC", None])
def test_convert_arrow_dtype(unit, to_tz):
    # tz_convert on an ArrowDtype timestamp must keep the result as
    # ArrowDtype (timestamp[unit, tz=...][pyarrow]) rather than falling
    # back to DatetimeTZDtype.
    # cudf cannot construct a tz-aware ArrowDtype Series via from_pandas,
    # so build it by localizing a naive ArrowDtype Series first.
    ps = pd.Series(
        ["2023-01-01 03:00", None, "2023-06-15 12:00"],
        dtype=pd.ArrowDtype(pa.timestamp(unit)),
    ).dt.tz_localize("US/Pacific")
    gs = cudf.from_pandas(
        pd.Series(
            ["2023-01-01 03:00", None, "2023-06-15 12:00"],
            dtype=pd.ArrowDtype(pa.timestamp(unit)),
        )
    ).dt.tz_localize("US/Pacific")

    expect = ps.dt.tz_convert(to_tz)
    got = gs.dt.tz_convert(to_tz)
    assert_eq(expect, got)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_localize_arrow_dtype(unit):
    # tz_localize on an ArrowDtype timestamp must keep the result as
    # ArrowDtype rather than DatetimeTZDtype.
    ps = pd.Series(
        ["2023-01-02 03:00", None, "2023-06-15 12:00"],
        dtype=pd.ArrowDtype(pa.timestamp(unit)),
    )
    gs = cudf.from_pandas(ps)

    expect = ps.dt.tz_localize("US/Pacific")
    got = gs.dt.tz_localize("US/Pacific")
    assert_eq(expect, got)


def test_convert_from_naive():
    gs = cudf.Series(cudf.date_range("2023-01-01", periods=3, freq="h"))
    with pytest.raises(TypeError):
        gs.dt.tz_convert("America/New_York")


@pytest.mark.parametrize(
    "data,original_timezone,target_timezone",
    [
        # DST transition:
        (["2023-03-12 01:30:00"], "America/New_York", "America/Los_Angeles"),
        # crossing the international date line:
        (["2023-05-17 23:30:00"], "Pacific/Auckland", "America/Los_Angeles"),
        # timezone with non-integer offset:
        (["2023-05-17 12:00:00"], "Asia/Kolkata", "Australia/Eucla"),
        # timezone with negative offset:
        (["2023-05-17 09:00:00"], "America/Los_Angeles", "Pacific/Auckland"),
        # conversion across multiple days:
        (["2023-05-16 23:30:00"], "America/New_York", "Asia/Kolkata"),
        # timezone with half-hour offset:
        (["2023-05-17 12:00:00"], "Asia/Kolkata", "Australia/Adelaide"),
        # timezone conversion with a timestamp in the future:
        (["2025-01-01 00:00:00"], "America/New_York", "Europe/London"),
        # timezone conversion with a timestamp in the past:
        (["2000-01-01 12:00:00"], "Europe/Paris", "America/Los_Angeles"),
        # timezone conversion with a timestamp at midnight:
        (["2023-05-17 00:00:00"], "Asia/Tokyo", "Europe/Paris"),
    ],
)
def test_convert_edge_cases(data, original_timezone, target_timezone):
    original_timezone = zoneinfo.ZoneInfo(original_timezone)
    target_timezone = zoneinfo.ZoneInfo(target_timezone)
    ps = pd.Series(data, dtype="datetime64[s]").dt.tz_localize(
        original_timezone
    )
    gs = cudf.Series(data, dtype="datetime64[s]").dt.tz_localize(
        original_timezone
    )
    expect = ps.dt.tz_convert(target_timezone)
    got = gs.dt.tz_convert(target_timezone)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        pd.date_range("2020-01-01", periods=10, freq="D"),
        pd.date_range("1990-01-01", periods=100, freq="ME"),
        pd.to_datetime(["2023-12-25", "2024-01-01", "2024-06-15", None]),
    ],
)
@pytest.mark.parametrize(
    "component",
    [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
        "weekday",
    ],
)
def test_dt_component_dtype_pandas_compat(data, component):
    ps = pd.Series(data)
    gs = cudf.Series(data)

    expect = getattr(ps.dt, component)
    got = getattr(gs.dt, component)
    assert_eq(expect, got, check_dtype=True)


@pytest.mark.parametrize(
    "unit",
    ["s", "ms", "us", "ns"],
)
@pytest.mark.parametrize(
    "component",
    [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
        "weekday",
        "day_of_year",
        "dayofyear",
        "days_in_month",
        "daysinmonth",
        "is_leap_year",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
    ],
)
def test_dt_component_arrow_dtype(unit, component):
    # Verify dt accessor results on ArrowDtype timestamps match pandas
    # in both dtype (e.g. int64[pyarrow], bool[pyarrow]) and values
    # (including null preservation for nullable extension dtype).
    # Use only second-resolution timestamps so the fixture data is valid
    # for every unit (pandas can't losslessly downcast ns -> s, etc).
    data = [
        "2023-01-02 03:04:05",
        None,
        "2024-02-29 23:59:59",
        "1969-12-31 12:00:00",
    ]
    ps = pd.Series(data, dtype=pd.ArrowDtype(pa.timestamp(unit)))
    gs = cudf.from_pandas(ps)

    expect = getattr(ps.dt, component)
    got = getattr(gs.dt, component)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtype",
    [
        "datetime64[ns]",
        "datetime64[us]",
        pd.ArrowDtype(pa.timestamp("ns")),
        pd.ArrowDtype(pa.timestamp("us")),
    ],
)
def test_to_pydatetime(dtype):
    # Verify Series.dt.to_pydatetime returns a Series of datetime.datetime
    # objects matching pandas, for both numpy datetime and ArrowDtype.
    data = [
        datetime.datetime(2022, 1, 1),
        datetime.datetime(2023, 6, 15, 12, 30, 45),
    ]
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.to_pydatetime()
    got = gs.dt.to_pydatetime()
    assert_eq(expect, got)
    assert all(type(v) is datetime.datetime for v in got)


@pytest.mark.parametrize(
    "dtype",
    [
        "timedelta64[ns]",
        "timedelta64[us]",
        pd.ArrowDtype(pa.duration("ns")),
        pd.ArrowDtype(pa.duration("us")),
    ],
)
def test_to_pytimedelta(dtype):
    # Verify Series.dt.to_pytimedelta returns an ndarray of
    # datetime.timedelta objects matching pandas.
    data = [
        datetime.timedelta(days=1, seconds=2, microseconds=3),
        datetime.timedelta(days=2, seconds=4, microseconds=5),
    ]
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    # pandas emits a Pandas4Warning here; ignore it because the function is
    # being deprecated upstream and behaviour parity is what we care about.
    with pytest.warns((DeprecationWarning, FutureWarning, Warning)):
        expect = ps.dt.to_pytimedelta()
    with pytest.warns((DeprecationWarning, FutureWarning, Warning)):
        got = gs.dt.to_pytimedelta()
    np.testing.assert_array_equal(expect, got)
    assert all(type(v) is datetime.timedelta for v in got)
