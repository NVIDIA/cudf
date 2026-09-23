# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime
from operator import methodcaller

import pytest

import polars as pl

from cudf_polars.dsl.expr import TemporalFunction
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Date(),
        pl.Datetime("ms"),
        pl.Datetime("us"),
        pl.Datetime("ns"),
        pl.Datetime("ms", time_zone="UTC"),
        pl.Datetime("us", time_zone="Europe/Dublin"),
        pl.Datetime("ns", time_zone="US/Pacific"),
        pl.Duration("ms"),
        pl.Duration("us"),
        pl.Duration("ns"),
    ],
    ids=repr,
)
def test_datetime_dataframe_scan(engine: pl.GPUEngine, dtype):
    ldf = pl.DataFrame(
        {
            "a": pl.Series([1, 2, 3, 4, 5, 6, 7], dtype=dtype),
            "b": pl.Series([3, 4, 5, 6, 7, 8, 9], dtype=pl.UInt16),
        }
    ).lazy()

    query = ldf.select(pl.col("b"), pl.col("a"))
    assert_gpu_result_equal(query, engine=engine)


datetime_extract_fields = [
    "year",
    "month",
    "day",
    "weekday",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
]

duration_extract_fields = [
    "total_seconds",
    "total_milliseconds",
    "total_microseconds",
    "total_nanoseconds",
    "total_days",
    "total_hours",
    "total_minutes",
]


@pytest.fixture(
    ids=datetime_extract_fields,
    params=[methodcaller(f) for f in datetime_extract_fields],
)
def field(request):
    return request.param


def test_datetime_extract(engine: pl.GPUEngine, field):
    ldf = pl.LazyFrame(
        {
            "datetimes": pl.datetime_range(
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2021, 12, 30),
                "3mo14h15s11ms33us999ns",
                eager=True,
            )
        }
    )

    q = ldf.select(field(pl.col("datetimes").dt))

    assert_gpu_result_equal(q, engine=engine)


def test_datetime_extra_unsupported(engine: pl.GPUEngine, monkeypatch):
    ldf = pl.LazyFrame(
        {
            "datetimes": pl.datetime_range(
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2021, 12, 30),
                "3mo14h15s11ms33us999ns",
                eager=True,
            )
        }
    )

    def unsupported_name_setter(self, value):
        pass

    def unsupported_name_getter(self):
        return "unsupported"

    monkeypatch.setattr(
        TemporalFunction,
        "name",
        property(unsupported_name_getter, unsupported_name_setter),
    )

    q = ldf.select(pl.col("datetimes").dt.nanosecond())

    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "field",
    [
        methodcaller("year"),
        methodcaller("month"),
        methodcaller("day"),
        methodcaller("weekday"),
    ],
)
def test_date_extract(engine: pl.GPUEngine, field):
    ldf = pl.LazyFrame(
        {
            "dates": [
                datetime.date(2024, 1, 1),
                datetime.date(2024, 10, 11),
            ]
        }
    )

    ldf = pl.LazyFrame(
        {"dates": [datetime.date(2024, 1, 1), datetime.date(2024, 10, 11)]}
    )

    q = ldf.select(field(pl.col("dates").dt))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("format", ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", ""])
def test_strftime_timestamp(engine: pl.GPUEngine, format):
    ldf = pl.LazyFrame(
        {
            "dates": [
                datetime.date(2024, 1, 1),
                datetime.date(2024, 10, 11),
            ]
        }
    )

    q = ldf.select(pl.col("dates").dt.strftime(format))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("format", ["iso", "polars"])
def test_strftime_duration(engine: pl.GPUEngine, format):
    ldf = pl.LazyFrame(
        {
            "durations": [
                datetime.timedelta(days=1, seconds=3600),
                datetime.timedelta(days=2, seconds=7200),
            ]
        }
    )

    q = ldf.select(pl.col("durations").dt.strftime(format))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("field", duration_extract_fields)
@pytest.mark.parametrize(
    "dtype", [pl.Duration("ms"), pl.Duration("us"), pl.Duration("ns")]
)
def test_duration_total_component_extract(engine: pl.GPUEngine, field, dtype):
    ldf = pl.LazyFrame(
        {
            "durations": pl.Series(
                [
                    0,
                    1,
                    15,
                    -1500,
                    1000,
                    1111,
                    1500,
                    11111,
                    -134234534,
                    134234534,
                    # values beyond float64's exact-integer range to guard
                    # against precision loss in the unit conversion
                    5857593848682946,
                    -5857593848682946,
                ],
                dtype=dtype,
            ),
        }
    )
    q = ldf.select(getattr(pl.col("durations").dt, field)())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("method", ["century", "millennium"])
@pytest.mark.parametrize(
    "dtype", [pl.Date(), pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
def test_century_millennium(engine: pl.GPUEngine, method, dtype):
    data = pl.Series(
        [
            datetime.date(1897, 5, 7),
            datetime.date(1900, 12, 31),
            datetime.date(1901, 1, 1),
            datetime.date(2000, 1, 1),
            datetime.date(2001, 7, 5),
            None,
        ],
        dtype=pl.Date(),
    ).cast(dtype)
    ldf = pl.LazyFrame({"dates": data})
    q = ldf.select(getattr(pl.col("dates").dt, method)())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("method", ["century", "millennium"])
@pytest.mark.parametrize(
    "days",
    [
        # ``Date`` supports a much wider year range than ``Datetime``; include
        # pre-year-1 offsets to exercise the floor-division branch of the
        # century/millennium formula.
        [-1_000_000, -800_000, -365, 0],
        [364_000, 376_000],  # years ~2966 and ~3000
    ],
)
def test_century_millennium_date_extreme_years(engine: pl.GPUEngine, method, days):
    dates = pl.Series(days, dtype=pl.Int32).cast(pl.Date())
    ldf = pl.LazyFrame({"dates": dates})
    q = ldf.select(getattr(pl.col("dates").dt, method)())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype", [pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
def test_datetime_date(engine: pl.GPUEngine, dtype):
    data = pl.Series(
        [
            datetime.datetime(1978, 1, 1, 1, 1, 1),
            datetime.datetime(1969, 12, 31, 23, 59, 59),  # pre-epoch (floors down)
            datetime.datetime(2024, 10, 13, 5, 30, 14, 500_000),
            datetime.datetime(2065, 1, 1, 10, 20, 30, 60_000),
            None,
        ],
        dtype=dtype,
    )
    ldf = pl.LazyFrame({"datetimes": data})
    q = ldf.select(pl.col("datetimes").dt.date())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype", [pl.Date(), pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
def test_datetime_month_start(engine: pl.GPUEngine, dtype):
    data = pl.DataFrame(
        {
            "dates": pl.Series(
                [
                    datetime.date(2024, 1, 1),
                    datetime.date(2024, 10, 11),
                    datetime.date(2024, 10, 31),
                    datetime.date(2000, 2, 1),
                    datetime.date(2000, 2, 29),
                    datetime.date(2000, 3, 1),
                ],
                dtype=dtype,
            )
        }
    ).lazy()

    q = data.select(pl.col("dates").dt.month_start())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype", [pl.Date(), pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
def test_datetime_month_end(engine: pl.GPUEngine, dtype):
    data = pl.DataFrame(
        {
            "dates": pl.Series(
                [
                    datetime.date(2024, 1, 1),
                    datetime.date(2024, 10, 11),
                    datetime.date(2024, 10, 31),
                    datetime.date(2000, 2, 1),
                    datetime.date(2000, 2, 29),
                    datetime.date(2000, 3, 1),
                ],
                dtype=dtype,
            )
        }
    ).lazy()

    q = data.select(pl.col("dates").dt.month_end())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "data",
    [
        [
            datetime.date(2000, 1, 1),
            datetime.date(2001, 1, 1),
            datetime.date(2004, 1, 1),
        ],
        [],
    ],
)
@pytest.mark.parametrize(
    "dtype", [pl.Date(), pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
def test_is_leap_year(engine: pl.GPUEngine, data, dtype):
    ldf = pl.LazyFrame({"dates": pl.Series(data, dtype=dtype)})

    q = ldf.select(pl.col("dates").dt.is_leap_year())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "start_date, end_date",
    [
        (datetime.date(2001, 12, 22), datetime.date(2001, 12, 25)),
        (datetime.date(2000, 2, 27), datetime.date(2000, 3, 1)),  # Leap year transition
        (datetime.date(1999, 12, 31), datetime.date(2000, 1, 2)),
        (datetime.date(2020, 2, 28), datetime.date(2020, 3, 1)),
        (datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)),
    ],
)
def test_ordinal_day(engine: pl.GPUEngine, start_date, end_date):
    df = pl.DataFrame({"date": pl.date_range(start_date, end_date, eager=True)}).lazy()

    q = df.with_columns(
        pl.col("date").dt.ordinal_day().alias("day_of_year"),
    )

    assert_gpu_result_equal(q, engine=engine)


def test_isoweek(engine: pl.GPUEngine):
    df = pl.DataFrame(
        {
            "date": [
                datetime.date(1999, 12, 27),
                datetime.date(2000, 1, 3),
                datetime.date(2000, 6, 15),
                datetime.date(2000, 12, 31),
                datetime.date(2001, 1, 1),
                datetime.date(2001, 12, 30),
                datetime.date(2002, 1, 1),
            ]
        }
    ).lazy()

    q = df.with_columns(pl.col("date").dt.week().alias("isoweek"))

    assert_gpu_result_equal(q, engine=engine)


def test_isoyear(engine: pl.GPUEngine):
    df = pl.DataFrame(
        {
            "date": [
                datetime.date(1999, 12, 27),
                datetime.date(2000, 1, 3),
                datetime.date(2000, 2, 29),
                datetime.date(2000, 6, 15),
                datetime.date(2000, 12, 31),
                datetime.date(2001, 1, 1),
                datetime.date(2001, 12, 30),
                datetime.date(2002, 1, 1),
            ]
        }
    ).lazy()

    q = df.with_columns(pl.col("date").dt.iso_year().alias("isoyear"))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype",
    [pl.Date(), pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")],
    ids=repr,
)
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns", "s", "d"])
def test_epoch(engine: pl.GPUEngine, dtype, time_unit):
    ldf = pl.LazyFrame(
        {
            "datetimes": pl.Series(
                [
                    datetime.datetime(2001, 1, 1),
                    datetime.datetime(2001, 1, 2, 12, 30, 15),
                    datetime.datetime(2020, 2, 29, 23, 59, 59),
                    datetime.datetime(2024, 12, 31, 23, 59, 59),
                ],
                dtype=dtype,
            )
        }
    )

    q = ldf.select(pl.col("datetimes").dt.epoch(time_unit))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype", [pl.Date(), pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_datetime_cast_time_unit_datetime(engine: pl.GPUEngine, dtype, time_unit):
    sr = pl.Series(
        "date",
        [
            datetime.datetime(1970, 1, 1, 0, 0, 0),
            datetime.datetime(1999, 12, 31, 23, 59, 59),
            datetime.datetime(2001, 1, 1, 12, 0, 0),
            datetime.datetime(2020, 2, 29, 23, 59, 59),
            datetime.datetime(2024, 12, 31, 23, 59, 59, 999999),
        ],
        dtype=dtype,
    )
    df = pl.DataFrame({"date": sr}).lazy()

    q = df.select(pl.col("date").dt.cast_time_unit(time_unit).alias("time_unit_ms"))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype", [pl.Duration("ms"), pl.Duration("us"), pl.Duration("ns")]
)
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_datetime_cast_time_unit_duration(engine: pl.GPUEngine, dtype, time_unit):
    sr = pl.Series(
        "date",
        [
            datetime.timedelta(days=1),
            datetime.timedelta(days=2),
            datetime.timedelta(days=3),
            datetime.timedelta(days=4),
            datetime.timedelta(days=5),
        ],
        dtype=dtype,
    )
    df = pl.DataFrame({"date": sr}).lazy()

    q = df.select(pl.col("date").dt.cast_time_unit(time_unit).alias("time_unit_ms"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "datetime_dtype",
    [
        pl.Datetime("ms"),
        pl.Datetime("us"),
        pl.Datetime("ns"),
    ],
)
@pytest.mark.parametrize(
    "integer_dtype",
    [
        pl.Int64(),
        pl.UInt64(),
        pl.Int32(),
        pl.UInt32(),
        pl.Int16(),
        pl.UInt16(),
        pl.Int8(),
        pl.UInt8(),
    ],
)
def test_datetime_from_integer(engine: pl.GPUEngine, datetime_dtype, integer_dtype):
    values = [
        0,
        1,
        100,
        pl.select(integer_dtype.max()).item(),
        pl.select(integer_dtype.min()).item(),
    ]
    df = pl.LazyFrame({"data": pl.Series(values, dtype=integer_dtype)})
    q = df.select(pl.col("data").cast(datetime_dtype).alias("datetime_from_int"))
    if integer_dtype == pl.UInt64():
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect()
        with pytest.raises(pl.exceptions.ComputeError):
            q.collect(engine=engine)
    else:
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype", [pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
# ``every`` finer than a column's storage unit panics in CPU polars, so only
# use frequencies that are coarser-or-equal to every tested resolution.
@pytest.mark.parametrize("every", ["1ms", "1s", "1m", "1h", "1d"])
def test_datetime_round(engine: pl.GPUEngine, dtype, every):
    # Use an irregular step so no timestamp lands exactly on a half-way point:
    # libcudf rounds half-to-even while polars rounds half-away, so only the
    # exact-tie behaviour differs.
    ldf = pl.LazyFrame(
        {
            "datetimes": pl.datetime_range(
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2020, 1, 2),
                "3h14m15s11ms33us999ns",
                eager=True,
            ).cast(dtype)
        }
    )

    q = ldf.select(pl.col("datetimes").dt.round(every))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("every", ["30m", "1mo"])
def test_datetime_round_unsupported(engine: pl.GPUEngine, every: str):
    ldf = pl.LazyFrame(
        {
            "datetimes": pl.datetime_range(
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2020, 1, 2),
                "30m",
                eager=True,
            )
        }
    )

    q = ldf.select(pl.col("datetimes").dt.round(every))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "dtype", [pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
@pytest.mark.parametrize("every", ["1ns", "1us", "1ms", "1s", "1m", "1h", "1d"])
def test_datetime_truncate(engine: pl.GPUEngine, dtype, every):
    ldf = pl.LazyFrame(
        {
            "datetimes": pl.datetime_range(
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2020, 1, 2),
                "3h14m15s11ms33us999ns",
                eager=True,
            ).cast(dtype)
        }
    )

    q = ldf.select(pl.col("datetimes").dt.truncate(every))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("every", ["30m", "1mo"])
def test_datetime_truncate_unsupported(engine: pl.GPUEngine, every: str):
    ldf = pl.LazyFrame(
        {
            "datetimes": pl.datetime_range(
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2020, 1, 2),
                "30m",
                eager=True,
            )
        }
    )

    q = ldf.select(pl.col("datetimes").dt.truncate(every))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "dtype", [pl.Date(), pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
def test_datetime_quarter(engine: pl.GPUEngine, dtype):
    data = pl.Series(
        [
            datetime.date(2001, 1, 1),
            datetime.date(2001, 3, 31),
            datetime.date(2001, 4, 1),
            datetime.date(2001, 6, 30),
            datetime.date(2001, 9, 15),
            datetime.date(2001, 12, 27),
            None,
        ],
        dtype=pl.Date(),
    ).cast(dtype)
    ldf = pl.LazyFrame({"dates": data})
    q = ldf.select(pl.col("dates").dt.quarter())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype", [pl.Date(), pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
def test_datetime_days_in_month(engine: pl.GPUEngine, dtype):
    data = pl.Series(
        [
            datetime.date(2001, 1, 15),
            datetime.date(2001, 2, 15),  # non-leap February
            datetime.date(2000, 2, 15),  # leap February
            datetime.date(2001, 4, 15),
            datetime.date(2001, 12, 31),
            None,
        ],
        dtype=pl.Date(),
    ).cast(dtype)
    ldf = pl.LazyFrame({"dates": data})
    q = ldf.select(pl.col("dates").dt.days_in_month())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "datetime_dtype",
    [
        pl.Datetime("ms"),
        pl.Datetime("us"),
        pl.Datetime("ns"),
    ],
)
@pytest.mark.parametrize(
    "integer_dtype",
    [
        pl.Int64(),
        pytest.param(
            pl.UInt64(), marks=pytest.mark.xfail(reason="INT64 can not fit max(UINT64)")
        ),
        pl.Int32(),
        pl.UInt32(),
        pl.Int16(),
        pl.UInt16(),
        pl.Int8(),
        pl.UInt8(),
    ],
)
def test_integer_from_datetime(engine: pl.GPUEngine, datetime_dtype, integer_dtype):
    values = [
        0,
        1,
        100,
        pl.select(integer_dtype.max()).item(),
        pl.select(integer_dtype.min()).item(),
    ]
    df = pl.LazyFrame({"data": pl.Series(values, dtype=datetime_dtype)})
    q = df.select(pl.col("data").cast(integer_dtype).alias("int_from_datetime"))
    assert_gpu_result_equal(q, engine=engine)
