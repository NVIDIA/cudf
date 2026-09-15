# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime
import zoneinfo
from typing import Literal, cast

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine


@pytest.fixture(params=["ms", "us", "ns"])
def units(request):
    return request.param


@pytest.fixture
def sample_datetimes(units):
    return pl.Series(
        [
            datetime.datetime(2020, 1, 15, 12, 0, 0),
            datetime.datetime(2020, 6, 15, 12, 30, 15),
            datetime.datetime(2019, 11, 15, 12, 45, 30),
            datetime.datetime(1969, 6, 15, 12, 0, 0),
            None,
        ],
        dtype=pl.Datetime(cast("Literal['ms', 'us', 'ns']", units)),
    )


@pytest.fixture
def utc_frame(sample_datetimes):
    return pl.LazyFrame({"a": sample_datetimes.dt.replace_time_zone("UTC")})


@pytest.fixture
def naive_frame(sample_datetimes):
    return pl.LazyFrame({"a": sample_datetimes})


@pytest.mark.parametrize(
    "target", ["Europe/London", "US/Pacific", "Asia/Kolkata", "UTC", "Etc/GMT"]
)
def test_convert_time_zone_from_utc(engine, utc_frame, target):
    q = utc_frame.select(pl.col("a").dt.convert_time_zone(target))
    assert_gpu_result_equal(q, engine=engine)


def test_convert_time_zone_from_naive(engine, naive_frame):
    q = naive_frame.select(pl.col("a").dt.convert_time_zone("Europe/London"))
    assert_gpu_result_equal(q, engine=engine)


def test_convert_time_zone_between_zones(engine, utc_frame):
    q = utc_frame.with_columns(
        pl.col("a").dt.convert_time_zone("Europe/London")
    ).select(pl.col("a").dt.convert_time_zone("US/Pacific"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "target",
    ["Europe/Amsterdam", "US/Pacific", "Asia/Kolkata", "Etc/GMT-5", "UTC", None],
)
def test_replace_time_zone_from_naive(engine, naive_frame, target):
    q = naive_frame.select(pl.col("a").dt.replace_time_zone(target))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("target", ["Europe/Amsterdam", "US/Pacific", "UTC", None])
def test_replace_time_zone_from_aware(engine, utc_frame, target):
    q = utc_frame.with_columns(
        pl.col("a").dt.convert_time_zone("Europe/London")
    ).select(pl.col("a").dt.replace_time_zone(target))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_time_zone_from_empty_table_zone(engine, utc_frame):
    q = utc_frame.with_columns(pl.col("a").dt.convert_time_zone("Etc/GMT")).select(
        pl.col("a").dt.replace_time_zone("Etc/GMT")
    )
    assert_gpu_result_equal(q, engine=engine)


def test_replace_time_zone_to_empty_table_zone(engine, naive_frame):
    q = naive_frame.select(pl.col("a").dt.replace_time_zone("Etc/GMT"))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_time_zone_from_empty_table_zone_to_other(engine, utc_frame):
    q = utc_frame.with_columns(pl.col("a").dt.convert_time_zone("Etc/GMT")).select(
        pl.col("a").dt.replace_time_zone("Europe/Amsterdam")
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("ambiguous", ["earliest", "latest", "null"])
def test_replace_time_zone_ambiguous(engine, ambiguous):
    q = pl.LazyFrame(
        {
            "a": [
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2020, 1, 1),
                None,
            ]
        }
    ).select(pl.col("a").dt.replace_time_zone("Europe/Amsterdam", ambiguous=ambiguous))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_time_zone_non_existent_null(engine):
    q = pl.LazyFrame(
        {
            "a": [
                datetime.datetime(2018, 3, 25, 2, 30),
                datetime.datetime(2020, 1, 1),
                None,
            ]
        }
    ).select(pl.col("a").dt.replace_time_zone("Europe/Amsterdam", non_existent="null"))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_time_zone_ambiguous_per_row(engine):
    q = pl.LazyFrame(
        {
            "a": [
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2020, 1, 1),
            ],
            "ambiguous": ["earliest", "latest", "null", "raise"],
        }
    ).select(
        pl.col("a").dt.replace_time_zone(
            "Europe/Amsterdam", ambiguous=pl.col("ambiguous")
        )
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("target", ["Europe/Amsterdam", "Etc/GMT"])
def test_replace_time_zone_nullable_ambiguous_per_row(engine, target):
    q = pl.LazyFrame(
        {
            "a": [
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2020, 1, 2),
            ],
            "ambiguous": [None, "earliest", "null"],
        }
    ).select(pl.col("a").dt.replace_time_zone(target, ambiguous=pl.col("ambiguous")))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("target", ["Europe/Amsterdam", "Etc/GMT"])
def test_replace_time_zone_invalid_ambiguous_scalar_raises(engine, target):
    q = pl.LazyFrame({"a": [datetime.datetime(2020, 1, 1)]}).select(
        pl.col("a").dt.replace_time_zone(target, ambiguous="bogus")
    )
    assert_ir_translation_raises(q, engine, pl.exceptions.InvalidOperationError)


@pytest.mark.parametrize("target", ["Europe/Amsterdam", "Etc/GMT"])
def test_replace_time_zone_invalid_ambiguous_per_row_raises(engine, target):
    q = pl.LazyFrame(
        {
            "a": [datetime.datetime(2020, 1, 1)],
            "ambiguous": ["bogus"],
        }
    ).select(pl.col("a").dt.replace_time_zone(target, ambiguous=pl.col("ambiguous")))
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)


@pytest.mark.parametrize("source", ["UTC", None])
@pytest.mark.parametrize("target", ["Europe/Amsterdam", "Etc/GMT", "UTC", None])
def test_replace_time_zone_non_string_ambiguous_raises(engine, source, target):
    q = pl.LazyFrame(
        {
            "a": [datetime.datetime(2018, 10, 28, 2, 30)],
            "int_col": [1],
        }
    ).select(
        pl.col("a")
        .dt.replace_time_zone(source)
        .dt.replace_time_zone(target, ambiguous=pl.col("int_col"))
    )
    assert_ir_translation_raises(q, engine, pl.exceptions.SchemaError)


@pytest.mark.parametrize("target", ["Europe/Amsterdam", "UTC"])
def test_replace_time_zone_null_ambiguous_literal_raises(engine, target):
    q = pl.LazyFrame({"a": [datetime.datetime(2018, 10, 28, 2, 30)]}).select(
        pl.col("a").dt.replace_time_zone(target, ambiguous=pl.lit(None))
    )
    assert_ir_translation_raises(q, engine, pl.exceptions.SchemaError)


def test_replace_time_zone_same_zone_ambiguous_instant(engine):
    q = (
        pl.LazyFrame({"a": [datetime.datetime(2018, 10, 28, 2, 30)]})
        .with_columns(
            pl.col("a").dt.replace_time_zone("Europe/Amsterdam", ambiguous="earliest")
        )
        .select(pl.col("a").dt.replace_time_zone("Europe/Amsterdam"))
    )
    assert_gpu_result_equal(q, engine=engine)


def test_replace_time_zone_ambiguous_raises(engine):
    q = pl.LazyFrame({"a": [datetime.datetime(2018, 10, 28, 2, 30)]}).select(
        pl.col("a").dt.replace_time_zone("Europe/Amsterdam")
    )
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.ComputeError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ComputeError):
            q.collect(engine=engine)


def test_replace_time_zone_non_existent_raises(engine):
    q = pl.LazyFrame({"a": [datetime.datetime(2018, 3, 25, 2, 30)]}).select(
        pl.col("a").dt.replace_time_zone("Europe/Amsterdam")
    )
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.ComputeError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ComputeError):
            q.collect(engine=engine)


def test_replace_time_zone_ambiguous_per_row_raises(engine):
    q = pl.LazyFrame(
        {
            "a": [
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2020, 1, 1),
            ],
            "ambiguous": ["raise", "raise"],
        }
    ).select(
        pl.col("a").dt.replace_time_zone(
            "Europe/Amsterdam", ambiguous=pl.col("ambiguous")
        )
    )
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.ComputeError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ComputeError):
            q.collect(engine=engine)


def test_replace_time_zone_unknown_zone_raises(engine, naive_frame, monkeypatch):
    monkeypatch.setattr(zoneinfo, "TZPATH", ())
    q = naive_frame.select(pl.col("a").dt.replace_time_zone("Europe/Amsterdam"))
    assert_ir_translation_raises(q, engine, NotImplementedError)
