# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture(params=[pl.Int64, pl.Int32, pl.UInt32, pl.Float64, pl.Float32])
def dtype(request):
    return request.param


@pytest.fixture
def df(dtype, with_nulls):
    values = [1, 2, 3, 4, 5, 23, 7, 8, 3, 0, 11, 6]
    if with_nulls:
        values = [None if i % 4 == 0 else v for i, v in enumerate(values)]
    return pl.LazyFrame({"a": pl.Series("a", values, dtype=dtype)})


@pytest.mark.parametrize("bias", [True, False])
def test_skew(df, bias, engine):
    q = df.select(pl.col("a").skew(bias=bias))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize("fisher", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_kurtosis(df, fisher, bias, engine):
    q = df.select(pl.col("a").kurtosis(fisher=fisher, bias=bias))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_skew_kurtosis_bool(engine):
    df = pl.LazyFrame({"a": [True, False, True, True, False, True, False, True]})
    q = df.select(
        pl.col("a").skew().alias("s"),
        pl.col("a").kurtosis().alias("k"),
    )
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_skew_kurtosis_constant(engine):
    df = pl.LazyFrame({"a": [5.0] * 8})
    q = df.select(
        pl.col("a").skew().alias("s"),
        pl.col("a").kurtosis().alias("k"),
    )
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_skew_kurtosis_large_offset(engine):
    df = pl.LazyFrame({"a": [1e12 + value for value in range(8)]})
    q = df.select(
        pl.col("a").skew().alias("s"),
        pl.col("a").kurtosis().alias("k"),
    )
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.1288e103],
        [0.0, 0.0, 1.1972e103],
        [6.0383e169],
        [1e200, -1e200, 1e200, -1e200],
    ],
)
def test_skew_kurtosis_overflow(engine, values):
    df = pl.LazyFrame({"a": pl.Series("a", values, dtype=pl.Float64)})
    q = df.select(
        pl.col("a").skew().alias("s"),
        pl.col("a").skew(bias=False).alias("sb"),
        pl.col("a").kurtosis().alias("k"),
        pl.col("a").kurtosis(bias=False).alias("kb"),
    )
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize("bias", [True, False])
def test_skew_small_sample(engine, bias):
    df = pl.LazyFrame({"a": [1.0, 2.0]})
    q = df.select(pl.col("a").skew(bias=bias))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize("bias", [True, False])
def test_kurtosis_small_sample(engine, bias):
    df = pl.LazyFrame({"a": [1.0, 2.0, 3.0]})
    q = df.select(pl.col("a").kurtosis(bias=bias))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_skew_kurtosis_empty(engine):
    df = pl.LazyFrame({"a": []}, schema={"a": pl.Float64})
    q = df.select(
        pl.col("a").skew().alias("s"),
        pl.col("a").kurtosis().alias("k"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_skew_kurtosis_all_null(engine):
    df = pl.LazyFrame({"a": [None] * 8}, schema={"a": pl.Float64})
    q = df.select(
        pl.col("a").skew().alias("s"),
        pl.col("a").kurtosis().alias("k"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_skew_kurtosis_groupby_unsupported(engine):
    df = pl.LazyFrame({"g": [1, 1, 2, 2], "a": [1.0, 2.0, 3.0, 4.0]})
    assert_ir_translation_raises(
        df.group_by("g").agg(pl.col("a").skew()), engine, NotImplementedError
    )
    assert_ir_translation_raises(
        df.group_by("g").agg(pl.col("a").kurtosis()), engine, NotImplementedError
    )


@pytest.mark.parametrize(
    "data",
    [
        pl.Series(
            [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 5), date(2020, 1, 10)],
            dtype=pl.Date,
        ),
        pl.Series(
            [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 5),
                datetime(2020, 1, 10),
            ],
            dtype=pl.Datetime("us"),
        ),
        pl.Series(
            [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                None,
                datetime(2020, 1, 5),
                datetime(2020, 1, 10),
            ],
            dtype=pl.Datetime("ms"),
        ),
        pl.Series(
            [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 5),
                datetime(2020, 1, 10),
            ],
            dtype=pl.Datetime("ns"),
        ),
        pl.Series(
            [
                timedelta(days=1),
                timedelta(days=2),
                None,
                timedelta(days=5),
                timedelta(days=10),
            ],
            dtype=pl.Duration("us"),
        ),
    ],
)
def test_skew_kurtosis_temporal(engine, data):
    df = pl.LazyFrame({"a": data})
    q = df.select(
        pl.col("a").skew().alias("s"),
        pl.col("a").kurtosis().alias("k"),
    )
    assert_gpu_result_equal(q, engine=engine, check_exact=False)
