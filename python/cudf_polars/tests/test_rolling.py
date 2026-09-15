# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine
from cudf_polars.utils.versions import POLARS_VERSION_LT_136


@pytest.fixture(params=[False, True], ids=["nosort", "sort"])
def presort(request):
    return request.param


@pytest.fixture
def df(presort):
    dates = pl.Series(
        [
            datetime.strptime("2020-01-01 13:45:48", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-01 16:42:13", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-01 16:45:09", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-02 18:12:48", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-03 19:45:32", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-08 23:16:43", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-10 23:16:43", "%Y-%m-%d %H:%M:%S"),
        ],
        dtype=pl.Datetime(time_unit="us"),
    )
    df = pl.LazyFrame(
        {
            "dt": dates,
            "values": [3, 7, 5, 9, 2, 1, 72],
            "floats": pl.Series(
                [float("nan"), 7, 5, 2, -10, 1, float("inf")], dtype=pl.Float64()
            ),
        }
    )
    if presort:
        return df.with_columns(pl.col("dt").set_sorted())
    return df


@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
@pytest.mark.parametrize("period", ["1w4d", "48h", "180s"])
def test_datetime_rolling(engine: pl.GPUEngine, df, closed, period):
    q = df.rolling("dt", period=period, closed=closed).agg(
        sum_a=pl.sum("values"),
        min_a=pl.min("values"),
        max_a=pl.max("values"),
        count=pl.len(),
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt32, pl.Int64, pl.UInt64])
def test_rolling_integral_orderby(engine: pl.GPUEngine, dtype):
    df = pl.LazyFrame(
        {
            "orderby": pl.Series([1, 4, 8, 10, 12, 13, 14, 22], dtype=dtype),
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.rolling("orderby", period="4i", closed="both").agg(pl.col("values").sum())

    assert_gpu_result_equal(q, engine=engine)


def test_rolling_collect_list_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "orderby": [1, 4, 8, 10, 12, 13, 14, 22],
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    assert_ir_translation_raises(
        df.rolling("orderby", period="4i").agg(pl.col("values")),
        engine,
        NotImplementedError,
    )


@pytest.mark.parametrize("with_slice", [False, True])
def test_rolling_empty_aggs(engine: pl.GPUEngine, with_slice):
    df = pl.LazyFrame(
        {
            "orderby": [1, 4, 8, 10, 12, 13, 14, 22],
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.rolling("orderby", period="2i", closed="none").agg()
    if with_slice:
        q = q.slice(2)

    assert_gpu_result_equal(q, engine=engine)


def test_calendrical_period_unsupported(engine: pl.GPUEngine, df):
    q = df.rolling("dt", period="1mo", closed="right").agg(sum=pl.sum("values"))

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_unsorted_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame({"orderby": [1, 2, 4, 2], "values": [1, 2, 3, 4]})
    q = df.rolling("orderby", period="2i").agg(sum=pl.sum("values"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(
                RuntimeError, match=r".*rolling is not sorted, please sort first"
            )
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(
            RuntimeError, match=r".*rolling is not sorted, please sort first"
        ):
            q.collect(engine=engine)


def test_grouped_rolling(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "keys": [1, None, 2, 1, 2, None],
            "orderby": [10, 2, -11, 11, -5, 3],
            "values": [1, 2, 3, 4, 5, 6],
        }
    )
    q = df.rolling("orderby", period="5i", group_by="keys").agg(pl.col("values").sum())

    assert_gpu_result_equal(q, engine=engine)


def test_grouped_rolling_unsorted_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "keys": [1, None, 2, 1, 2, None],
            "orderby": [10, 2, -11, 11, -5, -2],
            "values": [1, 2, 3, 4, 5, 6],
        }
    )
    q = df.rolling("orderby", period="5i", group_by="keys").agg(pl.col("values").sum())

    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine="in-memory")
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(
                RuntimeError, match="Input for grouped rolling is not sorted"
            )
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(
            RuntimeError, match="Input for grouped rolling is not sorted"
        ):
            q.collect(engine=engine)


def test_orderby_nulls_raises_computeerror(engine: pl.GPUEngine):
    df = pl.LazyFrame({"orderby": [1, 2, 4, None], "values": [1, 2, 3, 4]})
    q = df.rolling("orderby", period="2i").agg(sum=pl.sum("values"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(
                RuntimeError,
                match=r"Index column.*in rolling may not contain nulls",
            )
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(
            RuntimeError, match=r"Index column.*in rolling may not contain nulls"
        ):
            q.collect(engine=engine)


@pytest.mark.xfail(
    condition=not POLARS_VERSION_LT_136,
    reason="polars raises now",
)
def test_rolling_nested_raises(engine: pl.GPUEngine):
    q = (
        pl.LazyFrame(
            {
                "orderby": [1, 2, 3, 4, 10],
                "values": [6, 7, 8, 9, 10],
            }
        )
        .rolling("orderby", period="3i", closed="left")
        .agg(pl.col("values").rolling("orderby", period="2i").sum())
    )
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_unsupported_agg(engine: pl.GPUEngine):
    q = (
        pl.LazyFrame(
            {
                "orderby": [1, 2, 3, 4, 10],
                "values": [6, 7, 8, 9, 10],
            }
        )
        .rolling("orderby", period="3i", closed="left")
        .agg(pl.col("values").n_unique())
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_rolling_sum_all_null_window_returns_null(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "orderby": [1, 2, 3, 4, 5, 6],
            "null_windows": [None, None, 5, None, None, 1],
        }
    )
    q = df.rolling("orderby", period="2i", closed="both").agg(
        out=pl.col("null_windows").sum()
    )
    # Expected: [0, 0, 5, 5, 5, 1]
    assert_gpu_result_equal(q, engine=engine)


def test_rolling_null_count(engine: pl.GPUEngine, df):
    lf = df.with_columns(
        null=pl.when(pl.col("values") % 2 == 0).then(None).otherwise(pl.col("values"))
    )
    q = lf.rolling("dt", period="48h", closed="both").agg(
        nc=pl.col("null").null_count()
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.when(pl.col("values") > 5).then(pl.col("floats")).otherwise(None).sum(),
        pl.when(pl.col("values").count() > 0)
        .then(pl.col("values").sum())
        .otherwise(None),
        pl.when(pl.col("values").min() < pl.col("values").max())
        .then(pl.col("values").max() - pl.col("values").min())
        .otherwise(pl.lit(0)),
        pl.when(pl.col("values").count() > 0)
        .then(
            pl.col("values").cast(pl.Float64).sum()
            / pl.col("values").count().cast(pl.Float64)
        )
        .otherwise(None),
    ],
    ids=[
        "pre_pointwise_then_sum",
        "post_over_aggs",
        "post_multiple_aggs_range",
        "post_manually_compute_mean",
    ],
)
def test_rolling_ternary_supported(engine: pl.GPUEngine, df, expr):
    q = df.rolling("dt", period="48h", closed="both").agg(expr.alias("out"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.when(pl.col("values") > 3)
        .then(pl.col("values"))
        .otherwise(pl.lit(None, dtype=pl.Int64)),
        pl.when((pl.col("floats") - pl.col("floats").mean()) > 0)
        .then(pl.col("floats"))
        .otherwise(None)
        .sum(),
    ],
)
def test_rolling_ternary_unsupported(engine: pl.GPUEngine, df, expr):
    q = df.rolling("dt", period="48h", closed="both").agg(expr.alias("out"))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_rolling_rank_unsupported(engine: pl.GPUEngine, df):
    q = df.rolling("dt", period="48h", closed="both").agg(
        pl.col("values").rank(method="dense", descending=False)
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)
