# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import SPMDEngine
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import warns_on_spmd
from cudf_polars.utils.versions import POLARS_VERSION_LT_136, POLARS_VERSION_LT_139


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, fallback_mode="warn"),
    )


@pytest.mark.skipif(
    not POLARS_VERSION_LT_136 and POLARS_VERSION_LT_139,
    reason="Rolling window expressions are not accessible in polars 1.36-1.38",
)
def test_rolling_datetime(engine):
    dates = [
        "2020-01-01 13:45:48",
        "2020-01-01 16:42:13",
        "2020-01-01 16:45:09",
        "2020-01-02 18:12:48",
        "2020-01-03 19:45:32",
        "2020-01-08 23:16:43",
    ]
    df = (
        pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]})
        .with_columns(pl.col("dt").str.strptime(pl.Datetime("ns")))
        .lazy()
    )
    q = df.with_columns(pl.sum("a").rolling(index_column="dt", period="2d"))
    # HStack may redirect to Select before fallback; message differs by Polars IR / version.
    with warns_on_spmd(
        engine,
        UserWarning,
        match=r"This (HStack|selection) is not supported for multiple partitions\.",
    ):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.len().over("g"),
        pl.col("x").sum().over("g", "g2"),
        pl.col("x").sum().over("g_null"),
        pl.col("x").sum().over("g", order_by="s"),
        pl.col("x").rank(method="dense", descending=True).over("g"),
        pl.col("x").rank(method="min").over("g", "g2"),
        pl.col("x").cum_sum().over("g", order_by="s"),
        pl.col("x").diff().over("g", order_by="s"),
        pl.col("x").shift(1).over("g", order_by="s"),
        pl.col("x").shift(-1, fill_value=0).over("g", order_by="s"),
        pl.when((pl.col("x") % 2) == 0)
        .then(None)
        .otherwise(pl.col("x"))
        .fill_null(strategy="forward")
        .over("g", order_by="s"),
    ],
    ids=[
        "single_key_sum",
        "len_over",
        "multi_key",
        "null_keys",
        "order_by",
        "rank_dense",
        "rank_min_multi_key",
        "cum_sum_order_by",
        "diff_order_by",
        "shift_order_by",
        "shift_fill_order_by",
        "fill_null_forward",
    ],
)
def test_over_select(engine, expr):
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
            "g2": ["a", "b", "a", "b", "a", "b"],
            "g_null": [1, None, 1, None, 2, 1],
            "s": [6, 5, 4, 3, 2, 1],
        }
    )
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


@pytest.mark.skipif(
    not POLARS_VERSION_LT_136 and POLARS_VERSION_LT_139,
    reason="Rolling window expressions are not accessible in polars 1.36-1.38",
)
def test_rolling_sum_over(engine):
    df = (
        pl.LazyFrame(
            {
                "ric": ["A", "A", "A", "B", "B", "B"],
                "ts": [
                    dt.datetime(2025, 1, 1, 9, 0),
                    dt.datetime(2025, 1, 1, 9, 1),
                    dt.datetime(2025, 1, 1, 9, 3),
                    dt.datetime(2025, 1, 1, 9, 0),
                    dt.datetime(2025, 1, 1, 9, 2),
                    dt.datetime(2025, 1, 1, 9, 3),
                ],
                "price": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
                "volume": [100, 200, 300, 400, 500, 600],
            }
        )
        .with_columns(notional=pl.col("price") * pl.col("volume"))
        .sort("ric", "ts")
    )
    q = df.with_columns(
        volume_before=pl.col("volume")
        .sum()
        .rolling("ts", period="2m", offset="-2m", closed="left")
        .over("ric"),
        notional_before=pl.col("notional")
        .sum()
        .rolling("ts", period="2m", offset="-2m", closed="left")
        .over("ric"),
        volume_after=pl.col("volume")
        .sum()
        .rolling("ts", period="2m", closed="right")
        .over("ric"),
    ).select(
        "ric",
        "ts",
        "volume_before",
        "notional_before",
        "volume_after",
    )
    expected = pl.DataFrame(
        {
            "ric": ["A", "A", "A", "B", "B", "B"],
            "ts": [
                dt.datetime(2025, 1, 1, 9, 0),
                dt.datetime(2025, 1, 1, 9, 1),
                dt.datetime(2025, 1, 1, 9, 3),
                dt.datetime(2025, 1, 1, 9, 0),
                dt.datetime(2025, 1, 1, 9, 2),
                dt.datetime(2025, 1, 1, 9, 3),
            ],
            "volume_before": [0, 100, 200, 0, 400, 500],
            "notional_before": [0.0, 1000.0, 2200.0, 0.0, 8000.0, 10500.0],
            "volume_after": [100, 300, 300, 400, 500, 1100],
        }
    )
    assert_frame_equal(q.collect(engine=engine), expected)


@pytest.mark.skipif(
    not POLARS_VERSION_LT_136 and POLARS_VERSION_LT_139,
    reason="Rolling window expressions are not accessible in polars 1.36-1.38",
)
def test_rolling_common_aggs_over(engine):
    df = pl.LazyFrame(
        {
            "g": ["A", "A", "A", "B", "B", "B"],
            "ts": [1, 2, 4, 1, 3, 4],
            "x": [100, 200, 300, 400, 500, 600],
        }
    ).sort("g", "ts")
    q = df.select(
        pl.col("x").sum().rolling("ts", period="2i").over("g").alias("sum"),
        pl.col("x").min().rolling("ts", period="2i").over("g").alias("min"),
        pl.col("x").max().rolling("ts", period="2i").over("g").alias("max"),
        pl.col("x").mean().rolling("ts", period="2i").over("g").alias("mean"),
        pl.col("x").count().rolling("ts", period="2i").over("g").alias("count"),
        pl.len().rolling("ts", period="2i").over("g").alias("len"),
    )
    expected = pl.DataFrame(
        {
            "sum": [100, 300, 300, 400, 500, 1100],
            "min": [100, 100, 300, 400, 500, 500],
            "max": [100, 200, 300, 400, 500, 600],
            "mean": [100.0, 150.0, 300.0, 400.0, 500.0, 550.0],
            "count": pl.Series([1, 2, 1, 1, 1, 2], dtype=pl.UInt32),
            "len": pl.Series([1, 2, 1, 1, 1, 2], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(q.collect(engine=engine), expected)


@pytest.mark.parametrize("strategy", ["forward", "backward"])
def test_over_cum_sum_fill_null_per_partition(engine, strategy):
    df = pl.LazyFrame(
        {
            "g": [1, 1, 1, 2, 2],
            "s": [1, 2, 3, 1, 2],
            "x": [10.0, None, 5.0, 20.0, None],
        }
    )
    expr = pl.col("x").cum_sum().fill_null(strategy=strategy).over("g", order_by="s")
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


def test_over_shift_without_order_by_single_rank(spmd_engine_factory) -> None:
    engine = spmd_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="raise"),
    )
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    q = df.select(pl.col("x").shift(1).over("g"))
    assert_gpu_result_equal(q, engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").rank().fill_null(strategy="forward").over("g", order_by="x"),
        pl.col("x")
        .rank()
        .fill_null(strategy="forward")
        .fill_null(strategy="forward")
        .over("g", order_by="x"),
        pl.col("x").rank().abs().fill_null(strategy="forward").over("g", order_by="x"),
        pl.col("x")
        .cum_sum()
        .abs()
        .fill_null(strategy="forward")
        .over("g", order_by="x"),
        pl.col("x")
        .cum_sum()
        .fill_null(strategy="forward")
        .fill_null(strategy="forward")
        .over("g", order_by="x"),
        pl.col("x").shift(1).fill_null(strategy="forward").over("g"),
    ],
    ids=[
        "rank_fill",
        "rank_fill_fill",
        "rank_abs_fill",
        "cum_sum_abs_fill",
        "cum_sum_fill_fill",
        "shift_fill",
    ],
)
def test_over_fill_null_over_window_fails_translation(engine, expr):
    df = pl.LazyFrame({"g": [1, 1, 2, 2, 2, 1], "x": [1.0, None, 3, 4, None, 6]})
    assert_ir_translation_raises(df.select(expr), engine, NotImplementedError)


def test_over_with_columns(engine):
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    assert_gpu_result_equal(
        df.with_columns(pl.col("x").sum().over("g")),
        engine=engine,
        check_row_order=True,
    )


def test_over_colliding_internal_agg_names(engine):
    df = pl.LazyFrame(
        {
            "category": ["A", "A", "B", "B", "C"],
            "value": [20, 30, 15, 40, 35],
        }
    )
    q = df.select(
        pl.col("category"),
        pl.col("value"),
        pl.col("value").sum().over("category").alias("cat_sum"),
        pl.col("value").mean().over("category").alias("cat_avg"),
    ).sort("category", "value")
    assert_gpu_result_equal(q, engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over(pl.col("g") % 2),
        pl.col("x").sum().over("g", pl.col("x") % 2),
    ],
    ids=["noncol_key", "mixed_col_and_expr_key"],
)
def test_over_noncol_key_fallback(request, streaming_engine_factory, expr) -> None:
    # Non-Col and mixed Col/expr partition-by keys are not yet supported for
    # multi-partition streaming and should fall back to single-partition.
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="warn"),
    )
    if not isinstance(engine, SPMDEngine):
        # On Dask/Ray the fallback warning fires on worker processes and is
        # invisible to ``pytest.warns``.
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/NVIDIA/cudf/issues/22405",
                strict=False,
            )
        )
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    with pytest.warns(UserWarning, match=r"not supported for multiple partitions"):
        assert_gpu_result_equal(df.select(expr), engine=engine)


def test_over_mixed_keys(streaming_engine_factory) -> None:
    # Multiple over expressions with different partition-by keys are decomposed
    # into separate Over nodes (one per key group) and combined with HConcat.
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="warn"),
    )
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "g2": ["a", "b", "a", "b", "a", "b"],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    q = df.select(
        pl.col("x").sum().over("g").alias("s_g"),
        pl.col("x").sum().over("g2").alias("s_g2"),
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.len().over("g"),
        pl.col("x").rank(method="dense").over("g"),
        pl.col("x").cum_sum().over("g", order_by="s"),
        pl.col("x").shift(1).over("g", order_by="s"),
    ],
    ids=[
        "scalar_sum",
        "scalar_len",
        "nonscalar_rank",
        "nonscalar_cum_sum",
        "nonscalar_shift",
    ],
)
@pytest.mark.parametrize("max_rows_per_partition", [1, 2])
def test_over_many_partitions(
    streaming_engine_factory, max_rows_per_partition, expr
) -> None:
    # Small max_rows_per_partition forces many chunks, exercising the AllGather
    # (scalar broadcast) and sort-and-split (non-scalar) paths across many
    # partitions. Two values cover both single-row chunks and multi-row chunks
    # so the within-chunk position sort is also exercised.
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=max_rows_per_partition, fallback_mode="warn"
        ),
    )
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
            "s": [6, 5, 4, 3, 2, 1],
        }
    )
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.col("x").rank(method="dense").over("g"),
    ],
    ids=["scalar_sum", "nonscalar_rank"],
)
def test_over_empty_input(streaming_engine_factory, expr) -> None:
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, fallback_mode="warn"),
    )
    df = pl.LazyFrame(
        {
            "g": pl.Series([], dtype=pl.Int64),
            "x": pl.Series([], dtype=pl.Int64),
        }
    )
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.col("x").rank(method="dense").over("g"),
        pl.col("x").shift(1).over("g", order_by="x"),
    ],
    ids=["scalar_sum", "nonscalar_rank", "nonscalar_shift"],
)
def test_over_already_partitioned(streaming_engine_factory, expr) -> None:
    # broadcast_limit=0 disables broadcast joins entirely. Therefore, we should
    # already be shuffled on "g" after the join. The over("g") actor should
    # detect this and evaluate chunkwise without a second shuffle.
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3, broadcast_limit=0, fallback_mode="warn"
        ),
    )
    left = pl.LazyFrame({"g": [1, 1, 2, 2, 2, 1], "x": [1, 2, 3, 4, 5, 6]})
    right = pl.LazyFrame({"g": [1, 2], "y": [10, 20]})
    assert_gpu_result_equal(
        left.join(right, on="g").with_columns(expr),
        engine=engine,
        check_row_order=False,
    )


def test_over_in_filter_unsupported(request, streaming_engine_factory) -> None:
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1, fallback_mode="warn"),
    )
    if not isinstance(engine, SPMDEngine):
        # On Dask/Ray the fallback warning fires on worker processes and is
        # invisible to ``pytest.warns``; the multi-rank fallback also
        # doesn't preserve row order.
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/NVIDIA/cudf/issues/22405",
                strict=False,
            )
        )
    q = pl.concat(
        [
            pl.LazyFrame({"k": ["x", "y"], "v": [3, 2]}),
            pl.LazyFrame({"k": ["x", "y"], "v": [5, 7]}),
        ]
    ).filter(pl.len().over("k") == 2)

    with pytest.warns(
        UserWarning,
        match=r"over\(...\) inside filter is not supported for multiple partitions.*",
    ):
        assert_gpu_result_equal(q, engine=engine)
