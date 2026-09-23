# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars import polars as plrs  # type: ignore[attr-defined]
from polars.testing import assert_frame_equal

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.cuda_stream import get_cuda_stream

pytestmark = pytest.mark.skipif(
    not hasattr(plrs._expr_nodes, "RollingFunction"),
    reason="RollingFunction not available in this polars version",
)


@pytest.fixture
def df():
    return pl.LazyFrame(
        {
            "x": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        }
    )


@pytest.mark.parametrize(
    "rolling_fn",
    ["rolling_sum", "rolling_min", "rolling_max", "rolling_mean"],
)
def test_fixed_rolling_basic(df, engine: pl.GPUEngine, rolling_fn):
    q = df.select(getattr(pl.col("x"), rolling_fn)(window_size=3))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_mean_over_in_memory(in_memory_engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "g": ["A", "A", "B", "A", "B", "B"],
            "idx": [2, 1, 2, 3, 1, 3],
            "val": [20.0, 10.0, 50.0, 30.0, 40.0, 60.0],
        }
    )
    q = lf.select(
        pl.col("val").rolling_mean(window_size=2).over("g", order_by="idx").alias("rm")
    )
    assert_gpu_result_equal(q, engine=in_memory_engine)


def test_fixed_rolling_sum_over_casts_to_polars_dtype(
    in_memory_engine: pl.GPUEngine,
):
    lf = pl.LazyFrame(
        {
            "g": ["A", "A", "A", "B", "B", "B"],
            "x": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.Int32),
        }
    )
    q = lf.select(pl.col("x").rolling_sum(window_size=2).over("g").alias("rs"))
    assert_gpu_result_equal(q, engine=in_memory_engine)


def test_fixed_rolling_over_multiple_expressions_share_bounds(
    in_memory_engine: pl.GPUEngine,
):
    lf = pl.LazyFrame(
        {
            "g": ["A", "A", "A", "B", "B", "B"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    q = lf.select(
        pl.col("x").rolling_sum(window_size=2).over("g").alias("x_sum"),
        pl.col("y").rolling_mean(window_size=2).over("g").alias("y_mean"),
    )
    assert_gpu_result_equal(q, engine=in_memory_engine)


def test_fixed_rolling_over_reuses_bounds_for_same_window():
    dtype = DataType(pl.Float64())
    data = pl.DataFrame(
        {
            "g": ["A", "A", "A", "B", "B", "B"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    grouped_window = expr.GroupedWindow(
        dtype,
        ("groups_to_rows", False, False, False),
        [
            expr.NamedExpr(
                "_x",
                expr.FixedSizeRollingWindow(
                    dtype, "mean", 2, 0, 2, (), expr.Col(dtype, "x")
                ),
            ),
            expr.NamedExpr(
                "_y",
                expr.FixedSizeRollingWindow(
                    dtype, "mean", 2, 0, 2, (), expr.Col(dtype, "y")
                ),
            ),
        ],
        expr.NamedExpr(
            "out",
            expr.BinOp(
                dtype,
                plc.binaryop.BinaryOperator.ADD,
                expr.Col(dtype, "_x"),
                expr.Col(dtype, "_y"),
            ),
        ),
        1,
        expr.Col(DataType(pl.String()), "g"),
    )

    stream = get_cuda_stream()
    result = grouped_window.evaluate(
        DataFrame.from_polars(data, stream=stream), context=ExecutionContext.FRAME
    )
    got = DataFrame([result.rename("out")], stream=stream).to_polars()
    expected = data.lazy().select(
        (
            pl.col("x").rolling_mean(window_size=2).over("g")
            + pl.col("y").rolling_mean(window_size=2).over("g")
        ).alias("out")
    )
    assert_frame_equal(got, expected.collect())


@pytest.mark.parametrize(
    "lf,expected",
    [
        (
            pl.LazyFrame(
                {
                    "g": pl.Series([], dtype=pl.String),
                    "idx": pl.Series([], dtype=pl.Int64),
                    "val": pl.Series([], dtype=pl.Float64),
                }
            ),
            pl.DataFrame({"rm": pl.Series([], dtype=pl.Float64)}),
        ),
        (
            pl.LazyFrame(
                {
                    "g": ["A", "A", "B"],
                    "idx": [1, 2, 1],
                    "val": pl.Series([None, None, None], dtype=pl.Float64),
                }
            ),
            pl.DataFrame({"rm": [None, None, None]}, schema={"rm": pl.Float64}),
        ),
        (
            pl.LazyFrame(
                {
                    "g": ["A", "B"],
                    "idx": [1, 1],
                    "val": [10.0, 20.0],
                }
            ),
            pl.DataFrame({"rm": [None, None]}, schema={"rm": pl.Float64}),
        ),
    ],
    ids=["empty", "all_null", "single_row_groups"],
)
def test_fixed_rolling_mean_over_edge_cases(
    in_memory_engine: pl.GPUEngine,
    lf: pl.LazyFrame,
    expected: pl.DataFrame,
):
    q = lf.select(
        pl.col("val").rolling_mean(window_size=2).over("g", order_by="idx").alias("rm")
    )
    assert_gpu_result_equal(q, engine=in_memory_engine)
    result = q.collect(engine=in_memory_engine)
    assert_frame_equal(result, expected)


def test_fixed_rolling_in_groupby_raises(engine: pl.GPUEngine):
    q = (
        pl.LazyFrame({"g": ["A", "A", "B"], "x": [1.0, 2.0, 3.0]})
        .group_by("g")
        .agg(pl.col("x").rolling_mean(window_size=2))
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "window_expr",
    [
        pl.col("x").rolling_mean(window_size=2).rank().over("g"),
        pl.col("x").rolling_mean(window_size=2).fill_null(strategy="forward").over("g"),
        pl.col("x").rolling_mean(window_size=2).cum_sum().over("g"),
        pl.col("x").rolling_mean(window_size=2).shift(1).over("g"),
        pl.col("x").rolling_mean(window_size=2).shift(1, fill_value=0).over("g"),
    ],
    ids=[
        "rank",
        "fill_null_with_strategy",
        "cum_sum",
        "shift",
        "shift_and_fill",
    ],
)
def test_fixed_rolling_nested_under_unary_over_raises(
    engine: pl.GPUEngine,
    window_expr: pl.Expr,
):
    q = pl.LazyFrame({"g": ["A", "A", "A"], "x": [1.0, 2.0, 3.0]}).select(
        window_expr.alias("out")
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_unary_window_nested_under_fixed_rolling_over_raises(engine: pl.GPUEngine):
    q = pl.LazyFrame({"g": ["A", "A", "A"], "x": [1.0, 2.0, 3.0]}).select(
        pl.col("x").rank().rolling_mean(window_size=2).over("g").alias("out")
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("window_size", [1, 2, 4, 8])
def test_fixed_rolling_sum_window_sizes(df, engine: pl.GPUEngine, window_size):
    q = df.select(pl.col("x").rolling_sum(window_size=window_size))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_sum_centered(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_sum(window_size=3, center=True))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_sum_centered_even(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_sum(window_size=4, center=True))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_sum_min_samples(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_sum(window_size=3, min_samples=1))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_sum_with_nulls(engine: pl.GPUEngine):
    df = pl.LazyFrame({"x": [1.0, None, 3.0, None, 5.0, 6.0]})
    q = df.select(pl.col("x").rolling_sum(window_size=3, min_samples=1))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_sum_all_null_window(engine: pl.GPUEngine):
    df = pl.LazyFrame({"x": [None, None, None, 4.0, 5.0, 6.0]})
    q = df.select(pl.col("x").rolling_sum(window_size=3))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_mean_with_nulls(engine: pl.GPUEngine):
    df = pl.LazyFrame({"x": [1.0, None, 3.0, 4.0, None, 6.0]})
    q = df.select(pl.col("x").rolling_mean(window_size=3, min_samples=1))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_var(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_var(window_size=3))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_std(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_std(window_size=3))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_var_ddof(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_var(window_size=4, ddof=2))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_std_ddof(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_std(window_size=4, ddof=0))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.Float32, pl.Float64])
def test_fixed_rolling_sum_dtypes(engine: pl.GPUEngine, dtype):
    df = pl.LazyFrame({"x": pl.Series([1, 2, 3, 4, 5, 6], dtype=dtype)})
    q = df.select(pl.col("x").rolling_sum(window_size=3))
    assert_gpu_result_equal(q, engine=engine)


# TODO: Remove once fixed-size rolling supports multi-partition streaming.
@pytest.mark.filterwarnings(
    "ignore:This selection is not supported for multiple partitions\\.:UserWarning"
)
def test_fixed_rolling_large_window(engine: pl.GPUEngine):
    data = list(range(500))
    df = pl.LazyFrame({"x": [float(v) for v in data]})
    q = df.select(pl.col("x").rolling_sum(window_size=250))
    assert_gpu_result_equal(q, engine=engine)


def test_fixed_rolling_weighted_raises(df, engine: pl.GPUEngine):
    q = df.select(pl.col("x").rolling_mean(window_size=3, weights=[1.0, 2.0, 3.0]))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_fixed_rolling_invalid_dtype_raises(engine: pl.GPUEngine):
    q = pl.LazyFrame({"x": ["a", "b", "c"]}).select(
        pl.col("x").rolling_min(window_size=2)
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_fixed_rolling_unsupported_function_raises(engine: pl.GPUEngine):
    q = pl.LazyFrame({"x": [1.0, 2.0, 3.0]}).select(
        pl.col("x").rolling_quantile(0.5, window_size=2)
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)
