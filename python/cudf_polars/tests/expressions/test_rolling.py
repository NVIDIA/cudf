# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Literal, cast

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.utils.aggregations import decompose_single_agg
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine
from cudf_polars.typing import Duration
from cudf_polars.utils.versions import POLARS_VERSION_LT_136, POLARS_VERSION_LT_139

if TYPE_CHECKING:
    from cudf_polars.typing import RankMethod


# In polars 1.36-1.38, rolling window expressions (pl.col(...).rolling(...))
# are represented as an opaque Rust node that is not accessible via view_expression().
# Support requires polars <1.36 (via Window+RollingGroupOptions) or >=1.39 (via Rolling).
skip_rolling_expr_136_to_138 = pytest.mark.skipif(
    not POLARS_VERSION_LT_136 and POLARS_VERSION_LT_139,
    reason="Rolling window expressions are not accessible in polars 1.36-1.38, see https://github.com/pola-rs/polars/pull/25117",
)


@pytest.fixture
def df():
    return pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2],
            "x": [1, 2, 3, 4, 5],
            "x2": [1, 100, 3, 4, 50],
            "g2": ["a", "a", "b", "a", "a"],
            "g_null": [1, None, 1, None, 2],
        }
    )


def _range_rolling_sum(
    dtype: DataType, orderby: str, child: expr.Expr
) -> expr.RollingWindow:
    offset = Duration((0, 0, 0, 0, True, False))
    period = Duration((0, 0, 0, 2, True, False))
    return expr.RollingWindow(
        dtype,
        dtype.plc_type,
        offset,
        period,
        "right",
        orderby,
        expr.Agg(dtype, "sum", (), ExecutionContext.WINDOW, child),
    )


@skip_rolling_expr_136_to_138
@pytest.mark.parametrize("time_unit", ["ns", "us", "ms"])
def test_rolling_datetime(engine: pl.GPUEngine, time_unit):
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
        .with_columns(pl.col("dt").str.strptime(pl.Datetime(time_unit)))
        .lazy()
    )
    q = df.with_columns(
        sum_a=pl.sum("a").rolling(index_column="dt", period="2d"),
        min_a=pl.min("a").rolling(index_column="dt", period="5d"),
        max_a=pl.max("a").rolling(index_column="dt", period="10d", offset="2d"),
    )

    assert_gpu_result_equal(q, engine=engine)


@skip_rolling_expr_136_to_138
def test_rolling_date(engine: pl.GPUEngine):
    dates = [
        "2020-01-01",
        "2020-01-01",
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-08",
    ]
    df = (
        pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]})
        .with_columns(pl.col("dt").str.strptime(pl.Date()))
        .lazy()
    )
    q = df.with_columns(
        max_a=pl.max("a").rolling(index_column="dt", period="10d", offset="2d"),
    )

    assert_gpu_result_equal(q, engine=engine)


@skip_rolling_expr_136_to_138
@pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt32, pl.Int64, pl.UInt64])
def test_rolling_integral_orderby(engine: pl.GPUEngine, dtype):
    df = pl.LazyFrame(
        {
            "orderby": pl.Series([1, 4, 8, 10, 12, 13, 14, 22], dtype=dtype),
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.with_columns(
        pl.col("values").sum().rolling("orderby", period="4i", closed="both")
    )

    assert_gpu_result_equal(q, engine=engine)


@skip_rolling_expr_136_to_138
def test_rolling_agg_before_rolling(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    q = df.with_columns(pl.col("a").sum().rolling("b", period="2i"))
    assert_gpu_result_equal(q, engine=engine)


def test_rolling_collect_list_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "orderby": [1, 4, 8, 10, 12, 13, 14, 22],
            "values": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    assert_ir_translation_raises(
        df.with_columns(pl.col("values").rolling("orderby", period="4i")),
        engine,
        NotImplementedError,
    )


@skip_rolling_expr_136_to_138
def test_unsorted_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame({"orderby": [1, 2, 4, 2], "values": [1, 2, 3, 4]})
    q = df.select(pl.col("values").sum().rolling("orderby", period="2i"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    match = r"Index column.*in rolling is not sorted, please sort first"
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pytest.RaisesExc(RuntimeError, match=match)):
            q.collect(engine=engine)
    else:
        with pytest.raises(RuntimeError, match=match):
            q.collect(engine=engine)


@skip_rolling_expr_136_to_138
def test_orderby_nulls_raises_computeerror(engine: pl.GPUEngine):
    df = pl.LazyFrame({"orderby": [1, 2, 4, None], "values": [1, 2, 3, 4]})
    q = df.select(pl.col("values").sum().rolling("orderby", period="2i"))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")
    match = r"Index column.*in rolling may not contain nulls"
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pytest.RaisesExc(RuntimeError, match=match)):
            q.collect(engine=engine)
    else:
        with pytest.raises(RuntimeError, match=match):
            q.collect(engine=engine)


@skip_rolling_expr_136_to_138
def test_invalid_duration_spec_raises_in_translation(engine: pl.GPUEngine):
    df = pl.LazyFrame({"orderby": [1, 2, 4, 5], "values": [1, 2, 3, 4]})
    q = df.select(pl.col("values").sum().rolling("orderby", period="3d"))
    assert_ir_translation_raises(q, engine, pl.exceptions.InvalidOperationError)


@pytest.mark.xfail(condition=not POLARS_VERSION_LT_136, reason="not supported")
def test_rolling_inside_groupby_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {"keys": [1, 1, 1, 2], "orderby": [1, 2, 4, 2], "values": [1, 2, 3, 4]}
    )
    q = df.group_by("keys").agg(pl.col("values").rolling("orderby", period="2i").sum())

    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect(engine="in-memory")

    assert_ir_translation_raises(q, engine, NotImplementedError)


@skip_rolling_expr_136_to_138
def test_rolling_sum_all_null_window_returns_null(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "orderby": [1, 2, 3, 4, 5, 6],
            "null_windows": [None, None, 5, None, None, 1],
        }
    )
    q = df.select(
        out=pl.col("null_windows").sum().rolling("orderby", period="2i", closed="both")
    )
    # Expected: [null, null, 5, 5, 5, 1]
    assert_gpu_result_equal(q, engine=engine)


@skip_rolling_expr_136_to_138
def test_rolling_sum_over(engine: pl.GPUEngine) -> None:
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
    # Polars <1.36 cannot collect this expression on CPU. Switch to
    # assert_gpu_result_equal after we drop Polars 1.35.
    assert_frame_equal(q.collect(engine=engine), expected)


@skip_rolling_expr_136_to_138
def test_rolling_over_with_order_by_raises(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame(
        {
            "g": ["A", "A", "A"],
            "seq": [1, 2, 3],
            "ts": [1, 2, 3],
            "x": [10, 20, 30],
        }
    )
    q = df.select(
        pl.col("x").sum().rolling("ts", period="2i").over("g", order_by="seq")
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


@skip_rolling_expr_136_to_138
def test_rolling_common_aggs_over(engine: pl.GPUEngine) -> None:
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


@skip_rolling_expr_136_to_138
@pytest.mark.parametrize(
    "idx,period",
    [
        (pl.Series("idx", [1, 1, 3, 1, 2, 4, 5], dtype=pl.Int32), "2i"),
        (
            [
                dt.datetime(2025, 1, 1, 9, 0),
                dt.datetime(2025, 1, 1, 9, 0),
                dt.datetime(2025, 1, 1, 9, 2),
                dt.datetime(2025, 1, 1, 9, 0),
                dt.datetime(2025, 1, 1, 9, 1),
                dt.datetime(2025, 1, 1, 9, 3),
                dt.datetime(2025, 1, 1, 9, 4),
            ],
            "2m",
        ),
    ],
    ids=["integer_index", "datetime_index"],
)
def test_rolling_sum_over_index_types_and_group_sizes(
    engine: pl.GPUEngine,
    idx: pl.Series | list[dt.datetime],
    period: str,
) -> None:
    df = pl.LazyFrame(
        {
            "g": ["A", "B", "B", "C", "C", "C", "C"],
            "idx": idx,
            "x": [10, 20, 30, 40, 50, 60, 70],
        }
    )
    q = df.select(
        pl.col("x").sum().rolling("idx", period=period).over("g").alias("sum")
    )
    expected = pl.DataFrame({"sum": [10, 20, 30, 40, 90, 60, 130]})
    assert_frame_equal(q.collect(engine=engine), expected)


@skip_rolling_expr_136_to_138
def test_rolling_sum_over_null_index_raises(
    engine: pl.GPUEngine,
) -> None:
    df = pl.LazyFrame(
        {
            "g": ["A", "A", "A"],
            "idx": pl.Series([1, None, 3], dtype=pl.Int64),
            "x": [10, 20, 30],
        }
    )
    q = df.select(pl.col("x").sum().rolling("idx", period="2i").over("g").alias("sum"))
    match = "Index column 'idx' in rolling may not contain nulls"
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pytest.RaisesExc(RuntimeError, match=match)):
            q.collect(engine=engine)
    else:
        with pytest.raises(RuntimeError, match=match):
            q.collect(engine=engine)


def test_rolling_orderby_name_multiple_index_columns_raises() -> None:
    dtype = DataType(pl.Int64())
    col = expr.Col(dtype, "x")
    named_exprs = [
        expr.NamedExpr("x_sum", _range_rolling_sum(dtype, "t1", col)),
        expr.NamedExpr("y_sum", _range_rolling_sum(dtype, "t2", col)),
    ]
    with pytest.raises(
        NotImplementedError,
        match=r"rolling\(\.\.\.\)\.over\(\.\.\.\) only supports one rolling index column",
    ):
        expr.GroupedWindow._rolling_orderby_name(named_exprs)


@skip_rolling_expr_136_to_138
@pytest.mark.parametrize(
    "lf,expected",
    [
        (
            pl.LazyFrame(
                {
                    "g": pl.Series([], dtype=pl.String),
                    "ts": pl.Series([], dtype=pl.Int64),
                    "x": pl.Series([], dtype=pl.Int64),
                }
            ),
            pl.DataFrame(
                {
                    "sum": pl.Series([], dtype=pl.Int64),
                    "min": pl.Series([], dtype=pl.Int64),
                    "max": pl.Series([], dtype=pl.Int64),
                    "mean": pl.Series([], dtype=pl.Float64),
                    "count": pl.Series([], dtype=pl.UInt32),
                    "len": pl.Series([], dtype=pl.UInt32),
                }
            ),
        ),
        (
            pl.LazyFrame(
                {
                    "g": ["A", "A", "B"],
                    "ts": [1, 2, 1],
                    "x": pl.Series([None, None, None], dtype=pl.Int64),
                }
            ),
            pl.DataFrame(
                {
                    "sum": [0, 0, 0],
                    "min": [None, None, None],
                    "max": [None, None, None],
                    "mean": [None, None, None],
                    "count": pl.Series([0, 0, 0], dtype=pl.UInt32),
                    "len": pl.Series([1, 2, 1], dtype=pl.UInt32),
                },
                schema={
                    "sum": pl.Int64,
                    "min": pl.Int64,
                    "max": pl.Int64,
                    "mean": pl.Float64,
                    "count": pl.UInt32,
                    "len": pl.UInt32,
                },
            ),
        ),
        (
            pl.LazyFrame(
                {
                    "g": ["A", "A", "A", "B", "B"],
                    "ts": [1, 2, 3, 1, 3],
                    "x": pl.Series([10, None, 30, None, 50], dtype=pl.Int64),
                }
            ),
            pl.DataFrame(
                {
                    "sum": [10, 10, 30, 0, 50],
                    "min": [10, 10, 30, None, 50],
                    "max": [10, 10, 30, None, 50],
                    "mean": [10.0, 10.0, 30.0, None, 50.0],
                    "count": pl.Series([1, 1, 1, 0, 1], dtype=pl.UInt32),
                    "len": pl.Series([1, 2, 2, 1, 1], dtype=pl.UInt32),
                },
                schema={
                    "sum": pl.Int64,
                    "min": pl.Int64,
                    "max": pl.Int64,
                    "mean": pl.Float64,
                    "count": pl.UInt32,
                    "len": pl.UInt32,
                },
            ),
        ),
        (
            pl.LazyFrame(
                {
                    "g": ["A", "B"],
                    "ts": [1, 1],
                    "x": pl.Series([10, None], dtype=pl.Int64),
                }
            ),
            pl.DataFrame(
                {
                    "sum": [10, 0],
                    "min": [10, None],
                    "max": [10, None],
                    "mean": [10.0, None],
                    "count": pl.Series([1, 0], dtype=pl.UInt32),
                    "len": pl.Series([1, 1], dtype=pl.UInt32),
                },
                schema={
                    "sum": pl.Int64,
                    "min": pl.Int64,
                    "max": pl.Int64,
                    "mean": pl.Float64,
                    "count": pl.UInt32,
                    "len": pl.UInt32,
                },
            ),
        ),
    ],
    ids=["empty", "all_null", "mixed_null", "single_row_groups"],
)
def test_rolling_common_aggs_over_edge_cases(
    engine: pl.GPUEngine,
    lf: pl.LazyFrame,
    expected: pl.DataFrame,
) -> None:
    q = lf.sort("g", "ts").select(
        pl.col("x").sum().rolling("ts", period="2i").over("g").alias("sum"),
        pl.col("x").min().rolling("ts", period="2i").over("g").alias("min"),
        pl.col("x").max().rolling("ts", period="2i").over("g").alias("max"),
        pl.col("x").mean().rolling("ts", period="2i").over("g").alias("mean"),
        pl.col("x").count().rolling("ts", period="2i").over("g").alias("count"),
        pl.len().rolling("ts", period="2i").over("g").alias("len"),
    )
    assert_frame_equal(q.collect(engine=engine), expected)


@skip_rolling_expr_136_to_138
def test_range_rolling_nested_under_range_rolling_over_raises(
    engine: pl.GPUEngine,
) -> None:
    df = pl.LazyFrame(
        {
            "g": ["A", "A", "A"],
            "ts": [1, 2, 3],
            "x": [10, 20, 30],
        }
    )
    q = df.select(
        pl.col("x")
        .sum()
        .rolling("ts", period="2i")
        .sum()
        .rolling("ts", period="2i")
        .over("g")
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_range_rolling_nested_window_decomposition_raises() -> None:
    dtype = DataType(pl.Int64())
    child = expr.Col(dtype, "x")
    inner_rolling = _range_rolling_sum(dtype, "ts", child)
    outer_rolling = _range_rolling_sum(dtype, "ts", inner_rolling)

    with pytest.raises(
        NotImplementedError,
        match="Range rolling over a window does not support nested window expressions",
    ):
        decompose_single_agg(
            expr.NamedExpr("out", outer_rolling),
            (f"__{i}" for i in range(1)),
            is_top=True,
            context=ExecutionContext.WINDOW,
        )


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.len().over("g"),
        pl.col("x").cast(pl.Float64).mean().round(1).over("g"),
        pl.col("x2").quantile(0.5, interpolation="lower").over("g"),
        pl.col("x").sum().over("g", "g2"),
        pl.col("x").sum().over(pl.col("g") % 2),
        pl.col("x").sum().over("g_null"),
        pl.col("x").cast(pl.Float32).mean().over("g"),
        pl.col("x").sum().over(pl.lit(1)),
    ],
    ids=[
        "sum_broadcast",
        "len_broadcast",
        "mean_round",
        "quantile_lower",
        "multi_key_partition",
        "expr_partition",
        "null_keys",
        "mean_float32_promotion",
        "literal_partition",
    ],
)
def test_over_group_various(engine: pl.GPUEngine, df, expr):
    q = df.select(expr)
    assert_gpu_result_equal(q, engine=engine)


def test_window_over_group_sum_all_null_group_is_zero(engine: pl.GPUEngine, df):
    q = df.with_columns(
        pl.when(pl.col("g") == 1)
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col("x"))
        .alias("null")
    ).select(s=pl.col("null").sum().over("g"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "order_by",
    [
        "x",
        pl.col("x") * 2,
        pl.when((pl.col("x") % 2) == 0).then(pl.col("x")).otherwise(-pl.col("x")),
        ["x", "x2"],
        ["g_null", "g2", "x2"],
        [pl.col("g") + 7, (pl.col("x") * 3) - 2],
    ],
)
@pytest.mark.parametrize("order_by_descending", [False, True])
@pytest.mark.parametrize("order_by_nulls_last", [False, True])
def test_over_with_order_by(
    engine: pl.GPUEngine, df, order_by, order_by_descending, order_by_nulls_last
):
    q = df.select(
        pl.col("x")
        .sum()
        .over(
            "g",
            order_by=order_by,
            descending=order_by_descending,
            nulls_last=order_by_nulls_last,
        )
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("strategy", ["explode", "join"], ids=["explode", "join"])
def test_over_with_mapping_strategy_unsupported(engine: pl.GPUEngine, df, strategy):
    q = df.select(pl.col("x").sum().over("g", mapping_strategy=strategy))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_over_boolean_function_unsupported(engine: pl.GPUEngine, df):
    q = df.select(pl.col("x").not_().over("g"))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_over_ternary(engine: pl.GPUEngine, df):
    q = df.select(
        pl.when(pl.col("g") == 1)
        .then(pl.lit(None, dtype=pl.Int64))
        .otherwise(pl.col("x"))
        .sum()
        .over("g")
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skip_on_streaming_engine(
    "GroupedWindow not supported for multiple partitions"
)
def test_over_broadcast_input_row_group_indices_aligned(engine: pl.GPUEngine):
    num_rows, num_groups = 512, 64

    df = pl.LazyFrame(
        {
            "g": [(i * 31) % num_groups for i in range(num_rows)],
            "x": list(range(num_rows)),
        }
    )
    q = df.select(pl.col("x").sum().over("g"))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("order_by", [None, ["g2", pl.col("x2") * 2]])
def test_rank_over(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    method: RankMethod,
    *,
    descending: bool,
    order_by: list[str | pl.Expr] | None,
) -> None:
    q = df.select(
        pl.col("x")
        .rank(method=method, descending=descending)
        .over("g", order_by=order_by)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("order_by", [None, ["g2", pl.col("x2") * 2]])
def test_rank_over_with_ties(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    method: RankMethod,
    *,
    descending: bool,
    order_by: list[str | pl.Expr] | None,
) -> None:
    q = df.select(
        pl.when(pl.col("g") == 2)
        .then(pl.lit(4))
        .otherwise(pl.col("x"))
        .rank(method=method, descending=descending)
        .over("g", order_by=order_by)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("order_by", [None, ["g2", pl.col("x2") * 2]])
def test_rank_over_with_null_values(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    method: RankMethod,
    *,
    descending: bool,
    order_by: list[str | pl.Expr] | None,
) -> None:
    q = df.select(
        pl.when((pl.col("x") % 2) == 0)
        .then(None)
        .otherwise(pl.col("x"))
        .rank(method=method, descending=descending)
        .over("g", order_by=order_by)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("order_by", [None, ["g2", pl.col("x2") * 2]])
def test_rank_over_with_null_group_keys(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    method: RankMethod,
    *,
    descending: bool,
    order_by: list[str | pl.Expr] | None,
) -> None:
    q = df.select(
        pl.col("x")
        .rank(method=method, descending=descending)
        .over("g_null", order_by=order_by)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("strategy", ["forward", "backward"])
@pytest.mark.parametrize("order_by", [None, ["g2", pl.col("x2") * 2]])
@pytest.mark.parametrize(
    "group_key,expr",
    [
        pytest.param(
            "g",
            pl.when((pl.col("x") % 3) == 0).then(None).otherwise(pl.col("x")),
            id="fill_over",
        ),
        pytest.param(
            "g_null",
            pl.when((pl.col("x") % 2) == 0).then(None).otherwise(pl.col("x")),
            id="fill_over_with_null_group_keys",
        ),
    ],
)
def test_fill_over(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    strategy: str,
    order_by: list[str | pl.Expr] | None,
    group_key: str,
    expr: pl.Expr,
) -> None:
    q = df.select(
        expr.fill_null(strategy=cast("Literal['forward', 'backward']", strategy)).over(
            group_key, order_by=order_by
        )
    )
    assert_gpu_result_equal(q, engine=engine)


def test_fill_null_with_mean_over_unsupported(
    engine: pl.GPUEngine, df: pl.LazyFrame
) -> None:
    q = df.select(pl.col("x").fill_null(strategy="mean").over("g"))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "expr,group_key",
    [
        (pl.col("x"), "g"),
        (pl.when((pl.col("x") % 4) == 1).then(None).otherwise(pl.col("x")), "g"),
        (pl.col("x"), "g_null"),
    ],
)
@pytest.mark.parametrize(
    "order_by",
    [
        None,
        ["g2", pl.col("x2") * 2],
    ],
)
def test_cum_sum_over(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    *,
    expr: pl.Expr,
    group_key: str,
    order_by: list[str | pl.Expr] | None,
) -> None:
    q = df.select(expr.cum_sum().over(group_key, order_by=order_by))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [1, -1, 2])
@pytest.mark.parametrize(
    "expr,group_key",
    [
        (pl.col("x"), "g"),
        (pl.when((pl.col("x") % 2) == 0).then(None).otherwise(pl.col("x")), "g"),
        (pl.col("x"), "g_null"),
    ],
)
@pytest.mark.parametrize("order_by", ["x2", ["g2", pl.col("x2") * 2]])
def test_shift_over(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    n: int,
    expr: pl.Expr,
    group_key: str,
    order_by: str | list[str | pl.Expr],
) -> None:
    q = df.select(expr.shift(n).over(group_key, order_by=order_by))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n,fill_value", [(1, 0), (-1, 99)])
def test_shift_over_fill_value(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    n: int,
    fill_value: int,
) -> None:
    q = df.select(pl.col("x").shift(n, fill_value=fill_value).over("g", order_by="x2"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [1, -1, 2])
@pytest.mark.parametrize("order_by", ["x2", None])
def test_diff_over(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    n: int,
    order_by: str | None,
) -> None:
    expr = pl.col("x").diff(n=n).over("g")
    if order_by is not None:
        expr = pl.col("x").diff(n=n).over("g", order_by=order_by)
    q = df.select(expr)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").shift(pl.col("x2").min()).over("g"),
        pl.col("x").shift(1, fill_value=pl.col("x2").min()).over("g"),
    ],
    ids=["nonliteral_offset", "nonliteral_fill_value"],
)
def test_shift_over_nonliteral_args_raises(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    expr: pl.Expr,
) -> None:
    q = df.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").diff(n=pl.col("x2").min()).over("g"),
        pl.col("x").diff(null_behavior="drop").over("g"),
    ],
    ids=["nonliteral_offset", "drop_null_behavior"],
)
def test_diff_over_unsupported_args_raises(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    expr: pl.Expr,
) -> None:
    q = df.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("n", [1, -1])
def test_shift_over_without_order_by(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    n: int,
) -> None:
    q = df.select(pl.col("x").shift(n).over("g"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col(["x", "x2"]).first(),
        pl.col(["x", "x2"]).last(),
    ],
)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize(
    "order_by",
    [
        "g_null",
        ["g_null", "g2"],
    ],
)
def test_order_sensitive_over_scalar_aggs(
    engine: pl.GPUEngine, df, expr, descending, nulls_last, order_by
):
    q = df.select(
        expr.over(
            "g",
            order_by=order_by,
            descending=descending,
            nulls_last=nulls_last,
        )
    )
    if isinstance(order_by, list):
        assert_ir_translation_raises(q, engine, NotImplementedError)
    else:
        assert_gpu_result_equal(q, engine=engine)
