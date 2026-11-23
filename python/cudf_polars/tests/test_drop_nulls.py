# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl import expr
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.cuda_stream import get_cuda_stream


@pytest.fixture(
    params=[
        [1, 2, 1, 3, 5, None, None],
        [1.5, 2.5, None, 1.5, 3, float("nan"), 3],
        [],
        [None, None],
        [1, 2, 3, 4, 5],
    ]
)
def null_data(request):
    is_empty = pl.Series(request.param).dtype == pl.Null
    return pl.DataFrame(
        {
            "a": pl.Series(request.param, dtype=pl.Float64 if is_empty else None),
            "b": pl.Series(request.param, dtype=pl.Float64 if is_empty else None),
        }
    ).lazy()


def test_drop_null(engine: pl.GPUEngine, null_data):
    q = null_data.select(pl.col("a").drop_nulls())
    assert_gpu_result_equal(q, engine=engine)


@pytest.fixture(params=[False, True], ids=["ascending", "descending"])
def descending(request):
    return request.param


@pytest.fixture(params=["first", "last", "all", "one_null", "none"])
def nulls(request):
    return request.param


def sorted_null_frame(*, descending: bool, nulls: str) -> pl.LazyFrame:
    columns: dict[str, tuple[pl.DataType, list[Any]]] = {
        "i": (pl.Int64(), [1, 2, 3]),
        "f": (pl.Float64(), [1.0, 2.0, float("nan")]),
        "s": (pl.String(), ["x", "y", "z"]),
    }
    frame = {}
    for name, (dtype, ascending_values) in columns.items():
        values = ascending_values[::-1] if descending else ascending_values
        data = {
            "first": [None, None, *values],
            "last": [*values, None, None],
            "all": [None, None, None],
            "one_null": [None],
            "none": values,
        }[nulls]
        frame[name] = pl.Series(data, dtype=dtype)
    return pl.LazyFrame(frame)


@pytest.mark.parametrize("column", ["i", "f", "s"])
def test_drop_nulls_sorted(engine: pl.GPUEngine, descending, nulls, column):
    q = sorted_null_frame(descending=descending, nulls=nulls).select(
        pl.col(column).set_sorted(descending=descending).drop_nulls()
    )
    assert_gpu_result_equal(q, engine=engine)


def test_drop_nulls_sorted_then_reuse_null_mask(
    engine: pl.GPUEngine, descending, nulls
):
    q = sorted_null_frame(descending=descending, nulls=nulls).select(
        pl.col("i").set_sorted(descending=descending).drop_nulls().is_finite(),
        pl.col("f").set_sorted(descending=descending).drop_nulls().is_finite(),
        pl.col("s").set_sorted(descending=descending).drop_nulls().str.contains(""),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("column", ["i", "f", "s"])
def test_drop_nulls_sorted_keeps_sorted_flag(
    in_memory_engine: pl.GPUEngine, descending, nulls, column
):
    q = sorted_null_frame(descending=descending, nulls=nulls).select(
        pl.col(column).set_sorted(descending=descending).drop_nulls()
    )
    got = q.collect(engine=in_memory_engine)[column]
    expected = q.collect()[column]
    assert got.flags == expected.flags


def test_drop_nulls_after_unordered_unique(engine: pl.GPUEngine):
    n = 1_000
    q = (
        pl.LazyFrame(
            {
                "a": pl.Series([None] * 5 + list(range(n)), dtype=pl.Int64()),
                "b": list(range(n + 5))[::-1],
            }
        )
        .with_columns(pl.col("a").set_sorted())
        .unique()
        .select(pl.col("a").drop_nulls())
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "value",
    [0, pl.col("a").mean(), pl.col("b")],
    ids=["scalar", "aggregation", "column_expression"],
)
def test_fill_null(engine: pl.GPUEngine, null_data, value):
    q = null_data.select(pl.col("a").fill_null(value))
    assert_gpu_result_equal(q, engine=engine)


def test_fill_null_with_string(engine: pl.GPUEngine):
    q = pl.LazyFrame({"a": [None, "a"]}).select(pl.col("a").fill_null("b"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("scalar", [None, 1.5], ids=["null", "valid"])
@pytest.mark.parametrize(
    "values",
    [
        [0.5, None, 2.0],
        [None, None],
        [],
        [float(i) if i % 3 else None for i in range(9)],
    ],
    ids=["mixed", "all_null", "empty", "nine_rows"],
)
def test_fill_null_scalar_receiver_column_fill(engine: pl.GPUEngine, scalar, values):
    q = pl.LazyFrame({"a": pl.Series(values, dtype=pl.Float64)}).select(
        pl.lit(scalar, dtype=pl.Float64).fill_null(pl.col("a")).alias("a")
    )
    assert_gpu_result_equal(q, engine=engine)


def test_fill_null_null_scalar_receiver_keeps_fill_sortedness():
    dtype = DataType(pl.Float64())
    stream = get_cuda_stream()
    df = DataFrame.from_polars(pl.DataFrame({"a": [3.0, 2.0, 1.0]}), stream=stream)
    fill = df.column_map["a"].set_sorted(
        is_sorted=plc.types.Sorted.YES,
        order=plc.types.Order.DESCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    df = DataFrame([fill], stream=stream)
    result = expr.UnaryFunction(
        dtype, "fill_null", (), expr.Literal(dtype, None), expr.Col(dtype, "a")
    ).evaluate(df)
    assert result.is_sorted == plc.types.Sorted.YES
    assert result.order == plc.types.Order.DESCENDING


def test_fill_null_null_list_receiver_column_fill(engine: pl.GPUEngine):
    dtype = pl.List(pl.Int64)
    q = pl.LazyFrame({"a": pl.Series([[3], None, []], dtype=dtype)}).select(
        pl.lit(None, dtype=dtype).fill_null(pl.col("a")).alias("a")
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "data",
    [
        [[3]],
        [None],
        [[3], None, [], [1, 2], [4], None, [5], [6], [7]],
    ],
)
def test_fill_null_list_receiver_one_row_partition(engine: pl.GPUEngine, data):
    dtype = pl.List(pl.Int64)
    lf = pl.LazyFrame({"a": pl.Series(data, dtype=dtype)})
    q = lf.select(
        pl.lit(None, dtype=dtype).fill_null(pl.col("a")).alias("null_recv"),
        pl.col("a").head(1).fill_null(pl.col("a").head(1)).alias("valid_recv"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "strategy", ["forward", "backward", "min", "max", "mean", "zero", "one"]
)
def test_fill_null_with_strategy(engine: pl.GPUEngine, null_data, strategy):
    q = null_data.select(pl.col("a").fill_null(strategy=strategy))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("strategy", ["zero", "one"])
def test_fill_null_with_strategy_bool(engine: pl.GPUEngine, strategy):
    q = pl.LazyFrame({"a": [True, None, False]}).select(
        pl.col("a").fill_null(strategy=strategy)
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("strategy", ["forward", "backward"])
def test_fill_null_with_zero_limit(engine: pl.GPUEngine, null_data, strategy):
    q = null_data.select(pl.col("a").fill_null(strategy=strategy, limit=0))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("strategy", ["forward", "backward"])
@pytest.mark.parametrize("limit", [1, 2])
def test_fill_null_with_nonzero_limit_unsupported(in_memory_engine, strategy, limit):
    q = pl.LazyFrame({"a": [None, 1]}).select(
        pl.col("a").fill_null(strategy=strategy, limit=limit)
    )
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
