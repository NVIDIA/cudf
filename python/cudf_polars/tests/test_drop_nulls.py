# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


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
