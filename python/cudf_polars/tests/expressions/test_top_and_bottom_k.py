# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture
def df():
    return pl.LazyFrame(
        {
            "test": [2, 4, 1, 3],
            "val": [2, 4, 9, 3],
            "bool_val": [False, True, True, False],
            "str_value": ["d", "b", "a", "c"],
            "col_with_nulls": [2, 4, None, 3],
        }
    )


@pytest.mark.parametrize("col", ["test", "bool_val", "str_value", "col_with_nulls"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_top_k(engine: pl.GPUEngine, df, col, k):
    q = df.select(pl.col(col).top_k(k))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("col", ["test", "bool_val", "str_value", "col_with_nulls"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_bottom_k(engine: pl.GPUEngine, df, col, k):
    q = df.select(pl.col(col).bottom_k(k))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("by", ["val", "str_value", "col_with_nulls"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("reverse", [False, True])
def test_top_k_by(engine: pl.GPUEngine, df, by, k, reverse):
    q = df.select(pl.col("test").top_k_by(by, k, reverse=reverse))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("by", ["val", "str_value", "col_with_nulls"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("reverse", [False, True])
def test_bottom_k_by(engine: pl.GPUEngine, df, by, k, reverse):
    q = df.select(pl.col("test").bottom_k_by(by, k, reverse=reverse))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_top_k_by_multiple_by_unsupported(engine: pl.GPUEngine, df):
    q = df.select(pl.col("test").top_k_by(["val", "str_value"], 2))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "k_expr",
    [
        pl.col("test").max(),
        pl.col("test").min(),
        pl.lit(2) + pl.lit(1),
    ],
)
def test_top_k_expr_k(engine: pl.GPUEngine, df, k_expr):
    q = df.select(pl.col("val").top_k(k_expr))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "k_expr",
    [
        pl.col("test").max(),
        pl.col("test").min(),
        pl.lit(2) + pl.lit(1),
    ],
)
def test_bottom_k_expr_k(engine: pl.GPUEngine, df, k_expr):
    q = df.select(pl.col("val").bottom_k(k_expr))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "k_expr",
    [
        pl.col("test").max(),
        pl.col("test").min(),
        pl.lit(2) + pl.lit(1),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_top_k_by_expr_k(engine: pl.GPUEngine, df, k_expr, reverse):
    q = df.select(pl.col("val").top_k_by("col_with_nulls", k_expr, reverse=reverse))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
