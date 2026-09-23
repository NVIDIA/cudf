# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


def test_arg_unique(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 2, None, 3]})
    q = df.select(pl.col("a").arg_unique())
    assert_gpu_result_equal(q, engine=engine)


def test_unique_counts(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 2, None, 3]})
    q = df.select(pl.col("a").unique_counts())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("maintain_order", [False, True], ids=["unstable", "stable"])
@pytest.mark.parametrize("pre_sorted", [False, True], ids=["unsorted", "sorted"])
def test_unique(engine: pl.GPUEngine, maintain_order, pre_sorted):
    ldf = pl.DataFrame(
        {
            "b": [1.5, 2.5, None, 1.5, 3, float("nan"), 3],
        }
    ).lazy()
    if pre_sorted:
        ldf = ldf.sort("b")

    query = ldf.select(pl.col("b").unique(maintain_order=maintain_order))
    assert_gpu_result_equal(query, engine=engine, check_row_order=maintain_order)


def test_unique_on_sorted_expression(engine: pl.GPUEngine):
    # Sort expr produces a column with is_sorted=YES, covering the sorted fast-path
    ldf = pl.DataFrame({"b": [3.0, 1.0, 2.0, 1.0, 3.0]}).lazy()
    query = ldf.select(pl.col("b").sort().unique())
    assert_gpu_result_equal(query, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").rle(),
        pl.col("a").rle_id(),
    ],
    ids=["rle", "rle_id"],
)
def test_rle_unsupported(engine: pl.GPUEngine, expr: pl.Expr) -> None:
    df = pl.LazyFrame({"a": [1, 1, 2, 2, 3, 1]})
    q = df.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)
