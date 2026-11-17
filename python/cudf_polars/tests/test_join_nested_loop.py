# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for joins that Polars plans as a nested loop join (a cross join and a filter)."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_21

pytestmark = pytest.mark.skipif(
    POLARS_VERSION_LT_21,
    reason="Polars 2.0 does not expose nested loop joins to the visitor",
)


def not_in_query(outer: list, inner: list) -> pl.LazyFrame:
    partsupp = pl.LazyFrame({"ps_suppkey": outer}, schema={"ps_suppkey": pl.Int64})
    supplier = pl.LazyFrame({"s_suppkey": inner}, schema={"s_suppkey": pl.Int64})
    return pl.SQLContext(partsupp=partsupp, supplier=supplier).execute(
        "SELECT * FROM partsupp WHERE ps_suppkey NOT IN (SELECT s_suppkey FROM supplier)"
    )


@pytest.mark.parametrize(
    "outer,inner",
    [
        ([1, 2, 3, 4], [2, 4]),
        ([1, 2, 3, 4], [5, 6]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([1, 2, 3, 4], []),
        ([1, 2, None, 4], [2, 4]),
        ([1, 2, 3, 4], [2, None]),
        ([1, None], [None]),
        ([], [1]),
    ],
    ids=[
        "some-match",
        "no-match",
        "all-match",
        "empty-subquery",
        "null-in-outer",
        "null-in-subquery",
        "nulls-in-both",
        "empty-outer",
    ],
)
def test_not_in_subquery(engine: pl.GPUEngine, outer, inner):
    q = not_in_query(outer, inner)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_cross_join_filter_not_equal(engine: pl.GPUEngine):
    left = pl.LazyFrame({"x": [1, 2, 3], "u": ["a", "b", "c"]})
    right = pl.LazyFrame({"y": [2, 2, 5]})
    q = left.join(right, how="cross").filter(pl.col("x") != pl.col("y"))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_cross_join_filter_suffixed_columns(engine: pl.GPUEngine):
    left = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    right = pl.LazyFrame({"a": [2, 3, 4], "c": [1, 1, 2]})
    q = left.join(right, how="cross", suffix="_r").filter(
        (pl.col("a") != pl.col("a_r")) | (pl.col("b") > pl.col("c"))
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "maintain_order", ["none", "left", "right", "left_right", "right_left"]
)
def test_cross_join_filter_maintain_order(engine: pl.GPUEngine, maintain_order):
    left = pl.LazyFrame({"x": [0, 1, 2, 3, 4]})
    right = pl.LazyFrame({"y": [0, 1, 2, 3, 4]})
    q = left.join(right, how="cross", maintain_order=maintain_order).filter(
        (pl.col("x") + pl.col("y")) % 2 == 0
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=maintain_order != "none")


@pytest.mark.parametrize("maintain_order", ["none", "left", "right_left"])
@pytest.mark.parametrize("empty", ["left", "right"])
def test_cross_join_filter_maintain_order_empty(
    engine: pl.GPUEngine, maintain_order, empty
):
    schema = {"x": pl.Int64}
    left = pl.LazyFrame({"x": [] if empty == "left" else [0, 1, 2]}, schema=schema)
    right = pl.LazyFrame(
        {"y": [] if empty == "right" else [0, 1, 2]}, schema={"y": pl.Int64}
    )
    q = left.join(right, how="cross", maintain_order=maintain_order).filter(
        pl.col("x") != pl.col("y")
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=maintain_order != "none")
