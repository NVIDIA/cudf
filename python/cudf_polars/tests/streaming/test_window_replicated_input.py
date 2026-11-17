# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Window evaluation preserves replicated rows on every input owner."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_ordered_window_preserves_replicated_rows(ray_engine, empty, mixed):
    """Ordered windows and scalar windows align with replicated top-k rows."""
    frame = pl.LazyFrame(
        {
            "g": ["a", "a", "a", "b", "b", "b", "c"],
            "year": [2000, 2000, 2001, 2000, 2001, 2001, 2000],
            "month": [2, 1, 1, 1, 2, 1, 1],
            "x": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    if empty:
        frame = frame.clear()
    # Distributed top-k replicates its output. Keep g out of the leading
    # ordering keys to exercise the ordered window's forward/return shuffle.
    frame = frame.sort(["year", "month", "g"]).head(6)
    expressions = [
        pl.col("year")
        .rank(method="ordinal")
        .over("g", order_by=pl.struct("year", "month"))
        .alias("rn")
    ]
    if mixed:
        expressions.append(pl.col("x").mean().over(["g", "year"]).alias("avg"))
    assert_gpu_result_equal(
        frame.with_columns(expressions), engine=ray_engine, check_row_order=True
    )


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("ordered", [False, True])
def test_cumulative_window_preserves_replicated_rows(ray_engine, empty, ordered):
    """Ordered and input-order cumulative windows align with replicated top-k rows."""
    frame = pl.LazyFrame(
        {"g": ["a", "b", "a", "b"], "t": [3, 1, 4, 2], "x": [10, 40, 30, 20]}
    )
    if empty:
        frame = frame.clear()
    window = (
        pl.col("x").cum_sum().over("g", order_by="x")
        if ordered
        else pl.col("x").cum_sum().over("g")
    ).alias("running")
    query = frame.sort("t").head(3).with_columns(window)
    assert_gpu_result_equal(query, engine=ray_engine, check_row_order=True)
