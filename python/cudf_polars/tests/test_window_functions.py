# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, cast

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_136, POLARS_VERSION_LT_139


@pytest.fixture
def df():
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(6)]

    return pl.LazyFrame(
        {
            "date": dates,
            "a": ["a", "a", "b", "b", "b", "c"],
            "b": [1, 2, 3, 1, 3, 2],
            "c": [7, 4, 3, 2, 3, 2],
            "d": [1.0, 2, 3, 4, 5, 6],
        }
    )


@pytest.fixture(
    params=[
        pl.col("a"),
        pl.col("b"),
        [pl.col("a"), pl.col("b")],
        pl.col("b") + pl.col("c"),
    ]
)
def partition_by(request):
    return request.param


@pytest.fixture(
    params=[
        pl.col("b").max(),
        pl.col("b").min(),
        pl.col("b").sum(),
        pl.col("b").count(),
        pl.col("b").sum(),
        pl.col("b").mean(),
        pl.col("b").var(),
    ],
)
def agg_expr(request):
    return request.param


@pytest.fixture(
    params=[
        pl.col("b")
        + pl.col("c").sum(),  # Broadcasting aggregated values inside rolling
        pl.col("b").cum_sum(),  # libcudf doesn't support rolling cumulative sum
        pl.col("b").rank(),  # Rank not exposed in polars IR yet
        pl.col("b").rank(method="dense"),  # Rank not exposed in polars IR yet
        pl.col("b").n_unique(),  # libcudf doesn't support rolling nunique
        pl.col("b").first(),  # libcudf doesn't support rolling first
        pl.col("b").last(),  # libcudf doesn't support rolling last
        pl.col("b").median(),  # libcudf doesn't support rolling median
        pl.col("b").quantile(0.5),  # libcudf doesn't support rolling quantile
    ],
)
def unsupported_agg_expr(request):
    return request.param


def test_over(engine: pl.GPUEngine, df: pl.LazyFrame, partition_by, agg_expr):
    """Test window functions over partitions."""

    window_expr = agg_expr.over(partition_by)

    result_name = f"{agg_expr!s}_over_{partition_by!s}"
    window_expr = window_expr.alias(result_name)

    q = df.with_columns(window_expr)

    # CPU: 1.333333333333333
    # GPU: 1.333333333333334
    # Classic floating-point gotcha: looks the same, but the test fails
    assert_gpu_result_equal(
        q, engine=engine, check_exact=False, rtol=1e-15, atol=1e-15
    ) if "var" in str(agg_expr) else assert_gpu_result_equal(q, engine=engine)


def test_over_with_sort(engine: pl.GPUEngine, df: pl.LazyFrame):
    """Test window functions with sorting."""
    query = df.with_columns([pl.col("c").rank().sort().over(pl.col("a"))])
    assert_ir_translation_raises(query, engine, NotImplementedError)


@pytest.mark.parametrize("mapping_strategy", ["group_to_rows", "explode", "join"])
def test_over_mapping_strategy(
    engine: pl.GPUEngine, df: pl.LazyFrame, mapping_strategy: str
):
    """Test window functions with different mapping strategies."""
    # ignore is for polars' WindowMappingStrategy, which isn't publicly exported.
    # https://github.com/pola-rs/polars/issues/17420
    q = df.with_columns(
        [
            pl.col("b")
            .rank()
            .over(
                pl.col("a"),
                mapping_strategy=cast(
                    "Literal['group_to_rows', 'join', 'explode']", mapping_strategy
                ),
            )
        ]
    )
    if mapping_strategy == "group_to_rows":
        assert_gpu_result_equal(q, engine=engine)
    else:
        assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.skipif(
    not POLARS_VERSION_LT_136 and POLARS_VERSION_LT_139,
    reason="Rolling window expressions are not accessible in polars 1.36-1.38",
)
@pytest.mark.parametrize("period", ["2d", "3d"])
def test_rolling(
    engine: pl.GPUEngine, request, df: pl.LazyFrame, agg_expr, period: str
):
    """Test rolling window functions over time series."""
    window_expr = agg_expr.rolling(period=period, index_column="date")
    result_name = f"{agg_expr!s}_rolling_{period}"
    window_expr = window_expr.alias(result_name)

    query = df.with_columns(window_expr)

    assert_gpu_result_equal(query, engine=engine)


def test_rolling_unsupported(
    engine: pl.GPUEngine, df: pl.LazyFrame, unsupported_agg_expr
):
    """Test rolling window functions over time series."""
    window_expr = unsupported_agg_expr.rolling(period="2d", index_column="date")
    result_name = f"{unsupported_agg_expr!s}_rolling"
    window_expr = window_expr.alias(result_name)

    query = df.with_columns(window_expr)

    assert_ir_translation_raises(query, engine, NotImplementedError)


@pytest.mark.skipif(
    not POLARS_VERSION_LT_136 and POLARS_VERSION_LT_139,
    reason="Rolling window expressions are not accessible in polars 1.36-1.38",
)
@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_rolling_closed(engine: pl.GPUEngine, request, df: pl.LazyFrame, closed: str):
    """Test rolling window functions with different closed parameters."""
    # ignore is for polars' ClosedInterval, which isn't publicly exported.
    # https://github.com/pola-rs/polars/issues/17420
    query = df.with_columns(
        [
            pl.col("b")
            .sum()
            .rolling(
                period="2d",
                index_column="date",
                closed=cast("Literal['left', 'right', 'both', 'none']", closed),
            )
        ]
    )
    assert_gpu_result_equal(query, engine=engine)
