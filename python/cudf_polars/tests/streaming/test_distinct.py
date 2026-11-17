# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic Distinct operations using the rapidsmpf runtime."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(scope="module")
def df() -> pl.LazyFrame:
    """Create a test DataFrame with duplicates."""
    return pl.LazyFrame(
        {
            "x": list(range(50)) * 4,  # 200 rows, 50 unique
            "y": list(range(10)) * 20,  # 200 rows, 10 unique
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 40,
        }
    )


@pytest.mark.parametrize("subset", [None, ("y",), ("x", "y")])
@pytest.mark.parametrize("keep", ["any", "none"])
def test_dynamic_distinct_basic(df, streaming_engine, subset, keep):
    """Test dynamic distinct with various subset and keep options."""
    q = df.unique(subset=subset, keep=keep, maintain_order=False)

    # For keep="any" with subset, select only the subset columns
    # to avoid non-deterministic results for non-key columns
    if keep == "any" and subset:
        q = q.select(*(pl.col(col) for col in subset))

    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_distinct_tree_strategy(df, streaming_engine_factory):
    """Test that small output uses tree reduction (high target_partition_size)."""
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=100_000_000),
    )
    subset = ("y",)
    q = df.unique(subset=subset, keep="any", maintain_order=False)
    # With keep="any", non-subset columns are non-deterministic, so only check subset
    q = q.select(*(pl.col(col) for col in subset))
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_distinct_shuffle_strategy(streaming_engine_factory):
    """Test that large output uses shuffle (low target_partition_size)."""
    streaming_engine = streaming_engine_factory(
        StreamingOptions(target_partition_size=1000),
    )
    df = pl.LazyFrame({"x": range(1000), "y": range(1000)})
    q = df.unique(subset=None, keep="any", maintain_order=False)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_distinct_all_duplicates(streaming_engine):
    """Test dynamic distinct where all rows are duplicates."""
    df = pl.LazyFrame({"x": [1, 1, 1, 1], "y": [2, 2, 2, 2]})
    q = df.unique()
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_distinct_single_row(streaming_engine):
    """Test dynamic distinct on single-row DataFrame."""
    df = pl.LazyFrame({"x": [1], "y": [2]})
    q = df.unique()
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


def test_dynamic_distinct_chunkwise_after_groupby(streaming_engine):
    """Test distinct after group_by is handled chunkwise."""
    df = pl.LazyFrame(
        {
            "key": list(range(20)) * 10,  # 200 rows, 20 unique keys
            "value": range(200),
        }
    )
    # Groupby partitions data by "key", then unique on "key" should be chunkwise
    q = df.group_by("key").agg(pl.col("value").sum()).unique(subset=("key",))
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)
