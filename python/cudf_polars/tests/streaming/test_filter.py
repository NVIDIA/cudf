# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.testing.engine_utils import warns_on_spmd


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, fallback_mode="warn"),
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
            "c": [2, 4, 6, 8, 10, 12, 14],
        }
    )


def test_filter_pointwise(df, engine):
    query = df.filter(pl.col("a") > 3)
    assert_gpu_result_equal(query, engine=engine)


def test_filter_non_pointwise(df, engine):
    query = df.filter(pl.col("a") > pl.col("a").max())
    with warns_on_spmd(
        engine,
        UserWarning,
        match="This filter is not supported for multiple partitions.",
    ):
        assert_gpu_result_equal(query, engine=engine)
