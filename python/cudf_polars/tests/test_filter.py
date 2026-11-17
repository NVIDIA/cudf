# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("expr", [pl.col("c"), pl.col("b") < 1, pl.lit(value=True)])
@pytest.mark.parametrize("predicate_pushdown", [False, True])
def test_filter(engine: pl.GPUEngine, expr, predicate_pushdown):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
            "c": [True, False, False, True, True, True, None],
        }
    ).lazy()

    query = ldf.filter(expr)
    assert_gpu_result_equal(
        query,
        engine=engine,
        collect_kwargs={
            "optimizations": pl.QueryOptFlags(predicate_pushdown=predicate_pushdown)
        },
    )


def test_filter_drops_dynamic_predicate_hint(engine: pl.GPUEngine):
    ldf = pl.LazyFrame(
        {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": [1, 1, 3, 3, 5]}
    )
    # sort("b").head(3) causes Polars to inject a dynamic predicate hint into
    # the filter below: FILTER (a > 1) & (c == 3) & col("b").dynamic_predicate()
    # This test ensures we drop these dynamic predicate hints from the filter
    # before executing on the GPU.
    q = ldf.filter((pl.col("a") > 1) & (pl.col("c") == 3)).sort("b").head(3)
    assert_gpu_result_equal(q, engine=engine)
