# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "expr",
    [
        pl.lit(value=False),
        pl.lit(value=True),
        pl.col("c"),
        pl.col("b") > 2,
    ],
)
@pytest.mark.parametrize("predicate_pushdown", [False, True])
def test_filter_expression(engine: pl.GPUEngine, expr, predicate_pushdown):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, 3, 1, 5, 6, 1, 0],
            "c": [None, True, False, False, True, True, False],
        }
    )

    query = ldf.select(pl.col("a").filter(expr))
    assert_gpu_result_equal(
        query,
        engine=engine,
        collect_kwargs={
            "optimizations": pl.QueryOptFlags(predicate_pushdown=predicate_pushdown)
        },
    )


@pytest.mark.parametrize(
    "values",
    [
        pl.col("a"),  # length-N value
        pl.lit(2),  # scalar value
        pl.col("a").first(),  # scalar value
    ],
)
@pytest.mark.parametrize(
    "mask",
    [
        pl.col("b") > 1,  # length-N mask
        pl.lit(value=True),  # scalar mask
        pl.lit(value=False),  # scalar mask
    ],
)
def test_filter_broadcasts_scalar_operands(engine: pl.GPUEngine, values, mask):
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 1, 2, 1]})
    query = ldf.select(a=values.filter(mask))
    assert_gpu_result_equal(query, engine=engine)
