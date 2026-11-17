# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
)


def test_union(engine: pl.GPUEngine):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()
    ldf2 = ldf.select((pl.col("a") + pl.col("b")).alias("c"), pl.col("a"))
    query = pl.concat([ldf, ldf2], how="diagonal")
    assert_gpu_result_equal(query, engine=engine)


def test_concat_vertical(engine: pl.GPUEngine):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    )
    ldf2 = ldf.select(pl.col("a"), pl.col("b") * 2 + pl.col("a"))
    q = pl.concat([ldf, ldf2], how="vertical")

    assert_gpu_result_equal(q, engine=engine)


def test_concat_diagonal_empty(engine: pl.GPUEngine):
    df1 = pl.LazyFrame()
    df2 = pl.LazyFrame({"a": [1, 2]})

    q = pl.concat([df1, df2], how="diagonal_relaxed")

    assert_gpu_result_equal(
        q,
        engine=engine,
        collect_kwargs={"optimizations": pl.QueryOptFlags()},
    )
