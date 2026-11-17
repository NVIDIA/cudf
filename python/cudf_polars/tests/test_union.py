# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
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


def test_concat_single_empty_null_frame(engine: pl.GPUEngine):
    q = pl.concat([pl.LazyFrame({"A": []})])

    assert_gpu_result_equal(q, engine=engine)


def test_concat_nested_all_null_raises(in_memory_engine: pl.GPUEngine):
    q = pl.concat(
        [
            pl.LazyFrame({"A": [[1]]}),
            pl.LazyFrame({"A": [None]}),
        ]
    )

    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "value,dtype",
    [(1, pl.Int64()), ("foo", pl.String())],
    ids=["integer", "string"],
)
def test_concat_vertical_null_chunk(engine: pl.GPUEngine, value, dtype):
    concrete = pl.select(a=pl.lit(value, dtype)).lazy()
    null = pl.select(a=None).lazy()
    q = pl.concat([concrete, null])

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
