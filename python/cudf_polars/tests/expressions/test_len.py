# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("dtype", [pl.UInt32, pl.Int32, None])
@pytest.mark.parametrize("empty", [False, True])
def test_len(engine: pl.GPUEngine, dtype, empty):
    if empty:
        df = pl.LazyFrame({})
    else:
        df = pl.LazyFrame({"a": [1, 2, 3]})

    if dtype is None:
        q = df.select(pl.len())
    else:
        q = df.select(pl.len().cast(dtype))

    # Workaround for https://github.com/pola-rs/polars/issues/16904
    assert_gpu_result_equal(
        q,
        engine=engine,
        collect_kwargs={"optimizations": pl.QueryOptFlags(projection_pushdown=False)},
    )


@pytest.mark.parametrize("data", [[1, 2, 3], [1, 2, None]])
def test_col_len(engine: pl.GPUEngine, data):
    data = {"a": list("xyz"), "b": data}
    q = pl.LazyFrame(data).select(
        pl.col("a").len().alias("l"),
        (pl.col("a").len() * 2).alias("l2"),
        pl.col("b").len().alias("l3"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_concat_len(engine: pl.GPUEngine):
    # polars rewrites concat(...).select(len()) into
    # col("len").cast(UInt128).sum().cast(IDX_DTYPE); see
    # cudf_polars.dsl.translate._is_len_sum_uint128_node.
    df1 = pl.LazyFrame({"a": [1, 2, 3]})
    df2 = pl.LazyFrame({"a": [4, 5, 6, 7]})
    q = pl.concat([df1, df2]).select(pl.len())
    assert_gpu_result_equal(q, engine=engine)
