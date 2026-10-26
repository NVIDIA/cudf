# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "sort_keys",
    [
        (pl.col("a"),),
        (pl.col("d").abs(),),
        (pl.col("a"), pl.col("d")),
        (pl.col("b"),),
    ],
)
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("maintain_order", [False, True], ids=["unstable", "stable"])
def test_sort(engine: pl.GPUEngine, sort_keys, nulls_last, maintain_order):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 1, 3, 5, None, None],
            "b": [1.5, 2.5, None, 1.5, 3, float("nan"), 3],
            "c": [True, True, True, True, False, False, True],
            "d": [1, 2, -1, 10, 6, -1, -7],
        }
    ).lazy()

    query = ldf.sort(
        *sort_keys,
        descending=True,
        nulls_last=nulls_last,
        maintain_order=maintain_order,
    )
    assert_gpu_result_equal(query, engine=engine, check_row_order=maintain_order)


def test_duplicate_sort_keys_projection(engine):
    # https://github.com/pola-rs/polars/issues/30116
    query = (
        pl.LazyFrame({"c0": [1]}).sort(["c0", "c0"]).select(pl.col("c0").alias("c1"))
    )
    assert_gpu_result_equal(query, engine=engine)


def test_duplicate_sort_keys_ordering(engine):
    query = pl.LazyFrame(
        {
            "a": [2, None, 1, 2, 1, None, 2, 1, 2],
            "b": [3, 2, None, 1, 2, 1, None, 3, 2],
        }
    ).sort(
        pl.col("a").alias("key"),
        pl.col("a").alias("duplicate"),
        pl.col("b").alias("key"),
        descending=[False, True, False],
        nulls_last=[True, False, False],
    )
    assert_gpu_result_equal(query, engine=engine)
