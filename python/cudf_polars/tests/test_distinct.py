# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def assert_distinct_result(
    engine: pl.GPUEngine, subset, keep, maintain_order, pre_sorted
):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 1, 3, 5, None, None],
            "b": [1.5, 2.5, None, 1.5, 3, float("nan"), 3],
            "c": [True, True, True, True, False, False, True],
        }
    ).lazy()
    if pre_sorted:
        keys = ["a", "b", "c"] if subset is None else subset
        descending = False if len(keys) == 1 else [False, True, True][: len(keys)]
        ldf = ldf.sort(*keys, descending=descending)

    query = ldf.unique(subset=subset, keep=keep, maintain_order=maintain_order)
    assert_gpu_result_equal(query, engine=engine, check_row_order=maintain_order)


@pytest.mark.engine_params(["spmd", "spmd-small"])
@pytest.mark.parametrize("subset", [None, ["a"], ["a", "b"], ["b", "c"], ["c", "a"]])
@pytest.mark.parametrize("keep", ["any", "none", "first", "last"])
@pytest.mark.parametrize("maintain_order", [False, True], ids=["unstable", "stable"])
@pytest.mark.parametrize("pre_sorted", [False, True], ids=["unsorted", "sorted"])
def test_distinct(engine: pl.GPUEngine, subset, keep, maintain_order, pre_sorted):
    assert_distinct_result(engine, subset, keep, maintain_order, pre_sorted)


@pytest.mark.engine_params(["in-memory", "dask", "ray"])
@pytest.mark.parametrize(
    "subset,keep,maintain_order,pre_sorted",
    [
        pytest.param(None, "any", False, False, id="all-columns-any"),
        pytest.param(["a"], "first", True, False, id="single-column-stable"),
        pytest.param(["a", "b"], "last", True, True, id="multi-column-sorted"),
        pytest.param(["b", "c"], "none", False, True, id="none-sorted"),
        pytest.param(["c", "a"], "first", False, False, id="reordered-columns"),
    ],
)
def test_distinct_non_spmd(
    engine: pl.GPUEngine, subset, keep, maintain_order, pre_sorted
):
    assert_distinct_result(engine, subset, keep, maintain_order, pre_sorted)


@pytest.mark.parametrize("keep", ["any", "first", "last"])
@pytest.mark.parametrize("height", [0, 1, 5])
def test_distinct_zero_width(engine: pl.GPUEngine, keep, height):
    ldf = pl.LazyFrame(height=height)
    q = ldf.unique(keep=keep)
    assert_gpu_result_equal(q, engine=engine)
    assert_gpu_result_equal(q.head(1), engine=engine)


@pytest.mark.parametrize(
    "subset,maintain_order,expected",
    [
        pytest.param(["b"], False, False, id="unordered"),
        pytest.param(["a"], False, True, id="sorted-keys"),
        pytest.param(["b"], True, True, id="stable"),
    ],
)
def test_distinct_sorted_flag(
    in_memory_engine: pl.GPUEngine,
    subset: list[str],
    *,
    maintain_order: bool,
    expected: bool,
) -> None:
    ldf = pl.LazyFrame(
        {"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "b": [3, 8, 1, 9, 0, 5, 2, 7, 4, 6]}
    )
    q = ldf.with_columns(pl.col("a").set_sorted()).unique(
        subset=subset, maintain_order=maintain_order
    )
    result = q.collect(engine=in_memory_engine)
    assert result["a"].flags["SORTED_ASC"] is expected


def test_distinct_unordered_then_unique_expression(engine: pl.GPUEngine) -> None:
    ldf = pl.LazyFrame(
        {"a": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2], "b": [3, 8, 1, 9, 0, 5, 2, 7, 4, 6]}
    )
    q = (
        ldf.with_columns(pl.col("a").set_sorted())
        .unique(subset=["b"])
        .select(pl.col("a").unique())
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
