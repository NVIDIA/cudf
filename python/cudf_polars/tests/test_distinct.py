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
