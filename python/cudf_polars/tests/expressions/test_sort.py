# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_136, POLARS_VERSION_LT_140


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_sort_expression(engine: pl.GPUEngine, descending, nulls_last):
    ldf = pl.LazyFrame(
        {
            "a": [5, -1, 3, 4, None, 8, 6, 7, None],
        }
    )

    query = ldf.select(pl.col("a").sort(descending=descending, nulls_last=nulls_last))
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize(
    "descending", itertools.combinations_with_replacement([False, True], 3)
)
@pytest.mark.parametrize(
    "nulls_last", itertools.combinations_with_replacement([False, True], 3)
)
@pytest.mark.parametrize("maintain_order", [False, True], ids=["unstable", "stable"])
def test_sort_by_expression(
    engine: pl.GPUEngine, descending, nulls_last, maintain_order
):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [1, 2, 2, 3, 9, 5, -1, 2, -2, 16],
            "c": ["a", "A", "b", "b", "c", "d", "A", "Z", "ä", "̈Ä"],
        }
    )

    query = ldf.select(
        pl.col("a").sort_by(
            pl.col("b"),
            pl.col("c"),
            pl.col("b") + pl.col("a"),
            descending=descending,
            nulls_last=nulls_last,
            maintain_order=maintain_order,
        )
    )
    assert_gpu_result_equal(query, engine=engine, check_row_order=maintain_order)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("with_nulls", ["no_nulls", "nulls"])
def test_setsorted(engine: pl.GPUEngine, request, descending, nulls_last, with_nulls):
    if POLARS_VERSION_LT_136:
        request.applymarker(
            pytest.mark.xfail(
                reason="See https://github.com/pola-rs/polars/pull/24981, "
                "fixed in https://github.com/pola-rs/polars/pull/25250"
            )
        )
    elif not POLARS_VERSION_LT_140:
        # polars >= 1.40 keeps the hint_sorted node in the optimized plan for a
        # bare set_sorted; we do not support it, so it raises. 1.36-1.39 pruned
        # it during optimization and passed.
        request.applymarker(pytest.mark.xfail(reason="Hint sorted unsupported"))
    sorted_values = sorted([1, 2, 3, 4, 5, 6, -2], reverse=descending)
    values: list[int | None] = [*sorted_values]
    if with_nulls == "nulls":
        values[-1 if nulls_last else 0] = None
    ldf = pl.LazyFrame({"a": values})

    q = ldf.set_sorted("a", descending=descending)

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_setsorted_expr_with_nulls(engine: pl.GPUEngine, descending, nulls_last):
    sorted_values = sorted([1, 2, 3, 4, 5], reverse=descending)
    values: list[int | None] = [*sorted_values]
    if nulls_last:
        values.append(None)
    else:
        values.insert(0, None)
    ldf = pl.LazyFrame({"a": values})
    q = ldf.select(pl.col("a").set_sorted(descending=descending))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_arg_sort_expression(engine: pl.GPUEngine, descending, nulls_last):
    ldf = pl.LazyFrame(
        {
            "a": [5, -1, 3, 4, None, 8, 6, 7, None, 3],
        }
    )
    query = ldf.select(
        pl.col("a").arg_sort(descending=descending, nulls_last=nulls_last)
    )
    assert_gpu_result_equal(query, engine=engine)


def test_sort_concat_filtered_to_empty(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = pl.concat([df.filter(pl.col("a") == 0), df.filter(pl.col("a") == 4)]).sort("a")
    assert_gpu_result_equal(q, engine=engine)
