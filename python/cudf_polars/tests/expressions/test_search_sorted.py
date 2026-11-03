# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Literal

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(params=["any", "left", "right"])
def side(request: pytest.FixtureRequest) -> Literal["any", "left", "right"]:
    return request.param


@pytest.mark.parametrize("element", [0, 2, 3, 6, pl.Series([0, 2, 3, 6])])
def test_search_sorted(
    engine: pl.GPUEngine,
    side: Literal["any", "left", "right"],
    element: int | pl.Series,
) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 2, 2, 4, 5]})
    q = lf.select(pl.col("a").search_sorted(element, side=side))
    assert_gpu_result_equal(q, engine=engine)


def test_search_sorted_descending(
    engine: pl.GPUEngine, side: Literal["any", "left", "right"]
) -> None:
    lf = pl.LazyFrame({"a": [5, 4, 2, 2, 1]})
    q = lf.select(pl.col("a").search_sorted(2, side=side, descending=True))
    assert_gpu_result_equal(q, engine=engine)


def test_search_sorted_floats(
    engine: pl.GPUEngine, side: Literal["any", "left", "right"]
) -> None:
    lf = pl.LazyFrame({"a": [1.0, 1.5, 1.5, 3.0, 7.5]})
    q = lf.select(pl.col("a").search_sorted(1.5, side=side))
    assert_gpu_result_equal(q, engine=engine)


def test_search_sorted_empty(
    engine: pl.GPUEngine, side: Literal["any", "left", "right"]
) -> None:
    lf = pl.LazyFrame({"a": []}, schema={"a": pl.Int64})
    q = lf.select(pl.col("a").search_sorted(1, side=side))
    assert_gpu_result_equal(q, engine=engine)


def test_search_sorted_single_element(
    engine: pl.GPUEngine, side: Literal["any", "left", "right"]
) -> None:
    lf = pl.LazyFrame({"a": [2]})
    q = lf.select(pl.col("a").search_sorted(pl.Series([1, 2, 3]), side=side))
    assert_gpu_result_equal(q, engine=engine)


def test_search_sorted_nulls(
    engine: pl.GPUEngine, side: Literal["any", "left", "right"]
) -> None:
    lf = pl.LazyFrame({"a": [None, 1, 2, 2, 4]})
    q = lf.select(pl.col("a").search_sorted(pl.Series([None, 2, 3]), side=side))
    assert_gpu_result_equal(q, engine=engine)
