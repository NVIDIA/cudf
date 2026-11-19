# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize("inner", [pl.Int64, pl.String])
def test_list_len(engine: pl.GPUEngine, inner: pl.DataType) -> None:
    lf = pl.LazyFrame(
        {"a": pl.Series([[1, 2, 2], [None, 3], [], None, [5]]).cast(pl.List(inner))}
    )
    q = lf.select(pl.col("a").list.len())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", ["list", "struct"])
def test_list_len_nested(engine: pl.GPUEngine, inner: str) -> None:
    data: list[object]
    if inner == "list":
        data = [[[1], [2, 3]], None, [], [None]]
    else:
        data = [[{"x": 1}], None, [], [None, {"x": 2}]]
    lf = pl.LazyFrame({"a": data})
    q = lf.select(pl.col("a").list.len())
    assert_gpu_result_equal(q, engine=engine)


def test_list_len_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.List(pl.Int64))})
    q = lf.select(pl.col("a").list.len())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("inner", [pl.Int64, pl.String])
@pytest.mark.parametrize("how", ["slice", "tail"])
def test_list_len_sliced(
    engine: pl.GPUEngine, inner: pl.DataType, how: str, *, with_nulls: bool
) -> None:
    null_list = None if with_nulls else [6, 7]
    a = pl.Series([[1, 2, 2], [None, 3], [], null_list, [5]]).cast(pl.List(inner))
    lf = pl.LazyFrame({"k": [4, 3, 2, 1, 0], "a": a}).sort("k")
    lf = lf.slice(1, 3) if how == "slice" else lf.tail(3)
    q = lf.select(pl.col("a").list.len())
    assert_gpu_result_equal(q, engine=engine)


def test_list_len_group_by(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {"k": [1, 1, 2, 2, 3], "a": [[1, 2, 2], [None, 3], [], None, [5]]}
    )
    q = lf.group_by("k", maintain_order=True).agg(pl.col("a").list.len())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("values", [[1, 2, 3], [[1], None, [2, 3]]])
def test_list_len_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine, values: list
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": values})
    q = lf.group_by("k").agg(pl.col("a").implode().list.len())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_len_wrapped_implode_in_agg_not_supported(
    in_memory_engine: pl.GPUEngine,
) -> None:
    lf = pl.LazyFrame({"k": [1, 1, 2], "a": [1, 2, 3]})
    q = lf.group_by("k").agg(pl.col("a").implode().fill_null([0]).list.len())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_list_len_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.col("a"), pl.lit([1, 2]).list.len())
    assert_gpu_result_equal(q, engine=engine)


def test_list_function_not_supported(in_memory_engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [[1, 2], None]})
    q = lf.select(pl.col("a").list.sample(n=1, seed=0))
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)
