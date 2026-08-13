# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine


def test_replace(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 2, None, 3]})
    q = df.select(
        pl.col("a").replace([1, 2], [10, 20]).alias("list_replace"),
        pl.col("a").replace(2, 20).alias("scalar_replace"),
        pl.col("a").replace([1, 2], 99).alias("broadcast_replace"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_replace_strict(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 2, None, 3]})
    q = df.select(
        pl.col("a").replace_strict([1, 2], [10, 20], default=-1).alias("list_replace"),
        pl.col("a").replace_strict([1, 2], 99, default=-1).alias("broadcast_replace"),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_replace_non_literal_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3], "old": [1, 2, 3]})
    q = df.select(pl.col("a").replace(pl.col("old"), 0))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_replace_strict_without_default_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = df.select(pl.col("a").replace_strict([1, 2], [10, 20]))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_replace_strict_non_literal_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3], "old": [1, 2, 3]})
    q = df.select(pl.col("a").replace_strict(pl.col("old"), 0, default=-1))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("strict", [False, True])
def test_replace_new_length_mismatch(engine: pl.GPUEngine, strict: bool) -> None:  # noqa: FBT001
    df = pl.LazyFrame({"a": [1, 2, 3]})
    if strict:
        expr = pl.col("a").replace_strict([1, 2], [10, 20, 30], default=-1)
    else:
        expr = pl.col("a").replace([1, 2], [10, 20, 30])

    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            df.select(expr).collect(engine=engine)
    else:
        with pytest.raises(
            pl.exceptions.InvalidOperationError,
        ):
            df.select(expr).collect(engine=engine)


def test_replace_only_null_to_one(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, None, 2, None]})
    q = df.select(pl.col("a").replace(None, 100))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_many_with_null_to_one(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, None, 3]})
    q = df.select(pl.col("a").replace([1, None], 100))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_many_with_null_to_many(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, None, 3]})
    q = df.select(pl.col("a").replace([1, None], [10, 99]))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_value_to_null_and_null_to_value(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, None, 3]})
    q = df.select(pl.col("a").replace([1, None], [None, 100]))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_value_to_value_collision_with_null_fill(
    engine: pl.GPUEngine,
) -> None:
    df = pl.LazyFrame({"a": [1, None]})
    q = df.select(pl.col("a").replace([1, None], [2, 1]))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_new_all_null(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = df.select(pl.col("a").replace([1, 2], None))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_str_mapping_with_null_key(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": ["x", None, "y", "z"]})
    q = df.select(pl.col("a").replace({"x": "X", None: "missing"}))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_invalid_old_dtype(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3]})
    expr = pl.col("a").replace({"a": 10, "b": 20})
    match = "conversion from `str` to `i64` failed"
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(pl.exceptions.InvalidOperationError, match=match)
        ):
            df.select(expr).collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
            df.select(expr).collect(engine=engine)


def test_replace_strict_str_to_int(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": ["a", "b", "a", "c"]})
    q = df.select(pl.col("a").replace_strict(["a", "b"], [1, 2], default=-1))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_strict_null_key(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": ["a", None, "b", "c"]})
    q = df.select(pl.col("a").replace_strict(["a", None], [1, 99], default=-1))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_strict_many_to_null(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 2, 3]})
    q = df.select(pl.col("a").replace_strict([2, 3], None, default=-1))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_strict_only_null_old(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, None, 2]})
    q = df.select(pl.col("a").replace_strict(None, 100, default=-1))
    assert_gpu_result_equal(q, engine=engine)


def test_replace_strict_duplicate_old(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3, 2, 3]})
    expr = pl.col("a").replace_strict([2, 2], [10, 20], default=-1)
    match = "`old` input for `replace` must not contain duplicates"
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(pl.exceptions.InvalidOperationError, match=match)
        ):
            df.select(expr).collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
            df.select(expr).collect(engine=engine)
