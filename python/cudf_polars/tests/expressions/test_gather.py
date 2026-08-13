# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine


def test_gather(engine: pl.GPUEngine):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, 3, 1, 5, 6, 1, 0],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.col("b")))
    assert_gpu_result_equal(query, engine=engine)


def test_gather_with_nulls(engine: pl.GPUEngine):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, None, 1, None, 6, 1, 0],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.col("b")))

    assert_gpu_result_equal(query, engine=engine)


def test_gather_empty_indices(engine: pl.GPUEngine):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.lit(pl.Series("idx", [], dtype=pl.Int64))))
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("negative", [False, True])
def test_gather_out_of_bounds(engine_raise_on_fail: pl.GPUEngine, negative):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, -10 if negative else 10, 1, 2, 6, 1, 0],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.col("b")))

    with pytest.raises(ValueError, match="gather indices are out of bounds"):
        query.collect(engine=engine_raise_on_fail)


@pytest.mark.parametrize(
    "idx",
    [
        0,
        pl.lit(0),
        pl.col("a").first(),
    ],
)
@pytest.mark.parametrize(
    "lit",
    [
        pl.lit(7),
        pytest.param(
            pl.lit([7]),
            marks=pytest.mark.xfail(
                reason="List literal loses nesting in gather: https://github.com/rapidsai/cudf/issues/19610"
            ),
        ),
        pl.lit([[7]]),
        pl.lit(pl.Series([7, 8, 9])),
    ],
)
def test_gather_on_literal(
    engine: pl.GPUEngine,
    lit: pl.Expr,
    idx: pl.Expr,
) -> None:
    df = pl.LazyFrame(
        {
            "g": [10, 10, 10, 20, 20, 30],
            "a": [0, 0, 0, 0, 0, 0],
            "b": [1, 1, 1, 1, 1, 1],
            "c": [11, 12, 13, 21, 22, 31],
        }
    )

    q = df.select(lit.gather(idx))
    assert_gpu_result_equal(q, engine=engine)


def test_repeat_by(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, None, 3], "n": [2, None, 0, 1]})
    q = df.select(pl.col("a").repeat_by("n"))
    assert_gpu_result_equal(q, engine=engine)


def test_repeat_by_no_nulls(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": ["x", "y", "z"], "n": [0, 2, 1]})
    q = df.select(pl.col("a").repeat_by("n"))
    assert_gpu_result_equal(q, engine=engine)


def test_repeat_by_all_null_counts(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame(
        {"a": [1, 2, 3], "n": pl.Series([None, None, None], dtype=pl.Int32())}
    )
    q = df.select(pl.col("a").repeat_by("n"))
    assert_gpu_result_equal(q, engine=engine)


def test_repeat_by_negative_raises(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3], "n": [2, -1, 0]})
    q = df.select(pl.col("a").repeat_by("n"))
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)
    else:
        with pytest.raises(
            pl.exceptions.InvalidOperationError, match="must not be negative"
        ):
            q.collect(engine=engine)


def test_gather_non_integer_indices_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = df.select(pl.col("a").gather(pl.lit("y")))
    assert_ir_translation_raises(q, engine, NotImplementedError)
