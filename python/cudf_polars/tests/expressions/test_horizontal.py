# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import date

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture
def df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": [1, None, None, 4, 5],
            "b": [10, 20, None, None, 50],
            "c": [100, 200, None, 400, None],
        }
    )


def test_max_horizontal(df: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    q = df.select(pl.max_horizontal("a", "b", "c"))
    assert_gpu_result_equal(q, engine=engine)


def test_max_horizontal_single_column(df: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    q = df.select(pl.max_horizontal("a"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_max_horizontal_all_null_row(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame(
        {"a": [None, 1], "b": [None, 2]}, schema={"a": pl.Int64, "b": pl.Int64}
    )
    q = df.select(pl.max_horizontal("a", "b"))
    assert_gpu_result_equal(q, engine=engine)


def test_max_horizontal_string_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": ["x", None], "b": ["y", "z"]})
    q = df.select(pl.max_horizontal("a", "b"))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("ignore_nulls", [True, False])
def test_mean_horizontal(
    df: pl.LazyFrame,
    engine: pl.GPUEngine,
    ignore_nulls: bool,  # noqa: FBT001
) -> None:
    q = df.select(pl.mean_horizontal("a", "b", "c", ignore_nulls=ignore_nulls))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_mean_horizontal_single_column(df: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    q = df.select(pl.mean_horizontal("a"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_mean_horizontal_mixed_dtypes(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, None], "b": [1.5, None, 3.5]})
    q = df.select(pl.mean_horizontal("a", "b"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_mean_horizontal_bool(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [True, False, True], "b": [True, True, False]})
    q = df.select(pl.mean_horizontal("a", "b"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize(
    "df",
    [
        pl.LazyFrame({"a": ["1", "2"], "b": ["3", "4"]}),
        pl.LazyFrame({"a": [1, 2], "b": ["3", "4"]}),
        pl.LazyFrame({"a": [date(2020, 1, 1)], "b": [date(2020, 1, 3)]}),
    ],
    ids=["str", "mixed_int_str", "date"],
)
def test_mean_horizontal_non_numeric_raises(
    engine: pl.GPUEngine, df: pl.LazyFrame
) -> None:
    q = df.select(pl.mean_horizontal("a", "b"))
    assert_ir_translation_raises(q, engine, pl.exceptions.InvalidOperationError)


def test_min_horizontal(df: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    q = df.select(pl.min_horizontal("a", "b", "c"))
    assert_gpu_result_equal(q, engine=engine)


def test_min_horizontal_single_column(df: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    q = df.select(pl.min_horizontal("a"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_min_horizontal_all_null_row(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame(
        {"a": [None, 1], "b": [None, 2]}, schema={"a": pl.Int64, "b": pl.Int64}
    )
    q = df.select(pl.min_horizontal("a", "b"))
    assert_gpu_result_equal(q, engine=engine)


def test_min_horizontal_string_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": ["x", None], "b": ["y", "z"]})
    q = df.select(pl.min_horizontal("a", "b"))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("ignore_nulls", [True, False])
def test_sum_horizontal(
    df: pl.LazyFrame,
    engine: pl.GPUEngine,
    ignore_nulls: bool,  # noqa: FBT001
) -> None:
    q = df.select(pl.sum_horizontal("a", "b", "c", ignore_nulls=ignore_nulls))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_sum_horizontal_single_column(df: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    q = df.select(pl.sum_horizontal("a"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_sum_horizontal_mixed_dtypes(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, None], "b": [1.5, None, 3.5]})
    q = df.select(pl.sum_horizontal("a", "b"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_sum_horizontal_string_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": ["x", None], "b": ["y", "z"]})
    q = df.select(pl.sum_horizontal("a", "b"))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.fixture(
    params=[
        pl.max_horizontal,
        pl.mean_horizontal,
        pl.min_horizontal,
        pl.sum_horizontal,
    ],
    ids=["max", "mean", "min", "sum"],
)
def horizontal(request: pytest.FixtureRequest):
    return request.param


def test_horizontal_all_literals(
    df: pl.LazyFrame, engine: pl.GPUEngine, horizontal
) -> None:
    q = df.select(horizontal(pl.lit(1), pl.lit(2)))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_horizontal_all_aggregations(
    df: pl.LazyFrame, engine: pl.GPUEngine, horizontal
) -> None:
    q = df.select(horizontal(pl.col("a").sum(), pl.col("b").sum()))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_horizontal_literal_and_column(
    df: pl.LazyFrame, engine: pl.GPUEngine, horizontal
) -> None:
    q = df.select(horizontal(pl.lit(1), pl.col("b")))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_horizontal_all_literals_with_columns(
    df: pl.LazyFrame, engine: pl.GPUEngine, horizontal
) -> None:
    q = df.with_columns(horizontal(pl.lit(1), pl.lit(2)).alias("d"))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)
