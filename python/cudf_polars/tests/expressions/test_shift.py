# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize("n", [0, 1, 2, -1, -2, 5, -5])
def test_shift(engine: pl.GPUEngine, n):
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    q = df.select(pl.col("a").shift(n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "n,fill_value",
    [
        (1, 0),
        (2, 99),
        (-1, 0),
        (-2, -1),
        (0, 7),
    ],
)
def test_shift_and_fill(engine: pl.GPUEngine, n, fill_value):
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    q = df.select(pl.col("a").shift(n, fill_value=fill_value))
    assert_gpu_result_equal(q, engine=engine)


def test_shift_multiple_columns(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    q = df.select(pl.col("a").shift(2), pl.col("b").shift(-1))
    assert_gpu_result_equal(q, engine=engine)


def test_shift_float(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    q = df.select(pl.col("a").shift(2))
    assert_gpu_result_equal(q, engine=engine)


def test_shift_and_fill_float(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    q = df.select(pl.col("a").shift(2, fill_value=0.0))
    assert_gpu_result_equal(q, engine=engine)


def test_shift_and_fill_expr(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    q = df.select(pl.col("a").shift(n=2, fill_value=pl.col("a").min()))
    assert_gpu_result_equal(q, engine=engine)


def test_shift_string(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": ["x", "y", "z"]})
    q = df.select(pl.col("a").shift(1))
    assert_gpu_result_equal(q, engine=engine)


def test_shift_with_columns(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    q = df.with_columns(shift=pl.col("a").shift(-2, fill_value=100))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [1, -1, 2])
def test_shift_datetime(engine: pl.GPUEngine, n):
    df = pl.LazyFrame(
        {
            "a": [
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2021, 1, 2),
                datetime.datetime(2021, 1, 3),
            ],
            "b": [
                datetime.datetime(2022, 6, 1),
                datetime.datetime(2022, 6, 2),
                datetime.datetime(2022, 6, 3),
            ],
        }
    )
    q = df.select(pl.col("a").shift(n), pl.col("b").shift(-n))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n", [1, -1])
def test_shift_date(engine: pl.GPUEngine, n):
    df = pl.LazyFrame(
        {
            "a": [
                datetime.date(2021, 1, 1),
                datetime.date(2021, 1, 2),
                datetime.date(2021, 1, 3),
            ]
        }
    )
    q = df.select(pl.col("a").shift(n))
    assert_gpu_result_equal(q, engine=engine)


def test_shift_by_expression(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 1, 1, 1, 1]})
    q = df.select(pl.col("a").shift(n=pl.col("b").first()))
    assert_gpu_result_equal(q, engine=engine)


def test_shift_by_expression_last(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [2, 2, 2, 2, 2]})
    q = df.select(pl.col("a").shift(n=pl.col("b").last()))
    assert_gpu_result_equal(q, engine=engine)


def test_shift_by_expression_get(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [2, 2, 2, 2, 2]})
    q = df.select(pl.col("a").shift(n=pl.col("b").get(2)))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").interpolate(),
        pl.col("a").ewm_mean(com=1),
    ],
    ids=["interpolate", "ewm_mean"],
)
def test_shift_like_unsupported(engine: pl.GPUEngine, expr: pl.Expr) -> None:
    df = pl.LazyFrame({"a": [1.0, 2.0, 4.0, 8.0]})
    q = df.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)
