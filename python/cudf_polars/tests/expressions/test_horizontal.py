# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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
