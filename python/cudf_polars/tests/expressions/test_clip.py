# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def ldf() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": [1, 5, 9, None, 5, -3],
            "f": [1.5, 5.5, 9.5, float("nan"), None, -3.0],
            "lo": [2, 2, 2, 2, None, 0],
            "hi": [6, 6, 6, 6, 6, None],
        }
    )


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").clip(2, 6),
        pl.col("a").clip(2),
        pl.col("a").clip(upper_bound=6),
        pl.col("a").clip(None, 6),
        pl.col("f").clip(2.0, 6.0),
        pl.col("a").clip(pl.col("lo"), pl.col("hi")),
        pl.col("a").clip(pl.col("lo")),
        pl.col("f").clip(pl.col("lo"), pl.col("hi")),
    ],
)
def test_clip(engine: pl.GPUEngine, ldf: pl.LazyFrame, expr: pl.Expr) -> None:
    q = ldf.select(expr)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("a", [], dtype=pl.Int64),
        pl.Series("a", [], dtype=pl.Float64),
        pl.Series("a", [None, None, None], dtype=pl.Int64),
        pl.Series("a", [None, None, None], dtype=pl.Float64),
    ],
)
@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").clip(2, 6),
        pl.col("a").clip(2),
        pl.col("a").clip(upper_bound=6),
    ],
)
def test_clip_empty_and_all_null(
    engine: pl.GPUEngine, series: pl.Series, expr: pl.Expr
) -> None:
    q = pl.LazyFrame({"a": series}).select(expr)
    assert_gpu_result_equal(q, engine=engine)


def test_clip_non_literal_scalar_bound(engine: pl.GPUEngine, ldf: pl.LazyFrame) -> None:
    q = ldf.select(
        pl.col("a").clip(
            lower_bound=pl.col("lo").first(), upper_bound=pl.col("hi").last()
        )
    )
    assert_gpu_result_equal(q, engine=engine)
