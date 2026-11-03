# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("a", [1, 2, 3, 4, 5]),
        pl.Series("a", [10, 20, 30]),
        pl.Series("a", [1, None, 3, None, 5]),
        pl.Series("a", [1.5, float("nan"), 3.0, 0.0]),
        pl.Series("a", [], dtype=pl.Int64),
    ],
)
def test_arg_where(engine: pl.GPUEngine, series: pl.Series) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.arg_where(pl.col("a") % 2 == 0))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("a", [True, False, True, True, False]),
        pl.Series("a", [True, False, None, True, None]),
        pl.Series("a", [False, False, False]),
        pl.Series("a", [True, True, True]),
        pl.Series("a", [None, None], dtype=pl.Boolean),
    ],
)
def test_arg_where_boolean(engine: pl.GPUEngine, series: pl.Series) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.arg_where(pl.col("a")))
    assert_gpu_result_equal(q, engine=engine)


def test_arg_true(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 1, 2, 1]})
    q = lf.select((pl.col("a") == 1).arg_true())
    assert_gpu_result_equal(q, engine=engine)
