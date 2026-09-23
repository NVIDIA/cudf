# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "series",
    [
        pl.Series([1, 2, 4, 7, 11]),
        pl.Series([1.0, 2.0, None, 8.0, 16.0]),
        pl.Series([], dtype=pl.Float64),
    ],
)
@pytest.mark.parametrize("n", [1, 2, -1])
def test_pct_change(engine: pl.GPUEngine, series: pl.Series, n: int) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").pct_change(n=n))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_pct_change_non_literal_n(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1.0, 2.0, 4.0, 8.0, 16.0], "n": [2, 2, 2, 2, 2]})
    q = lf.select(pl.col("a").pct_change(n=pl.col("n").first()))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)
