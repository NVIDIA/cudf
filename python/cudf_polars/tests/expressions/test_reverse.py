# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "series",
    [
        pl.Series([1, 2, 3, 4, 5, 6, 7]),
        pl.Series([1, 2, None, 4, None, 6, 7]),
        pl.Series([1.5, 2.5, float("nan"), -3.0, float("inf"), 0.0, -0.0]),
        pl.Series(["a", "bb", None, "dddd", "e", "ff", "ggg"]),
        pl.Series([1]),
        pl.Series([None, None, None], dtype=pl.Int64),
        pl.Series([], dtype=pl.Int64),
    ],
)
def test_reverse(engine: pl.GPUEngine, series: pl.Series) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").reverse())
    assert_gpu_result_equal(q, engine=engine)
