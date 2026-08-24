# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("a", [1, 2, 3, None, 5], dtype=pl.Int64),
        pl.Series("a", [1.5, float("nan"), None, -3.0]),
        pl.Series("a", ["a", "bb", None, "dddd"]),
        pl.Series("a", [], dtype=pl.Int64),
    ],
)
def test_rechunk(engine: pl.GPUEngine, series: pl.Series) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").rechunk())
    assert_gpu_result_equal(q, engine=engine)
