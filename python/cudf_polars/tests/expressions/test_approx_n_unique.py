# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("a", [1, 2, 2, 3, None], dtype=pl.Int64),
        pl.Series("a", [1.0, float("nan"), float("nan"), None]),
        pl.Series("a", ["a", "a", "b", None]),
        pl.Series("a", [1, 1, 1, 1], dtype=pl.Int64),
        pl.Series("a", [None, None], dtype=pl.Int64),
        pl.Series("a", [], dtype=pl.Int64),
    ],
)
def test_approx_n_unique(engine: pl.GPUEngine, series: pl.Series) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").approx_n_unique())
    assert_gpu_result_equal(q, engine=engine)
