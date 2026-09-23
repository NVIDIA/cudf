# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_139


@pytest.mark.skipif(
    POLARS_VERSION_LT_139, reason="truncate is not supported in polars <1.39"
)
@pytest.mark.parametrize(
    "series",
    [
        pl.Series("a", [1.7, -1.7, 2.5, -2.5, 0.0, 0.99, -0.99]),
        pl.Series("a", [1.98765, -1.98765, 123.456, -123.456]),
        pl.Series("a", [1.5, None, float("nan"), -3.0]),
        pl.Series("a", [3, -3, 5, 0], dtype=pl.Int64),
        pl.Series("a", [None, None, None], dtype=pl.Float64),
        pl.Series("a", [], dtype=pl.Float64),
    ],
)
@pytest.mark.parametrize("decimals", [0, 1, 2, 5])
def test_truncate(engine: pl.GPUEngine, series: pl.Series, decimals: int) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").truncate(decimals))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)
