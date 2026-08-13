# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("a", [1.0, float("nan"), 3.0, None, float("nan")]),
        pl.Series("a", [1.0, 2.0, 3.0]),
        pl.Series("a", [float("nan"), float("nan")]),
        pl.Series("a", [None, None], dtype=pl.Float64),
        pl.Series("a", [1, 2, None, 3], dtype=pl.Int64),
        pl.Series("a", [], dtype=pl.Float64),
    ],
)
def test_drop_nans(engine: pl.GPUEngine, series: pl.Series) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").drop_nans())
    assert_gpu_result_equal(q, engine=engine)
