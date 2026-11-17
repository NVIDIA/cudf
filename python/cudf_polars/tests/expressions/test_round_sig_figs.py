# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "series",
    [
        pl.Series([123.456, 0.0012345, -987.0, 0.0, None, float("nan")]),
        pl.Series([1.0, 10.0, 100.0, 1000.0, 0.1, 0.01]),
        pl.Series([12345, 678, 9, 0, -54321]),
        pl.Series([], dtype=pl.Float64),
    ],
)
@pytest.mark.parametrize("digits", [1, 2, 3, 5])
def test_round_sig_figs(engine: pl.GPUEngine, series: pl.Series, digits: int) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").round_sig_figs(digits))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)
