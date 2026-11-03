# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Literal

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "series",
    [
        pl.Series([1, 2, 4, 7, 11]),
        pl.Series([1, 2, None, 7, 11]),
        pl.Series([1.0, 3.0, 6.0, None, 10.0]),
        pl.Series([], dtype=pl.Int64),
    ],
)
@pytest.mark.parametrize("n", [1, 2, -1, 3])
@pytest.mark.parametrize("null_behavior", ["ignore", "drop"])
def test_diff(
    engine: pl.GPUEngine,
    series: pl.Series,
    n: int,
    null_behavior: Literal["ignore", "drop"],
) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").diff(n=n, null_behavior=null_behavior))
    assert_gpu_result_equal(q, engine=engine)


def test_diff_non_literal_n(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 4, 7, 11], "n": [2, 2, 2, 2, 2]})
    q = lf.select(pl.col("a").diff(n=pl.col("n").first()))
    assert_gpu_result_equal(q, engine=engine)
