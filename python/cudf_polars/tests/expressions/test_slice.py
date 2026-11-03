# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "zlice",
    [
        (1,),
        (1, 3),
        (-1,),
    ],
)
def test_slice(engine: pl.GPUEngine, zlice):
    df = pl.LazyFrame({"a": [0, 1, 2, 3], "b": [1, 2, 3, 4]})
    q = df.select(pl.col("a").slice(*zlice))

    assert_gpu_result_equal(q, engine=engine)
