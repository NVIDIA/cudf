# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(params=[pl.Float32, pl.Float64])
def dtype(request):
    return request.param


@pytest.fixture
def df(dtype, with_nulls):
    a = [-math.e, 10, 22.5, 1.5, 2.5, -1.5, math.pi, 8]
    if with_nulls:
        a[2] = None
        a[-1] = None
    return pl.LazyFrame({"a": a}, schema={"a": dtype})


@pytest.mark.parametrize("decimals", [0, 2, 4])
def test_round(engine: pl.GPUEngine, df, decimals):
    q = df.select(pl.col("a").round(decimals=decimals))

    assert_gpu_result_equal(q, engine=engine, check_exact=False)
