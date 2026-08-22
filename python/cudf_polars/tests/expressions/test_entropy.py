# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture(params=[pl.Int32, pl.Int64, pl.UInt32, pl.Float32, pl.Float64])
def dtype(request):
    return request.param


@pytest.fixture(
    params=[
        [1, 2, 3, 4, 5, 6],
        [1, 2, None, 4, None, 6],
        [0, 1, 2, 3, 4, 5],
        [-1, 2, 3, 4, 5, 6],
    ],
    ids=["values", "with_nulls", "with_zero", "with_negative"],
)
def values(request):
    return request.param


@pytest.fixture
def df(dtype, values):
    if dtype in {pl.UInt32} and any(v is not None and v < 0 for v in values):
        pytest.skip("negative values with unsigned dtype")
    return pl.LazyFrame({"a": pl.Series(values, dtype=dtype)})


@pytest.mark.parametrize("base", [math.e, 2.0, 10.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_entropy(engine, df, base, normalize):
    q = df.select(pl.col("a").entropy(base=base, normalize=normalize))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_entropy_default(engine, df):
    q = df.select(pl.col("a").entropy())
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize("data", [[5.0], [None], [], [None, None]])
@pytest.mark.parametrize("normalize", [True, False])
def test_entropy_edge_cases(engine, data, normalize):
    q = pl.LazyFrame({"a": pl.Series(data, dtype=pl.Float64)}).select(
        pl.col("a").entropy(normalize=normalize)
    )
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize(
    "data",
    [
        pl.Series(["a", "b"], dtype=pl.String),
        pl.Series([True, False], dtype=pl.Boolean),
    ],
    ids=["string", "boolean"],
)
def test_entropy_non_numeric_unsupported(engine, data):
    q = pl.LazyFrame({"a": data}).select(pl.col("a").entropy())
    assert_ir_translation_raises(q, engine, NotImplementedError)
