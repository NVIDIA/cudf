# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.mark.parametrize(
    "dtype,bits",
    [
        (pl.Int8, 8),
        (pl.Int16, 16),
        (pl.Int32, 32),
        (pl.Int64, 64),
        (pl.UInt8, 8),
        (pl.UInt16, 16),
        (pl.UInt32, 32),
        (pl.UInt64, 64),
    ],
)
def test_bitwise_count_ones(
    engine: pl.GPUEngine, dtype: pl.DataType, bits: int
) -> None:
    if dtype.is_signed_integer():
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lo, hi = 0, (1 << bits) - 1
    patterns = (
        0xFCEF_0123,
        0xFFFF_FFFF,
        0xFFF0_FFE1_ABCD_EF01,
        0xAAAA_AAAA_AAAA_AAAA,
    )
    masked = tuple(pattern & ((1 << bits) - 1) for pattern in patterns)
    values = tuple(value - (1 << bits) if value > hi else value for value in masked)
    lf = pl.LazyFrame({"a": pl.Series([0, 1, 6, lo, hi, *values, None], dtype=dtype)})
    q = lf.select(pl.col("a").bitwise_count_ones())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_count_ones_boolean(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [True, False, None]})
    q = lf.select(pl.col("a").bitwise_count_ones())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_count_ones_sql(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"x": [20, 32, 50, 88, 128], "y": [-128, 0, 10, -1, None]})
    q = lf.sql("SELECT BITCOUNT(x) AS x_bits_set, BIT_COUNT(y) AS y_bits_set FROM self")
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_count_ones_literal(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.with_columns(pl.lit(-1, dtype=pl.Int16).bitwise_count_ones().alias("b"))
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_count_ones_all_scalar(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.select(pl.lit(-1, dtype=pl.Int8).bitwise_count_ones())
    assert_gpu_result_equal(q, engine=engine)


def test_bitwise_count_ones_empty(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame({"a": pl.Series([], dtype=pl.Int32)})
    q = lf.select(pl.col("a").bitwise_count_ones())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "func",
    [
        "bitwise_count_zeros",
        "bitwise_leading_ones",
        "bitwise_leading_zeros",
        "bitwise_trailing_ones",
        "bitwise_trailing_zeros",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
    ],
)
def test_bitwise_function_not_supported(
    in_memory_engine: pl.GPUEngine, func: str
) -> None:
    lf = pl.LazyFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int32)})
    q = lf.select(getattr(pl.col("a"), func)())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_bitwise_count_ones_dtype_not_supported() -> None:
    child = expr.Col(DataType(pl.Float64()), "a")
    with pytest.raises(NotImplementedError, match="Float64"):
        expr.BitwiseFunction(
            DataType(pl.UInt32()), expr.BitwiseFunction.Name.CountOnes, (), child
        )
