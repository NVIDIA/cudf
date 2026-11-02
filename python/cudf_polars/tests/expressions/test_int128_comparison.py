# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for comparisons that Polars widens to Int128.

Polars has no 64-bit type that holds both a UInt64 and an Int64 value, so it casts
both sides of such a comparison to Int128. libcudf has no 128-bit integer type.
"""

from __future__ import annotations

import operator

import pytest

import polars as pl

from cudf_polars.dsl.translate import Translator
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)

COMPARISONS = [
    operator.gt,
    operator.ge,
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
]


@pytest.fixture
def df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "u": [0, 1, 5, 2**63, 2**64 - 1, None, 7],
            "i": [-1, 1, 7, 2**63 - 1, -(2**63), 3, None],
            "i8": [-1, 1, 7, 127, -128, 3, None],
        },
        schema={"u": pl.UInt64, "i": pl.Int64, "i8": pl.Int8},
    )


@pytest.mark.parametrize("op", COMPARISONS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("signed", ["i", "i8"])
def test_uint64_signed_comparison(engine: pl.GPUEngine, df, op, signed):
    q = df.select(op(pl.col("u"), pl.col(signed)).alias("r"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("method", ["eq_missing", "ne_missing"])
def test_uint64_signed_comparison_missing(engine: pl.GPUEngine, df, method):
    q = df.select(getattr(pl.col("u"), method)(pl.col("i")).alias("r"))
    assert_gpu_result_equal(q, engine=engine)


def test_uint64_signed_comparison_filter(engine: pl.GPUEngine, df):
    q = df.filter(pl.col("u") > pl.col("i"))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_count_above_threshold(engine: pl.GPUEngine):
    # The shape of TPC-DS query 9 with Polars' 64-bit index type: a count is
    # compared with a threshold column.
    sales = pl.LazyFrame({"bucket": [1, 1, 2, 2, 2, 3], "value": [1.0, 2, 3, 4, 5, 6]})
    thresholds = pl.LazyFrame({"bucket": [1, 2, 3], "threshold": [1, 5, 0]})
    q = (
        sales.group_by("bucket")
        .agg(pl.len().cast(pl.UInt64).alias("count"), pl.col("value").mean())
        .join(thresholds, on="bucket")
        .select(
            pl.col("bucket"),
            pl.when(pl.col("count") > pl.col("threshold"))
            .then(pl.col("value"))
            .otherwise(pl.lit(-1.0))
            .alias("result"),
        )
    )
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("closed", ["both", "left", "right", "none"])
def test_uint64_signed_is_between(engine: pl.GPUEngine, df, closed):
    q = df.select(
        pl.col("u").is_between(pl.col("i8"), pl.col("i"), closed=closed).alias("r")
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "bounds",
    [
        (0, pl.col("i")),
        (pl.col("i"), 100),
        (2**63, pl.col("u")),
        (pl.col("i8"), 2**63),
    ],
    ids=["literal-low", "literal-high", "uint64-literal-low", "uint64-literal-high"],
)
def test_uint64_signed_is_between_literal_bound(engine: pl.GPUEngine, df, bounds):
    low, high = bounds
    q = df.select(
        pl.col("u").is_between(low, high).alias("unsigned"),
        pl.col("i").is_between(low, high).alias("signed"),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "bounds",
    [
        (pl.col("i8"), pl.col("i")),
        (0, pl.col("i")),
        (pl.col("i"), 100),
        (2**63, pl.col("u")),
    ],
    ids=["columns", "literal-low", "literal-high", "uint64-literal"],
)
def test_uint64_signed_is_between_translates(
    in_memory_engine: pl.GPUEngine, df, bounds
):
    q = df.select(pl.col("u").is_between(*bounds).alias("r"))
    translator = Translator(q._ldf.visit(), in_memory_engine)
    translator.translate_ir()
    assert not translator.errors


@pytest.mark.parametrize("signed", ["i", "i8"])
def test_uint64_signed_comparison_translates(
    in_memory_engine: pl.GPUEngine, df, signed
):
    q = df.select((pl.col("u") > pl.col(signed)).alias("r"))
    translator = Translator(q._ldf.visit(), in_memory_engine)
    translator.translate_ir()
    assert not translator.errors


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("u") + pl.col("i"),
        pl.col("u").cast(pl.Int128),
        (pl.col("u") + pl.col("i")) > 0,
        pl.max_horizontal("u", "i"),
        pl.col("i").cast(pl.Float64).cast(pl.Int128) > pl.col("i").cast(pl.Int128),
    ],
    ids=[
        "arithmetic",
        "cast",
        "arithmetic-in-comparison",
        "max_horizontal",
        "float-cast",
    ],
)
def test_int128_elsewhere_falls_back(in_memory_engine: pl.GPUEngine, df, expr):
    assert_ir_translation_raises(df.select(expr), in_memory_engine, NotImplementedError)
