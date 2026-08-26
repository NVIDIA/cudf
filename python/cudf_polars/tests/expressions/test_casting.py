# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine

_supported_dtypes = [
    (pl.Int8(), pl.Int64()),
    (pl.Date(), pl.Float64()),
    (pl.Datetime("us"), pl.Float64()),
    (pl.Duration("ms"), pl.Float64()),
]

_unsupported_dtypes = [
    (pl.Boolean(), pl.Datetime("ns")),
]


@pytest.fixture
def dtypes(request):
    return request.param


@pytest.fixture
def tests(dtypes):
    fromtype, totype = dtypes
    if fromtype == pl.String():
        data = ["a", "b", "c"]
    elif fromtype == pl.Boolean():
        data = [True, False, True]
    else:
        data = [1, 2, 3]
    return pl.DataFrame(
        {
            "a": pl.Series(data, dtype=fromtype),
        }
    ).lazy(), totype


@pytest.mark.parametrize("dtypes", _supported_dtypes, indirect=True)
def test_cast_supported(engine: pl.GPUEngine, tests):
    df, totype = tests
    q = df.select(pl.col("a").cast(totype))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("dtypes", _unsupported_dtypes, indirect=True)
def test_cast_unsupported(engine: pl.GPUEngine, tests):
    df, totype = tests
    assert_ir_translation_raises(
        df.select(pl.col("a").cast(totype)), engine, NotImplementedError
    )


def test_allow_double_cast(engine: pl.GPUEngine):
    df = pl.LazyFrame({"c0": [1000]})
    query = df.select(pl.col("c0").cast(pl.Boolean).cast(pl.Int8))
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("dtype", [pl.Int64(), pl.Float64()])
@pytest.mark.parametrize("strict", [True, False])
def test_cast_strict_false_string_to_numeric(engine: pl.GPUEngine, dtype, strict):
    df = pl.LazyFrame({"c0": ["1969-12-08 17:00:01", "1", None]})
    query = df.with_columns(pl.col("c0").cast(dtype, strict=strict))
    if strict:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            query.collect()
        if is_streaming_engine(engine):
            with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
                query.collect(engine=engine)
        else:
            with pytest.raises(pl.exceptions.InvalidOperationError):
                query.collect(engine=engine)
    else:
        assert_gpu_result_equal(query, engine=engine)


def test_cast_from_string_unsupported(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": ["True"]})
    query = df.select(pl.col("a").cast(pl.Boolean()))
    assert_ir_translation_raises(query, engine, NotImplementedError)


def test_cast_to_string_unsupported(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [True]})
    query = df.select(pl.col("a").cast(pl.String()))
    assert_ir_translation_raises(query, engine, NotImplementedError)


def test_float_to_decimal_rounding(engine: pl.GPUEngine):
    # See https://github.com/NVIDIA/cudf/pull/21450
    df = pl.LazyFrame(
        {
            "foo": [Decimal("16954168.35")],
            "bar": [Decimal("436736374.77")],
        }
    )
    q = df.select(pl.col("foo") / pl.col("bar"))
    assert_gpu_result_equal(q, engine=engine)
