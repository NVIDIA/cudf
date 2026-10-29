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


def test_cast_bool_to_string(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [True, False, None]})
    query = df.select(pl.col("a").cast(pl.String()))
    assert_gpu_result_equal(query, engine=engine)


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


@pytest.mark.parametrize("dtype", [pl.UInt16, pl.Int32, pl.Int64])
def test_cast_integer_to_date(engine: pl.GPUEngine, dtype):
    df = pl.LazyFrame({"a": [0, 1, 2, 35, None]}, schema={"a": dtype})
    query = df.select(pl.col("a").cast(pl.Date()))
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize(
    "from_dtype,to_dtype,data",
    [
        (pl.Int16, pl.Int8, [-1, 100, 127, 128, -128, -129, 200, None]),
        (pl.Int16, pl.UInt8, [-1, 0, 255, 256, None]),
        (pl.UInt8, pl.Int8, [0, 127, 128, 255, None]),
        (pl.Int64, pl.UInt64, [-1, 0, 2**62, None]),
        (pl.UInt64, pl.Int64, [0, 2**63 - 1, 2**63, None]),
        (pl.Int8, pl.Int64, [-128, 127, None]),
        (pl.Float64, pl.UInt64, [1.0, -1.0, 2.0**64, float("nan"), None]),
        (pl.Float64, pl.Int32, [2.0**31, -(2.0**31), 1.9, float("inf"), None]),
        (pl.Float32, pl.Int16, [40000.0, -40000.0, 3.5, None]),
    ],
)
def test_non_strict_integer_narrowing_cast_overflow_is_null(
    engine: pl.GPUEngine, from_dtype, to_dtype, data
):
    df = pl.LazyFrame({"a": data}, schema={"a": from_dtype})
    q = df.select(pl.col("a").cast(to_dtype, strict=False))
    assert_gpu_result_equal(q, engine=engine)


def _assert_collect_raises(q, engine, exception):
    with pytest.raises(exception):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(exception):
            q.collect(engine=engine)
    else:
        with pytest.raises(exception):
            q.collect(engine=engine)


@pytest.mark.parametrize(
    "from_dtype,to_dtype,data",
    [
        (pl.Int64, pl.UInt32, [1, -1, None]),
        (pl.Int64, pl.Int32, [2**31, 0]),
        (pl.Int64, pl.Int8, [0, 128, None]),
        (pl.Int16, pl.UInt8, [255, 256]),
        (pl.UInt64, pl.Int64, [0, 2**63]),
        (pl.Int64, pl.UInt64, [0, -1]),
        (pl.Float64, pl.UInt64, [1.0, -1.0, None]),
        (pl.Float64, pl.Int32, [0.5, 2.0**31]),
        (pl.Float64, pl.Int64, [2.0**63, None]),
        (pl.Float64, pl.Int8, [float("nan")]),
        (pl.Float64, pl.Int64, [float("inf")]),
        (pl.Float32, pl.Int16, [40000.0]),
    ],
)
def test_strict_integer_cast_out_of_range_raises(
    engine: pl.GPUEngine, from_dtype, to_dtype, data
):
    df = pl.LazyFrame({"a": data}, schema={"a": from_dtype})
    q = df.select(pl.col("a").cast(to_dtype))
    _assert_collect_raises(q, engine, pl.exceptions.InvalidOperationError)


@pytest.mark.parametrize(
    "from_dtype,to_dtype,data",
    [
        (pl.Int64, pl.Int32, [-(2**31), 2**31 - 1, None]),
        (pl.Int16, pl.UInt8, [0, 255, None]),
        (pl.UInt64, pl.Int64, [0, 2**63 - 1]),
        (pl.Int8, pl.Int64, [-128, 127]),
        (pl.Float64, pl.Int32, [-(2.0**31), 2.0**31 - 1, 1.9, -1.9, None]),
        (pl.Float64, pl.UInt64, [0.0, 2.0**63, 1.0e19, None]),
        (pl.Float64, pl.Int64, [-(2.0**63), 9.0e18]),
        (pl.Float32, pl.Int16, [32767.0, -32768.0, None]),
        (pl.Int64, pl.UInt8, []),
    ],
)
def test_strict_integer_cast_in_range(engine: pl.GPUEngine, from_dtype, to_dtype, data):
    df = pl.LazyFrame({"a": data}, schema={"a": from_dtype})
    q = df.select(pl.col("a").cast(to_dtype))
    assert_gpu_result_equal(q, engine=engine)


def test_wrapping_integer_cast_still_wraps(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [-1, 200, None]}, schema={"a": pl.Int16})
    q = df.select(pl.col("a").cast(pl.Int8, wrap_numerical=True))
    assert_gpu_result_equal(q, engine=engine)
