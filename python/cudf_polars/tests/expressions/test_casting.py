# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import date
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
    ],
)
def test_non_strict_integer_narrowing_cast_overflow_is_null(
    engine: pl.GPUEngine, from_dtype, to_dtype, data
):
    df = pl.LazyFrame({"a": data}, schema={"a": from_dtype})
    q = df.select(pl.col("a").cast(to_dtype, strict=False))
    assert_gpu_result_equal(q, engine=engine)


def _overflowing_temporal_series(case):
    return {
        "date_to_ns": pl.Series([date(2500, 1, 1), date(2000, 1, 1), None]),
        "date_to_us": pl.Series([2**30, 0, -(2**30), None], dtype=pl.Int32).cast(
            pl.Date
        ),
        "date_boundary": pl.Series(
            [106751, 106752, -106751, -106752], dtype=pl.Int32
        ).cast(pl.Date),
        "datetime_ms_to_us": pl.Series([2**62, 0, None]).cast(pl.Datetime("ms")),
        "duration_us_to_ns": pl.Series([2**62, 5, None]).cast(pl.Duration("us")),
    }[case]


TEMPORAL_OVERFLOW_CASES = [
    ("date_to_ns", pl.Datetime("ns")),
    ("date_to_us", pl.Datetime("us", "UTC")),
    ("date_boundary", pl.Datetime("ns")),
    ("datetime_ms_to_us", pl.Datetime("us")),
    ("duration_us_to_ns", pl.Duration("ns")),
]


@pytest.mark.parametrize("case,to_dtype", TEMPORAL_OVERFLOW_CASES)
def test_non_strict_temporal_cast_overflow_is_null(
    engine: pl.GPUEngine, case, to_dtype
):
    df = pl.LazyFrame({"t": _overflowing_temporal_series(case)})
    q = df.select(pl.col("t").cast(to_dtype, strict=False))
    assert_gpu_result_equal(q, engine=engine)


# Polars CPU panics instead of raising for the other cases.
@pytest.mark.parametrize(
    "case,to_dtype",
    [
        case
        for case in TEMPORAL_OVERFLOW_CASES
        if case[0] in {"date_to_ns", "date_boundary", "duration_us_to_ns"}
    ],
)
def test_strict_temporal_cast_overflow_raises(engine: pl.GPUEngine, case, to_dtype):
    df = pl.LazyFrame({"t": _overflowing_temporal_series(case)})
    q = df.select(pl.col("t").cast(to_dtype))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        q.collect()
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)


@pytest.mark.parametrize(
    "from_dtype,to_dtype,data",
    [
        (pl.Date, pl.Datetime("ms"), [date(2500, 1, 1), None]),
        (pl.Datetime("us"), pl.Datetime("ms"), [2**62, None]),
        (pl.Datetime("ns"), pl.Datetime("ns"), [2**62, None]),
        (pl.Duration("ns"), pl.Duration("us"), [2**62, None]),
        (pl.Datetime("ms"), pl.Datetime("us"), [10**12, -(10**12), None]),
    ],
)
def test_temporal_cast_without_overflow(
    engine: pl.GPUEngine, from_dtype, to_dtype, data
):
    series = (
        pl.Series(data)
        if from_dtype == pl.Date
        else pl.Series(data).cast(pl.Int64).cast(from_dtype)
    )
    q = pl.LazyFrame({"t": series}).select(pl.col("t").cast(to_dtype))
    assert_gpu_result_equal(q, engine=engine)


def test_wrapping_integer_cast_still_wraps(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [-1, 200, None]}, schema={"a": pl.Int16})
    q = df.select(pl.col("a").cast(pl.Int8, wrap_numerical=True))
    assert_gpu_result_equal(q, engine=engine)
