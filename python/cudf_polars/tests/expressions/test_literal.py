# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine


@pytest.fixture(
    params=[
        None,
        pl.Int8(),
        pl.Int16(),
        pl.Int32(),
        pl.Int64(),
        pl.UInt8(),
        pl.UInt16(),
        pl.UInt32(),
        pl.UInt64(),
    ]
)
def integer(request):
    return pl.lit(10, dtype=request.param)


@pytest.fixture(params=[None, pl.Float32(), pl.Float64()])
def float(request):
    return pl.lit(1.0, dtype=request.param)


def test_numeric_literal(engine: pl.GPUEngine, integer, float):
    df = pl.LazyFrame({})

    q = df.select(integer=integer, float_=float, sum_=integer + float)

    assert_gpu_result_equal(q, engine=engine)


@pytest.fixture(
    params=[pl.Date(), pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
def timestamp(request):
    return pl.lit(10_000, dtype=request.param)


@pytest.fixture(params=[pl.Duration("ms"), pl.Duration("us"), pl.Duration("ns")])
def timedelta(request):
    return pl.lit(9_000, dtype=request.param)


def test_timelike_literal(engine: pl.GPUEngine, timestamp, timedelta):
    df = pl.LazyFrame({})

    q = df.select(
        time=timestamp,
        delta=timedelta,
        adjusted=timestamp + timedelta,
        two_delta=timedelta + timedelta,
    )
    schema = {k: DataType(v).plc_type for k, v in q.collect_schema().items()}
    if plc.binaryop.is_supported_operation(
        schema["adjusted"],
        schema["time"],
        schema["delta"],
        plc.binaryop.BinaryOperator.ADD,
    ) and plc.binaryop.is_supported_operation(
        schema["two_delta"],
        schema["delta"],
        schema["delta"],
        plc.binaryop.BinaryOperator.ADD,
    ):
        assert_gpu_result_equal(q, engine=engine)
    else:
        assert_ir_translation_raises(q, engine, NotImplementedError)


def test_select_literal_series(engine: pl.GPUEngine):
    df = pl.LazyFrame({})

    q = df.select(
        a=pl.Series(["a", "b", "c"], dtype=pl.String()),
        b=pl.Series([[1, 2], [3], None], dtype=pl.List(pl.UInt16())),
        c=pl.Series([[[1]], [], [[1, 2, 3, 4]]], dtype=pl.List(pl.List(pl.Float32()))),
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "value",
    [
        5,
        "abc",
    ],
)
@pytest.mark.parametrize("n", [0, 1, 3])
def test_repeat_literal(engine: pl.GPUEngine, value, n):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = df.select(pl.repeat(value, n))
    assert_gpu_result_equal(q, engine=engine)


def test_repeat_len(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = df.select(pl.repeat("abc", pl.len()))
    assert_gpu_result_equal(
        q,
        engine=engine,
        collect_kwargs={"optimizations": pl.QueryOptFlags(projection_pushdown=False)},
    )


def test_repeat_null_unsupported(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = df.select(pl.repeat(None, 2))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_repeat_negative_literal_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = df.select(pl.repeat(5, -1))
    assert_ir_translation_raises(q, engine, pl.exceptions.InvalidOperationError)


def test_repeat_negative_expression_result_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [-3, 1, 2]})
    q = df.select(pl.repeat("x", pl.col("a").min()))
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.InvalidOperationError):
            q.collect(engine=engine)
    else:
        with pytest.raises(
            pl.exceptions.InvalidOperationError, match="must not be negative"
        ):
            q.collect(engine=engine)


@pytest.mark.parametrize(
    "expr", [pl.lit(None), pl.lit(datetime.time(12, 0), dtype=pl.Time())]
)
def test_unsupported_literal_raises(engine: pl.GPUEngine, expr):
    df = pl.LazyFrame({})

    q = df.select(expr)

    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "dtype,val",
    [
        (pl.Int64(), 42),
        (pl.Struct({"a": pl.Int64()}), {"a": 1}),
        (pl.List(pl.Int64()), [1, 2, 3]),
    ],
    ids=["int", "dict", "list"],
)
def test_literal_hash(dtype, val):
    assert isinstance(hash(Literal(DataType(dtype), val)), int)


def test_struct_literal_not_supported(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    q = df.select(pl.lit({"x": 1, "y": "foo"}))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_coalesce(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame(
        {
            "a": [None, 2, None, None],
            "b": [1, None, None, 4],
            "c": [10, 20, None, 40],
        }
    )
    q = df.select(pl.coalesce("a", "b", "c"))
    assert_gpu_result_equal(q, engine=engine)


def test_coalesce_with_literal_fill(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [None, 2, None], "b": [1, None, None]})
    q = df.select(pl.coalesce("a", "b", 0))
    assert_gpu_result_equal(q, engine=engine)


def test_coalesce_first_column_no_nulls(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [None, 20, None]})
    q = df.select(pl.coalesce("a", "b"))
    assert_gpu_result_equal(q, engine=engine)


def test_coalesce_mixed_dtypes(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [None, 2, None], "b": [1.5, None, 3.5]})
    q = df.select(pl.coalesce("a", "b"))
    assert_gpu_result_equal(q, engine=engine)


def test_coalesce_strings(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": ["x", None, None], "b": ["p", "q", None]})
    q = df.select(pl.coalesce("a", "b"))
    assert_gpu_result_equal(q, engine=engine)


def test_coalesce_scalar_first(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"b": [1, None, 3]})
    q = df.select(pl.coalesce(pl.lit(5, dtype=pl.Int64), "b"))
    assert_gpu_result_equal(q, engine=engine)


def test_coalesce_null_scalar_first(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"b": [1, None, 3]})
    q = df.select(pl.coalesce(pl.lit(None, dtype=pl.Int64), "b"))
    assert_gpu_result_equal(q, engine=engine)


def test_concat_list_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [[1, 2], [3]], "b": [[4], [5, 6]]})
    q = df.select(pl.concat_list("a", "b"))
    assert_ir_translation_raises(q, engine, NotImplementedError)
