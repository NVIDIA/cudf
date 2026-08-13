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
from cudf_polars.utils.versions import POLARS_VERSION_LT_140, POLARS_VERSION_LT_142

dtypes = [
    pl.Int8,
    pl.Int16,
    pl.Int64,
    pl.UInt8,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
]


@pytest.fixture(params=dtypes)
def ltype(request):
    return request.param


@pytest.fixture(params=dtypes)
def rtype(request):
    return request.param


@pytest.fixture(
    params=[
        pl.Expr.eq,
        pl.Expr.eq_missing,
        pl.Expr.ne,
        pl.Expr.ne_missing,
        pl.Expr.lt,
        pl.Expr.le,
        pl.Expr.gt,
        pl.Expr.ge,
        pl.Expr.add,
        pl.Expr.sub,
        pl.Expr.mul,
        pl.Expr.truediv,
        pl.Expr.floordiv,
        pl.Expr.mod,
    ],
    ids=lambda fn: fn.__name__,
)
def binop(request):
    return request.param


@pytest.fixture
def df(request, ltype, rtype, with_nulls, binop):
    a = [1, 2, 3, 5, 8]
    if with_nulls:
        a[2] = None
        a[-1] = None
    b = [10, 20, 30, 50, 0]
    if with_nulls:
        b[1] = None
        b[3] = None
        b[-1] = None

    return pl.LazyFrame({"a": a, "b": b}, schema={"a": ltype, "b": rtype})


def test_numeric_binop(engine: pl.GPUEngine, df, binop):
    left = pl.col("a")
    right = pl.col("b")

    q = df.select(binop(left, right))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("left_scalar", [False, True])
@pytest.mark.parametrize("right_scalar", [False, True])
def test_binop_with_scalar(engine: pl.GPUEngine, left_scalar, right_scalar):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [5, 6, 7]})

    lop = pl.lit(2) if left_scalar else pl.col("a")
    rop = pl.lit(6) if right_scalar else pl.col("b")
    q = df.select(lop / rop)

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("zero", [0, pl.lit(0)])
def test_floor_div_binop_by_zero(engine: pl.GPUEngine, zero, ltype):
    df = pl.LazyFrame({"a": [1, 0, 3]}, schema={"a": ltype})

    q = df.select(pl.col("a") // zero)

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("divisor", [1, 2.0])
def test_true_div_boolean_column(engine: pl.GPUEngine, divisor):
    df = pl.LazyFrame({"a": [True, False]})

    q = df.select(pl.col("a") / divisor)

    assert_gpu_result_equal(q, engine=engine)


def test_true_div_with_decimals(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "foo": [Decimal("1.00"), Decimal("2.00"), Decimal("3.00"), None],
            "bar": [Decimal("4.00"), Decimal("5.00"), Decimal("6.00"), Decimal("1.00")],
        },
        schema={"foo": pl.Decimal(15, 2), "bar": pl.Decimal(15, 2)},
    )
    q = df.select(pl.col("bar") / pl.col("foo"))
    assert_gpu_result_equal(q, engine=engine)


def test_multiply_with_decimals(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "x": [Decimal("1.23"), Decimal("4.56"), Decimal("7.89")],
            "y": [Decimal("2.00"), Decimal("3.00"), Decimal("4.00")],
        },
        schema={"x": pl.Decimal(10, 2), "y": pl.Decimal(10, 3)},
    )

    q = df.select(pl.col("x") * pl.col("y"))
    assert_gpu_result_equal(q, engine=engine)


def test_sum_decimal_widens_precision(request) -> None:
    if POLARS_VERSION_LT_140:
        request.applymarker(
            pytest.mark.xfail(
                reason="Polars does not widen precision for sum of decimals."
            )
        )
    df = pl.LazyFrame(
        {"x": [Decimal("5000000000000.00"), Decimal("5000000000000.00")]},
        schema={"x": pl.Decimal(15, 2)},
    )
    assert df.select(pl.sum("x")).collect_schema()["x"] == pl.Decimal(38, 2)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a") * pl.col("b") + pl.col("c"),
        pl.col("a") - pl.col("b") * pl.col("c"),
        pl.col("a") * pl.col("b") - pl.col("c"),
    ],
    ids=["fma", "fsm", "fms"],
)
def test_fused_arithmetic(engine: pl.GPUEngine, expr: pl.Expr) -> None:
    # fma: fused multiply add
    # fsm: fused subtract multiply
    # fms: fused multiply subtract
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [5, 5, 5]})
    q = df.select(expr)
    if POLARS_VERSION_LT_142:
        assert_ir_translation_raises(q, engine, NotImplementedError)
    else:
        assert_gpu_result_equal(q, engine=engine)
