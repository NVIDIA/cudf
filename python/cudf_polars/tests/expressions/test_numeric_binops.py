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

dtypes = [
    pl.Int8,
    pl.Int16,
    pl.Int64,
    pl.UInt8,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
]

_BINOPS = [
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
]

_COMPARISONS = _BINOPS[:8]

# Preserve the complete promotion matrix for both SPMD configurations. Other
# backends exercise every operator and nullability path with pairs covering
# each dtype in both operand positions, without duplicating the full matrix.
_REPRESENTATIVE_DTYPE_PAIRS = [
    (pl.Int8, pl.Int16),
    (pl.Int16, pl.Int64),
    (pl.Int64, pl.UInt8),
    (pl.UInt8, pl.UInt64),
    (pl.UInt64, pl.Float32),
    (pl.Float32, pl.Float64),
    (pl.Float64, pl.Int8),
]


_SIGNED_INTEGERS = {pl.Int8, pl.Int16, pl.Int64}


def _promotes_to_int128(ltype, rtype):
    return {ltype, rtype} & _SIGNED_INTEGERS and pl.UInt64 in {ltype, rtype}


@pytest.fixture(params=dtypes)
def ltype(request):
    return request.param


@pytest.fixture(params=dtypes)
def rtype(request):
    return request.param


@pytest.fixture(
    params=_BINOPS,
    ids=lambda fn: fn.__name__,
)
def binop(request):
    return request.param


def _make_df(ltype, rtype, with_nulls):
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


@pytest.fixture
def df(ltype, rtype, with_nulls):
    return _make_df(ltype, rtype, with_nulls)


@pytest.mark.engine_params(["spmd", "spmd-small"])
def test_numeric_binop(engine: pl.GPUEngine, df, binop, ltype, rtype):
    left = pl.col("a")
    right = pl.col("b")

    q = df.select(binop(left, right))

    if _promotes_to_int128(ltype, rtype) and binop not in _COMPARISONS:
        assert_ir_translation_raises(q, engine, NotImplementedError)
    else:
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.engine_params(["in-memory", "dask", "ray"])
@pytest.mark.parametrize(
    "ltype,rtype",
    _REPRESENTATIVE_DTYPE_PAIRS,
    ids=[
        f"{ltype.__name__}-{rtype.__name__}"
        for ltype, rtype in _REPRESENTATIVE_DTYPE_PAIRS
    ],
)
@pytest.mark.parametrize("with_nulls", [False, True], ids=["no_nulls", "nulls"])
def test_numeric_binop_non_spmd(engine: pl.GPUEngine, ltype, rtype, with_nulls, binop):
    """Exercise non-SPMD backends with representative promotion pairs."""
    q = _make_df(ltype, rtype, with_nulls).select(binop(pl.col("a"), pl.col("b")))
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


def test_sql_decimal_div(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "revenue": [Decimal("10.00"), Decimal("20.00")],
            "total": [Decimal("40.00"), Decimal("50.00")],
        },
        schema={"revenue": pl.Decimal(15, 2), "total": pl.Decimal(15, 2)},
    )
    q = df.sql("SELECT revenue / total AS share FROM self")
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("scales", [(2, 2), (2, 4), (4, 2)])
def test_sql_decimal_div_rounding_nulls(engine: pl.GPUEngine, scales):
    left_scale, right_scale = scales
    df = pl.LazyFrame(
        {
            "x": [Decimal("1"), Decimal("2"), Decimal("-1"), Decimal("5"), None],
            "y": [Decimal("3"), Decimal("3"), Decimal("3"), None, Decimal("7")],
        },
        schema={"x": pl.Decimal(15, left_scale), "y": pl.Decimal(15, right_scale)},
    )
    q = df.sql("SELECT x / y AS share FROM self")
    assert_gpu_result_equal(q, engine=engine)


def assert_decimal_div_by_zero_raises(q: pl.LazyFrame, engine: pl.GPUEngine):
    match = "division by zero"
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(
            pytest.RaisesExc(pl.exceptions.ComputeError, match=match)
        ):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            q.collect(engine=engine)


def test_sql_decimal_div_by_zero_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {"x": [Decimal("1.00"), Decimal("2.00")], "y": [Decimal("3.00"), Decimal("0")]},
        schema={"x": pl.Decimal(15, 2), "y": pl.Decimal(15, 2)},
    )
    q = df.sql("SELECT x / y AS share FROM self")
    assert_decimal_div_by_zero_raises(q, engine)


def test_sql_decimal_div_by_zero_null_dividend(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {"x": [Decimal("1.00"), None], "y": [Decimal("3.00"), Decimal("0")]},
        schema={"x": pl.Decimal(15, 2), "y": pl.Decimal(15, 2)},
    )
    q = df.sql("SELECT x / y AS share FROM self")
    assert_gpu_result_equal(q, engine=engine)


def test_sql_decimal_div_by_zero_literal(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {"x": [Decimal("1.00"), Decimal("2.00")]}, schema={"x": pl.Decimal(15, 2)}
    )
    q = df.sql("SELECT x / 0.00 AS share FROM self")
    assert_decimal_div_by_zero_raises(q, engine)


@pytest.mark.parametrize("rows", [1, 3], ids=["one-row", "many-rows"])
def test_sql_decimal_div_by_zero_literal_row_counts(engine: pl.GPUEngine, rows):
    df = pl.LazyFrame({"x": [Decimal("1.00")] * rows}, schema={"x": pl.Decimal(15, 2)})
    q = df.sql("SELECT x / 0.00 AS share FROM self")
    assert_decimal_div_by_zero_raises(q, engine)


def test_sql_decimal_div_by_zero_literal_null_dividends(engine: pl.GPUEngine):
    df = pl.LazyFrame({"x": [None, None]}, schema={"x": pl.Decimal(15, 2)})
    q = df.sql("SELECT x / 0.00 AS share FROM self")
    assert_gpu_result_equal(q, engine=engine)


def test_sql_decimal_div_by_nonzero_literal(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {"x": [Decimal("1.00"), None, Decimal("-3.00")]},
        schema={"x": pl.Decimal(15, 2)},
    )
    q = df.sql("SELECT x / 2.00 AS share FROM self")
    assert_gpu_result_equal(q, engine=engine)


def test_sql_decimal_div_by_zero_single_row_column(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {"x": [Decimal("1.00")], "y": [Decimal("0")]},
        schema={"x": pl.Decimal(15, 2), "y": pl.Decimal(15, 2)},
    )
    q = df.sql("SELECT x / y AS share FROM self")
    assert_decimal_div_by_zero_raises(q, engine)


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


def test_sum_decimal_widens_precision() -> None:
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
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "left_dtype,right_dtype,fallback",
    [
        # Both fit in Decimal(38, 4) after rescaling.
        (pl.Decimal(10, 2), pl.Decimal(38, 4), False),
        # No Decimal(38, s) holds both operands; polars 2.1 compares exactly,
        # libcudf would wrap when rescaling.
        (pl.Decimal(38, 2), pl.Decimal(38, 12), True),
        (pl.Decimal(38, 12), pl.Decimal(38, 2), True),
    ],
)
def test_decimal_comparison_mixed_scale_overflow(
    engine: pl.GPUEngine, left_dtype, right_dtype, fallback
):
    big = "9" * (left_dtype.precision - left_dtype.scale) + "." + "9" * left_dtype.scale
    left = pl.DataFrame({"a": ["-2.5", "1", big, None]}).cast(left_dtype)
    right = pl.DataFrame({"b": ["1", "-1.5", "0.5", "3"]}).cast(right_dtype)
    q = (
        pl.concat([left, right], how="horizontal")
        .lazy()
        .select(gt=pl.col("a") > pl.col("b"), eq=pl.col("a") == pl.col("b"))
    )
    if fallback:
        assert_ir_translation_raises(q, engine, NotImplementedError)
    else:
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("dtype", [pl.Decimal(15, 2), pl.Decimal(38, 2)])
@pytest.mark.parametrize("literal", [0, 24, -6, 10_000])
def test_decimal_comparison_with_integer_literal(engine: pl.GPUEngine, dtype, literal):
    # polars types an integer literal as Decimal(38, 0), but its value needs
    # far fewer digits than that, so rescaling it cannot overflow.
    biggest = "9" * (dtype.precision - dtype.scale) + "." + "9" * dtype.scale
    df = pl.DataFrame({"a": ["-2.5", "0", "24", str(literal), biggest, None]}).cast(
        dtype
    )
    a = pl.col("a")
    q = df.lazy().select(
        lt=a < literal,
        le=a <= literal,
        gt=a > literal,
        ge=a >= literal,
        eq=a == literal,
        ne=a != literal,
    )
    assert_gpu_result_equal(q, engine=engine)


def test_decimal_comparison_of_expression_with_integer_literal(
    engine: pl.GPUEngine,
):
    df = pl.DataFrame({"a": ["1.50", "2.25", "3.00"]}).cast(pl.Decimal(7, 2)).lazy()
    q = df.select(gt=(pl.col("a") * 2).sum() > 0, lt=pl.col("a") * 2 < 5)
    assert_gpu_result_equal(q, engine=engine)


def test_decimal_comparison_with_literal_that_may_overflow(engine: pl.GPUEngine):
    # The literal has a finer scale than the column, so rescaling the column
    # to the literal's scale needs 36 + 3 digits.
    df = pl.DataFrame({"a": ["1.00", "2.00"]}).cast(pl.Decimal(38, 2)).lazy()
    q = df.select(gt=pl.col("a") > pl.lit(Decimal("1.005")))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_decimal_comparison_with_literal_of_maximum_precision(engine: pl.GPUEngine):
    # Neither operand needs rescaling, so a literal of 38 digits is supported.
    df = pl.DataFrame({"a": ["1", "9" * 38]}).cast(pl.Decimal(38, 0)).lazy()
    q = df.select(lt=pl.col("a") < pl.lit(Decimal("9" * 38)))
    assert_gpu_result_equal(q, engine=engine)
