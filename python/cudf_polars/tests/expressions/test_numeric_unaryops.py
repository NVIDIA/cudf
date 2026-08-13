# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)

if TYPE_CHECKING:
    from cudf_polars.typing import RankMethod, RoundMethod


@pytest.fixture(
    params=[
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "arcsinh",
        "arccosh",
        "arctanh",
        "exp",
        "sqrt",
        "cbrt",
        "ceil",
        "floor",
        "abs",
        "sign",
        "cot",
        "log1p",
        "degrees",
        "radians",
    ]
)
def op(request):
    return request.param


@pytest.fixture(params=[pl.Int32, pl.Float32])
def dtype(request):
    return request.param


@pytest.fixture
def ldf(with_nulls, dtype):
    values = [1, 2, 4, 5, -2, -4, 0]
    if with_nulls:
        values.append(None)
    if dtype == pl.Float32:
        values.append(-float("inf"))
        values.append(float("nan"))
        values.append(float("inf"))
    elif dtype == pl.Int32:
        values.append(-2_147_483_648)
        values.append(2_147_483_647)
    return pl.LazyFrame(
        {
            "a": pl.Series(values, dtype=dtype),
            "b": pl.Series([i - 4 for i in range(len(values))], dtype=pl.Float32),
            "c": pl.Series([timedelta(hours=i) for i in range(len(values))]),
        }
    )


def test_unary(engine: pl.GPUEngine, ldf, op):
    expr = getattr(pl.col("a"), op)()
    q = ldf.select(expr)
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize("base_literal", [False, True])
@pytest.mark.parametrize("exponent_literal", [False, True])
def test_pow(engine: pl.GPUEngine, ldf, base_literal, exponent_literal):
    base = pl.lit(2) if base_literal else pl.col("a")
    exponent = pl.lit(-3, dtype=pl.Float32) if exponent_literal else pl.col("b")

    q = ldf.select(base.pow(exponent))

    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize("natural", [True, False])
def test_log(engine: pl.GPUEngine, ldf, natural):
    if natural:
        expr = pl.col("a").log()
    else:
        expr = pl.col("a").log(10)

    q = ldf.select(expr)

    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize("col", ["a", "b", "c"])
def test_negate(engine: pl.GPUEngine, ldf, col):
    q = ldf.select(-pl.col(col))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.Int64, pl.UInt8, pl.UInt32, pl.UInt64, pl.Float64],
)
def test_sign_dtypes(engine: pl.GPUEngine, dtype: pl.DataType) -> None:
    values: list[float | None] = [0, 1, 2, 5, None]
    if dtype in (pl.Int8, pl.Int64, pl.Float64):
        values += [-1, -3]
    q = pl.LazyFrame({"a": pl.Series(values, dtype=dtype)}).select(pl.col("a").sign())
    assert_gpu_result_equal(q, engine=engine)


def test_null_count(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "foo": [1, None, 3],
            "bar": [None, None, 1],
            "baz": [1, 2, 3],
        }
    )
    q = lf.select(
        pl.col("foo").is_null().sum(),
        pl.col("bar").is_null().sum(),
        pl.col("baz").is_null().sum(),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
def test_rank_supported(
    engine: pl.GPUEngine,
    ldf: pl.LazyFrame,
    method: RankMethod,
    *,
    descending: bool,
):
    expr = pl.col("a").rank(method=method, descending=descending)
    q = ldf.select(expr)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("method", ["ordinal", "dense", "min", "max", "average"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("test", ["with_nulls", "with_ties"])
def test_rank_methods_with_nulls_or_ties(
    engine: pl.GPUEngine,
    ldf: pl.LazyFrame,
    method: RankMethod,
    *,
    descending: bool,
    test: str,
) -> None:
    base = pl.col("a")
    if test == "with_nulls":
        expr = pl.when((base % 2) == 0).then(None).otherwise(base)
    else:
        expr = pl.when((base % 2) == 0).then(pl.lit(-5)).otherwise(base)

    q = ldf.select(expr.rank(method=method, descending=descending))
    assert_gpu_result_equal(q, engine=engine)


def test_rank_unsupported(engine: pl.GPUEngine, ldf: pl.LazyFrame) -> None:
    expr = pl.col("a").rank(method="random", seed=42)
    q = ldf.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("mode", ["half_to_even", "half_away_from_zero"])
def test_round(engine: pl.GPUEngine, ldf: pl.LazyFrame, mode: RoundMethod) -> None:
    q = ldf.select(pl.col("a").sin().round(2, mode=mode))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Date,
        pl.Datetime("us"),
        pl.Datetime("us", "America/New_York"),
        pl.Duration("us"),
        pl.Int64,
        pl.Float64,
    ],
)
def test_to_physical(engine: pl.GPUEngine, dtype: pl.DataType) -> None:
    values = [1, None, 3]
    s = pl.Series(values, dtype=pl.Int64).cast(dtype)
    df = pl.LazyFrame({"a": s})
    q = df.select(pl.col("a").to_physical())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype",
    [pl.List(pl.Date), pl.Struct({"d": pl.Date})],
)
def test_to_physical_nested_logical_unsupported(
    engine: pl.GPUEngine, dtype: pl.DataType
) -> None:
    df = pl.LazyFrame({"a": pl.Series([None], dtype=dtype)})
    q = df.select(pl.col("a").to_physical())
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "seeds",
    [(0,), (10, 20, 30, 40)],
)
def test_hash(engine: pl.GPUEngine, seeds: tuple[int, ...]) -> None:
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, None], dtype=pl.Int64),
            "b": pl.Series(["x", None, "z"], dtype=pl.String),
        }
    )
    q = df.select(pl.col("a").hash(*seeds), pl.col("b").hash(*seeds))
    # CPU vs GPU hash implementations not guaranteed to be the same
    # Check alignment of result type, stability across GPU collect calls
    result = q.collect(engine=engine)
    assert result.schema == {"a": pl.UInt64, "b": pl.UInt64}
    next_result = q.collect(engine=engine)
    assert_frame_equal(result, next_result)


def test_hash_seed_sensitivity(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": pl.Series([1, 2, 3, None], dtype=pl.Int64)})
    q = df.select(
        same_a=pl.col("a").hash(0, 1, 2, 3),
        same_b=pl.col("a").hash(0, 1, 2, 3),
        diff_tail=pl.col("a").hash(0, 4, 5, 6),
    )
    result = q.collect(engine=engine)
    assert result["same_a"].equals(result["same_b"])
    assert not result["same_a"].equals(result["diff_tail"])


def test_atan2_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]})
    q = df.select(pl.arctan2("y", "x"))
    assert_ir_translation_raises(q, engine, NotImplementedError)
