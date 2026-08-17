# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.dsl import expr
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_136,
)


@pytest.fixture(
    params=[
        # regular aggs from Agg
        "min",
        "max",
        "median",
        "n_unique",
        "first",
        "last",
        "mean",
        "sum",
        "count",
        "std",
        "var",
        # scan aggs from UnaryFunction
        "cum_count",
        "cum_min",
        "cum_max",
        "cum_prod",
        "cum_sum",
    ]
)
def agg(request):
    return request.param


@pytest.fixture(params=[pl.Int32, pl.Float32, pl.Int16, pl.Int8, pl.UInt16])
def dtype(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["unsorted", "sorted"])
def is_sorted(request):
    return request.param


@pytest.fixture
def xfail_if_sorted(is_sorted, request):
    # See https://github.com/rapidsai/cudf/pull/20791#issuecomment-3750528419
    if is_sorted:
        request.applymarker(
            pytest.mark.xfail(reason="See https://github.com/pola-rs/polars/pull/24981")
        )


@pytest.fixture
def df(dtype, with_nulls, is_sorted):
    values = [-10, 4, 5, 2, 3, 6, 8, 9, 4, 4, 5, 2, 3, 7, 3, 6, -10, -11]
    if with_nulls:
        values = [None if v % 5 == 0 else v for v in values]

    if is_sorted:
        values = sorted(values, key=lambda x: -1000 if x is None else x)

    if dtype.is_unsigned_integer():
        values = pl.Series(values).abs()
        if is_sorted:
            values = values.sort()

    df = pl.LazyFrame({"a": values}, schema={"a": dtype})
    if is_sorted:
        return df.set_sorted("a")
    return df


@pytest.fixture
def decimal_df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": pl.Series(
                "a",
                [Decimal("0.10"), Decimal("1.10"), Decimal("100.10")],
                dtype=pl.Decimal(precision=9, scale=2),
            ),
        }
    )


def test_agg(engine: pl.GPUEngine, df, agg, xfail_if_sorted):
    expr = getattr(pl.col("a"), agg)()
    q = df.select(expr)
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_bool_agg(engine: pl.GPUEngine, agg):
    if agg == "cum_min" or agg == "cum_max":
        pytest.skip("Does not apply")
    df = pl.LazyFrame({"a": [True, False, None, True]})
    expr = getattr(pl.col("a"), agg)()
    q = df.select(expr)

    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize(
    "data,dtype",
    [
        ([1, 2, 3, 4], pl.Int32),
        ([1, 2, None, 4], pl.Int32),
        ([1, 0, 3], pl.Int32),
        ([2, 3, 4], pl.Int8),
        ([1.5, 2.0, 3.0], pl.Float64),
        ([1.5, None, 3.0], pl.Float64),
        ([True, False, True], pl.Boolean),
        ([], pl.Int32),
        ([None, None], pl.Int32),
    ],
)
def test_product(engine: pl.GPUEngine, data, dtype):
    df = pl.LazyFrame({"a": pl.Series(data, dtype=dtype)})
    q = df.select(pl.col("a").product())
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_implode(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, None, 3]})
    q = df.select(pl.col("a").implode())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "data,allow_empty",
    [
        ([42], False),
        ([], True),
    ],
)
def test_item(engine: pl.GPUEngine, data, allow_empty) -> None:
    df = pl.LazyFrame({"a": pl.Series(data, dtype=pl.Int64)})
    q = df.select(pl.col("a").item(allow_empty=allow_empty))
    assert_gpu_result_equal(q, engine=engine)


def test_mode(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 2, None, 3, 1]})
    q = df.select(pl.col("a").mode())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.skipif(
    POLARS_VERSION_LT_136, reason="maintain_order added in Polars 1.36.0"
)
def test_mode_maintain_order_unsupported(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"a": [1, 2, 2, None, 3, 1]})
    q = df.select(pl.col("a").mode(maintain_order=True))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("agg", ["arg_max", "arg_min"])
def test_arg_max_min(engine: pl.GPUEngine, agg: str) -> None:
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, 2, None, 3], dtype=pl.Int64),
            "b": pl.Series([None, None, None, None, None], dtype=pl.Int64),
        }
    )
    q = df.select(getattr(pl.col("a"), agg)(), getattr(pl.col("b"), agg)())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "data",
    [
        [1, None, 3, None, 5],
        [None, None, None],
        [1, 2, 3],
        [],
    ],
)
def test_cum_count(engine: pl.GPUEngine, data):
    df = pl.LazyFrame({"a": pl.Series(data, dtype=pl.Int64())})
    q = df.select(pl.col("a").cum_count())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("cum_agg", sorted(expr.UnaryFunction._supported_cum_aggs))
@pytest.mark.parametrize(
    "data,dtype",
    [
        ([1, 2, 3, 4, 5], pl.Int32),
        ([1, None, 3, None, 5], pl.Int32),
        ([None, None, None], pl.Int32),
        ([2, 3, 4], pl.Int8),
        ([1.5, 2.0, 0.5, 4.0], pl.Float64),
        ([], pl.Int32),
    ],
)
def test_cum_agg_reverse(engine: pl.GPUEngine, cum_agg, data, dtype):
    df = pl.LazyFrame({"a": pl.Series(data, dtype=dtype)})
    q = df.select(getattr(pl.col("a"), cum_agg)(reverse=True))
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize("q", [0.5, pl.lit(0.5)])
@pytest.mark.parametrize("interp", ["nearest", "higher", "lower", "midpoint", "linear"])
def test_quantile(engine: pl.GPUEngine, df, q, interp, xfail_if_sorted):
    expr = pl.col("a").quantile(q, interp)
    q = df.select(expr)
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_quantile_invalid_q(engine: pl.GPUEngine, df):
    expr = pl.col("a").quantile(pl.col("a"))
    q = df.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_quantile_equiprobable_unsupported(engine: pl.GPUEngine, df):
    expr = pl.col("a").quantile(0.5, interpolation="equiprobable")
    q = df.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_quantile_duration_unsupported(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": pl.Series([1, 2, 3, 4], dtype=pl.Duration("ns"))})
    q = df.select(pl.col("a").quantile(0.5))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "op", [pl.Expr.min, pl.Expr.nan_min, pl.Expr.max, pl.Expr.nan_max]
)
def test_agg_float_with_nans(engine: pl.GPUEngine, op):
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, float("nan")], dtype=pl.Float64()),
            "b": pl.Series([1, 2, None], dtype=pl.Int8()),
        }
    )
    q = df.select(op(pl.col("a")), op(pl.col("b")))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.xfail(reason="https://github.com/pola-rs/polars/issues/17513")
@pytest.mark.parametrize("op", [pl.Expr.max, pl.Expr.min])
def test_agg_singleton(engine: pl.GPUEngine, op):
    df = pl.LazyFrame({"a": pl.Series([float("nan")])})

    q = df.select(op(pl.col("a")))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("data", [[], [None], [None, 2, 3, None]])
def test_sum_empty_zero(engine: pl.GPUEngine, data):
    df = pl.LazyFrame({"a": pl.Series(values=data, dtype=pl.Int32())})
    q = df.select(pl.col("a").sum())
    assert_gpu_result_equal(q, engine=engine)


def test_decimal_aggs(
    engine: pl.GPUEngine,
    decimal_df: pl.LazyFrame,
    xfail_decimal_sum_precision_polars_140,
) -> None:
    q = decimal_df.with_columns(
        sum=pl.col("a").sum(),
        min=pl.col("a").min(),
        max=pl.col("a").max(),
        mean=pl.col("a").mean(),
        median=pl.col("a").median(),
        mean_f32=pl.col("a").mean().cast(pl.Float32),
        median_f32=pl.col("a").median().cast(pl.Float32),
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("interp", ["nearest", "higher", "lower", "midpoint", "linear"])
def test_decimal_quantile(engine: pl.GPUEngine, decimal_df, interp):
    q = decimal_df.select(pl.col("a").quantile(0.5, interpolation=interp))
    assert_gpu_result_equal(q, engine=engine)


def test_decimal_std_var(engine: pl.GPUEngine, decimal_df):
    q = decimal_df.select(
        std=pl.col("a").std(),
        var=pl.col("a").var(),
    )
    assert_gpu_result_equal(q, engine=engine)


def test_invalid_agg(engine: pl.GPUEngine, request):
    request.applymarker(
        pytest.mark.xfail(
            condition=not POLARS_VERSION_LT_136,
            reason="polars raises now",
        )
    )
    df = pl.LazyFrame({"s": pl.Series(["a", "b", "c"], dtype=pl.String())})
    q = df.select(pl.col("s").sum())
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_sum_all_null_decimal_dtype(
    engine: pl.GPUEngine, xfail_decimal_sum_precision_polars_140
):
    df = pl.LazyFrame({"foo": pl.Series([None], dtype=pl.Decimal(9, 2))})
    q = df.select(pl.col("foo").sum())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("expr", [pl.col("a").median(), pl.col("a").quantile(0.5)])
def test_temporal_quantile_median_not_supported(engine: pl.GPUEngine, expr):
    df = pl.LazyFrame({"a": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]})
    q = df.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").entropy(),
        pl.col("a").skew(),
        pl.col("a").kurtosis(),
    ],
    ids=[
        "entropy",
        "skew",
        "kurtosis",
    ],
)
def test_agg_unsupported(engine: pl.GPUEngine, expr: pl.Expr) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3, 2, 1]})
    q = df.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)
