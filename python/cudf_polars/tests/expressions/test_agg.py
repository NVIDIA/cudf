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
from cudf_polars.testing.engine_utils import is_streaming_engine
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_136,
    POLARS_VERSION_LT_138,
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
    # See https://github.com/NVIDIA/cudf/pull/20791#issuecomment-3750528419
    if is_sorted and POLARS_VERSION_LT_138:
        request.applymarker(
            pytest.mark.xfail(reason="set_sorted lowers to unsupported hint ir")
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


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="polars 1.38.0 introduced max_by and min_by"
)
@pytest.mark.parametrize("expr", ["max_by", "min_by"])
@pytest.mark.parametrize(
    "data,by_dtype",
    [
        ({"a": [1, 2, 2, None, 3, 1], "b": [5, 4, 3, 2, 1, 6]}, pl.Int64),
        ({"a": [1, 2, 3, 4], "b": [5, None, 3, None]}, pl.Int64),
        ({"a": [10, 20, 30, 40], "b": [1, 5, 5, 2]}, pl.Int64),
        ({"a": [None, None, None], "b": [None, None, None]}, pl.Int64),
        ({"a": [7], "b": [3]}, pl.Int64),
        ({"a": [], "b": []}, pl.Int64),
        ({"a": [], "b": []}, pl.Float64),
        ({"a": [None, None, None], "b": [None, None, None]}, pl.Float64),
        ({"a": [7], "b": [3.0]}, pl.Float64),
        ({"a": [10, 20, 30], "b": [None, float("nan"), None]}, pl.Float64),
        ({"a": [10, 20, 30], "b": [float("nan"), None, float("nan")]}, pl.Float64),
        ({"a": [10, 20, 30], "b": [None, None, float("nan")]}, pl.Float64),
    ],
)
def test_max_min_by(engine: pl.GPUEngine, expr: str, data, by_dtype) -> None:
    df = pl.LazyFrame(
        {
            "a": pl.Series(data["a"], dtype=pl.Int64),
            "b": pl.Series(data["b"], dtype=by_dtype),
        }
    )
    q = df.select(getattr(pl.col("a"), expr)("b"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="polars 1.38.0 introduced max_by and min_by"
)
@pytest.mark.parametrize("expr", ["max_by", "min_by"])
@pytest.mark.parametrize(
    "by",
    [
        [1.0, float("nan"), 2.0],
        [float("nan"), 1.0, 2.0],
        [1.0, 2.0, float("nan")],
        [-1.0, float("nan"), 1.0],
        [1.0, float("nan"), float("nan")],
        [float("nan"), float("nan"), 1.0],
        [float("nan"), float("nan"), float("nan")],
    ],
)
def test_max_min_by_float_nan(engine: pl.GPUEngine, expr: str, by) -> None:
    df = pl.LazyFrame(
        {
            "a": pl.Series([10, 20, 30], dtype=pl.Int64),
            "b": pl.Series(by, dtype=pl.Float64),
        }
    )
    q = df.select(getattr(pl.col("a"), expr)("b"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="polars 1.38.0 introduced max_by and min_by"
)
@pytest.mark.parametrize("expr", ["max_by", "min_by"])
@pytest.mark.parametrize(
    "by",
    [
        pl.col("b").filter(pl.col("b") > 20),
        pl.col("b").head(2),
        pl.lit(1),
    ],
)
def test_max_min_by_mismatched_length_raises(
    engine: pl.GPUEngine, expr: str, by
) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    q = df.select(getattr(pl.col("a"), expr)(by))
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.ShapeError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ShapeError):
            q.collect(engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="polars 1.38.0 introduced max_by and min_by"
)
@pytest.mark.parametrize("expr", ["max_by", "min_by"])
def test_max_min_by_scalar_value(engine: pl.GPUEngine, expr: str) -> None:
    df = pl.LazyFrame({"b": [1, 5, 2]})
    q = df.select(getattr(pl.lit(99), expr)("b"))
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pl.exceptions.ShapeError):
            q.collect(engine=engine)
    else:
        with pytest.raises(pl.exceptions.ShapeError):
            q.collect(engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="polars 1.38.0 introduced max_by and min_by"
)
@pytest.mark.parametrize("agg", ["max_by", "min_by"])
@pytest.mark.parametrize(
    "g,a,b,by_dtype",
    [
        ([1, 1, 1, 2, 2, 2], [10, 20, 30, 40, 50, 60], [1, 5, 5, 2, None, 1], pl.Int64),
        (
            [1, 1, 1, 2, 2, 2],
            [10, 20, 30, 40, 50, 60],
            [1.0, float("nan"), 2.0, 5.0, None, 1.0],
            pl.Float64,
        ),
        (
            [1, 1, 1, 2, 2, 2],
            [10, 20, 30, 40, 50, 60],
            [None, None, None, 2.0, 5.0, 1.0],
            pl.Float64,
        ),
        (
            [1, 1, 1, 2, 2, 2],
            [10, 20, 30, 40, 50, 60],
            [1.0, 5.0, 2.0, float("nan"), float("nan"), float("nan")],
            pl.Float64,
        ),
        ([], [], [], pl.Int64),
        ([], [], [], pl.Float64),
        ([1, 2, 3], [10, 20, 30], [None, None, None], pl.Float64),
        ([1], [10], [3.0], pl.Float64),
        ([1], [10], [None], pl.Float64),
        ([1], [10], [float("nan")], pl.Float64),
        (
            [1, 1, 1, 2, 2, 2],
            [10, 20, 30, 40, 50, 60],
            [None, float("nan"), None, float("nan"), None, float("nan")],
            pl.Float64,
        ),
        (
            [1, 1, 1, 2, 2],
            [10, 20, 30, 40, 50],
            [None, None, float("nan"), float("nan"), None],
            pl.Float64,
        ),
    ],
)
def test_groupby_max_min_by(engine: pl.GPUEngine, agg: str, g, a, b, by_dtype) -> None:
    df = pl.LazyFrame(
        {
            "g": pl.Series(g, dtype=pl.Int64),
            "a": pl.Series(a, dtype=pl.Int64),
            "b": pl.Series(b, dtype=by_dtype),
        }
    )
    q = df.group_by("g").agg(getattr(pl.col("a"), agg)("b")).sort("g")
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="polars 1.38.0 introduced max_by and min_by"
)
@pytest.mark.parametrize("agg", ["max_by", "min_by"])
def test_groupby_max_min_by_nested_unsupported(engine: pl.GPUEngine, agg: str) -> None:
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2],
            "a": [10, 20, 30, 40],
            "b": [1, 5, 2, 1],
        }
    )
    q = df.group_by("g").agg(getattr(pl.col("a"), agg)(pl.col("b").sum()))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="polars 1.38.0 introduced max_by and min_by"
)
@pytest.mark.parametrize("agg", ["max_by", "min_by"])
@pytest.mark.parametrize(
    "by",
    [
        pl.lit(1),
        pl.lit(None),
        pl.lit(1) + pl.lit(2),
        pl.lit(pl.Series("s", [7, 3])),
        pl.lit(pl.Series("s", [7, 3, 1, 9])),
    ],
)
def test_groupby_max_min_by_group_length_unsupported(
    engine: pl.GPUEngine, agg: str, by
) -> None:
    df = pl.LazyFrame({"g": [1, 1, 2, 2], "a": [10, 20, 30, 40], "b": [1, 5, 2, 1]})
    q = df.group_by("g").agg(getattr(pl.col("a"), agg)(by))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="polars 1.38.0 introduced max_by and min_by"
)
@pytest.mark.parametrize("agg", ["max_by", "min_by"])
def test_groupby_max_min_by_literal_column_value_unsupported(
    engine: pl.GPUEngine, agg: str
) -> None:
    df = pl.LazyFrame({"g": [1, 1, 2, 2], "a": [10, 20, 30, 40], "b": [1, 5, 2, 1]})
    q = df.group_by("g").agg(
        getattr(pl.lit(pl.Series("s", [7, 3, 1, 9])), agg)(pl.col("b"))
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="polars 1.38.0 introduced max_by and min_by"
)
@pytest.mark.parametrize("agg", ["max_by", "min_by"])
def test_groupby_max_min_by_scalar_value(engine: pl.GPUEngine, agg: str) -> None:
    df = pl.LazyFrame({"g": [1, 1, 2, 2], "a": [10, 20, 30, 40], "b": [1, 5, 2, 1]})
    q = df.group_by("g").agg(getattr(pl.lit(99), agg)(pl.col("b")).alias("v")).sort("g")
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138, reason="polars 1.38.0 introduced max_by and min_by"
)
@pytest.mark.parametrize("agg", ["max_by", "min_by"])
def test_rolling_max_min_by_unsupported(engine: pl.GPUEngine, agg: str) -> None:
    df = pl.LazyFrame(
        {
            "ts": [1, 2, 3, 4],
            "a": [10, 20, 30, 40],
            "b": [1, 5, 2, 3],
        }
    )
    q = df.rolling("ts", period="2i").agg(getattr(pl.col("a"), agg)("b"))
    assert_ir_translation_raises(q, engine, NotImplementedError)


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


@pytest.mark.parametrize("bin_count", [0, 1, 3, 10])
@pytest.mark.parametrize(
    "data",
    [
        pl.Series([1, 2, 2, None, 3, 1], dtype=pl.Int64),
        pl.Series([17, 12, 10, 5, 6, 0, 1, 0, 3, 16], dtype=pl.Int64),
        pl.Series([-3, -1, -2, 0, 2], dtype=pl.Int32),
        pl.Series([1.5, 2.0, 3.0, 2.5, 3.0], dtype=pl.Float64),
        pl.Series([1.0, 2.0, float("nan"), 3.0, None], dtype=pl.Float64),
        pl.Series([5, 5, 5], dtype=pl.Int64),
        pl.Series([7], dtype=pl.Int64),
        pl.Series([], dtype=pl.Int64),
        pl.Series([None, None], dtype=pl.Int64),
        pl.Series([float("nan"), float("nan")], dtype=pl.Float64),
        pl.Series([1.0, float("inf")], dtype=pl.Float64),
        pl.Series([float("-inf"), 1.0], dtype=pl.Float64),
        pl.Series([float("-inf"), float("inf")], dtype=pl.Float64),
        pl.Series([1.0, 2.0, float("inf"), 3.0], dtype=pl.Float64),
        pl.Series([float("inf"), float("inf")], dtype=pl.Float64),
        pl.Series([float("-inf"), float("-inf")], dtype=pl.Float64),
    ],
)
def test_hist(engine: pl.GPUEngine, bin_count: int, data: pl.Series) -> None:
    df = pl.LazyFrame({"a": data})
    q = df.select(pl.col("a").hist(bin_count=bin_count))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_category": True},
        {"include_breakpoint": True},
    ],
)
def test_hist_category_breakpoint_unsupported(engine: pl.GPUEngine, kwargs):
    df = pl.LazyFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int64())})
    q = df.select(pl.col("a").hist(bin_count=3, **kwargs))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "data",
    [
        pl.Series([True, False, True], dtype=pl.Boolean),
        pl.Series([Decimal("1.00"), Decimal("2.00")], dtype=pl.Decimal(10, 2)),
    ],
)
def test_hist_invalid_dtype_raises(engine: pl.GPUEngine, data: pl.Series) -> None:
    df = pl.LazyFrame({"a": data})
    q = df.select(pl.col("a").hist(bin_count=3))
    assert_ir_translation_raises(q, engine, pl.exceptions.InvalidOperationError)


def test_hist_no_bin_count_unsupported(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int64())})
    q = df.select(pl.col("a").hist(include_category=False, include_breakpoint=False))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("expr", [pl.col("a").median(), pl.col("a").quantile(0.5)])
def test_temporal_quantile_median_not_supported(engine: pl.GPUEngine, expr):
    df = pl.LazyFrame({"a": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]})
    q = df.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)
