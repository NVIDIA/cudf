# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_141,
    POLARS_VERSION_LT_142,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(params=[False, True], ids=["no_nulls", "nulls"])
def has_nulls(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[False, True], ids=["include_nulls", "ignore_nulls"])
def ignore_nulls(request: pytest.FixtureRequest) -> bool:
    return request.param


def test_booleanfunction_reduction(engine: pl.GPUEngine, *, ignore_nulls: bool) -> None:
    ldf = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, 3.0, 2, 5], dtype=pl.Float64()),
            "b": [0, 3, 1, -1, None],
            "c": [1, 6, 5, 3, 2],
        }
    )

    query = ldf.select(
        (pl.col("a") > 3).any(ignore_nulls=ignore_nulls),
        (pl.col("b") > 2).all(ignore_nulls=ignore_nulls),
    )

    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("expr", [pl.Expr.any, pl.Expr.all])
def test_booleanfunction_all_any_kleene(engine: pl.GPUEngine, expr, ignore_nulls):
    ldf = pl.LazyFrame(
        {
            "a": [False, None],
            "b": [False, False],
            "c": [False, True],
            "d": [None, False],
            "e": pl.Series([None, None], dtype=pl.Boolean()),
            "f": [None, True],
            "g": [True, False],
            "h": [True, None],
            "i": [True, True],
        }
    )
    q = ldf.select(expr(pl.col("*"), ignore_nulls=ignore_nulls))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.Expr.is_null,
        pl.Expr.is_not_null,
        pl.Expr.is_nan,
        pl.Expr.is_not_nan,
    ],
    ids=lambda f: f"{f.__name__}()",
)
@pytest.mark.parametrize("has_nans", [False, True], ids=["no_nans", "nans"])
def test_boolean_function_unary(
    engine: pl.GPUEngine,
    expr: Callable[[pl.Expr], pl.Expr],
    *,
    has_nans: bool,
    has_nulls: bool,
) -> None:
    values: list[float | None] = [1, 2, 3, 4, 5]
    if has_nans:
        values[3] = float("nan")
    if has_nulls:
        values[0] = None

    df = pl.LazyFrame({"a": pl.Series(values, dtype=pl.Float32())})

    q = df.select(expr(pl.col("a")), expr(pl.col("a")).not_().alias("b"))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(lambda e: e.is_nan(), id="is_nan"),
        pytest.param(lambda e: e.is_not_nan(), id="is_not_nan"),
        pytest.param(lambda e: e.is_finite(), id="is_finite"),
    ],
)
def test_nan_in_non_floating_point_column(engine: pl.GPUEngine, expr):
    ldf = pl.LazyFrame({"int": [-1, 1, None]}).with_columns(
        float=pl.col("int").cast(pl.Float64),
        float_na=pl.col("int") ** 0.5,
    )

    q = ldf.select(
        [
            expr(pl.col("int")),
            expr(pl.col("float")),
            expr(pl.col("float_na")),
        ]
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").is_finite(),
        pl.col("a").is_infinite(),
        [pl.col("a").is_infinite(), pl.col("b").is_finite()],
    ],
)
def test_boolean_finite(engine: pl.GPUEngine, expr):
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, float("nan"), 2, float("inf")], dtype=pl.Float64()),
            "b": [1, 2, 3, 4],
            "c": pl.Series([1, 2, 3, 4], dtype=pl.Float64()),
        }
    )

    q = df.select(expr)

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("closed", ["both", "left", "right", "none"])
@pytest.mark.parametrize(
    "bounds", [(1, 2), (-1, 10), (11, 10), (pl.col("lo"), pl.col("hi"))]
)
def test_boolean_isbetween(engine: pl.GPUEngine, closed, bounds):
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, float("nan"), 2, 4], dtype=pl.Float32()),
            "lo": [1, 2, 2, 3],
            "hi": [10, 4, 2, 4],
        }
    )

    lower, upper = bounds
    q = df.select(pl.col("a").is_between(lower, upper, closed=closed))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("closed", ["both", "left", "right", "none"])
def test_boolean_isbetween_decimal_float(engine: pl.GPUEngine, closed):
    df = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, 3, 4], dtype=pl.Decimal(scale=2)),
            "lo": pl.Series([0.5, 1.5, 2.5, 3.5], dtype=pl.Float64),
            "hi": pl.Series([1.5, 2.5, 3.5, 4.5], dtype=pl.Float64),
        }
    )

    q = df.select(pl.col("a").is_between(pl.col("lo"), pl.col("hi"), closed=closed))

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr", [pl.any_horizontal("*"), pl.all_horizontal("*")], ids=["any", "all"]
)
@pytest.mark.parametrize("wide", [False, True], ids=["narrow", "wide"])
def test_boolean_horizontal(engine: pl.GPUEngine, expr, has_nulls, wide):
    ldf = pl.LazyFrame(
        {
            "a": [False, False, False, False, False, True],
            "b": [False, False, True, None, None, True],
            "c": [False, True, True, False, True, True],
        }
    )
    if not has_nulls:
        ldf = ldf.select(pl.col("a"), pl.col("c"))

    # To see the All/Any Horizontal nodes, we need a dataframe with
    # more than 128 columns
    if wide:
        ldf = ldf.with_columns(pl.col("c").alias(f"col{i}") for i in range(128))
    q = ldf.select(expr)

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(
            pl.col("a").is_in(pl.col("b").implode()),
            id="implode",
        ),
        pytest.param(pl.col("a").is_in([1, 2, 3]), id="list_small"),
        pytest.param(pl.col("a").is_in([]), id="list_empty"),
        pytest.param(pl.col("a").is_in([3, 4, 2]), id="list_shuffled"),
        pytest.param(pl.col("c").is_in([10, None, 11]), id="list_with_nulls"),
    ],
)
def test_boolean_is_in(engine: pl.GPUEngine, expr):
    ldf = pl.LazyFrame(
        {
            "a": pl.Series([1, 2, 3], dtype=pl.Int64()),
            "b": pl.Series([3, 4, 2], dtype=pl.Int64()),
            "c": pl.Series([1, None, 3], dtype=pl.Int64()),
            "d": pl.Series([10, None, 11], dtype=pl.Int64()),
        }
    )

    q = ldf.select(expr)

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("expr", [pl.Expr.and_, pl.Expr.or_, pl.Expr.xor])
def test_boolean_kleene_logic(engine: pl.GPUEngine, expr):
    ldf = pl.LazyFrame(
        {
            "a": [False, False, False, None, None, None, True, True, True],
            "b": [False, None, True, False, None, True, False, None, True],
        }
    )
    q = ldf.select(expr(pl.col("a"), pl.col("b")))
    assert_gpu_result_equal(q, engine=engine)


def test_boolean_is_in_raises_unsupported(engine: pl.GPUEngine):
    # Needs implode agg
    ldf = pl.LazyFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int64)})
    q = ldf.select(pl.col("a").is_in(pl.lit(1, dtype=pl.Int32())))

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_boolean_is_in_with_nested_list_raises(engine: pl.GPUEngine):
    ldf = pl.LazyFrame({"x": [1, 2, 3], "y": [[1, 2], [2, 3], [4]]})
    q = ldf.select(pl.col("x").is_in(pl.col("y")))
    with pytest.raises(AssertionError, match="DataFrames are different"):
        assert_gpu_result_equal(q, engine=engine)


def test_expr_is_in_empty_list(engine: pl.GPUEngine):
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4]})
    q = ldf.select(pl.col("a").is_in([]))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "needles,haystack",
    [
        (pl.Series([0, 1, 2, 3, 4]), pl.Series([[3], [1]])),
        (pl.Series([1, 2, 3]), pl.Series([[1, 2], [3, 4]])),
        (pl.Series(["a", "b", "c", "d"]), pl.Series([["a"], ["b"]])),
    ],
)
def test_is_in_shape_mismatch_raises(engine: pl.GPUEngine, needles, haystack):
    q = pl.LazyFrame().select(pl.lit(needles).is_in(haystack))
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_boolean_is_close(engine: pl.GPUEngine):
    ldf = pl.LazyFrame({"a": [1.0, 1.2, 1.4, 1.45, 1.6]})
    q = ldf.select(pl.col("a").is_close(1.4, abs_tol=0.1))

    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.skipif(
    POLARS_VERSION_LT_141,
    reason="has_nulls added to polars' BooleanFunction in 1.41",
)
@pytest.mark.parametrize("data", [[1, None, 3], [1, 2, 3], []])
def test_boolean_has_nulls(engine: pl.GPUEngine, data):
    ldf = pl.LazyFrame({"a": pl.Series(data, dtype=pl.Int64)})
    q = ldf.select(pl.col("a").has_nulls())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_141,
    reason="is_empty added to polars' BooleanFunction in 1.41",
)
@pytest.mark.parametrize("data", [[1, 2, 3], [None, None], []])
def test_boolean_is_empty(engine: pl.GPUEngine, data):
    ldf = pl.LazyFrame({"a": pl.Series(data, dtype=pl.Int64)})
    q = ldf.select(pl.col("a").is_empty())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "dtype, col",
    [
        (pl.Int64(), [1, 0, None, -1, 7]),
        (pl.UInt8(), [1, 0, None, 255, 2]),
    ],
)
def test_boolean_not_with_integers(engine: pl.GPUEngine, dtype, col):
    ldf = pl.LazyFrame({"a": pl.Series(col, dtype=dtype)})
    q = ldf.select(~pl.col("a"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_142, reason="IsSorted BooleanFunction was added in polars 1.42"
)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_boolean_is_sorted(
    engine: pl.GPUEngine, *, descending: bool, nulls_last: bool, has_nulls: bool
) -> None:
    values: list[int | None] = [1, 2, 3, 4, 5]
    if has_nulls:
        values[2] = None

    ldf = pl.LazyFrame(
        {
            "asc": pl.Series(values, dtype=pl.Int64()),
            "desc": pl.Series(list(reversed(values)), dtype=pl.Int64()),
            "unsorted": pl.Series([3, 1, None, 4, 2], dtype=pl.Int64()),
        }
    )

    q = ldf.select(
        pl.col("asc").is_sorted(descending=descending, nulls_last=nulls_last),
        pl.col("desc").is_sorted(descending=descending, nulls_last=nulls_last),
        pl.col("unsorted").is_sorted(descending=descending, nulls_last=nulls_last),
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.sum_horizontal("a", "b"),
        pl.mean_horizontal("a", "b"),
        pl.min_horizontal("a", "b"),
    ],
    ids=["sum_horizontal", "mean_horizontal", "min_horizontal"],
)
def test_numeric_horizontal_unsupported(engine: pl.GPUEngine, expr: pl.Expr) -> None:
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    q = df.select(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)
