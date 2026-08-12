# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import random
from datetime import date
from decimal import Decimal

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.dsl.expressions.aggregation import Agg
from cudf_polars.dsl.expressions.base import Col, ExecutionContext, NamedExpr
from cudf_polars.dsl.utils.aggregations import decompose_single_agg
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_136,
    POLARS_VERSION_LT_140,
)


@pytest.fixture
def df():
    lf = pl.LazyFrame(
        {
            "key1": [1, 1, 1, 2, 3, 1, 4, 6, 7],
            "key2": [2, 2, 2, 2, 6, 1, 4, 6, 8],
            "int": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "int32": pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=pl.Int32()),
            "uint16_with_null": pl.Series(
                [1, None, 2, None, None, None, 4, 5, 6], dtype=pl.UInt16()
            ),
            "float": [7.0, 1, 2, 3, 4.5, 5, 6, 7, 8],
            "string": ["abc", "def", "hijk", "lmno", "had", "to", "be", "or", "not"],
            "datetime": [
                date(1970, 1, 1),
                date(1972, 1, 10),
                date(2000, 1, 1),
                date(2004, 12, 1),
                date(2004, 10, 1),
                date(1971, 2, 1),
                date(2003, 12, 1),
                date(2001, 1, 1),
                date(1999, 12, 31),
            ],
        }
    )
    return lf.with_columns(
        pl.col("float").cast(pl.Decimal(precision=9, scale=2)).alias("decimal")
    )


@pytest.fixture(
    params=[
        [pl.col("key1")],
        [pl.col("key2")],
        [pl.col("key1"), pl.lit(1)],
        [pl.col("key1") * pl.col("key2")],
        [pl.col("key1"), pl.col("key2")],
        [pl.col("key1") == pl.col("key2")],
        [pl.col("key2"), pl.col("key1") == pl.lit(1, dtype=pl.Int64)],
    ],
    ids=lambda keys: "-".join(map(str, keys)),
)
def keys(request):
    return request.param


_EXPRS: list[list[pl.Expr | str]] = [
    [],
    ["int"],
    ["float", "int"],
    [pl.col("float") + pl.col("int")],
    [pl.col("float").is_not_null()],
    [pl.col("int32").sum()],
    [pl.col("int32").mean()],
    [
        pl.col("uint16_with_null").sum(),
        pl.col("uint16_with_null").mean().alias("mean"),
    ],
    [pl.col("float").max() - pl.col("int").min() + pl.col("int").max()],
    [pl.col("float").mean(), pl.col("int").std()],
    [(pl.col("float") - pl.lit(2)).max()],
    [pl.lit(10).alias("literal_value")],
    [pl.col("float").sum().round(decimals=1)],
    [pl.col("float").round(decimals=1).sum()],
    [pl.col("float").sum().round()],
    [pl.col("float").round().sum()],
    [pl.col("int").first(), pl.col("float").last()],
    [pl.col("int").sum(), pl.col("string").str.replace("h", "foo", literal=True)],
    [pl.col("float").quantile(0.3, interpolation="nearest")],
    [pl.col("float").quantile(0.3, interpolation="higher")],
    [pl.col("float").quantile(0.3, interpolation="lower")],
    [pl.col("float").quantile(0.3, interpolation="midpoint")],
    [pl.col("float").quantile(0.3, interpolation="linear")],
    [
        pl.col("datetime").max(),
        pl.col("datetime").max().dt.is_leap_year().alias("leapyear"),
    ],
    # polars gives us precision=None, which we
    # do not support
    [pl.col("decimal").median()],
]


@pytest.fixture(
    params=_EXPRS,
    ids=lambda aggs: "-".join(map(str, aggs)),
)
def exprs(request):
    return request.param


@pytest.fixture(
    params=[False, True],
    ids=["no_maintain_order", "maintain_order"],
)
def maintain_order(request):
    return request.param


def test_groupby(engine: pl.GPUEngine, df: pl.LazyFrame, maintain_order, keys, exprs):
    q = df.group_by(*keys, maintain_order=maintain_order).agg(*exprs)

    if not maintain_order:
        sort_keys = list(q.collect_schema().keys())[: len(keys)]
        q = q.sort(*sort_keys)

    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.Int32, pl.Int64, pl.UInt16, pl.Float64],
)
def test_groupby_product_all_null_1(engine: pl.GPUEngine, dtype: pl.DataType) -> None:
    lf = pl.LazyFrame(
        {
            "key": [1, 1, 2, 2, 3, 3, 3],
            "value": pl.Series([2, 3, 4, None, None, None, None], dtype=dtype),
        }
    )
    q = lf.group_by("key").agg(pl.col("value").product()).sort("key")
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_groupby_sorted_keys(
    engine: pl.GPUEngine,
    df: pl.LazyFrame,
    keys,
    exprs,
    request,
):
    request.applymarker(
        pytest.mark.xfail(
            is_streaming_engine(engine),
            strict=False,
            reason="https://github.com/rapidsai/cudf/issues/21642 -  no deterministic sort for keys",
        )
    )
    sorted_keys = [
        key.sort(descending=descending)
        for key, descending in zip(keys, itertools.cycle([False, True]))
    ]

    q = df.group_by(*sorted_keys).agg(*exprs)

    schema = q.collect_schema()
    sort_keys = list(schema.keys())[: len(keys)]
    # Multiple keys don't do sorting
    qsorted = q.sort(*sort_keys)
    if len(keys) > 1:
        # https://github.com/pola-rs/polars/issues/17556
        # Can't assert that the query without post-sorting fails,
        # since it _might_ pass.
        assert_gpu_result_equal(qsorted, engine=engine, check_exact=False)
    elif schema[sort_keys[0]] == pl.Boolean():
        # Boolean keys don't do sorting, so we get random order
        assert_gpu_result_equal(qsorted, engine=engine, check_exact=False)
    else:
        assert_gpu_result_equal(q, engine=engine, check_exact=False)


def test_groupby_len(engine: pl.GPUEngine, df, keys):
    q = df.group_by(*keys).agg(pl.len())

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_item(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "bucket": [1, 2, 3],
            "value": [10.0, None, 30.0],
        }
    )
    q = lf.group_by("bucket").agg(pl.col("value").item())

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_item_empty_result(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        schema={
            "bucket": pl.Int64,
            "value": pl.Float64,
        }
    )
    q = lf.group_by("bucket").agg(pl.col("value").item(allow_empty=True))

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_filtered_item_allow_empty(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "bucket": [1, 1, 2, 2, 3],
            "exchange": ["A", "B", "A", "B", "A"],
            "value": [10.0, 20.0, 30.0, 40.0, None],
        }
    )
    q = lf.group_by("bucket").agg(
        A=pl.col("value").filter(pl.col("exchange") == "A").item(allow_empty=True),
        B=pl.col("value").filter(pl.col("exchange") == "B").item(allow_empty=True),
        C=pl.col("value").filter(pl.col("exchange") == "C").item(allow_empty=True),
    )

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_filtered_item_missing_cell_raise(engine_raise_on_fail):
    lf = pl.LazyFrame(
        {
            "bucket": [1],
            "exchange": ["B"],
            "value": [10.0],
        }
    )
    q = lf.group_by("bucket").agg(
        A=pl.col("value").filter(pl.col("exchange") == "A").item(),
    )

    with pytest.raises(
        pl.exceptions.ComputeError,
        match="aggregation 'item' expected a single value, got none",
    ):
        q.collect(engine=engine_raise_on_fail)


def test_groupby_filtered_item_nested_agg_raises(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "bucket": [1, 1],
            "value": [10.0, 20.0],
        }
    )
    q = lf.group_by("bucket").agg(
        pl.col("value").filter(pl.col("value").sum() > 0).item(allow_empty=True),
    )

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_groupby_item_nested_agg_raises(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "bucket": [1, 1],
            "value": [10.0, 20.0],
        }
    )
    q = lf.group_by("bucket").agg(
        pl.col("value").sum().item(allow_empty=True),
    )

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_item_decomposition_outside_groupby_raises():
    dtype = DataType(pl.Int64())
    allow_empty = True
    named_expr = NamedExpr(
        "value",
        Agg(dtype, "item", allow_empty, ExecutionContext.FRAME, Col(dtype, "value")),
    )
    name_generator = (f"__{i}" for i in itertools.count())

    with pytest.raises(
        NotImplementedError, match="item is only supported in groupby context"
    ):
        decompose_single_agg(
            named_expr,
            name_generator,
            is_top=True,
            context=ExecutionContext.FRAME,
        )


@pytest.mark.skipif(
    POLARS_VERSION_LT_136, reason="LazyFrame.pivot added in Polars 1.36"
)
def test_groupby_pivot_item(engine: pl.GPUEngine):
    lf = pl.LazyFrame(
        {
            "bucket": [1, 1, 2, 2, 3],
            "exchange": ["A", "B", "A", "B", "A"],
            "value": [10.0, 20.0, 30.0, 40.0, None],
        }
    )
    q = lf.pivot(
        on="exchange",
        on_columns=["A", "B", "C"],
        index="bucket",
        values="value",
    )

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_filtered_item_duplicate_cells_raise(engine_raise_on_fail):
    lf = pl.LazyFrame(
        {
            "bucket": [1, 1],
            "exchange": ["A", "A"],
            "value": [10.0, 20.0],
        }
    )
    q = lf.group_by("bucket").agg(
        A=pl.col("value").filter(pl.col("exchange") == "A").item(allow_empty=True),
    )

    with pytest.raises(
        pl.exceptions.ComputeError,
        match="aggregation 'item' expected no or a single value",
    ):
        q.collect(engine=engine_raise_on_fail)


@pytest.mark.parametrize(
    "values, expr",
    [
        ([10.0], pl.col("value").item()),
        ([], pl.col("value").item(allow_empty=True)),
    ],
)
def test_select_item(engine_raise_on_fail, values, expr):
    lf = pl.LazyFrame({"value": pl.Series(values, dtype=pl.Float64)})
    q = lf.select(expr)

    assert_gpu_result_equal(q, engine=engine_raise_on_fail)


@pytest.mark.parametrize(
    "values, match",
    [
        ([], "aggregation 'item' expected a single value, got none"),
        ([10.0, 20.0], "aggregation 'item' expected a single value, got 2 values"),
    ],
)
def test_select_item_raises(engine_raise_on_fail, values, match):
    lf = pl.LazyFrame({"value": pl.Series(values, dtype=pl.Float64)})
    q = lf.select(pl.col("value").item())

    with pytest.raises(pl.exceptions.ComputeError, match=match):
        q.collect(engine=engine_raise_on_fail)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [10.0, 20.0],
        [None, 10.0, 20.0],
    ],
)
def test_select_first_non_null(engine_raise_on_fail, values):
    lf = pl.LazyFrame({"value": pl.Series(values, dtype=pl.Float64)})
    q = lf.select(pl.col("value").drop_nulls().first())

    assert_gpu_result_equal(q, engine=engine_raise_on_fail)


@pytest.mark.parametrize(
    "expr",
    [
        (pl.col("int").max() + pl.col("float").min()).max(),
        (
            pl.when((pl.col("float") - pl.col("float").mean()) > 0)
            .then(pl.col("float"))
            .otherwise(None)
            .sum()
        ),
        (pl.when(pl.col("int") > 5).then(pl.col("float")).otherwise(pl.lit(0.0))),
        (pl.when(pl.col("int").min() >= 3).then(pl.col("float"))),
    ],
)
def test_groupby_unsupported(
    engine: pl.GPUEngine, df: pl.LazyFrame, expr: pl.Expr
) -> None:
    q = df.group_by("key1").agg(expr)

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_groupby_null_keys(engine: pl.GPUEngine, maintain_order):
    df = pl.LazyFrame(
        {
            "key": pl.Series([1, float("nan"), 2, None, 2, None], dtype=pl.Float64()),
            "value": [-1, 2, 1, 2, 3, 4],
        }
    )

    q = df.group_by("key", maintain_order=maintain_order).agg(pl.col("value").min())
    if not maintain_order:
        q = q.sort("key")

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.xfail(reason="https://github.com/pola-rs/polars/issues/17513")
def test_groupby_minmax_with_nan(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {"key": [1, 2, 2, 2], "value": [float("nan"), 1, -1, float("nan")]}
    )

    q = df.group_by("key").agg(
        pl.col("value").max().alias("max"), pl.col("value").min().alias("min")
    )

    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("op", [pl.Expr.nan_max, pl.Expr.nan_min])
def test_groupby_nan_minmax_raises(engine: pl.GPUEngine, op):
    df = pl.LazyFrame(
        {"key": [1, 2, 2, 2], "value": [float("nan"), 1, -1, float("nan")]}
    )

    q = df.group_by("key").agg(op(pl.col("value")))

    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "key",
    [1, pl.col("key1")],
)
@pytest.mark.parametrize(
    "expr",
    [
        pl.lit(1).alias("value"),
        pl.lit([[4, 5, 6]]).alias("value"),
        pytest.param(
            pl.Series("value", [[4, 5, 6]], dtype=pl.List(pl.Int32)),
            marks=pytest.mark.xfail(
                condition=POLARS_VERSION_LT_140,
                reason="polars < 1.40 emits the list literal at its inner dtype, "
                "nested per group, which we mishandle: "
                "https://github.com/rapidsai/cudf/issues/19610",
            ),
        ),
        pl.col("float") * (1 - pl.col("int")),
        [pl.lit(2).alias("value"), pl.col("float") * 2],
    ],
)
def test_groupby_literal_in_agg(engine: pl.GPUEngine, df, key, expr, request):
    # check_row_order=False doesn't work for list aggregations
    # so just sort by the group key
    q = df.group_by(key).agg(expr).sort(key, maintain_order=True)
    if not POLARS_VERSION_LT_140 and isinstance(key, int):
        translator = Translator(q._ldf.visit(), pl.GPUEngine())
        translator.translate_ir()
        if any("implode" in str(e) for e in translator.errors):
            request.applymarker(
                pytest.mark.xfail(
                    reason="group_by(<literal>) wraps element-wise aggs in an "
                    "unsupported implode on polars >= 1.40"
                )
            )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [pl.col("int").unique(), pl.col("int").drop_nulls(), pl.col("int").cum_max()],
)
def test_groupby_unary_non_pointwise_raises(engine: pl.GPUEngine, df, expr):
    q = df.group_by("key1").agg(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_groupby_agg_broadcast_raises(engine: pl.GPUEngine, df):
    q = df.group_by("key1").agg(pl.col("int") + pl.col("float").max())
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "dtype",
    [
        pl.List(pl.Struct({"foo": pl.Int64, "bar": pl.String})),
        pl.List(pl.List(pl.Struct({"foo": pl.Int64, "bar ": pl.String}))),
        pl.List(pl.List(pl.List(pl.Struct({"foo": pl.Int64, "bar": pl.String})))),
    ],
)
def test_groupby_nested_list_struct_raises(engine: pl.GPUEngine, dtype):
    ldf = pl.LazyFrame(
        {
            "key": [1, 2, 3],
            "value": pl.Series([[], [], []], dtype=dtype),
        }
    )
    q = ldf.group_by("key").agg(pl.col("value"))
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize("nkeys", [1, 2, 4])
def test_groupby_maintain_order_random(
    engine: pl.GPUEngine,
    nkeys: int,
    with_nulls: bool,  # noqa: FBT001
):
    nrows = 30
    key_names = [f"key{key}" for key in range(nkeys)]
    rng = random.Random(2)
    key_values = [rng.choices(range(100), k=nrows) for _ in key_names]
    value = rng.choices(range(-100, 100), k=nrows)
    df = pl.DataFrame(dict(zip(key_names, key_values, strict=True), value=value))
    if with_nulls:
        df = df.with_columns(
            *(
                pl.when(pl.col(name) == 1)
                .then(None)
                .otherwise(pl.col(name))
                .alias(name)
                for name in key_names
            )
        )
    q = df.lazy().group_by(key_names, maintain_order=True).agg(pl.col("value").sum())
    assert_gpu_result_equal(q, engine=engine)


def test_groupby_len_with_nulls(engine: pl.GPUEngine):
    df = pl.DataFrame({"a": [1, 1, 1, 2], "b": [1, None, 2, 3]})
    q = df.lazy().group_by("a").agg(pl.col("b").len())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("column", ["int", "string", "uint16_with_null"])
def test_groupby_nunique(engine: pl.GPUEngine, df: pl.LazyFrame, column):
    q = df.group_by("key1").agg(pl.col(column).n_unique())

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("column", ["int", "string", "uint16_with_null"])
def test_groupby_nunique_drop_nulls(engine: pl.GPUEngine, df: pl.LazyFrame, column):
    q = df.group_by("key1").agg(pl.col(column).drop_nulls().n_unique())

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_null_count(engine: pl.GPUEngine, df: pl.LazyFrame):
    q = df.group_by("key1").agg(pl.col("uint16_with_null").null_count())

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("int").all(),
        pl.col("int").any(),
        pl.col("int").is_duplicated(),
        pl.col("int").is_first_distinct(),
        pl.col("int").is_last_distinct(),
        pl.col("int").is_unique(),
    ],
    ids=[
        "all_horizontal",
        "any_horizontal",
        "is_duplicated",
        "is_first_distinct",
        "is_last_distinct",
        "is_unique",
    ],
)
def test_groupby_unsupported_non_pointwise_boolean_function(
    engine: pl.GPUEngine, df: pl.LazyFrame, expr
):
    q = df.group_by("key1").agg(expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_groupby_mean_type_promotion(engine: pl.GPUEngine, df: pl.LazyFrame) -> None:
    df = df.with_columns(pl.col("float").cast(pl.Float32))

    q = df.group_by("key1").agg(pl.col("float").mean())

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_sum_all_null_group_returns_null(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "key": ["a", "a", "b", "b", "c"],
            "null_groups": [None, None, None, 2, None],
        }
    )

    q = df.group_by("key").agg(out=pl.col("null_groups").sum())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "agg_expr",
    [
        pl.all().sum(),
        pl.all().mean(),
        pl.all().median(),
        pl.all().quantile(0.5),
    ],
    ids=["sum", "mean", "median", "quantile-0.5"],
)
def test_groupby_aggs_keep_unsupported_as_null(
    engine: pl.GPUEngine, request, df: pl.LazyFrame, agg_expr
) -> None:
    request.applymarker(
        pytest.mark.xfail(
            condition="sum" in str(agg_expr) and not POLARS_VERSION_LT_136,
            reason="polars raises now",
        )
    )
    lf = df.filter(pl.col("datetime") == date(2004, 12, 1))
    q = lf.group_by("datetime").agg(agg_expr)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.when(pl.col("int") > 5).then(pl.col("float")).otherwise(None).sum(),
        pl.when(pl.col("float").count() > 0)
        .then(pl.col("float").sum())
        .otherwise(None),
        (
            pl.when(pl.col("float").min() < pl.col("float").max())
            .then(pl.col("float").max() - pl.col("float").min())
            .otherwise(pl.lit(0.0))
        ),
        (
            pl.when(pl.col("int").count() > 0)
            .then(
                pl.col("int").cast(pl.Float64).sum()
                / pl.col("int").count().cast(pl.Float64)
            )
            .otherwise(None)
        ),
    ],
    ids=[
        "pre_pointwise_then_sum",
        "post_over_aggs",
        "post_multiple_aggs_range",
        "post_manually_compute_mean",
    ],
)
def test_groupby_ternary_supported(
    engine: pl.GPUEngine, df: pl.LazyFrame, expr: pl.Expr
) -> None:
    q = df.group_by("key1").agg(expr)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize(
    "strategy", ["forward", "backward", "min", "max", "mean", "zero", "one"]
)
def test_groupby_fill_null_with_strategy(engine: pl.GPUEngine, strategy):
    lf = pl.LazyFrame(
        {
            "key": [1, 1, 2, 2, 2],
            "val": [None, 2, None, 4, None],
        }
    )

    q = lf.group_by("key").agg(pl.col("val").fill_null(strategy=strategy))

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_groupby_rank_raises(engine: pl.GPUEngine, df: pl.LazyFrame) -> None:
    q = df.group_by("key1").agg(pl.col("int").rank())

    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_groupby_sum_decimal_null_group(
    engine: pl.GPUEngine, xfail_decimal_sum_precision_polars_140
) -> None:
    df = pl.LazyFrame(
        {"key1": [1, 1, 2, 3], "foo": [None, None, Decimal("1.00"), Decimal("2.00")]},
        schema={"key1": pl.Int32, "foo": pl.Decimal(9, 2)},
    )
    q = df.group_by("key1").agg(pl.col("foo").sum())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_literal_agg(engine: pl.GPUEngine):
    df = pl.LazyFrame({"c0": [True, False]})
    q = df.group_by("c0").agg(pl.lit(1).is_not_null())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_empty_keys_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame({"x": [1, 2, 3]})
    q = df.group_by([]).agg(pl.len())
    if POLARS_VERSION_LT_140:
        assert_ir_translation_raises(q, engine, NotImplementedError)
    else:
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("foo", [[]], dtype=pl.List(pl.Int64())),
        pl.Series("foo", [[[], [1]]], dtype=pl.List(pl.List(pl.Int64()))),
        pl.Series("foo", [[1, 2, 3]], dtype=pl.List(pl.Int64())),
        pl.Series("foo", [[[1, 2], [4]]], dtype=pl.List(pl.List(pl.Int64()))),
        pl.Series(
            "foo",
            [[[[1], [2]], [[3], [4]]]],
            dtype=pl.List(pl.List(pl.List(pl.Int64()))),
        ),
    ],
)
def test_groupby_agg_list_literal(engine: pl.GPUEngine, series: pl.Series) -> None:
    q = pl.LazyFrame({"a": [1, 1, 2]}).group_by("a").agg(series)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
