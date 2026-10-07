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
from cudf_polars.utils.versions import POLARS_VERSION_LT_138


@pytest.mark.parametrize(
    "subset",
    [
        None,
        ["a", "c"],
        ["b", "c", "d"],
        ["b", "d"],
        ["b", "c"],
        ["c", "e"],
        ["d", "e"],
        pl.selectors.string(),
        pl.selectors.integer(),
    ],
)
@pytest.mark.parametrize("predicate_pushdown", [False, True])
def test_scan_drop_nulls(engine: pl.GPUEngine, subset, predicate_pushdown):
    df = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [None, 4, 5, None],
            "c": [6, 7, None, None],
            "d": [8, None, 9, 10],
            "e": [None, None, "A", None],
        }
    )
    # Drop nulls are pushed into filters
    q = df.drop_nulls(subset)

    assert_gpu_result_equal(
        q,
        engine=engine,
        collect_kwargs={
            "optimizations": pl.QueryOptFlags(predicate_pushdown=predicate_pushdown)
        },
    )


def test_can_convert_lists(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "a": pl.Series([[1, 2], [3]], dtype=pl.List(pl.Int8())),
            "b": pl.Series([[1], [2]], dtype=pl.List(pl.UInt16())),
            "c": pl.Series(
                [
                    [["1", "2", "3"], ["4", "567"]],
                    [["8", "9"], []],
                ],
                dtype=pl.List(pl.List(pl.String())),
            ),
            "d": pl.Series([[[1, 2]], []], dtype=pl.List(pl.List(pl.UInt16()))),
        }
    )

    assert_gpu_result_equal(df, engine=engine)


def test_array_pass_through(in_memory_engine: pl.GPUEngine):
    q = (
        pl.LazyFrame(
            {
                "keep": [False, True, True, True],
                "embedding": pl.Series(
                    [[0.0, 0.0], [1.0, None], None, [5.0, 6.0]],
                    dtype=pl.Array(pl.Float32, 2),
                ),
            }
        )
        .filter("keep")
        .select("embedding")
        .slice(0, 2)
    )
    assert_gpu_result_equal(q, engine=in_memory_engine)


def test_array_select_pass_through(in_memory_engine: pl.GPUEngine):
    """Check a direct Array projection alongside a scalar expression."""
    q = pl.LazyFrame(
        {
            "a": pl.Series([[1, 2]], dtype=pl.Array(pl.Int8, 2)),
            "value": [1],
        }
    ).select(
        pl.col("a").alias("renamed"),
        (pl.col("value") + 1).alias("value"),
    )

    assert_gpu_result_equal(q, engine=in_memory_engine)


@pytest.mark.parametrize(
    "array_expr",
    [
        pl.col("a").arr.sum(),
        pl.col("a") == pl.col("b"),
        pl.when("keep").then("a").otherwise("b"),
        pl.col("a").cast(pl.List(pl.Int8)),
        pl.col("values").cast(pl.Array(pl.Int8, 2)),
        pl.col("a").count(),
    ],
    ids=[
        "function",
        "binary",
        "non-column",
        "cast-from",
        "cast-to",
        "count",
    ],
)
def test_array_expression_falls_back(
    in_memory_engine: pl.GPUEngine,
    array_expr: pl.Expr,
):
    """Check that unsupported Array consumers fail during translation."""
    q = pl.LazyFrame(
        {
            "keep": [True],
            "a": pl.Series([[1, 2]], dtype=pl.Array(pl.Int8, 2)),
            "b": pl.Series([[3, 4]], dtype=pl.Array(pl.Int8, 2)),
            "values": pl.Series([[1, 2]], dtype=pl.List(pl.Int8)),
        }
    ).select(array_expr)

    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "array_expr",
    [
        pl.when("keep").then("a").otherwise("b").is_null(),
        pl.col("a").arr.sum().is_null(),
    ],
    ids=["computed-array", "array-reduction"],
)
def test_unsupported_array_expression_null_check_falls_back(
    in_memory_engine: pl.GPUEngine,
    array_expr: pl.Expr,
) -> None:
    """Check that a null predicate cannot hide a computed Array input."""
    q = pl.LazyFrame(
        {
            "keep": [True, False],
            "a": pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int8, 2)),
            "b": pl.Series([[5, 6], [7, 8]], dtype=pl.Array(pl.Int8, 2)),
        }
    ).select(array_expr)

    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "values,dtype",
    [
        (
            [[[1, 2], [3, 4]], None],
            pl.Array(pl.Array(pl.Int8, 2), 2),
        ),
        (
            [["a", "b"], None],
            pl.Array(pl.String, 2),
        ),
    ],
    ids=["nested-array", "variable-width-inner"],
)
@pytest.mark.parametrize(
    "predicate",
    [pl.Expr.is_null, pl.Expr.is_not_null],
    ids=lambda f: f"{f.__name__}()",
)
def test_unsupported_array_dtype_null_check_falls_back(
    in_memory_engine: pl.GPUEngine,
    values: list,
    dtype: pl.DataType,
    predicate,
) -> None:
    """Check that null predicates do not enable unsupported Array dtypes."""
    q = pl.LazyFrame({"a": pl.Series(values, dtype=dtype)}).select(
        predicate(pl.col("a"))
    )

    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("combine", [False, True], ids=["siblings", "combined"])
def test_supported_and_unsupported_array_consumers_fall_back(
    in_memory_engine: pl.GPUEngine,
    *,
    combine: bool,
) -> None:
    """Check that a supported Array consumer cannot hide an unsupported one."""
    df = pl.LazyFrame(
        {
            "a": pl.Series([[1, 2], None], dtype=pl.Array(pl.Int8, 2)),
        }
    )
    is_null = pl.col("a").is_null()
    count = pl.col("a").count()
    q = (
        df.select(is_null | (count > 0))
        if combine
        else df.select(is_null, count.alias("count"))
    )

    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "array_expr",
    [
        pl.col("a").null_count(),
        pl.col("a").has_nulls(),
    ],
    ids=["null-count", "has-nulls"],
)
def test_array_null_reduction_falls_back(
    in_memory_engine: pl.GPUEngine,
    array_expr: pl.Expr,
) -> None:
    """Check that direct Array null reductions remain unsupported."""
    q = pl.LazyFrame(
        {"a": pl.Series([[1, 2], None], dtype=pl.Array(pl.Int8, 2))}
    ).select(array_expr)

    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "array_expr",
    [
        pl.col("a").is_null().sum(),
        pl.col("a").is_not_null().sum(),
        pl.col("a").is_null().any(),
        pl.col("a").is_not_null().all(),
    ],
    ids=["is-null-sum", "is-not-null-sum", "is-null-any", "is-not-null-all"],
)
def test_optimized_array_null_reduction_falls_back(
    in_memory_engine: pl.GPUEngine,
    array_expr: pl.Expr,
) -> None:
    """Check rejection after Polars rewrites Array null-check reductions."""
    q = pl.LazyFrame(
        {"a": pl.Series([[1, 2], None], dtype=pl.Array(pl.Int8, 2))}
    ).select(array_expr)

    with pytest.raises(NotImplementedError, match="unsupported operations"):
        q.collect(engine=in_memory_engine)


@pytest.mark.parametrize(
    "values,dtype",
    [
        ([[[1, 2], [3, 4]]], pl.List(pl.Array(pl.Int8, 2))),
        ([{"x": [1, 2]}], pl.Struct({"x": pl.Array(pl.Int8, 2)})),
    ],
    ids=["list", "struct"],
)
def test_nested_array_falls_back(
    in_memory_engine: pl.GPUEngine,
    values: list,
    dtype: pl.DataType,
):
    q = pl.LazyFrame({"a": pl.Series(values, dtype=dtype)}).select("a")

    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_array_grouped_collection_falls_back(in_memory_engine: pl.GPUEngine):
    q = (
        pl.LazyFrame(
            {
                "key": [1, 1],
                "a": pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int8, 2)),
            }
        )
        .group_by("key")
        .agg("a")
    )

    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize(
    "operation",
    ["sort", "group-by", "join", "unique"],
)
def test_array_consumer_falls_back(
    in_memory_engine: pl.GPUEngine,
    operation: str,
):
    q = pl.LazyFrame(
        {
            "a": pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int8, 2)),
            "value": [1, 2],
        }
    )

    if operation == "sort":
        q = q.sort("a")
    elif operation == "group-by":
        q = q.group_by("a").agg(pl.len())
    elif operation == "join":
        q = q.join(q, on="a")
    else:
        q = q.unique(subset=["a"])

    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_array_explode_falls_back(in_memory_engine: pl.GPUEngine):
    q = pl.LazyFrame(
        {"a": pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int8, 2))}
    ).explode("a")

    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_dataframescan_with_decimals(engine: pl.GPUEngine):
    q = pl.LazyFrame(
        {
            "foo": [1, 2],
            "bar": [Decimal("1.23"), Decimal("4.56")],
        },
        schema={"foo": pl.Int64, "bar": pl.Decimal(precision=15, scale=2)},
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_138,
    reason="height parameter added in Polars 1.38",
)
def test_dataframescan_zero_width_with_rows(engine: pl.GPUEngine):
    df = pl.LazyFrame(height=5)
    q = df.select(pl.len())
    assert_gpu_result_equal(q, engine=engine)


def test_struct_literal_not_supported(engine: pl.GPUEngine):
    dtype = pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.String)])
    q = pl.LazyFrame().select(pl.lit(None, dtype=pl.Null).cast(dtype, strict=True))
    assert_ir_translation_raises(q, engine, NotImplementedError)
