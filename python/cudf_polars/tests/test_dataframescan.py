# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import Select
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_138, POLARS_VERSION_LT_143


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
        pl.col("a").is_null(),
        pl.col("a").count(),
    ],
    ids=[
        "function",
        "binary",
        "non-column",
        "cast-from",
        "cast-to",
        "is-null",
        "count",
    ],
)
def test_array_expression_falls_back(
    in_memory_engine: pl.GPUEngine,
    array_expr: pl.Expr,
):
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


@pytest.mark.skipif(
    POLARS_VERSION_LT_143,
    reason="Polars < 1.43 does not push len() below the union",
)
def test_len_does_not_materialize_dataframescan(
    in_memory_engine: pl.GPUEngine, monkeypatch: pytest.MonkeyPatch
):
    # Using the same frame twice puts a Cache node between each Select(len)
    # and the DataFrameScan, and keeps every column in the scan's projection,
    # so the scan is not zero-width. The row count is known from the polars
    # frame and must not require copying the frame to the GPU.
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    q = pl.concat([df, df]).select(pl.len())

    def fail(*args, **kwargs):
        raise AssertionError("DataFrameScan was materialized to count its rows")

    monkeypatch.setattr(DataFrame, "from_polars", fail)
    assert_gpu_result_equal(q, engine=in_memory_engine)


@pytest.mark.parametrize("len_first", [True, False])
def test_len_and_data_share_dataframescan_cache(
    in_memory_engine: pl.GPUEngine, monkeypatch: pytest.MonkeyPatch, *, len_first
):
    # One consumer of the shared cache only needs the row count, the other
    # needs the data. Whichever is evaluated first, both must be correct. If
    # the data consumer runs first the frame is already cached, so the count
    # must come from the cached frame rather than the fast path.
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    count = df.select(pl.len().alias("x"))
    total = df.select(pl.col("a").sum().cast(pl.UInt32).alias("x"))
    q = pl.concat([count, total] if len_first else [total, count])

    fast_path_calls = []
    len_frame = Select._len_frame

    def spy(self, count, context):
        fast_path_calls.append(count)
        return len_frame(self, count, context)

    monkeypatch.setattr(Select, "_len_frame", spy)
    assert_gpu_result_equal(q, engine=in_memory_engine, check_row_order=False)
    if not POLARS_VERSION_LT_143:
        # Polars < 1.43 plans a projection between the len and the Cache, so
        # the fast path is not reached there; results are still checked above.
        assert len(fast_path_calls) == (1 if len_first else 0)


def _virtual_frame(n: int) -> pl.LazyFrame:
    # A polars frame of n rows that is never materialized on the CPU either.
    return pl.Series([{}], dtype=pl.Struct({})).new_from_index(0, n).to_frame().lazy()


def test_len_of_dataframescan_beyond_size_type(in_memory_engine: pl.GPUEngine):
    # More rows than cudf::size_type can hold; the count must not need the rows.
    n = 2**31 + 1
    q = _virtual_frame(n).select(pl.len())
    assert q.collect(engine=in_memory_engine).item() == n


@pytest.mark.skipif(
    POLARS_VERSION_LT_143,
    reason="Polars < 1.43 does not push len() below the union",
)
def test_len_of_union_exceeding_index_dtype_raises(in_memory_engine: pl.GPUEngine):
    if pl.get_index_type() != pl.UInt32:
        pytest.skip("requires a 32-bit polars index type")
    # Each branch fits the index dtype, their combined length does not. Polars
    # sums in UInt128 and raises when narrowing back; wrapping would be silent.
    lf = _virtual_frame(2**32 - 2)
    q = pl.concat([lf, lf]).select(pl.len())
    with pytest.raises(
        pl.exceptions.InvalidOperationError, match=r"conversion.*failed"
    ):
        q.collect(engine=in_memory_engine)
