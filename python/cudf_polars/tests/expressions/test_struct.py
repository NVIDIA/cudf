# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pickle

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars import Translator
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


@pytest.fixture
def ldf():
    return pl.LazyFrame(
        {"a": [{"b": "c", "d": "e", "f": "g"}, {"b": None, "d": "g", "f": "h"}]}
    )


def test_field_getitem(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").struct[0])
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("fields", [("b",), ("b", "d"), ("^b.*|f.*$",)])
def test_field(engine: pl.GPUEngine, ldf, fields):
    q = ldf.select(pl.col("a").struct.field(*fields))
    assert_gpu_result_equal(q, engine=engine)


def test_unnest(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").struct.unnest())
    assert_gpu_result_equal(q, engine=engine)


def test_json_encode(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").struct.json_encode())
    assert_gpu_result_equal(q, engine=engine)

    ldf_newlines = pl.LazyFrame({"a": [{"b": "c\nd", "d": "\r\nz"}]})
    q = ldf_newlines.select(pl.col("a").struct.json_encode())
    assert_gpu_result_equal(q, engine=engine)


def test_json_encode_empty(engine: pl.GPUEngine):
    # ``write_json`` emits no lines for a zero-row input, so the
    # ``from_iterable_of_py(buff.split())`` round-trip cannot infer a
    # dtype. The expression short-circuits to an empty string column.
    ldf = pl.LazyFrame(
        {"a": pl.Series([], dtype=pl.Struct({"b": pl.String, "d": pl.String}))}
    )
    q = ldf.select(pl.col("a").struct.json_encode())
    assert_gpu_result_equal(q, engine=engine)


def test_rename_fields(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").struct.rename_fields(["1", "2", "3"]).struct.unnest())
    assert_gpu_result_equal(q, engine=engine)


def test_with_fields(engine: pl.GPUEngine, ldf):
    q = ldf.select(
        pl.col("a").struct.with_fields(pl.field("b").str.len_chars()).struct.unnest()
    )
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "expr",
    [pl.col("a").name.prefix_fields, pl.col("a").name.suffix_fields],
    ids=lambda x: x.__name__,
)
def test_prefix_suffix_fields(engine: pl.GPUEngine, ldf, expr):
    q = ldf.select(expr("foo").struct.unnest())
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("unnest", [False, True])
def test_map_field_names(
    engine: pl.GPUEngine, *, with_nulls: bool, unnest: bool
) -> None:
    rows: list[dict[str, str | None] | None] = [
        {"b": "c", "d": None},
        {"b": None, "d": "g"},
    ]
    if with_nulls:
        rows.append(None)
    ldf = pl.LazyFrame({"a": rows})
    expr = pl.col("a").name.map_fields(str.upper)
    if unnest:
        expr = expr.struct.unnest()
    q = ldf.select(expr)
    assert_gpu_result_equal(q, engine=engine)


def test_map_field_names_nested(engine: pl.GPUEngine) -> None:
    rows = [
        {"x": {"i": 0, "j": None}, "y": 1},
        None,
        {"x": {"i": None, "j": 2}, "y": None},
        {"x": None, "y": 3},
    ]
    ldf = pl.LazyFrame({"a": rows})
    q = ldf.select(pl.col("a").name.map_fields(lambda x: f"{x}_new"))
    assert_gpu_result_equal(q, engine=engine)


def test_map_field_names_empty(engine: pl.GPUEngine) -> None:
    ldf = pl.LazyFrame(
        {"a": pl.Series([], dtype=pl.Struct({"b": pl.String, "d": pl.Int64}))}
    )
    q = ldf.select(pl.col("a").name.map_fields(str.upper))
    assert_gpu_result_equal(q, engine=engine)


def test_map_field_names_pickle(in_memory_engine: pl.GPUEngine, ldf) -> None:
    q = ldf.select(pl.col("a").name.map_fields(lambda x: x.upper()))
    ir = Translator(q._ldf.visit(), in_memory_engine).translate_ir()
    assert pickle.loads(pickle.dumps(ir)) == ir


@pytest.mark.parametrize("method", ["map_fields", "rename_fields"])
def test_rename_duplicate_field_names(
    in_memory_engine: pl.GPUEngine, method: str
) -> None:
    ldf = pl.LazyFrame({"a": [{"b": 1, "d": 2}]})
    if method == "map_fields":
        expr = pl.col("a").name.map_fields(lambda x: "same")
    else:
        expr = pl.col("a").struct.rename_fields(["x", "x"])
    q = ldf.select(expr)
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


def test_map_field_names_then_field(engine: pl.GPUEngine, ldf) -> None:
    q = ldf.select(pl.col("a").name.map_fields(str.upper).struct.field("D"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").name.map_fields(str.upper),
        pl.col("a").struct.rename_fields(["x"]),
        pl.col("a").name.prefix_fields("p_"),
        pl.col("a").name.suffix_fields("_s"),
    ],
    ids=["map_fields", "rename_fields", "prefix_fields", "suffix_fields"],
)
def test_rename_fields_preserves_sortedness(
    in_memory_engine: pl.GPUEngine, expr: pl.Expr
) -> None:
    ldf = pl.LazyFrame({"a": [{"b": 1}, {"b": 2}]}).set_sorted("a")
    node = Translator(ldf.select(expr)._ldf.visit(), in_memory_engine).translate_ir()
    result = node.evaluate(cache={}, context=IRExecutionContext())
    assert result.columns[0].is_sorted == plc.types.Sorted.YES


@pytest.mark.parametrize("name", [None, "my_count"])
@pytest.mark.parametrize("normalize", [True, False])
def test_value_counts(engine: pl.GPUEngine, ldf, name, normalize):
    # sort=True since order is non-deterministic
    q = ldf.select(pl.col("a").value_counts(sort=True, name=name, normalize=normalize))
    assert_gpu_result_equal(q, engine=engine)


def test_value_counts_normalize_div_by_zero(engine: pl.GPUEngine):
    ldf = pl.LazyFrame({"a": []}, schema={"a": pl.Int64()})
    q = ldf.select(pl.col("a").value_counts(normalize=True))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "n_values, n_nulls",
    [(6, 4), (0, 3), (1, 0), (0, 0)],
    ids=["values_and_nulls", "all_nulls", "one_row", "empty"],
)
@pytest.mark.parametrize("dtype", ["int", "float"])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("set_sorted_on", ["frame", "expr"])
def test_value_counts_sorted(
    engine: pl.GPUEngine,
    *,
    n_values: int,
    n_nulls: int,
    dtype: str,
    sort: bool,
    descending: bool,
    nulls_last: bool,
    set_sorted_on: str,
) -> None:
    if dtype == "float":
        nan = float("nan")
        ascending_values = [1.0, 2.0, 2.0, nan, nan, nan]
        polars_dtype: pl.DataType = pl.Float64()
    else:
        ascending_values = [1, 2, 2, 3, 3, 3]
        polars_dtype = pl.Int64()
    values = ascending_values[:n_values]
    if descending:
        values.reverse()
    nulls = [None] * n_nulls
    ldf = pl.LazyFrame(
        {"a": values + nulls if nulls_last else nulls + values},
        schema={"a": polars_dtype},
    )
    sorted_a = pl.col("a").set_sorted(descending=descending)
    if set_sorted_on == "frame":
        q = ldf.with_columns(sorted_a).select(pl.col("a").value_counts(sort=sort))
    else:
        q = ldf.select(sorted_a.value_counts(sort=sort))
    assert_gpu_result_equal(q, engine=engine, check_row_order=sort)


def test_value_counts_derived_expression(engine: pl.GPUEngine):
    ldf = pl.LazyFrame({"a": [1, 2, 2, 3, 3, 3]})
    q = ldf.select((pl.col("a") + 1).value_counts(sort=True))
    assert_gpu_result_equal(q, engine=engine)


def test_groupby_value_counts_notimplemented(engine: pl.GPUEngine):
    lgb = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]}).group_by("a")
    value_counts_expr = pl.col("b").value_counts()
    q = lgb.agg(value_counts_expr)
    assert_ir_translation_raises(q, engine, NotImplementedError)

    q = lgb.agg(value_counts_expr.first())
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_struct(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.struct(pl.all()))
    assert_gpu_result_equal(q, engine=engine)


def test_nested_struct(engine: pl.GPUEngine):
    ldf = pl.LazyFrame({"a": [{"x": {"i": 0, "j": 0}, "y": {"i": 0, "k": 1}}]})
    q = ldf.select(pl.struct(pl.all()))
    assert_gpu_result_equal(q, engine=engine)


def test_value_counts_with_nulls(engine: pl.GPUEngine, ldf):
    ldf_with_nulls = ldf.select(c=pl.Series(["x", None, "y", "x", None, "x"]))
    q = ldf_with_nulls.select(pl.col("c").value_counts(sort=True))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("literal", [pl.lit(value=False), pl.lit(7)])
def test_struct_literal_field_is_broadcast(
    engine: pl.GPUEngine, literal: pl.Expr
) -> None:
    df = pl.LazyFrame(
        {"a": [1, 2, 3], "b": ["dogs", "cats", None], "c": ["play", "swim", "walk"]}
    )
    q = df.select(pl.struct("a", x="c", y=literal).alias("struct").struct.field("y"))
    assert_gpu_result_equal(q, engine=engine)


def test_group_by_struct_of_literals_empty_frame(engine: pl.GPUEngine) -> None:
    df = pl.LazyFrame({"g": [10, 10, 20], "n": [1, 2, 3]}).head(0)
    q = df.group_by("g").agg(pl.struct(pl.lit(7).alias("lhs"), pl.lit(7).alias("rhs")))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_struct_of_sliced_children(engine: pl.GPUEngine) -> None:
    # Sorting first makes the slice a view with a non-zero offset.
    df = (
        pl.LazyFrame(
            {
                "k": [1, 1, 1, 2, 1, 2, 2, 2, 2, 1],
                "a": [10, None, 30, 40, 50, 60, 20, 70, 90, 80],
                "b": ["x", "y", None, "w", "v", "u", "t", "s", "r", "q"],
                "c": [1.0, 5.0, 2.0, 9.0, 3.0, 4.0, 7.0, 1.5, 2.5, 8.0],
            }
        )
        .sort("c")
        .slice(2, 7)
    )
    q = df.select(pl.struct("a", "b", lit=pl.lit(7)))
    assert_gpu_result_equal(q, engine=engine)

    q_corr = df.group_by("k").agg(pl.corr("a", "c")).sort("k")
    assert_gpu_result_equal(q_corr, engine=engine, check_exact=False)

    # Exercises _StructCreate on a sliced Float64 column in streaming mode.
    q_std = df.group_by("k").agg(pl.col("c").std()).sort("k")
    assert_gpu_result_equal(q_std, engine=engine, check_exact=False)
