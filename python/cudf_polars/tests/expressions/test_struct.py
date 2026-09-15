# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

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


def test_map_field_names(engine: pl.GPUEngine, ldf):
    q = ldf.select(pl.col("a").name.map_fields(lambda x: x.upper()).struct.unnest())
    assert_ir_translation_raises(q, engine, NotImplementedError)


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
