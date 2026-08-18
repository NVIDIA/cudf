# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

import pylibcudf as plc

import cudf_polars.dsl.expr as expr_nodes
import cudf_polars.dsl.ir as ir_nodes
from cudf_polars import Translator
from cudf_polars.containers import DataType
from cudf_polars.containers.dataframe import DataFrame, NamedColumn
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.dsl.to_ast import insert_colrefs, to_ast, to_parquet_filter
from cudf_polars.dsl.traversal import traversal
from cudf_polars.utils.cuda_stream import get_cuda_stream


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "c": ["a", "b", "c", "d", "e", "f"],
            "a": [1, 2, 3, None, 4, 5],
            "b": pl.Series([None, None, 3, float("inf"), 4, 0], dtype=pl.Float64),
            "d": [False, True, True, None, False, False],
        }
    )


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("a").is_in([0, 1]),
        pl.col("a").is_between(0, 2),
        (pl.col("a") < pl.col("b")).not_(),
        pl.lit(2) > pl.col("a"),
        pl.lit(2) >= pl.col("a"),
        pl.lit(2) < pl.col("a"),
        pl.lit(2) <= pl.col("a"),
        pl.lit(0) == pl.col("a"),
        pl.lit(1) != pl.col("a"),
        (pl.col("b") < pl.lit(2, dtype=pl.Float64).sqrt()),
        (pl.col("a") >= pl.lit(2)) & (pl.col("b") > 0),
        pl.col("a").is_null(),
        pl.col("a").is_not_null(),
        pl.col("b").is_finite(),
        pytest.param(
            pl.col("a").sin(),
            marks=pytest.mark.xfail(reason="Need to insert explicit casts"),
        ),
        pl.col("b").cos(),
        pl.col("a").abs().is_between(0, 2),
        pl.col("a").ne_missing(pl.lit(None, dtype=pl.Int64)),
        [pl.col("a") * 2, pl.col("b") + pl.col("a")],
        pl.col("d").not_(),
    ],
)
def test_compute_column(expr, df):
    stream = get_cuda_stream()

    q = df.select(expr)
    ir = Translator(q._ldf.visit(), pl.GPUEngine()).translate_ir()

    assert isinstance(ir, ir_nodes.Select)
    table = ir.children[0].evaluate(
        cache={}, timer=None, context=IRExecutionContext(get_cuda_stream=lambda: stream)
    )
    name_to_index = {c.name: i for i, c in enumerate(table.columns)}

    def compute_column(e):
        e_with_colrefs = insert_colrefs(
            e.value,
            table_ref=plc.expressions.TableReference.LEFT,
            name_to_index=name_to_index,
        )
        with pytest.raises(NotImplementedError):
            e_with_colrefs.evaluate(table)
        ast = to_ast(e_with_colrefs, stream=stream)
        if ast is not None:
            return NamedColumn(
                plc.transform.compute_column(table.table, ast, stream=stream),
                name=e.name,
                dtype=e.value.dtype,
            )
        return e.evaluate(table)

    got = DataFrame(map(compute_column, ir.exprs), stream=stream).to_polars()

    expect = q.collect()

    assert_frame_equal(expect, got)


def test_invalid_colref_construction_raises():
    literal = expr_nodes.Literal(DataType(pl.datatypes.Int8()), 1)
    with pytest.raises(TypeError):
        expr_nodes.ColRef(
            literal.dtype, 0, plc.expressions.TableReference.LEFT, literal
        )


def test_to_ast_without_colref_raises():
    stream = get_cuda_stream()
    col = expr_nodes.Col(DataType(pl.datatypes.Int8()), "a")

    with pytest.raises(TypeError, match="Should always be wrapped"):
        to_ast(col, stream=stream)


def test_to_parquet_filter_with_colref_raises():
    col = expr_nodes.Col(DataType(pl.datatypes.Int8()), "a")
    colref = expr_nodes.ColRef(col.dtype, 0, plc.expressions.TableReference.LEFT, col)

    with pytest.raises(TypeError):
        to_parquet_filter(colref, stream=get_cuda_stream())


@pytest.mark.parametrize(
    "name",
    [
        expr_nodes.BooleanFunction.Name.IsNull,
        expr_nodes.BooleanFunction.Name.IsNotNull,
    ],
)
def test_to_parquet_filter_null_checks_on_column(name):
    col = expr_nodes.Col(DataType(pl.datatypes.Int64()), "a")
    fn = expr_nodes.BooleanFunction(DataType(pl.datatypes.Boolean()), name, (), col)
    filter_expr, residual = to_parquet_filter(fn, stream=get_cuda_stream())
    assert filter_expr is not None
    assert residual is None


@pytest.mark.parametrize(
    "name",
    [
        expr_nodes.BooleanFunction.Name.IsNull,
        expr_nodes.BooleanFunction.Name.IsNotNull,
    ],
)
def test_to_parquet_filter_null_checks_on_nested_column_not_pushed(name):
    # See https://github.com/rapidsai/cudf/issues/23397
    struct_dtype = DataType(pl.Struct({"a": pl.Int64}))
    col = expr_nodes.Col(struct_dtype, "s")
    fn = expr_nodes.BooleanFunction(DataType(pl.datatypes.Boolean()), name, (), col)
    filter_expr, residual = to_parquet_filter(fn, stream=get_cuda_stream())
    assert filter_expr is None
    assert residual is None


@pytest.mark.parametrize(
    "predicate, pushed, exact",
    [
        (pl.col("a") >= 2, True, True),
        ((pl.col("a") >= 2) & pl.col("s").str.contains("b"), True, False),
        ((pl.col("a") >= 2) | pl.col("s").str.contains("b"), False, False),
    ],
)
def test_to_parquet_filter_conjunction_splitting(predicate, pushed, exact):
    lf = pl.LazyFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})
    ir = Translator(lf.filter(predicate)._ldf.visit(), pl.GPUEngine()).translate_ir()
    mask = next(n.mask.value for n in traversal([ir]) if isinstance(n, ir_nodes.Filter))
    filter_expr, residual = to_parquet_filter(mask, stream=get_cuda_stream())
    assert (filter_expr is not None) == pushed
    assert (filter_expr is not None and residual is None) == exact
