# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the extra condition of an equi join."""

from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.ir import DataFrameScan, Join
from cudf_polars.dsl.to_ast import JoinPredicate, split_join_predicate

I32 = DataType(pl.Int32())
I64 = DataType(pl.Int64())
BOOL = DataType(pl.Boolean())
LEFT = plc.expressions.TableReference.LEFT
RIGHT = plc.expressions.TableReference.RIGHT

COLUMNS = {
    "k": (LEFT, "k"),
    "a": (LEFT, "a"),
    "k_right": (RIGHT, "k"),
    "b": (RIGHT, "b"),
}


def make_scan(dtype: DataType, *names: str) -> DataFrameScan:
    frame = pl.DataFrame({name: [1] for name in names})
    return DataFrameScan(dict.fromkeys(names, dtype), frame._df, None)


def make_join(how: str, predicate: JoinPredicate | None) -> Join:
    left = make_scan(I32, "k", "a")
    right = make_scan(I64, "k", "b")
    key = (expr.NamedExpr("k", expr.Col(I32, "k")),)
    right_key = (expr.NamedExpr("k", expr.Col(I64, "k")),)
    return Join(
        {"k": I32, "a": I32, "k_right": I64, "b": I64},
        key,
        right_key,
        (how, False, None, "_right", False, "none"),
        predicate,
        left,
        right,
    )


def split(node: expr.Expr) -> JoinPredicate:
    return split_join_predicate(
        node, columns=COLUMNS, left_names=["k", "a"], right_names=["k", "b"]
    )


def reference(node: expr.Expr) -> tuple[plc.expressions.TableReference, int]:
    assert isinstance(node, expr.ColRef)
    return node.table_ref, node.index


def test_comparison_of_columns_needs_no_extra_columns() -> None:
    node = expr.BinOp(
        BOOL,
        plc.binaryop.BinaryOperator.LESS,
        expr.Col(I32, "a"),
        expr.Col(I64, "b"),
    )
    predicate = split(node)
    assert predicate.left_exprs == ()
    assert predicate.right_exprs == ()
    left, right = predicate.predicate.children
    assert reference(left) == (LEFT, 1)
    assert reference(right) == (RIGHT, 1)


def test_expressions_of_one_input_become_extra_columns() -> None:
    one = expr.Literal(I64, 1)
    a_plus_one = expr.BinOp(
        I64, plc.binaryop.BinaryOperator.ADD, expr.Col(I64, "a"), one
    )
    b_plus_one = expr.BinOp(
        I64, plc.binaryop.BinaryOperator.ADD, expr.Col(I64, "b"), one
    )
    node = expr.BinOp(BOOL, plc.binaryop.BinaryOperator.LESS, a_plus_one, b_plus_one)
    # The same expression used twice is computed once.
    node = expr.BinOp(BOOL, plc.binaryop.BinaryOperator.LOGICAL_OR, node, node)
    predicate = split(node)
    ((left_named,),) = (predicate.left_exprs,)
    ((right_named,),) = (predicate.right_exprs,)
    assert left_named.value == a_plus_one
    assert right_named.value == b_plus_one
    left, right = predicate.predicate.children[0].children
    # Extra columns come after the input's own columns.
    assert reference(left) == (LEFT, 2)
    assert reference(right) == (RIGHT, 2)


def test_expression_is_renamed_to_the_input_column() -> None:
    # The output column "k_right" is the column "k" of the right input.
    node = expr.BinOp(
        BOOL,
        plc.binaryop.BinaryOperator.LESS,
        expr.Col(I64, "a"),
        expr.UnaryFunction(I64, "abs", (), expr.Col(I64, "k_right")),
    )
    predicate = split(node)
    (right_named,) = predicate.right_exprs
    assert right_named.value.children == (expr.Col(I64, "k"),)


def test_non_pointwise_expression_of_one_input_is_not_supported() -> None:
    total = expr.Agg(I64, "sum", (), ExecutionContext.FRAME, expr.Col(I64, "b"))
    node = expr.BinOp(BOOL, plc.binaryop.BinaryOperator.LESS, expr.Col(I64, "a"), total)
    with pytest.raises(NotImplementedError, match="non-pointwise"):
        split(node)


def test_predicate_is_only_supported_for_inner_joins() -> None:
    node = expr.BinOp(
        BOOL,
        plc.binaryop.BinaryOperator.LESS,
        expr.Col(I64, "a"),
        expr.Col(I64, "b"),
    )
    with pytest.raises(NotImplementedError, match="join type Left"):
        make_join("Left", split(node))


def test_operands_of_different_types_are_not_supported() -> None:
    # Polars casts such operands itself, so this guards against a plan it
    # leaves uncast, which libcudf's AST would only reject once the join runs.
    node = expr.BinOp(
        BOOL,
        plc.binaryop.BinaryOperator.LESS,
        expr.ColRef(I32, 1, LEFT, expr.Col(I32, "a")),
        expr.ColRef(I64, 1, RIGHT, expr.Col(I64, "b")),
    )
    with pytest.raises(NotImplementedError, match="different types"):
        make_join("Inner", JoinPredicate(node, (), ()))


@pytest.mark.parametrize("decimal_side", [LEFT, RIGHT])
def test_decimal_and_float_columns_are_cast_to_match(
    decimal_side: plc.expressions.TableReference,
) -> None:
    decimal = DataType(pl.Decimal(10, 2))
    float64 = DataType(pl.Float64())
    left_dtype, right_dtype = (
        (decimal, float64) if decimal_side == LEFT else (float64, decimal)
    )
    node = expr.BinOp(
        BOOL,
        plc.binaryop.BinaryOperator.LESS,
        expr.ColRef(left_dtype, 1, LEFT, expr.Col(left_dtype, "a")),
        expr.ColRef(right_dtype, 1, RIGHT, expr.Col(right_dtype, "b")),
    )
    join = make_join("Inner", JoinPredicate(node, (), ()))
    assert join.predicate is not None
