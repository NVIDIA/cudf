# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


def test_literal_construction_invalid():
    with pytest.raises(ValueError):
        plc.expressions.Literal(
            plc.Scalar.from_arrow(pa.scalar(None, type=pa.list_(pa.int64())))
        )


@pytest.mark.parametrize(
    "tableref",
    [
        plc.expressions.TableReference.LEFT,
        plc.expressions.TableReference.RIGHT,
    ],
)
def test_columnref_construction(tableref):
    plc.expressions.ColumnReference(1, tableref)


def test_columnnameref_construction():
    plc.expressions.ColumnNameReference("abc")


@pytest.mark.parametrize(
    "kwargs",
    [
        # Unary op
        {
            "op": plc.expressions.ASTOperator.IDENTITY,
            "left": plc.expressions.ColumnReference(1),
        },
        # Binop
        {
            "op": plc.expressions.ASTOperator.ADD,
            "left": plc.expressions.ColumnReference(1),
            "right": plc.expressions.ColumnReference(2),
        },
        # Ternary op
        {
            "op": plc.expressions.ASTOperator.IF_ELSE,
            "left": plc.expressions.ColumnReference(0),
            "right": plc.expressions.ColumnReference(1),
            "third": plc.expressions.ColumnReference(2),
        },
    ],
)
def test_astoperation_construction(kwargs):
    plc.expressions.Operation(**kwargs)


def test_astoperation_third_operand_requires_right():
    with pytest.raises(ValueError):
        plc.expressions.Operation(
            plc.expressions.ASTOperator.IF_ELSE,
            plc.expressions.ColumnReference(0),
            third=plc.expressions.ColumnReference(1),
        )


def test_evaluation():
    table_h = pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    lit = pa.scalar(42, type=pa.int64())
    table = plc.Table.from_arrow(table_h)
    # expr = abs(b * c - (a + 42))
    expr = plc.expressions.Operation(
        plc.expressions.ASTOperator.ABS,
        plc.expressions.Operation(
            plc.expressions.ASTOperator.SUB,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.MUL,
                plc.expressions.ColumnReference(1),
                plc.expressions.ColumnReference(2),
            ),
            plc.expressions.Operation(
                plc.expressions.ASTOperator.ADD,
                plc.expressions.ColumnReference(0),
                plc.expressions.Literal(plc.Scalar.from_arrow(lit)),
            ),
        ),
    )

    expect = pc.abs(
        pc.subtract(
            pc.multiply(table_h["b"], table_h["c"]), pc.add(table_h["a"], lit)
        )
    )
    got = plc.transform.compute_column(table, expr)

    assert_column_eq(expect, got)


def test_evaluation_if_else():
    table_h = pa.table(
        {
            "cond": [True, False, None, True],
            "a": [1, 2, 3, 4],
            "b": [10, 20, 30, 40],
        }
    )
    table = plc.Table.from_arrow(table_h)
    expr = plc.expressions.Operation(
        plc.expressions.ASTOperator.IF_ELSE,
        plc.expressions.ColumnReference(0),
        plc.expressions.ColumnReference(1),
        plc.expressions.ColumnReference(2),
    )

    # A null condition selects the else branch.
    expect = pa.array([1, 20, 30, 4], type=pa.int64())
    got = plc.transform.compute_column(table, expr)

    assert_column_eq(expect, got)


def test_evaluation_if_else_nested():
    table_h = pa.table(
        {
            "a": [1, 2, 3, 4],
            "b": [10, 20, 30, 40],
        }
    )
    table = plc.Table.from_arrow(table_h)
    ref_a = plc.expressions.ColumnReference(0)
    ref_b = plc.expressions.ColumnReference(1)
    lit = plc.expressions.Literal(
        plc.Scalar.from_arrow(pa.scalar(2, type=pa.int64()))
    )
    # a > 2 ? (a > 3 ? 100 : b) : a
    inner_cond = plc.expressions.Operation(
        plc.expressions.ASTOperator.GREATER,
        ref_a,
        plc.expressions.Literal(
            plc.Scalar.from_arrow(pa.scalar(3, type=pa.int64()))
        ),
    )
    hundred = plc.expressions.Literal(
        plc.Scalar.from_arrow(pa.scalar(100, type=pa.int64()))
    )
    inner = plc.expressions.Operation(
        plc.expressions.ASTOperator.IF_ELSE, inner_cond, hundred, ref_b
    )
    outer_cond = plc.expressions.Operation(
        plc.expressions.ASTOperator.GREATER, ref_a, lit
    )
    expr = plc.expressions.Operation(
        plc.expressions.ASTOperator.IF_ELSE, outer_cond, inner, ref_a
    )

    expect = pa.array([1, 2, 30, 100], type=pa.int64())
    got = plc.transform.compute_column(table, expr)

    assert_column_eq(expect, got)
