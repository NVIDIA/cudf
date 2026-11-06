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
    ],
)
def test_astoperation_construction(kwargs):
    plc.expressions.Operation(**kwargs)


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


def test_expression_nan_semantics_default():
    expr = plc.expressions.ColumnReference(0)
    assert expr.nan_semantics == plc.types.NanSemantics.DEFAULT
    expr.nan_semantics = plc.types.NanSemantics.ETL
    assert expr.nan_semantics == plc.types.NanSemantics.ETL


@pytest.mark.parametrize(
    "op, ieee, etl",
    [
        (
            plc.expressions.ASTOperator.EQUAL,
            [False, False, True],
            [True, False, True],
        ),
        (
            plc.expressions.ASTOperator.LESS,
            [False, False, False],
            [False, True, False],
        ),
        (
            plc.expressions.ASTOperator.GREATER_EQUAL,
            [False, False, True],
            [True, False, True],
        ),
    ],
)
def test_compute_column_nan_semantics(op, ieee, etl):
    nan = float("nan")
    table = plc.Table.from_arrow(
        pa.table({"a": [nan, 1.0, 2.0], "b": [nan, nan, 2.0]})
    )
    expr = plc.expressions.Operation(
        op,
        plc.expressions.ColumnReference(0),
        plc.expressions.ColumnReference(1),
    )
    got = plc.transform.compute_column(table, expr)
    assert_column_eq(pa.array(ieee), got)

    expr.nan_semantics = plc.types.NanSemantics.ETL
    got = plc.transform.compute_column(table, expr)
    assert_column_eq(pa.array(etl), got)
