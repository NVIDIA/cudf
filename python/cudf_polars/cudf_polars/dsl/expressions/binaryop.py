# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""BinaryOp DSL nodes."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

from polars import _plr as polars
from polars.exceptions import ComputeError

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

pl_expr = polars._expr_nodes

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["BinOp"]


class BinOp(Expr):
    __slots__ = ("op",)
    _non_child = ("dtype", "op")

    def __init__(
        self,
        dtype: DataType,
        op: plc.binaryop.BinaryOperator,
        left: Expr,
        right: Expr,
    ) -> None:
        self.dtype = dtype
        if plc.traits.is_boolean(self.dtype.plc_type):
            # For boolean output types, bitand and bitor implement
            # boolean logic, so translate. bitxor also does, but the
            # default behaviour is correct.
            op = BinOp._BOOL_KLEENE_MAPPING.get(op, op)
        self.op = op
        self.children = (left, right)
        self.is_pointwise = True
        if not plc.binaryop.is_supported_operation(
            self.dtype.plc_type, left.dtype.plc_type, right.dtype.plc_type, op
        ):
            raise NotImplementedError(
                f"Operation {op.name} not supported "
                f"for types {left.dtype.id().name} and {right.dtype.id().name} "
                f"with output type {self.dtype.id().name}"
            )

    _BOOL_KLEENE_MAPPING: ClassVar[
        dict[plc.binaryop.BinaryOperator, plc.binaryop.BinaryOperator]
    ] = {
        plc.binaryop.BinaryOperator.BITWISE_AND: plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
        plc.binaryop.BinaryOperator.BITWISE_OR: plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
        plc.binaryop.BinaryOperator.LOGICAL_AND: plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
        plc.binaryop.BinaryOperator.LOGICAL_OR: plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
    }

    _MAPPING: ClassVar[
        dict[polars._expr_nodes.Operator, plc.binaryop.BinaryOperator]
    ] = {
        pl_expr.Operator.Eq: plc.binaryop.BinaryOperator.EQUAL,
        pl_expr.Operator.EqValidity: plc.binaryop.BinaryOperator.NULL_EQUALS,
        pl_expr.Operator.NotEq: plc.binaryop.BinaryOperator.NOT_EQUAL,
        pl_expr.Operator.NotEqValidity: plc.binaryop.BinaryOperator.NULL_NOT_EQUALS,
        pl_expr.Operator.Lt: plc.binaryop.BinaryOperator.LESS,
        pl_expr.Operator.LtEq: plc.binaryop.BinaryOperator.LESS_EQUAL,
        pl_expr.Operator.Gt: plc.binaryop.BinaryOperator.GREATER,
        pl_expr.Operator.GtEq: plc.binaryop.BinaryOperator.GREATER_EQUAL,
        pl_expr.Operator.Plus: plc.binaryop.BinaryOperator.ADD,
        pl_expr.Operator.Minus: plc.binaryop.BinaryOperator.SUB,
        pl_expr.Operator.Multiply: plc.binaryop.BinaryOperator.MUL,
        pl_expr.Operator.Divide: plc.binaryop.BinaryOperator.DIV,
        pl_expr.Operator.TrueDivide: plc.binaryop.BinaryOperator.TRUE_DIV,
        pl_expr.Operator.FloorDivide: plc.binaryop.BinaryOperator.FLOOR_DIV,
        pl_expr.Operator.Modulus: plc.binaryop.BinaryOperator.PYMOD,
        pl_expr.Operator.And: plc.binaryop.BinaryOperator.BITWISE_AND,
        pl_expr.Operator.Or: plc.binaryop.BinaryOperator.BITWISE_OR,
        pl_expr.Operator.Xor: plc.binaryop.BinaryOperator.BITWISE_XOR,
        pl_expr.Operator.LogicalAnd: plc.binaryop.BinaryOperator.LOGICAL_AND,
        pl_expr.Operator.LogicalOr: plc.binaryop.BinaryOperator.LOGICAL_OR,
    }

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        left, right = (child.evaluate(df, context=context) for child in self.children)
        lop: plc.Column | plc.Scalar = left.obj
        rop: plc.Column | plc.Scalar = right.obj
        if left.size != right.size:
            if left.is_scalar:
                lop = left.obj_scalar(stream=df.stream)
            elif right.is_scalar:
                rop = right.obj_scalar(stream=df.stream)
        if plc.traits.is_integral_not_bool(self.dtype.plc_type) and self.op in {
            plc.binaryop.BinaryOperator.FLOOR_DIV,
            plc.binaryop.BinaryOperator.PYMOD,
        }:
            if (
                right.obj.size() == 1
                and right.obj.to_scalar(stream=df.stream).to_py(stream=df.stream) == 0
            ):
                return Column(
                    plc.Column.all_null_like(
                        left.obj, left.obj.size(), stream=df.stream
                    ),
                    dtype=self.dtype,
                )

            if right.obj.size() > 1:
                rop = plc.replace.find_and_replace_all(
                    right.obj,
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(
                            0, dtype=self.dtype.plc_type, stream=df.stream
                        ),
                        1,
                        stream=df.stream,
                    ),
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(
                            None, dtype=self.dtype.plc_type, stream=df.stream
                        ),
                        1,
                        stream=df.stream,
                    ),
                    stream=df.stream,
                )
        result = plc.binaryop.binary_operation(
            lop, rop, self.op, self.dtype.plc_type, stream=df.stream
        )
        if self.op == plc.binaryop.BinaryOperator.DIV and plc.traits.is_fixed_point(
            self.dtype.plc_type
        ):
            self._raise_on_decimal_division_by_zero(right, result, df.stream)
        return Column(result, dtype=self.dtype)

    @staticmethod
    def _raise_on_decimal_division_by_zero(
        right: Column, result: plc.Column, stream: Stream
    ) -> None:
        bool_type = plc.DataType(plc.TypeId.BOOL8)

        def any_true(mask: plc.Column) -> bool:
            return bool(
                plc.reduce.reduce(
                    mask, plc.aggregation.any(), bool_type, stream=stream
                ).to_py(stream=stream)
            )

        is_zero = plc.binaryop.binary_operation(
            right.obj,
            plc.Scalar.from_py(Decimal(0), right.obj.type(), stream=stream),
            plc.binaryop.BinaryOperator.EQUAL,
            bool_type,
            stream=stream,
        )
        if right.size == 1:
            # A scalar divisor: one value decides, so the result is only
            # looked at when that value is zero.
            if not any_true(is_zero):
                return
            divides_by_zero = plc.unary.is_valid(result, stream=stream)
        else:
            divides_by_zero = plc.binaryop.binary_operation(
                is_zero,
                plc.unary.is_valid(result, stream=stream),
                plc.binaryop.BinaryOperator.LOGICAL_AND,
                bool_type,
                stream=stream,
            )
        if any_true(divides_by_zero):
            raise ComputeError("division by zero Decimal")
