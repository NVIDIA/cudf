# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from typing import Self

    from cudf_polars.containers import DataFrame

__all__ = ["BitwiseFunction"]


class BitwiseFunction(Expr):
    """Bitwise counting functions on integer and Boolean columns."""

    class Name(IntEnum):
        """Internal and picklable representation of polars' `BitwiseFunction`."""

        CountOnes = auto()
        CountZeros = auto()
        LeadingOnes = auto()
        LeadingZeros = auto()
        TrailingOnes = auto()
        TrailingZeros = auto()

        @classmethod
        def from_polars(cls, obj: Any) -> Self:
            """Convert from polars' `BitwiseFunction`."""
            _, name = str(obj).split(".", maxsplit=1)
            try:
                return cls[name]
            except KeyError:
                raise NotImplementedError(f"{obj} is not supported") from None

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    _valid_ops: ClassVar[frozenset[Name]] = frozenset({Name.CountOnes})

    def __init__(
        self,
        dtype: DataType,
        name: BitwiseFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = True
        if self.name not in self._valid_ops:
            raise NotImplementedError(
                f"BitwiseFunction.{self.name.name} is not supported"
            )
        (child,) = children
        input_type = child.dtype.polars_type
        if not (input_type.is_integer() or isinstance(input_type, pl.Boolean)):
            raise NotImplementedError(
                f"BitwiseFunction.{self.name.name} is not supported "
                f"for {input_type} input"
            )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context)
        return Column(
            plc.unary.unary_operation(
                column.obj, plc.unary.UnaryOperator.BIT_COUNT, stream=df.stream
            ),
            dtype=DataType(pl.Int32()),
        ).astype(self.dtype, df.stream)
