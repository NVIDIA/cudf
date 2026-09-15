# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for ternary operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.exceptions import ShapeError

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import (
    ExecutionContext,
    Expr,
)
from cudf_polars.dsl.utils.reshape import broadcast

if TYPE_CHECKING:
    from cudf_polars.containers import DataFrame, DataType


__all__ = ["Ternary"]


class Ternary(Expr):
    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(
        self, dtype: DataType, when: Expr, then: Expr, otherwise: Expr
    ) -> None:
        self.dtype = dtype
        self.children = (when, then, otherwise)
        self.is_pointwise = True

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        when, then, otherwise = (
            child.evaluate(df, context=context) for child in self.children
        )

        if (
            not then.is_scalar
            and not otherwise.is_scalar
            and then.size != otherwise.size
            and then.size != 1
            and otherwise.size != 1
        ):
            raise ShapeError(
                "then and otherwise branches have incompatible lengths "
                f"({then.size} and {otherwise.size})"
            )

        if when.is_scalar:
            # For scalar predicates: lowering to copy_if_else would require
            # materializing an all true/false mask column. Instead, just pick
            # the correct branch.
            when_predicate = when.obj_scalar(stream=df.stream).to_py(stream=df.stream)
            pick, other = (then, otherwise) if when_predicate else (otherwise, then)

            pick_col = (
                broadcast(
                    pick,
                    target_length=1 if other.is_scalar else other.size,
                    stream=df.stream,
                )[0]
                if pick.is_scalar
                else pick
            )
            return Column(pick_col.obj, dtype=self.dtype)

        then_obj = then.obj_scalar(stream=df.stream) if then.is_scalar else then.obj
        otherwise_obj = (
            otherwise.obj_scalar(stream=df.stream)
            if otherwise.is_scalar
            else otherwise.obj
        )

        return Column(
            plc.copying.copy_if_else(
                then_obj,
                otherwise_obj,
                when.obj,
                stream=df.stream,
            ),
            dtype=self.dtype,
        )
