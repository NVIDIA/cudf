# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for selection operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.utils.reshape import broadcast

if TYPE_CHECKING:
    from cudf_polars.containers import DataFrame, DataType

__all__ = ["Filter", "Gather"]


class Gather(Expr):
    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(self, dtype: DataType, values: Expr, indices: Expr) -> None:
        self.dtype = dtype
        self.children = (values, indices)
        self.is_pointwise = False
        if not plc.traits.is_integral_not_bool(indices.dtype.plc_type):
            raise NotImplementedError("Gather with non-integer indices")

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        values, indices = (
            child.evaluate(df, context=context) for child in self.children
        )
        n = values.size
        if indices.size == 0:
            return Column(
                plc.column_factories.make_empty_column(
                    self.dtype.plc_type, stream=df.stream
                ),
                dtype=self.dtype,
            )
        lo, hi = plc.reduce.minmax(indices.obj, stream=df.stream)
        if hi.to_py(stream=df.stream) >= n or lo.to_py(stream=df.stream) < -n:  # type: ignore[operator]
            raise ValueError("gather indices are out of bounds")
        if indices.null_count:
            bounds_policy = plc.copying.OutOfBoundsPolicy.NULLIFY
            obj = plc.replace.replace_nulls(
                indices.obj,
                plc.Scalar.from_py(n, dtype=indices.obj.type(), stream=df.stream),
                stream=df.stream,
            )
        else:
            bounds_policy = plc.copying.OutOfBoundsPolicy.DONT_CHECK
            obj = indices.obj
        table = plc.copying.gather(
            plc.Table([values.obj]), obj, bounds_policy, stream=df.stream
        )
        return Column(table.columns()[0], dtype=self.dtype)


class Filter(Expr):
    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(self, dtype: DataType, values: Expr, indices: Expr):
        self.dtype = dtype
        self.children = (values, indices)
        self.is_pointwise = False

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        values, mask = (child.evaluate(df, context=context) for child in self.children)
        # polars type-puns length-1 columns as scalars.
        values, mask = broadcast(values, mask, stream=df.stream)
        table = plc.stream_compaction.apply_boolean_mask(
            plc.Table([values.obj]), mask.obj, stream=df.stream
        )
        return Column(table.columns()[0], dtype=self.dtype).sorted_like(values)
