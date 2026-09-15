# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Sorting DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.utils import sorting

if TYPE_CHECKING:
    from cudf_polars.containers import DataFrame, DataType

__all__ = ["Sort", "SortBy"]


class Sort(Expr):
    __slots__ = ("options",)
    _non_child = ("dtype", "options")

    def __init__(
        self, dtype: DataType, options: tuple[bool, bool, bool], column: Expr
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.children = (column,)
        self.is_pointwise = False

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context)
        (stable, nulls_last, descending) = self.options
        order, null_order = sorting.sort_order(
            [descending], nulls_last=[nulls_last], num_keys=1
        )
        do_sort = plc.sorting.stable_sort if stable else plc.sorting.sort
        table = do_sort(plc.Table([column.obj]), order, null_order, stream=df.stream)
        return Column(
            table.columns()[0],
            is_sorted=plc.types.Sorted.YES,
            order=order[0],
            null_order=null_order[0],
            dtype=self.dtype,
        )


class SortBy(Expr):
    __slots__ = ("options",)
    _non_child = ("dtype", "options")

    def __init__(
        self,
        dtype: DataType,
        options: tuple[bool, tuple[bool], tuple[bool]],
        column: Expr,
        *by: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.children = (column, *by)
        self.is_pointwise = False

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        column, *by = (child.evaluate(df, context=context) for child in self.children)
        (stable, nulls_last, descending) = self.options
        order, null_order = sorting.sort_order(
            descending, nulls_last=nulls_last, num_keys=len(by)
        )
        do_sort = plc.sorting.stable_sort_by_key if stable else plc.sorting.sort_by_key
        table = do_sort(
            plc.Table([column.obj]),
            plc.Table([c.obj for c in by]),
            order,
            null_order,
            stream=df.stream,
        )
        return Column(table.columns()[0], dtype=self.dtype)
