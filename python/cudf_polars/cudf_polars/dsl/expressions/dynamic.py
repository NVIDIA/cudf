# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Window boundary expressions for ``group_by_dynamic``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.datetime import _truncate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame, DataType
    from cudf_polars.dsl.expressions.datetime import _TruncatePlan

__all__ = ["DynamicWindowEnd", "DynamicWindowStart"]

_INT32 = plc.DataType(plc.TypeId.INT32)
_INT64 = plc.DataType(plc.TypeId.INT64)
_BOOL8 = plc.DataType(plc.TypeId.BOOL8)
_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1


def _is_sorted(
    columns: Sequence[plc.Column], null_order: plc.types.NullOrder, stream: Stream
) -> bool:
    """Whether ``columns`` are lexicographically ascending."""
    n = len(columns)
    return plc.sorting.is_sorted(
        plc.Table(columns),
        [plc.types.Order.ASCENDING] * n,
        [null_order] * n,
        stream=stream,
    )


def _is_sorted_within_groups(index: Column, keys: list[Column], stream: Stream) -> bool:
    """
    Whether ``index`` (without nulls) is ascending within each group of ``keys``.

    The common input layouts are recognized in linear time: an index sorted
    overall, or rows sorted by the keys and then the index. Otherwise only the
    keys and the index are stably sorted by the keys and checked.
    """
    if index.check_sorted(
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.BEFORE,
        stream=stream,
    ):
        return True
    columns = [*(k.obj for k in keys), index.obj]
    nulls = (plc.types.NullOrder.BEFORE, plc.types.NullOrder.AFTER)
    if any(_is_sorted(columns, null_order, stream) for null_order in nulls):
        return True
    n = len(keys)
    grouped = plc.sorting.stable_sort_by_key(
        plc.Table(columns),
        plc.Table(columns[:n]),
        [plc.types.Order.ASCENDING] * n,
        [plc.types.NullOrder.BEFORE] * n,
        stream=stream,
    )
    return _is_sorted(grouped.columns(), plc.types.NullOrder.BEFORE, stream)


def _to_ticks(column: plc.Column, stream: Stream) -> plc.Column:
    """Return the physical values of an integer or timestamp column as INT64."""
    type_id = column.type().id()
    if type_id == plc.TypeId.INT64:
        return column
    if type_id == plc.TypeId.INT32:
        return plc.unary.cast(column, _INT64, stream=stream)
    if type_id == plc.TypeId.TIMESTAMP_DAYS:
        return plc.unary.cast(
            plc.unary.bit_cast(column, _INT32, stream=stream), _INT64, stream=stream
        )
    return plc.unary.bit_cast(column, _INT64, stream=stream)


def _from_ticks(ticks: plc.Column, dtype: plc.DataType, stream: Stream) -> plc.Column:
    """Inverse of :func:`_to_ticks`."""
    type_id = dtype.id()
    if type_id == plc.TypeId.INT64:
        return ticks
    if type_id == plc.TypeId.INT32:
        return plc.unary.cast(ticks, dtype, stream=stream)
    if type_id == plc.TypeId.TIMESTAMP_DAYS:
        return plc.unary.bit_cast(
            plc.unary.cast(ticks, _INT32, stream=stream), dtype, stream=stream
        )
    return plc.unary.bit_cast(ticks, dtype, stream=stream)


def _add_ticks(ticks: plc.Column, value: int, stream: Stream) -> plc.Column:
    """
    Add a host integer to an INT64 column, with an overflowing result as null.

    ``value`` is static (from the window's offset or length, never the data),
    so whether ``ticks + value`` overflows INT64 depends only on ``ticks``,
    checked here without relying on the wrapped sum itself.
    """
    if value == 0:
        return ticks
    result = plc.binaryop.binary_operation(
        ticks,
        plc.Scalar.from_py(value, _INT64, stream=stream),
        plc.binaryop.BinaryOperator.ADD,
        _INT64,
        stream=stream,
    )
    bound, op = (
        (_I64_MAX - value, plc.binaryop.BinaryOperator.LESS_EQUAL)
        if value > 0
        else (_I64_MIN - value, plc.binaryop.BinaryOperator.GREATER_EQUAL)
    )
    safe = plc.binaryop.binary_operation(
        ticks,
        plc.Scalar.from_py(bound, _INT64, stream=stream),
        op,
        _BOOL8,
        stream=stream,
    )
    return plc.copying.copy_if_else(
        result, plc.Scalar.from_py(None, _INT64, stream=stream), safe, stream=stream
    )


def _checked_narrow(
    ticks: plc.Column, dtype: plc.DataType, stream: Stream
) -> plc.Column:
    """
    Narrow ``ticks`` to ``dtype``, with out-of-range ``INT32``/Date values as null.

    This matches polars' checked conversion when materializing a window's
    label or boundary, where libcudf's native narrowing cast would wrap. It is
    only safe where the result is not reused as a grouping or sort key, since
    overflowing but distinct windows would collide into a single null group.
    """
    narrowed = _from_ticks(ticks, dtype, stream)
    if dtype.id() not in (plc.TypeId.INT32, plc.TypeId.TIMESTAMP_DAYS):
        return narrowed
    in_range = plc.binaryop.binary_operation(
        plc.binaryop.binary_operation(
            ticks,
            plc.Scalar.from_py(-(2**31), _INT64, stream=stream),
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            _BOOL8,
            stream=stream,
        ),
        plc.binaryop.binary_operation(
            ticks,
            plc.Scalar.from_py(2**31 - 1, _INT64, stream=stream),
            plc.binaryop.BinaryOperator.LESS_EQUAL,
            _BOOL8,
            stream=stream,
        ),
        plc.binaryop.BinaryOperator.LOGICAL_AND,
        _BOOL8,
        stream=stream,
    )
    return plc.copying.copy_if_else(
        narrowed,
        plc.Scalar.from_py(None, dtype, stream=stream),
        in_range,
        stream=stream,
    )


class DynamicWindowStart(Expr):
    """
    Start of the ``group_by_dynamic`` window that each index value falls in.

    Notes
    -----
    Windows tile the index: with ``closed="left"`` a value ``t`` belongs to the
    window starting at ``truncate(t - offset, every) + offset``. With
    ``closed="right"`` windows are ``(start, start + every]``, so the value is
    first moved back by one tick. This matches polars for ``period == every``.
    """

    __slots__ = ("closed", "offset", "plan")
    _non_child = ("dtype", "plan", "offset", "closed")
    plan: _TruncatePlan
    """Truncation applied to the shifted index."""
    offset: int
    """Window offset, in ticks of the index column."""
    closed: Literal["left", "right"]
    """Which window endpoint is inclusive."""

    def __init__(
        self,
        dtype: DataType,
        plan: _TruncatePlan,
        offset: int,
        closed: Literal["left", "right"],
        index: Expr,
        *keys: Expr,
    ) -> None:
        self.dtype = dtype
        self.plan = plan
        self.offset = offset
        self.closed = closed
        # The group keys are only inputs to the sortedness check.
        self.children = (index, *keys)
        self.is_pointwise = True

    @staticmethod
    def _validate(index: Column, keys: list[Column], stream: Stream) -> None:
        """
        Raise the errors polars raises for an invalid index.

        The streaming engine evaluates this one partition at a time, so an
        index that only decreases across partitions is not detected yet.
        """
        if not keys:
            if index.null_count > 0 or not index.check_sorted(
                order=plc.types.Order.ASCENDING,
                null_order=plc.types.NullOrder.BEFORE,
                stream=stream,
            ):
                raise pl.exceptions.InvalidOperationError(
                    "argument in operation 'group_by_dynamic' is not sorted, "
                    "please sort the 'expr/series/column' first"
                )
        elif index.null_count > 0:
            raise pl.exceptions.ComputeError(
                "null values in dynamic group_by not supported, fill nulls."
            )
        elif not _is_sorted_within_groups(index, keys, stream):
            raise pl.exceptions.ComputeError("input data is not sorted")

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        index, *keys = (child.evaluate(df, context=context) for child in self.children)
        stream = df.stream
        self._validate(index, keys, stream)
        dtype = self.dtype.plc_type
        shift = self.offset + (1 if self.closed == "right" else 0)
        if self.offset == 0 and shift == 0 and plc.traits.is_timestamp(dtype):
            start = _truncate(index.obj, self.plan, stream)
        else:
            ticks = _add_ticks(_to_ticks(index.obj, stream), -shift, stream)
            if plc.traits.is_timestamp(dtype):
                ticks = _to_ticks(
                    _truncate(_from_ticks(ticks, dtype, stream), self.plan, stream),
                    stream,
                )
            else:
                ticks = _truncate(ticks, self.plan, stream)
            start = _from_ticks(_add_ticks(ticks, self.offset, stream), dtype, stream)
        # Window starts are non-decreasing in the index.
        return Column(start, dtype=self.dtype).sorted_like(index)


class DynamicWindowEnd(Expr):
    """End of a ``group_by_dynamic`` window, given the window start."""

    __slots__ = ("months", "ticks")
    _non_child = ("dtype", "months", "ticks")
    months: int
    """Calendar months in the window length."""
    ticks: int
    """Fixed part of the window length, in ticks of the index column."""

    def __init__(self, dtype: DataType, months: int, ticks: int, start: Expr) -> None:
        self.dtype = dtype
        self.months = months
        self.ticks = ticks
        self.children = (start,)
        self.is_pointwise = True

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        start = child.evaluate(df, context=context)
        stream = df.stream
        dtype = self.dtype.plc_type
        end = start.obj
        if self.months != 0:
            end = plc.datetime.add_calendrical_months(
                end,
                plc.Scalar.from_py(self.months, _INT32, stream=stream),
                stream=stream,
            )
        if self.ticks != 0 or end.type().id() != dtype.id():
            # Also narrows a wider hidden window start (see rewrite_groupby_dynamic)
            # to the index dtype, with null on overflow instead of wrapping.
            end = _checked_narrow(
                _add_ticks(_to_ticks(end, stream), self.ticks, stream), dtype, stream
            )
        return Column(end, dtype=self.dtype).sorted_like(start)
