# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite ``group_by_dynamic`` into a grouped aggregation over window starts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.expressions.datetime import (
    _NANOSECONDS_PER_DAY,
    _TIMESTAMP_TICK_NANOSECONDS,
    _plan_integer_truncate,
    _plan_truncate,
)
from cudf_polars.dsl.expressions.dynamic import (
    DynamicWindowEnd,
    DynamicWindowStart,
)
from cudf_polars.dsl.utils.groupby import rewrite_groupby
from cudf_polars.dsl.utils.naming import unique_names

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_polars.dsl.expressions.datetime import _TruncatePlan
    from cudf_polars.typing import Schema

__all__ = ["rewrite_groupby_dynamic"]

# A polars duration: (months, weeks, days, nanoseconds, parsed_int, negative).
Duration = tuple[int, int, int, int, bool, bool]

# Nanoseconds in one tick of each timestamp type that can index the windows.
_TICK_NANOSECONDS = {
    **_TIMESTAMP_TICK_NANOSECONDS,
    plc.TypeId.TIMESTAMP_DAYS: _NANOSECONDS_PER_DAY,
}


class _Window(NamedTuple):
    """Device description of the windows, in the units of the index column."""

    plan: _TruncatePlan
    """Truncation that maps a (shifted) index value to its window start."""
    offset: int
    """Window offset, in ticks."""
    months: int
    """Calendar months in the window length."""
    ticks: int
    """Fixed part of the window length, in ticks."""


def _signed(value: int, *, negative: bool) -> int:
    return -value if negative else value


def _is_single_unit(duration: Duration) -> bool:
    months, weeks, days, nanoseconds, _, _ = duration
    return (months != 0) + (weeks != 0) + (days != 0) + (nanoseconds != 0) <= 1


def _null_order(dtype: DataType) -> plc.types.NullOrder:
    """Where polars places the null group of a ``group_by_dynamic`` key."""
    if isinstance(dtype.polars_type, (pl.String, pl.Binary)):
        return plc.types.NullOrder.AFTER
    return plc.types.NullOrder.BEFORE


def _plan_window(index_dtype: DataType, every: Duration, offset: Duration) -> _Window:
    """
    Plan the windows of a supported ``group_by_dynamic``.

    Raises
    ------
    NotImplementedError
        For window and offset combinations evaluated on the CPU instead.
    """
    months, weeks, days, nanoseconds, _, every_negative = every
    offset_months, offset_weeks, offset_days, offset_nanoseconds, _, offset_negative = (
        offset
    )
    if every_negative or months == weeks == days == nanoseconds == 0:
        raise NotImplementedError(  # pragma: no cover; polars raises first
            "group_by_dynamic with a non-positive every"
        )
    type_id = index_dtype.id()
    if type_id in {plc.TypeId.INT32, plc.TypeId.INT64}:
        # Integer windows are "Ni" durations, stored in the nanosecond field.
        return _Window(
            _plan_integer_truncate(nanoseconds),
            _signed(offset_nanoseconds, negative=offset_negative),
            0,
            nanoseconds,
        )
    if not (_is_single_unit(every) and _is_single_unit(offset)):
        # Polars raises when mixing months, weeks, days and sub-daily units.
        raise NotImplementedError("group_by_dynamic with a mixed-unit duration")
    tick = _TICK_NANOSECONDS[type_id]
    if offset_months != 0:
        raise NotImplementedError("group_by_dynamic with a calendar offset")
    offset_ns = (7 * offset_weeks + offset_days) * _NANOSECONDS_PER_DAY + (
        offset_nanoseconds
    )
    if offset_ns % tick != 0:
        raise NotImplementedError(
            "group_by_dynamic with an offset finer than the index resolution"
        )
    every_string = "".join(
        f"{count}{unit}"
        for count, unit in (
            (months, "mo"),
            (weeks, "w"),
            (days, "d"),
            (nanoseconds, "ns"),
        )
        if count != 0
    )
    # Polars' window starts are its truncation, so reuse that (it raises
    # NotImplementedError for durations and time zones it cannot evaluate).
    plan = _plan_truncate(index_dtype, every_string)
    if months != 0:
        if offset_ns != 0:
            raise NotImplementedError(
                "group_by_dynamic with an offset and a calendar every"
            )
        return _Window(plan, 0, months, 0)
    every_ns = (7 * weeks + days) * _NANOSECONDS_PER_DAY + nanoseconds
    if every_ns % tick != 0:
        raise NotImplementedError(
            "group_by_dynamic with an every finer than the index resolution"
        )
    return _Window(
        plan, _signed(offset_ns // tick, negative=offset_negative), 0, every_ns // tick
    )


def _check_supported(dynamic: Any) -> None:
    if dynamic.period != dynamic.every:
        raise NotImplementedError("group_by_dynamic with period != every")
    if dynamic.closed_window not in {"left", "right"}:
        raise NotImplementedError(
            f"group_by_dynamic with closed={dynamic.closed_window!r}"
        )
    if dynamic.start_by not in {"window", "window_bound"}:
        raise NotImplementedError(
            f"group_by_dynamic with start_by={dynamic.start_by!r}"
        )


def rewrite_groupby_dynamic(
    options: Any,
    schema: Schema,
    keys: Sequence[expr.NamedExpr],
    aggs: Sequence[expr.NamedExpr],
    inp: ir.IR,
) -> ir.IR:
    """
    Rewrite a ``group_by_dynamic`` plan node into nodes we can handle.

    Parameters
    ----------
    options
        Polars group-by options, with dynamic-window options.
    schema
        Schema of the dynamic group-by plan node.
    keys
        Grouping keys, excluding the index column.
    aggs
        Requested aggregations.
    inp
        Input plan node.

    Returns
    -------
    New plan node computing the dynamic group-by.

    Raises
    ------
    NotImplementedError
        For unsupported options or aggregations.

    Notes
    -----
    When ``period == every`` the windows tile the index, so every row falls in
    exactly one window. The rewrite computes each row's window start, runs an
    ordinary grouped aggregation on ``(*keys, start)``, and sorts the result by
    the keys and the start, which is the order polars produces (null keys
    sort last for strings and first otherwise).
    Any slice applies to that sorted result. Window labels and boundaries are
    derived from the start afterwards.
    """
    dynamic = options.dynamic
    _check_supported(dynamic)
    index_name = dynamic.index_column
    index_dtype = inp.schema[index_name]
    if dynamic.include_boundaries and index_dtype.id() == plc.TypeId.TIMESTAMP_DAYS:
        # Polars' schema says Date, but it computes microsecond Datetimes.
        raise NotImplementedError("group_by_dynamic boundaries on a Date index")
    window = _plan_window(index_dtype, dynamic.every, dynamic.offset)
    # An Int32 window start may not fit back into Int32 (polars nulls it, we
    # cannot): group by and sort on it as wide INT64 ticks instead, and only
    # narrow to Int32 (with null on overflow) for the materialized columns
    # below. Date and Int64 starts already fit, so this is a no-op for them.
    key_dtype = (
        DataType(pl.Int64())
        if index_dtype.plc_type.id() == plc.TypeId.INT32
        else index_dtype
    )
    names = unique_names((*inp.schema, *schema))

    start_name = next(names)
    start = DynamicWindowStart(
        key_dtype,
        window.plan,
        window.offset,
        dynamic.closed_window,
        expr.Col(index_dtype, index_name),
        *(k.value for k in keys),
    )
    with_start = ir.HStack(
        {**inp.schema, start_name: key_dtype},
        [expr.NamedExpr(start_name, start)],
        True,  # noqa: FBT003
        inp,
    )
    start_col = expr.Col(key_dtype, start_name)
    narrow_start = DynamicWindowEnd(index_dtype, 0, 0, start_col)

    end = DynamicWindowEnd(index_dtype, window.months, window.ticks, start_col)
    group_aggs = list(aggs)
    group_schema = {k.name: k.value.dtype for k in keys}
    group_schema[start_name] = key_dtype
    group_schema.update((agg.name, schema[agg.name]) for agg in aggs)
    label: expr.Expr
    if dynamic.label == "left":
        label = narrow_start
    elif dynamic.label == "right":
        label = end
    else:
        # The first index value in each window, which is its minimum because
        # the index is sorted within the window.
        datapoint_name = next(names)
        group_aggs.append(
            expr.NamedExpr(
                datapoint_name,
                expr.Agg(
                    index_dtype,
                    "min",
                    False,  # noqa: FBT003
                    ExecutionContext.GROUPBY,
                    expr.Col(index_dtype, index_name),
                ),
            )
        )
        group_schema[datapoint_name] = index_dtype
        label = expr.Col(index_dtype, datapoint_name)
    grouped = rewrite_groupby(
        group_schema,
        [*keys, expr.NamedExpr(start_name, start_col)],
        group_aggs,
        False,  # noqa: FBT003
        None,
        with_start,
    )

    sort_by = [expr.NamedExpr(k.name, expr.Col(k.value.dtype, k.name)) for k in keys]
    ordered = ir.Sort(
        grouped.schema,
        [*sort_by, expr.NamedExpr(start_name, start_col)],
        [plc.types.Order.ASCENDING] * (len(keys) + 1),
        [*(_null_order(k.value.dtype) for k in keys), plc.types.NullOrder.BEFORE],
        False,  # noqa: FBT003
        options.slice,
        grouped,
    )

    outputs: dict[str, expr.Expr] = {index_name: label}
    if dynamic.include_boundaries:
        outputs["_lower_boundary"] = narrow_start
        outputs["_upper_boundary"] = end
    return ir.Select(
        schema,
        [
            expr.NamedExpr(name, outputs.get(name, expr.Col(dtype, name)))
            for name, dtype in schema.items()
        ],
        False,  # noqa: FBT003
        ordered,
    )
