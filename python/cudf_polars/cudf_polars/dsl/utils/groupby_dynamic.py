# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rewrite non-overlapping dynamic windows as an ordinary groupby."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr, ir
from cudf_polars.dsl.utils.groupby import rewrite_groupby
from cudf_polars.dsl.utils.naming import unique_names

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_polars.typing import Duration, Schema

__all__ = ["rewrite_groupby_dynamic"]


def _duration_string(duration: Duration) -> str:
    """Return a positive fixed duration as a Polars duration string."""
    months, weeks, days, nanoseconds, parsed_int, negative = duration
    if months or weeks or days or parsed_int or negative or nanoseconds <= 0:
        raise NotImplementedError("group_by_dynamic with a non-fixed duration")
    return f"{nanoseconds}ns"


def _check_supported(dynamic: Any) -> None:
    """Validate the supported dynamic-window options."""
    if dynamic.period != dynamic.every:
        raise NotImplementedError("group_by_dynamic with period != every")
    if dynamic.offset != (0, 0, 0, 0, False, False):
        raise NotImplementedError("group_by_dynamic with a nonzero offset")
    if dynamic.closed_window != "left":
        raise NotImplementedError(
            f"group_by_dynamic with closed={dynamic.closed_window!r}"
        )
    if dynamic.label != "left":
        raise NotImplementedError(f"group_by_dynamic with label={dynamic.label!r}")
    if dynamic.include_boundaries:
        raise NotImplementedError("group_by_dynamic with boundary columns")
    if dynamic.start_by != "window_bound":
        raise NotImplementedError(
            f"group_by_dynamic with start_by={dynamic.start_by!r}"
        )


def _null_order(dtype: DataType) -> plc.types.NullOrder:
    """Return the null order Polars uses for dynamic-groupby keys."""
    if isinstance(dtype.polars_type, (pl.String, pl.Binary)):
        return plc.types.NullOrder.AFTER
    return plc.types.NullOrder.BEFORE


def rewrite_groupby_dynamic(
    schema: Schema,
    keys: Sequence[expr.NamedExpr],
    aggs: Sequence[expr.NamedExpr],
    inp: ir.IR,
    options: Any,
) -> ir.Select:
    """Rewrite a supported ``group_by_dynamic`` as a grouped truncation."""
    dynamic = options.dynamic
    _check_supported(dynamic)
    index_name = dynamic.index_column
    index_dtype = inp.schema[index_name]
    if not isinstance(index_dtype.polars_type, pl.Datetime):
        raise NotImplementedError("group_by_dynamic with a non-Datetime index")
    every = _duration_string(dynamic.every)

    start_name = next(unique_names((*inp.schema, *schema)))
    start = expr.TemporalFunction(
        index_dtype,
        expr.TemporalFunction.Name.Truncate,
        (),
        expr.Col(index_dtype, index_name),
        expr.Literal(DataType(pl.String()), every),
    )
    with_start = ir.HStack(
        {**inp.schema, start_name: index_dtype},
        [expr.NamedExpr(start_name, start)],
        True,  # noqa: FBT003
        inp,
    )
    start_col = expr.Col(index_dtype, start_name)
    group_schema = {start_name: index_dtype}
    group_schema.update((k.name, k.value.dtype) for k in keys)
    group_schema.update((agg.name, schema[agg.name]) for agg in aggs)
    grouped = rewrite_groupby(
        group_schema,
        [expr.NamedExpr(start_name, start_col), *keys],
        aggs,
        with_start,
        maintain_order=True,
        zlice=None,
    )

    sort_by = [expr.NamedExpr(k.name, expr.Col(k.value.dtype, k.name)) for k in keys]
    # TODO: Remove this compatibility sort once grouped aggregation can produce
    # Polars' key-first, window-ordered output directly.
    ordered = ir.Sort(
        grouped.schema,
        [*sort_by, expr.NamedExpr(start_name, start_col)],
        [plc.types.Order.ASCENDING] * (len(keys) + 1),
        [*(_null_order(k.value.dtype) for k in keys), plc.types.NullOrder.BEFORE],
        False,  # noqa: FBT003
        options.slice,
        grouped,
    )
    return ir.Select(
        schema,
        [
            expr.NamedExpr(
                name,
                start_col if name == index_name else expr.Col(dtype, name),
            )
            for name, dtype in schema.items()
        ],
        False,  # noqa: FBT003
        ordered,
    )
