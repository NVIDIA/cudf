# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sorting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_polars.containers import Column


def sort_order(
    descending: Sequence[bool], *, nulls_last: Sequence[bool], num_keys: int
) -> tuple[list[plc.types.Order], list[plc.types.NullOrder]]:
    """
    Produce sort order arguments.

    Parameters
    ----------
    descending
        List indicating order for each column
    nulls_last
        Should nulls sort last or first?
    num_keys
        Number of sort keys

    Returns
    -------
    tuple of column_order and null_precedence
    suitable for passing to sort routines
    """
    # Mimicking polars broadcast handling of descending
    if num_keys > (n := len(descending)) and n == 1:
        descending = [descending[0]] * num_keys
    if num_keys > (n := len(nulls_last)) and n == 1:
        nulls_last = [nulls_last[0]] * num_keys
    column_order = [
        plc.types.Order.DESCENDING if d else plc.types.Order.ASCENDING
        for d in descending
    ]
    null_precedence = []
    if len(descending) != len(nulls_last) or len(descending) != num_keys:
        raise ValueError("Mismatching length of arguments in sort_order")
    for asc, null_last in zip(column_order, nulls_last, strict=True):
        if (asc == plc.types.Order.ASCENDING) ^ (not null_last):
            null_precedence.append(plc.types.NullOrder.AFTER)
        elif (asc == plc.types.Order.ASCENDING) ^ null_last:
            null_precedence.append(plc.types.NullOrder.BEFORE)
    return column_order, null_precedence


def nulls_at_start(column: Column) -> bool:
    """
    Return whether the nulls of a sorted column are at its start.

    Parameters
    ----------
    column
        Column with ``is_sorted == plc.types.Sorted.YES``.

    Returns
    -------
    True if the nulls are at the start, False if they are at the end.
    """
    return (column.order == plc.types.Order.ASCENDING) == (
        column.null_order == plc.types.NullOrder.BEFORE
    )
