# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Testing utilities for cudf_streaming."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf

from rmm.pylibrmm.stream import DEFAULT_STREAM

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream


def assert_eq(
    left: pylibcudf.Table,
    right: pylibcudf.Table,
    *,
    sort_rows: int | None = None,
    stream: Stream | None = None,
) -> None:
    """
    Assert that two tables are equivalent using pylibcudf.

    Parameters
    ----------
    left
        plc.Table to compare.
    right
        plc.Table to compare.
    sort_rows
        If not None, sort both tables by this column before comparing.
        An ``int`` is treated as a column index.
    stream
        CUDA stream to use for the comparison.

    Raises
    ------
    AssertionError
        If the two tables do not compare equal.
    """
    if stream is None:
        stream = DEFAULT_STREAM

    if sort_rows is not None:
        column_order = [pylibcudf.types.Order.ASCENDING]
        null_precedence = [pylibcudf.types.NullOrder.BEFORE]
        left = pylibcudf.sorting.stable_sort_by_key(
            left,
            pylibcudf.Table([left.columns()[sort_rows]]),
            column_order,
            null_precedence,
            stream=stream,
        )
        right = pylibcudf.sorting.stable_sort_by_key(
            right,
            pylibcudf.Table([right.columns()[sort_rows]]),
            column_order,
            null_precedence,
            stream=stream,
        )
    if not pylibcudf.table_equality.tables_equal(left, right, stream=stream):
        raise AssertionError(f"Table are not equal with {sort_rows=}")
