# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for reshaping Columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_polars.containers import Column

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream


def broadcast(
    *columns: Column, target_length: int | None = None, stream: Stream
) -> list[Column]:
    """
    Broadcast a sequence of columns to a common length.

    Parameters
    ----------
    columns
        Columns to broadcast.
    target_length
        Optional length to broadcast to. If not provided, uses the
        non-unit length of existing columns.
    stream
        CUDA stream used for device memory operations and kernel launches
        on this dataframe.

    Returns
    -------
    List of broadcasted columns all of the same length.

    Raises
    ------
    RuntimeError
        If broadcasting is not possible.

    Notes
    -----
    In evaluation of a set of expressions, polars type-puns length-1
    columns with scalars. When we insert these into a DataFrame
    object, we need to ensure they are of equal length. This function
    takes some columns, some of which may be length-1 and ensures that
    all length-1 columns are broadcast to the length of the others.

    Broadcasting is only possible if the set of lengths of the input
    columns is a subset of ``{1, n}`` for some (fixed) ``n``. If
    ``target_length`` is provided and not all columns are length-1
    (i.e. ``n != 1``), then ``target_length`` must be equal to ``n``.
    """
    if len(columns) == 0:
        return []
    lengths: set[int] = {column.size for column in columns}
    if lengths == {1}:
        if target_length is None:
            return list(columns)
        nrows = target_length
    else:
        try:
            (nrows,) = lengths.difference([1])
        except ValueError as e:
            raise RuntimeError("Mismatching column lengths") from e
        if target_length is not None and nrows != target_length:
            raise RuntimeError(
                f"Cannot broadcast columns of length {nrows=} to {target_length=}"
            )
    return [
        column
        if column.size == nrows
        else Column(
            # Can be removed if https://github.com/NVIDIA/cudf/issues/24763 is addressed
            plc.filling.repeat(plc.Table([column.obj]), nrows, stream=stream).columns()[
                0
            ]
            if column.obj.type().id() == plc.TypeId.DICTIONARY32
            else plc.Column.from_scalar(
                column.obj_scalar(stream=stream), nrows, stream=stream
            ),
            is_sorted=plc.types.Sorted.YES,
            order=plc.types.Order.ASCENDING,
            null_order=plc.types.NullOrder.BEFORE,
            name=column.name,
            dtype=column.dtype,
        )
        for column in columns
    ]
