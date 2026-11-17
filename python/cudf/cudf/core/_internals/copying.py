# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pylibcudf as plc

from cudf.core.column.utils import access_columns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf.core.column import ColumnBase
    from cudf.core.column.numerical import NumericalColumn


def gather(
    columns: Sequence[ColumnBase],
    gather_map: NumericalColumn,
    nullify: bool = False,
) -> tuple[plc.Column, ...]:
    with access_columns(
        *columns, gather_map, mode="read", scope="internal"
    ) as (*columns, gather_map):
        plc_tbl = plc.copying.gather(
            plc.Table([col.plc_column for col in columns]),
            gather_map.plc_column,
            plc.copying.OutOfBoundsPolicy.NULLIFY
            if nullify
            else plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        )
    return plc_tbl.columns()


def scatter(
    sources: list[ColumnBase] | list[plc.Scalar],
    scatter_map: NumericalColumn,
    target_columns: list[ColumnBase],
    bounds_check: bool = True,
) -> tuple[plc.Column, ...]:
    """
    Scattering source into target as per the scatter map.
    `source` can be a list of scalars, or a list of columns. The number of
    items in `sources` must equal the number of `target_columns` to scatter.
    """
    # TODO: Only single column scatter is used, we should explore multi-column
    # scatter for frames for performance increase.

    if len(sources) != len(target_columns):
        raise ValueError("Mismatched number of source and target columns.")

    if len(sources) == 0:
        return ()

    if bounds_check:
        n_rows = len(target_columns[0])
        if not (
            (scatter_map >= -n_rows).all() and (scatter_map < n_rows).all()
        ):
            raise IndexError(
                f"index out of bounds for column of size {n_rows}"
            )

    with access_columns(  # type: ignore[assignment]
        *target_columns, mode="write", scope="internal"
    ) as target_columns:
        plc_tbl = plc.copying.scatter(
            cast("list[plc.Scalar]", sources)
            if isinstance(sources[0], plc.Scalar)
            else plc.Table(
                [col.plc_column for col in cast("list[ColumnBase]", sources)]
            ),
            scatter_map.plc_column,
            plc.Table([col.plc_column for col in target_columns]),
        )

    return plc_tbl.columns()


def columns_split(
    input_columns: Sequence[ColumnBase], splits: list[int]
) -> list[tuple[plc.Column, ...]]:
    with access_columns(
        *input_columns, mode="read", scope="internal"
    ) as input_columns:
        return [
            plc_tbl.columns()
            for plc_tbl in plc.copying.split(
                plc.Table([col.plc_column for col in input_columns]),
                splits,
            )
        ]
