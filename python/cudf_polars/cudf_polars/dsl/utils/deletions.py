# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-level deletions of an Iceberg or Delta Lake scan."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.containers.datatype import DataType

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.utils.lake import LakeScanOptions

__all__ = ["apply_deletions", "deletion_mask"]


def deletion_mask(
    paths: Sequence[str],
    rows_per_path: Sequence[int],
    lake_options: LakeScanOptions,
    *,
    stream: Stream,
) -> Column | None:
    """
    Build the mask of rows a lake scan keeps.

    Parameters
    ----------
    paths
        Data files of the scan, in scan order.
    rows_per_path
        Number of rows each path contributed to the frame.
    lake_options
        Options of the scan.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    Boolean column that is true for the rows to keep, or ``None`` when
    nothing is deleted.
    """
    positions: list[plc.Column] = []
    puffin_cache: dict[str, dict[str, pa.ChunkedArray]] = {}
    offset = 0
    for index, (path, num_rows) in enumerate(zip(paths, rows_per_path, strict=True)):
        selection = lake_options.delta_selections.get(index)
        if selection is None:
            file_positions = _iceberg_positions(
                index, path, lake_options, puffin_cache, stream=stream
            )
            if file_positions is not None:
                positions.append(
                    file_positions
                    if offset == 0
                    else plc.binaryop.binary_operation(
                        file_positions,
                        plc.Scalar.from_py(
                            offset, plc.DataType(plc.TypeId.INT64), stream=stream
                        ),
                        plc.binaryop.BinaryOperator.ADD,
                        plc.DataType(plc.TypeId.INT64),
                        stream=stream,
                    )
                )
        else:
            deleted = (
                selection.fill_null(value=False).not_().arg_true().cast(pl.Int64)
                + offset
            )
            positions.append(plc.Column.from_arrow(deleted.to_arrow(), stream=stream))
        offset += num_rows
    if not positions:
        return None
    (mask,) = plc.copying.scatter(
        [
            plc.Scalar.from_py(
                False,  # noqa: FBT003
                plc.DataType(plc.TypeId.BOOL8),
                stream=stream,
            )
        ],
        plc.unary.cast(
            positions[0]
            if len(positions) == 1
            else plc.concatenate.concatenate(positions, stream=stream),
            plc.DataType(plc.types.SIZE_TYPE_ID),
            stream=stream,
        ),
        plc.Table(
            [
                plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        True,  # noqa: FBT003
                        plc.DataType(plc.TypeId.BOOL8),
                        stream=stream,
                    ),
                    offset,
                    stream=stream,
                )
            ]
        ),
        stream=stream,
    ).columns()
    return Column(mask, dtype=DataType(pl.Boolean()))


def apply_deletions(
    df: DataFrame,
    paths: Sequence[str],
    rows_per_path: Sequence[int],
    lake_options: LakeScanOptions,
    *,
    stream: Stream,
) -> DataFrame:
    """
    Drop the rows an Iceberg or Delta scan deletes.

    Parameters
    ----------
    df
        Frame holding the rows of ``paths``, in scan order.
    paths
        Data files of the scan, in scan order.
    rows_per_path
        Number of rows each path contributed to ``df``.
    lake_options
        Options of the scan.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    The frame without the deleted rows.
    """
    mask = deletion_mask(paths, rows_per_path, lake_options, stream=stream)
    if mask is None:
        return df
    if not df.columns:
        # Preserve the row counts
        kept_count = plc.stream_compaction.apply_retention_mask(
            plc.Table([mask.obj]), mask.obj, stream=stream
        ).num_rows()
        return DataFrame([], num_rows=kept_count, stream=stream)
    return df.filter(mask)


def _iceberg_positions(
    index: int,
    path: str,
    lake_options: LakeScanOptions,
    puffin_cache: dict[str, dict[str, pa.ChunkedArray]],
    *,
    stream: Stream,
) -> plc.Column | None:
    """Positions of the rows deleted from one Iceberg data file."""
    delete_files = lake_options.position_deletes.get(index)
    if delete_files is not None:
        options = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(list(delete_files))
        ).build()
        options.set_column_names(["pos"])
        (positions,) = plc.io.parquet.read_parquet(options, stream=stream).tbl.columns()
        return positions
    puffin = lake_options.deletion_vectors.get(index)
    if puffin is None:
        return None
    if puffin not in puffin_cache:
        # polars.scan_iceberg imports pyiceberg, so pyiceberg should be available
        from pyiceberg.table.deletion_vector import deletion_vectors_from_puffin_file
        from pyiceberg.table.puffin import PuffinFile

        puffin_cache[puffin] = {
            vector.referenced_data_file: vector.to_vector()
            for vector in deletion_vectors_from_puffin_file(
                PuffinFile(Path(puffin).read_bytes())
            )
        }
    vector = puffin_cache[puffin].get(path)
    if vector is None:  # pragma: no cover; polars keys on the scan path
        return None
    return plc.Column.from_arrow(vector.cast("int64"), stream=stream)
