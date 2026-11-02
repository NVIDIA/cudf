# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-level deletions of an Iceberg or Delta Lake scan."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.containers.datatype import DataType
from cudf_polars.utils.versions import POLARS_VERSION_LT_21

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.utils.lake import LakehouseScanOptions

__all__ = ["apply_deletions", "deletion_mask"]


def deletion_mask(
    paths: Sequence[str],
    rows_per_path: Sequence[int],
    lake_options: LakehouseScanOptions,
    *,
    stream: Stream,
) -> Column:
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
    Boolean column that is true for the rows to keep.
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
        else:
            file_positions = plc.Column.from_arrow(
                selection.fill_null(value=False).not_().arg_true().cast(pl.Int64),
                stream=stream,
            )
        if file_positions is not None:
            positions.append(
                _scan_positions(file_positions, num_rows, offset, stream=stream)
            )
        offset += num_rows
    keep = plc.Column.from_scalar(
        plc.Scalar.from_py(
            True,  # noqa: FBT003
            plc.DataType(plc.TypeId.BOOL8),
            stream=stream,
        ),
        offset,
        stream=stream,
    )
    if not positions:  # pragma: no cover; only Polars 2.0 has no missing Puffin error
        return Column(keep, dtype=DataType(pl.Boolean()))
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
        plc.Table([keep]),
        stream=stream,
    ).columns()
    return Column(mask, dtype=DataType(pl.Boolean()))


def apply_deletions(
    df: DataFrame,
    paths: Sequence[str],
    rows_per_path: Sequence[int],
    lake_options: LakehouseScanOptions,
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
    if not df.columns:
        # Preserve the row counts
        kept_count = plc.stream_compaction.apply_retention_mask(
            plc.Table([mask.obj]), mask.obj, stream=stream
        ).num_rows()
        return DataFrame([], num_rows=kept_count, stream=stream)
    return df.filter(mask)


def _scan_positions(
    file_positions: plc.Column, num_rows: int, offset: int, *, stream: Stream
) -> plc.Column:
    """Positions within the scan of the deleted rows that exist in one data file."""
    int64 = plc.DataType(plc.TypeId.INT64)
    position = plc.expressions.ColumnReference(0)
    in_file = plc.transform.compute_column(
        plc.Table([file_positions]),
        plc.expressions.Operation(
            plc.expressions.ASTOperator.LOGICAL_AND,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.GREATER_EQUAL,
                position,
                plc.expressions.Literal(plc.Scalar.from_py(0, int64, stream=stream)),
            ),
            plc.expressions.Operation(
                plc.expressions.ASTOperator.LESS,
                position,
                plc.expressions.Literal(
                    plc.Scalar.from_py(num_rows, int64, stream=stream)
                ),
            ),
        ),
        stream=stream,
    )
    (kept,) = plc.stream_compaction.apply_retention_mask(
        plc.Table([file_positions]), in_file, stream=stream
    ).columns()
    if offset == 0:
        return kept
    return plc.transform.compute_column(
        plc.Table([kept]),
        plc.expressions.Operation(
            plc.expressions.ASTOperator.ADD,
            position,
            plc.expressions.Literal(plc.Scalar.from_py(offset, int64, stream=stream)),
        ),
        stream=stream,
    )


def _iceberg_positions(
    index: int,
    path: str,
    lake_options: LakehouseScanOptions,
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
        options.set_column_names(["file_path", "pos"])
        deletes = plc.io.parquet.read_parquet(options, stream=stream)
        file_paths, positions = deletes.tbl.columns()
        per_delete_file = plc.copying.split(
            file_paths,
            list(itertools.accumulate(deletes.num_rows_per_source[:-1])),
            stream=stream,
        )
        for delete_file, referenced in zip(delete_files, per_delete_file, strict=True):
            if (
                plc.reduce.distinct_count(
                    referenced,
                    plc.types.NullPolicy.INCLUDE,
                    plc.types.NanPolicy.NAN_IS_VALID,
                    stream=stream,
                )
                > 1
            ):
                raise pl.exceptions.ComputeError(
                    f"iceberg: position delete file {delete_file} references "
                    "more than one data file"
                )
        return positions
    puffin = lake_options.deletion_vectors.get(index)
    if puffin is None:
        return None
    if puffin not in puffin_cache:
        # polars.scan_iceberg imports pyiceberg, so pyiceberg should be available
        from pyiceberg.table.deletion_vector import deletion_vectors_from_puffin_file
        from pyiceberg.table.puffin import PuffinFile

        # Scan paths have the file:// scheme stripped, so the keys must too.
        puffin_cache[puffin] = {
            vector.referenced_data_file.removeprefix("file://"): vector.to_vector()
            for vector in deletion_vectors_from_puffin_file(
                PuffinFile(Path(puffin).read_bytes())
            )
        }
    vector = puffin_cache[puffin].get(path)
    if vector is None:
        if POLARS_VERSION_LT_21:  # pragma: no cover; CI tests the newest polars
            return None
        raise pl.exceptions.ComputeError(
            f"iceberg: deletion vector file {puffin} has no deletion vector "
            f"for data file {path}"
        )
    return plc.Column.from_arrow(vector.cast("int64"), stream=stream)
