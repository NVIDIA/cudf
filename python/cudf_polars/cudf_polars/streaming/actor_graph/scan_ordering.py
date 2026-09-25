# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Infer ordering from Parquet scan-task metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.sort import (
    _extract_boundaries_from_endpoint_rows as _extract_ordering_boundaries,
)
from cudf_polars.streaming.io import ParquetScanTask
from cudf_polars.streaming.partitioning_requests import OrderPartitioningRequest
from cudf_polars.utils.dtypes import is_order_preserving_cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import CachedParquetInfo, IRExecutionContext, Scan
    from cudf_polars.streaming.io import StreamingScan
    from cudf_polars.streaming.partitioning_requests import PartitioningRequest


def _null_column(ir: StreamingScan, name: str, size: int, stream: Stream) -> plc.Column:
    return plc.Column.from_scalar(
        plc.Scalar.from_py(None, ir.schema[name].plc_type, stream=stream),
        size,
        stream=stream,
    )


def _gather_rows(table: plc.Table, rows: list[int], stream: Stream) -> plc.Table:
    return plc.copying.gather(
        table,
        plc.Column.from_iterable_of_py(
            rows, plc.DataType(plc.TypeId.INT32), stream=stream
        ),
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )


def _get_ordering_candidates(
    ir: StreamingScan,
    requests: tuple[PartitioningRequest, ...],
) -> list[tuple[str, OrderKey]]:
    """Return the distinct leading keys requested downstream."""
    candidates: list[tuple[str, OrderKey]] = []
    for request in requests:
        if not isinstance(request, OrderPartitioningRequest):
            continue
        assert request.keys, "Order partitioning requests must have at least one key."
        key = request.keys[0]
        assert key.name in ir.schema, (
            f"Ordering request key {key.name!r} must be present in scan schema."
        )
        (column_index,) = names_to_indices((key.name,), ir.schema)
        candidate = (key.name, OrderKey(column_index, key.order, key.null_order))
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _get_rank_parquet_info_map(
    base_scan: Scan,
    paths: list[str],
) -> dict[str, CachedParquetInfo]:
    path_set = set(paths)
    cached_parquet_info_map = {
        info.path: info
        for info in (base_scan.cached_parquet_info or ())
        if info.path in path_set
    }
    if set(cached_parquet_info_map) == path_set:
        return cached_parquet_info_map

    from cudf_polars.dsl.utils.io import _prefetch_parquet_footers_for_paths

    fetched = _prefetch_parquet_footers_for_paths(paths)
    rank_parquet_info_map = {info.path: info for info in fetched}
    assert all(path in rank_parquet_info_map for path in paths), (
        "Ordering footer metadata must contain all rank-local scan paths."
    )
    return rank_parquet_info_map


def _stats_are_safe(
    rank_row_group_metadata: Sequence[plc.io.parquet_metadata.RowGroup],
    name: str,
    indices: list[int],
) -> bool:
    for index in indices:
        row_group = rank_row_group_metadata[index]
        column_chunk = next(
            (
                column_chunk
                for column_chunk in row_group.columns
                if ".".join(column_chunk.meta_data.path_in_schema) == name
            ),
            None,
        )
        stats = None if column_chunk is None else column_chunk.meta_data.statistics
        # Nulls are safe when they all belong in the first or last task,
        # depending on the requested null order. Bail for now to keep the
        # inference logic simple.
        if (
            stats is None
            or stats.null_count is None
            or stats.null_count != 0
            or stats.is_min_value_exact is False
            or stats.is_max_value_exact is False
        ):
            return False
    return True


def _candidate_task_bounds(
    file_metadata: list[plc.io.parquet_metadata.FileMetaData],
    rank_row_group_metadata: Sequence[plc.io.parquet_metadata.RowGroup],
    task_row_group_indices: Sequence[list[int] | None],
    name: str,
    key: OrderKey,
    dtype: plc.DataType,
    stream: Stream,
) -> plc.Column | None:
    """
    Return start and end bounds for each local task.

    Bounds alternate: ``[task0_start, task0_end, task1_start, task1_end, ...]``.
    Parquet stores min/max statistics as encoded values. libcudf decodes them
    into typed device columns used by libcudf sorting operations.
    """
    try:
        bounds = plc.io.parquet_metadata.read_parquet_column_chunk_bounds(
            file_metadata, columns=[name], stream=stream
        )
    except ValueError:
        return None

    columns = bounds.columns()[2:]
    assert len(columns) == 2, "Single-column parquet bounds must have min/max columns."
    min_col, max_col = columns
    assert min_col.size() == max_col.size(), "Parquet min/max columns must align."
    bounds_type = min_col.type()
    assert max_col.type() == bounds_type, (
        "Parquet min/max columns must have matching types."
    )
    assert len(rank_row_group_metadata) == min_col.size(), (
        "Decoded parquet bounds must match footer row-group metadata."
    )

    if min_col.null_count() or max_col.null_count():
        return None

    valid_task_row_group_indices: list[list[int]] = []
    for row_group_indices in task_row_group_indices:
        if row_group_indices is None or not _stats_are_safe(
            rank_row_group_metadata, name, row_group_indices
        ):
            return None
        valid_task_row_group_indices.append(row_group_indices)

    if bounds_type != dtype:
        if not is_order_preserving_cast(bounds_type, dtype):
            return None
        min_col = plc.unary.cast(min_col, dtype, stream=stream)
        max_col = plc.unary.cast(max_col, dtype, stream=stream)

    def invalidate() -> plc.Column:
        return plc.Column.all_null_like(
            min_col, 2 * len(task_row_group_indices), stream=stream
        )

    start, end = (
        (max_col, min_col)
        if key.order == plc.types.Order.DESCENDING
        else (
            min_col,
            max_col,
        )
    )
    row_group_bounds = plc.concatenate.concatenate(
        [plc.Table([start]), plc.Table([end])], stream=stream
    )

    task_bounds: list[plc.Table] = []
    for row_group_indices in valid_task_row_group_indices:
        selected = _gather_rows(
            row_group_bounds,
            [
                i
                for group in row_group_indices
                for i in (group, len(rank_row_group_metadata) + group)
            ],
            stream,
        )
        if not plc.sorting.is_sorted(
            selected, [key.order], [key.null_order], stream=stream
        ):
            return invalidate()
        task_bounds.append(_gather_rows(selected, [0, selected.num_rows() - 1], stream))

    return plc.concatenate.concatenate(task_bounds, stream=stream).columns()[0]


def _extract_local_task_bounds(
    ir: StreamingScan,
    tasks: Sequence[ParquetScanTask],
    candidates: list[tuple[str, OrderKey]],
    rank_parquet_info_map: dict[str, CachedParquetInfo],
    stream: Stream,
) -> plc.Table:
    """Return two endpoint rows per task for each candidate."""
    paths = list(dict.fromkeys(path for task in tasks for path in task.paths))
    rank_row_group_offset_map: dict[str, int] = {}
    rank_row_group_metadata: list[plc.io.parquet_metadata.RowGroup] = []
    for path in paths:
        info = rank_parquet_info_map[path]
        rank_row_group_offset_map[path] = len(rank_row_group_metadata)
        rank_row_group_metadata.extend(info.file_metadata.row_groups)
    task_row_group_indices = [
        task.absolute_row_group_indices(
            rank_parquet_info_map, rank_row_group_offset_map
        )
        for task in tasks
    ]

    file_metadata = [rank_parquet_info_map[path].file_metadata for path in paths]
    bound_count = 2 * len(tasks)
    columns: list[plc.Column] = []
    for name, key in candidates:
        column = (
            _candidate_task_bounds(
                file_metadata,
                rank_row_group_metadata,
                task_row_group_indices,
                name,
                key,
                ir.schema[name].plc_type,
                stream,
            )
            if tasks
            else None
        )
        if column is None:
            column = _null_column(ir, name, bound_count, stream)
        columns.append(column)
    return plc.Table(columns)


def _partitioning_from_task_bounds(
    context: Context,
    candidates: list[tuple[str, OrderKey]],
    bounds: plc.Table,
    global_task_count: int,
    stream: Stream,
) -> Partitioning | None:
    """Infer global partitioning from task bounds in rank order."""
    for i, (_, key) in enumerate(candidates):
        column = bounds.columns()[i]
        if column.null_count():
            continue

        candidate_bounds = plc.Table([column])
        if not plc.sorting.is_sorted(
            candidate_bounds, [key.order], [key.null_order], stream=stream
        ):
            continue

        if global_task_count < 2:
            ordering_boundaries = plc.Table(
                [plc.Column.from_iterable_of_py([], column.type(), stream=stream)]
            )
            strict = True
        else:
            ordering_boundaries, strict = _extract_ordering_boundaries(
                candidate_bounds, global_task_count, stream
            )
        return Partitioning(
            inter_rank=OrderScheme(
                [
                    Ordering(
                        [key],
                        TableChunk.from_pylibcudf_table(
                            ordering_boundaries,
                            stream,
                            exclusive_view=True,
                            br=context.br(),
                        ),
                        strict_boundaries=strict,
                        locally_ordered=False,
                    )
                ]
            ),
            local="inherit",
        )
    return None


async def parquet_metadata_ordering(
    context: Context,
    comm: Communicator,
    ir: StreamingScan,
    global_chunk_count: int,
    requests: tuple[PartitioningRequest, ...],
    ir_context: IRExecutionContext,
    collective_id: int,
) -> Partitioning | None:
    """
    Return partitioning inferred from Parquet footer statistics.

    Only the leading key of each ordering request is inspected. For a request
    on ``[a, b]``, this inspects ``a`` but not ``b``, because single-column
    statistics cannot prove ordering on ``[a, b]``. Inference succeeds only
    when task bounds are globally ordered.
    """
    assert ir.base_scan.typ == "parquet", (
        f"Expected parquet Scan, got {ir.base_scan.typ}."
    )
    assert global_chunk_count > 0, "Scan partition count must be positive."

    if not (candidates := _get_ordering_candidates(ir, requests)):
        return None

    tasks: list[ParquetScanTask] = []
    for task in ir.tasks:
        assert isinstance(task, ParquetScanTask)
        tasks.append(task)

    paths = list(dict.fromkeys(path for task in tasks for path in task.paths))
    rank_parquet_info_map = await ir_context.to_thread(
        _get_rank_parquet_info_map, ir.base_scan, paths
    )
    stream = ir_context.get_cuda_stream()
    local_task_bounds = _extract_local_task_bounds(
        ir, tasks, candidates, rank_parquet_info_map, stream
    )
    global_task_bounds = local_task_bounds
    if comm.nranks > 1:
        local_chunk = TableChunk.from_pylibcudf_table(
            local_task_bounds, stream, exclusive_view=True, br=context.br()
        )
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, local_chunk)
        global_task_bounds = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )

    assert global_task_bounds.num_rows() == 2 * global_chunk_count, (
        "Ordering task bounds must contain first/last rows for every scan task."
    )
    return _partitioning_from_task_bounds(
        context, candidates, global_task_bounds, global_chunk_count, stream
    )
