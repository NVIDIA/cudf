# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sort logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.dsl.ir import Empty, Sort
from cudf_polars.dsl.utils.naming import names_to_indices, unique_names
from cudf_polars.streaming.actor_graph.collectives import external_sort
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.shuffle import ShuffleManager
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.nodes import (
    default_node_single,
    shutdown_on_error,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    ChunkStore,
    NormalizedPartitioning,
    _sample_chunks,
    allgather_reduce,
    chunk_to_frame,
    chunkwise_evaluate,
    concat_batch,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    process_children,
    recv_metadata,
    send_metadata,
)
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.sort import (
    _get_final_sort_boundaries,
    _has_simple_zlice,
    _select_local_split_candidates,
    find_sort_splits,
)
from cudf_polars.utils.cuda_stream import get_joined_cuda_stream, stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import StreamingExecutor


@dataclass(frozen=True)
class OrderSchemePartitioningResult:
    """OrderScheme partitioning and buffered chunks consumed to derive it."""

    partitioning: Partitioning | None
    """The extracted partitioning, or ``None`` if extraction was not possible."""
    chunks: ChunkStore
    """The consumed chunks, stored in replay order."""


def _extract_boundaries_from_endpoint_rows(
    endpoint_key_rows: plc.Table,
    num_partitions: int,
    stream: Stream,
) -> tuple[plc.Table, bool]:
    """
    Extract boundaries from alternating partition endpoint rows.

    Parameters
    ----------
    endpoint_key_rows
        The table containing key-column rows ordered as
        ``[start0, end0, start1, end1, ...]``.
    num_partitions
        The number of partitions.
    stream
        The CUDA stream to use for the operation.

    Returns
    -------
    The boundaries table and whether they are strict.
    """
    # Boundaries are the first endpoint of all partitions except the first.
    # Strictness compares each partition end with the following partition start.
    previous_partition_ends = plc.concatenate.concatenate(
        plc.copying.slice(
            endpoint_key_rows, list(range(1, 2 * num_partitions - 1)), stream=stream
        ),
        stream=stream,
    )
    next_partition_starts = plc.concatenate.concatenate(
        plc.copying.slice(
            endpoint_key_rows, list(range(2, 2 * num_partitions)), stream=stream
        ),
        stream=stream,
    )

    # Single-kernel row-equality check across all key columns using AST
    num_cols = previous_partition_ends.num_columns()
    combined = plc.Table(
        list(previous_partition_ends.columns()) + list(next_partition_starts.columns())
    )
    eq_exprs = [
        plc.expressions.Operation(
            plc.expressions.ASTOperator.NULL_EQUAL,
            plc.expressions.ColumnReference(j),
            plc.expressions.ColumnReference(num_cols + j),
        )
        for j in range(num_cols)
    ]
    row_eq_expr = eq_exprs[0]
    for expr in eq_exprs[1:]:
        row_eq_expr = plc.expressions.Operation(
            plc.expressions.ASTOperator.NULL_LOGICAL_AND,
            row_eq_expr,
            expr,
        )
    row_eq_col = plc.transform.compute_column(combined, row_eq_expr, stream=stream)
    strict = not plc.reduce.reduce(
        row_eq_col,
        plc.aggregation.any(),
        plc.DataType(plc.TypeId.BOOL8),
        stream=stream,
    ).to_py(stream=stream)
    return next_partition_starts, strict


async def extract_orderscheme_partitioning(
    context: Context,
    comm: Communicator,
    schema_ir: IR,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    order_keys: Sequence[OrderKey],
    collective_id: int,
) -> OrderSchemePartitioningResult:
    """
    Extract the partitioning metadata for a sorted channel.

    Parameters
    ----------
    context
        The RapidsMPF context.
    comm
        The RapidsMPF communicator.
    schema_ir
        The IR reference to use for the boundary table schema.
    ir_context
        The IR execution context.
    ch_in
        The channel to collect boundaries from.
    order_keys
        The order keys associated with the boundaries.
    collective_id
        The collective ID for the allgather.

    Returns
    -------
    A result containing a ``Partitioning`` whose ``inter_rank`` is an
    ``OrderScheme`` built from the observed boundaries and ``local`` is
    ``"inherit"``, or ``None`` if the channel contains insufficient data
    or the data is not globally sorted. The result also contains the
    consumed chunks in replay order.

    Notes
    -----
    This utility does not collect channel metadata, nor does
    it push any messages into an output channel. All data
    messages from the input channel are consumed.

    Boundaries are not collected for empty chunks.
    """
    # Collect the first and last row from each non-empty sorted chunk.
    endpoint_rows: list[plc.Table] = []
    chunks = ChunkStore(context)
    stream = ir_context.get_cuda_stream()
    row_indices = plc.Column.from_iterable_of_py(
        [0, -1],
        plc.DataType(plc.TypeId.INT32),
        stream=stream,
    )
    while (msg := await ch_in.recv(context)) is not None:
        chunk, _ = await make_table_chunks_available_or_wait(
            context,
            TableChunk.from_message(msg, br=context.br()),
            reserve_extra=0,
            net_memory_delta=0,
        )
        tbl = chunk.table_view()
        if tbl.num_rows() == 0:
            chunks.insert(Message(msg.sequence_number, chunk))
            continue
        with stream_ordered_after(lambda: stream, upstreams=(chunk.stream,)):
            endpoint_rows.append(
                plc.copying.gather(
                    tbl,
                    row_indices,
                    plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                    stream=stream,
                )
            )
        chunks.insert(Message(msg.sequence_number, chunk))
    endpoint_table: plc.Table | None = (
        plc.concatenate.concatenate(endpoint_rows, stream=stream)
        if endpoint_rows
        else None
    )
    del endpoint_rows

    # Allgather endpoint rows across all ranks.
    if comm.nranks > 1:
        local_chunk = (
            TableChunk.from_pylibcudf_table(
                endpoint_table, stream, exclusive_view=True, br=context.br()
            )
            if endpoint_table is not None
            else empty_table_chunk(schema_ir, context, stream)
        )
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, local_chunk)
        endpoint_table = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )

    # Return None if there are insufficient endpoints to process.
    if endpoint_table is None or (num_partitions := endpoint_table.num_rows() // 2) < 2:
        return OrderSchemePartitioningResult(None, chunks)

    key_indices = [key.column_index for key in order_keys]
    endpoint_key_rows = plc.Table(
        [endpoint_table.columns()[index] for index in key_indices]
    )
    del endpoint_table

    # Return None if chunk endpoints are not globally sorted.
    column_order = [key.order for key in order_keys]
    null_order = [key.null_order for key in order_keys]
    if not plc.sorting.is_sorted(
        endpoint_key_rows, column_order, null_order, stream=stream
    ):
        return OrderSchemePartitioningResult(None, chunks)

    # Extract boundaries and construct the Partitioning
    boundaries, strict = _extract_boundaries_from_endpoint_rows(
        endpoint_key_rows, num_partitions, stream
    )
    del endpoint_key_rows
    boundaries_chunk = TableChunk.from_pylibcudf_table(
        boundaries, stream, exclusive_view=True, br=context.br()
    )
    return OrderSchemePartitioningResult(
        Partitioning(
            inter_rank=OrderScheme(
                [Ordering(order_keys, boundaries_chunk, strict_boundaries=strict)]
            ),
            local="inherit",
        ),
        chunks,
    )


async def _simple_top_or_bottom_k(
    context: Context,
    comm: Communicator,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    ir: Sort,
    ir_context: IRExecutionContext,
    metadata_in: ChannelMetadata,
    collective_ids: list[int],
    tracer: ActorTracer | None,
) -> None:
    """Sort + simple head/tail slice."""
    # TODO: We may need to gate this optimization on the slice size.
    await send_metadata(
        ch_out,
        context,
        ChannelMetadata(local_count=1, partitioning=None, duplicated=True),
    )

    chunks: list[TableChunk] = []
    while (msg := await ch_in.recv(context)) is not None:
        chunks.append(
            await evaluate_chunk(
                context,
                TableChunk.from_message(msg, br=context.br()),
                ir,
                ir_context=ir_context,
            )
        )
    chunk: TableChunk
    if chunks:
        chunk = await evaluate_batch(chunks, context, ir, ir_context=ir_context)
    else:
        # This rank received no input partitions. Produce an empty chunk
        # with the IR's output schema so the AllGather below still has
        # something to insert (and other ranks don't deadlock waiting).
        chunk = empty_table_chunk(ir, context, ir_context.get_cuda_stream())
    chunks.clear()

    if comm.nranks > 1 and not metadata_in.duplicated:
        allgather = AllGatherManager(context, comm, collective_ids.pop())
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, chunk)

        stream = ir_context.get_cuda_stream()
        chunk = await evaluate_chunk(
            context,
            TableChunk.from_pylibcudf_table(
                await allgather.extract_concatenated(
                    stream, ordered=True, ir_context=ir_context
                ),
                stream,
                exclusive_view=True,
                br=context.br(),
            ),
            ir,
            ir_context=ir_context,
        )

    await send_chunk(context, ch_out, chunk, comm.rank, tracer=tracer)

    await ch_out.drain(context)


def _boundary_schema(by: list[str], by_dtypes: list[DataType]) -> Schema:
    """Schema of boundaries table."""
    name_gen = unique_names(by)
    part_id_dtype = DataType(pl.UInt32())
    return dict(
        zip(
            [*by, next(name_gen), next(name_gen)],
            [*by_dtypes, part_id_dtype, part_id_dtype],
            strict=True,
        )
    )


async def _compute_sort_boundaries(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    local_candidates_list: list[TableChunk],
    ir: Sort,
    by: list[str],
    num_partitions: int,
    allgather_id: int | None,
) -> DataFrame:
    """Compute global sort boundaries."""
    column_order = list(ir.order)
    null_order = list(ir.null_order)
    by_dtypes = [ir.schema[b] for b in by]
    boundary_ir = Empty(_boundary_schema(by, by_dtypes))
    local_boundaries_df = _get_final_sort_boundaries(
        chunk_to_frame(
            await concat_batch(
                local_candidates_list,
                context,
                boundary_ir.schema,
                ir_context,
            )
            if local_candidates_list
            else empty_table_chunk(
                boundary_ir,
                context,
                ir_context.get_cuda_stream(),
            ),
            boundary_ir,
        ),
        column_order,
        null_order,
        num_partitions,
    )
    stream = local_boundaries_df.stream

    if allgather_id is not None:
        chunk = TableChunk.from_pylibcudf_table(
            local_boundaries_df.table,
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        allgather = AllGatherManager(context, comm, allgather_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, chunk)
        concat_table = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )
        return _get_final_sort_boundaries(
            DataFrame.from_table(
                concat_table,
                list(boundary_ir.schema.keys()),
                list(boundary_ir.schema.values()),
                stream=stream,
            ),
            column_order,
            null_order,
            num_partitions,
        )
    else:
        return local_boundaries_df


# Number of leading chunks sampled per rank to derive range-partition boundaries
# when dynamic planning is disabled (with dynamic planning,
# ``DynamicPlanningOptions.sample_chunk_count`` is used). Boundaries only need to
# be *consistent* across ranks for correctness; sample size affects balance.
_BOUNDARY_SAMPLE_CHUNKS = 8


async def _sample_chunks_for_boundaries(
    context: Context,
    comm: Communicator,
    ch_in: Channel[TableChunk],
    num_partitions: int,
    metadata_in: ChannelMetadata,
    executor: StreamingExecutor,
    collective_ids: list[int],
) -> tuple[ChunkStore, int]:
    """
    Sample an input prefix for range boundaries and the size estimate.

    With dynamic planning the same sample also derives ``num_partitions``.

    Unlike the previous ``ChunkStore``-based sort, the sample is the *only*
    input buffered before the shuffle starts; every later chunk is
    range-split and inserted into the (disk-spilling) Shuffler as it arrives.
    """
    target_partition_size = executor.target_partition_size
    if executor.dynamic_planning is not None:
        sample_chunk_count = executor.dynamic_planning.sample_chunk_count
    else:
        sample_chunk_count = _BOUNDARY_SAMPLE_CHUNKS

    sample = await _sample_chunks(
        context,
        ch_in,
        sample_chunk_count,
        # Cap the sample at one target partition per sampled chunk.
        sample_chunk_count * target_partition_size,
        metadata_in.local_count,
    )

    if executor.dynamic_planning is not None:
        size_estimate_id = collective_ids.pop()
        if comm.nranks > 1 and not metadata_in.duplicated:
            (global_size,) = await allgather_reduce(
                context, comm, size_estimate_id, sample.total_size
            )
        else:
            global_size = sample.total_size
        num_partitions = max(1, math.ceil(global_size / target_partition_size))

    return sample.chunks, num_partitions


class _ChunkPreparer:
    """
    Locally sort input chunks and tag them for a stable sort.

    For stable sorts a sequence column is appended that keeps the post-shuffle
    sort stable across chunks and ranks.

    Chunks must be prepared in arrival order: the sequence column encodes
    ``(rank << 48) + running local row offset``.
    """

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        ir: Sort,
        ir_context: IRExecutionContext,
    ) -> None:
        self._context = context
        self._comm = comm
        self._ir = ir
        self._ir_context = ir_context
        self._local_row_offset = 0

    async def prepare(self, msg: Message) -> tuple[int, DataFrame, plc.Table]:
        """Return ``(seq_num, sorted frame, table with optional sequence column)``."""
        df = chunk_to_frame(
            # Make sure chunks are pre-sorted
            await evaluate_chunk(
                self._context,
                TableChunk.from_message(msg, br=self._context.br()),
                self._ir,
                ir_context=self._ir_context,
            ),
            self._ir,
        )
        if self._ir.stable:
            nrows = df.table.num_rows()
            start = (self._comm.rank * (1 << 48)) + self._local_row_offset
            seq_id_col = plc.filling.sequence(
                nrows,
                plc.Scalar.from_py(
                    start, plc.DataType(plc.TypeId.UINT64), stream=df.stream
                ),
                plc.Scalar.from_py(
                    1, plc.DataType(plc.TypeId.UINT64), stream=df.stream
                ),
                stream=df.stream,
            )
            self._local_row_offset += nrows
            tbl = plc.Table([*df.table.columns(), seq_id_col])
        else:
            tbl = df.table
        return msg.sequence_number, df, tbl


async def _prepare_sample_and_collect_candidates(
    context: Context,
    sampled_chunks: ChunkStore,
    preparer: _ChunkPreparer,
    by: list[str],
    num_partitions: int,
) -> tuple[ChunkStore, list[TableChunk]]:
    """
    Sort the sampled chunks and collect split candidates from them.

    The prepared chunks (bounded by the sample size) are kept for insertion
    once the boundaries are known.
    """
    prepared_sample = ChunkStore(context)
    local_candidates_list: list[TableChunk] = []
    for msg in sampled_chunks:
        seq_num, df, tbl = await preparer.prepare(msg)
        local_candidates_list.append(
            TableChunk.from_pylibcudf_table(
                _select_local_split_candidates(
                    df.select(by), by, num_partitions, seq_num
                ).table,
                df.stream,
                exclusive_view=True,
                br=context.br(),
            )
        )
        prepared_sample.insert(
            Message(
                seq_num,
                TableChunk.from_pylibcudf_table(
                    tbl, df.stream, exclusive_view=True, br=context.br()
                ),
            )
        )
        del df
    return prepared_sample, local_candidates_list


def _post_sort_ir(ir: Sort) -> Sort:
    """The Sort applied to each received partition (adds the stable-sort key)."""
    if not ir.stable:
        return ir
    assert ir.zlice is None
    seq_id_name = next(unique_names(ir.schema.keys()))
    return Sort(
        ir.schema | {seq_id_name: DataType(pl.UInt64())},
        (
            *ir.by,
            NamedExpr(seq_id_name, Col(DataType(pl.UInt64()), seq_id_name)),
        ),
        (*ir.order, plc.types.Order.ASCENDING),
        (*ir.null_order, plc.types.NullOrder.AFTER),
        ir.stable,
        None,
        ir.children[0],
    )


async def _stream_chunks_into_shuffle(
    context: Context,
    comm: Communicator,
    ir: Sort,
    ir_context: IRExecutionContext,
    prepared_sample: ChunkStore,
    preparer: _ChunkPreparer,
    ch_in: Channel[TableChunk],
    num_partitions: int,
    collective_ids: list[int],
    metadata_in: ChannelMetadata,
    sort_boundaries_df: DataFrame,
    by: list[str],
) -> ShuffleManager:
    """
    Range-split and insert the sample, then every remaining chunk.

    Chunks after the sample are prepared and inserted one at a time as they
    arrive.

    At most one input chunk (plus shuffler-owned packed buffers, which
    rapidsmpf spills — to disk when configured) is resident per rank.
    """
    column_order = list(ir.order)
    null_order = list(ir.null_order)
    by_indices = names_to_indices(tuple(by), ir.schema)

    skip_insert = metadata_in.duplicated and comm.rank != 0

    shuffle = ShuffleManager(
        context,
        comm,
        num_partitions,
        collective_ids.pop(),
        partition_assignment=PartitionAssignment.CONTIGUOUS,
    )

    async def insert_prepared(seq_num: int, chunk: TableChunk) -> None:
        # The chunk's data moves into shuffler-owned packed buffers,
        # nothing lasting is added.
        available_chunk, _ = await make_table_chunks_available_or_wait(
            context, chunk, reserve_extra=0, net_memory_delta=0
        )
        tbl = available_chunk.table_view()
        sort_cols_tbl = plc.Table([tbl.columns()[i] for i in by_indices])
        stream = get_joined_cuda_stream(
            ir_context.get_cuda_stream,
            upstreams=(available_chunk.stream, sort_boundaries_df.stream),
        )
        splits = find_sort_splits(
            sort_cols_tbl,
            sort_boundaries_df.table,
            seq_num,
            column_order,
            null_order,
            stream=stream,
            chunk_relative=True,
        )
        await inserter.insert_split(available_chunk, splits)

    async with shuffle.inserting() as inserter:
        # 1. The sampled prefix, already sorted and sequence-tagged.
        for msg in prepared_sample:
            if skip_insert:
                continue
            await insert_prepared(
                msg.sequence_number, TableChunk.from_message(msg, br=context.br())
            )
        # 2. Everything else, one chunk at a time, straight from the input.
        while (msg := await ch_in.recv(context)) is not None:
            if skip_insert:
                continue
            seq_num, df, tbl = await preparer.prepare(msg)
            chunk = TableChunk.from_pylibcudf_table(
                tbl, df.stream, exclusive_view=True, br=context.br()
            )
            del df
            await insert_prepared(seq_num, chunk)

    return shuffle


async def _global_sort(
    context: Context,
    comm: Communicator,
    ir: Sort,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    prepared_sample: ChunkStore,
    preparer: _ChunkPreparer,
    metadata_in: ChannelMetadata,
    by: list[str],
    num_partitions: int,
    sort_boundaries_df: DataFrame,
    collective_ids: list[int],
    executor: StreamingExecutor,
    *,
    tracer: ActorTracer | None,
) -> None:
    """Global sort: range-shuffle the input, then sort each owned partition."""
    output_metadata = ChannelMetadata(
        local_count=max(1, num_partitions // comm.nranks),
        partitioning=Partitioning(
            _build_order_scheme(context, _sort_to_order_keys(ir), sort_boundaries_df),
            "inherit",
        ),
    )
    await send_metadata(ch_out, context, output_metadata)

    shuffle = await _stream_chunks_into_shuffle(
        context,
        comm,
        ir,
        ir_context,
        prepared_sample,
        preparer,
        ch_in,
        num_partitions,
        collective_ids,
        metadata_in,
        sort_boundaries_df,
        by,
    )
    await _extract_partitions_and_send(
        context,
        comm,
        ch_out,
        shuffle,
        _post_sort_ir(ir),
        ir_context,
        ir.schema,
        executor,
        tracer=tracer,
    )


async def _extract_partitions_and_send(
    context: Context,
    comm: Communicator,
    ch_out: Channel[TableChunk],
    shuffle: ShuffleManager,
    post_sort_ir: Sort,
    ir_context: IRExecutionContext,
    output_schema: Schema,
    executor: StreamingExecutor,
    *,
    tracer: ActorTracer | None,
) -> None:
    """
    Sort each owned partition and send it.

    Partitions that fit the run budget are unpacked and sorted in memory as
    before; larger ones go through the external merge sort (sorted runs on
    disk, streamed k-way merge) so a partition is never resident whole.
    """
    order_keys = _sort_to_order_keys(post_sort_ir)
    batch_bytes = external_sort.batch_bytes_for(executor)
    run_root = external_sort.make_run_directory(context.options(), comm.rank)
    decisions: set[str] = set()
    try:
        for partition_id in shuffle.local_partitions():
            decisions.add(
                await external_sort.sort_partition(
                    context,
                    ir_context,
                    ch_out,
                    shuffle,
                    post_sort_ir,
                    order_keys,
                    output_schema,
                    partition_id,
                    batch_bytes=batch_bytes,
                    run_root=run_root,
                    tracer=tracer,
                )
            )
    finally:
        await ir_context.to_thread(shutil.rmtree, run_root, ignore_errors=True)
    if tracer is not None and "external" in decisions:
        tracer.decision = "external_sort"

    await ch_out.drain(context)


def _sort_by_column_names(ir: Sort) -> list[str]:
    """
    Resolve the underlying column names for a ``Sort`` node's keys.

    A sort key's ``NamedExpr.name`` reflects its output alias, which may differ from
    the referenced column after upstream renames. E.g., a join dedup may leave
    ``ORDER BY df2.text`` represented as ``Col('text:df2').alias('text')``. Resolve
    through the underlying ``Col`` so schema lookups use the actual column name.

    Raises
    ------
    NotImplementedError
        If any sort key is not a bare column reference.
    """
    by = [ne.value.name for ne in ir.by if isinstance(ne.value, Col)]
    if len(by) != len(ir.by):
        raise NotImplementedError("Sorting columns must be column names.")
    return by


def _sort_to_order_keys(ir: Sort) -> list[OrderKey]:
    """Convert Sort IR to list of OrderKeys."""
    return [
        OrderKey(index, order, null_order)
        for index, order, null_order in zip(
            names_to_indices(tuple(_sort_by_column_names(ir)), ir.schema),
            ir.order,
            ir.null_order,
            strict=False,
        )
    ]


def _can_sort_chunkwise(
    ordering: Ordering | None, order_keys: Sequence[OrderKey]
) -> bool:
    """Return true when ordering avoids a global sort."""
    if ordering is None:
        return False
    keys = tuple(ordering.keys)
    return keys == tuple(order_keys[: len(keys)]) and (
        len(keys) == len(order_keys) or ordering.strict_boundaries
    )


def _build_order_scheme(
    context: Context,
    order_keys: list[OrderKey],
    sort_boundaries_df: DataFrame,
) -> OrderScheme:
    """Build output OrderScheme metadata."""
    n_keys = len(order_keys)
    stream = sort_boundaries_df.stream
    # sort_boundaries_df will contain a tie-breaker column
    by_table = plc.Table(sort_boundaries_df.table.columns()[:n_keys])
    n_rows = by_table.num_rows()

    strict_boundaries = (
        n_rows == 0
        # TODO: Use unique_count_table
        # Requires https://github.com/NVIDIA/cudf/pull/22487
        or plc.stream_compaction.unique(
            by_table,
            list(range(n_keys)),
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            stream=stream,
        ).num_rows()
        == n_rows
    )

    boundaries_chunk = TableChunk.from_pylibcudf_table(
        by_table, stream, exclusive_view=False, br=context.br()
    )
    return OrderScheme(
        [Ordering(order_keys, boundaries_chunk, strict_boundaries=strict_boundaries)]
    )


@define_actor()
async def sort_actor(
    context: Context,
    comm: Communicator,
    ir: Sort,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    by: list[str],
    num_partitions: int,
    executor: StreamingExecutor,
    collective_ids: list[int],
) -> None:
    """Streaming sort actor."""
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        # TODO: Skip sort if OrderScheme metadata is present and compatible.
        metadata_in = await recv_metadata(ch_in, context)

        if ir.zlice is not None:
            assert _has_simple_zlice(ir.zlice), (
                f"This slice not supported in `sort_actor`: {ir.zlice}."
            )
            await _simple_top_or_bottom_k(
                context,
                comm,
                ch_in,
                ch_out,
                ir,
                ir_context,
                metadata_in,
                collective_ids,
                tracer,
            )
            return

        order_keys = _sort_to_order_keys(ir)
        partitioning = NormalizedPartitioning.from_keys(
            metadata_in.partitioning, comm.nranks, keys=order_keys
        )
        ordering = partitioning.get_ordering(
            level="local" if metadata_in.duplicated else "flat"
        )
        if _can_sort_chunkwise(ordering, order_keys):
            if tracer is not None:
                tracer.decision = "already_sorted"
            await chunkwise_evaluate(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                tracer=tracer,
            )
            return

        sampled_chunks, num_partitions = await _sample_chunks_for_boundaries(
            context, comm, ch_in, num_partitions, metadata_in, executor, collective_ids
        )

        # Range-partition boundaries come from the sampled prefix only (as in
        # Spark's RangePartitioner); the rest of the input is never buffered.
        preparer = _ChunkPreparer(context, comm, ir, ir_context)
        (
            prepared_sample,
            local_candidates_list,
        ) = await _prepare_sample_and_collect_candidates(
            context, sampled_chunks, preparer, by, num_partitions
        )

        need_allgather = comm.nranks > 1 and not metadata_in.duplicated
        sort_boundaries_df = await _compute_sort_boundaries(
            context,
            comm,
            ir_context,
            local_candidates_list,
            ir,
            by,
            num_partitions,
            collective_ids.pop() if need_allgather else None,
        )

        await _global_sort(
            context,
            comm,
            ir,
            ir_context,
            ch_out,
            ch_in,
            prepared_sample,
            preparer,
            metadata_in,
            by,
            num_partitions,
            sort_boundaries_df,
            collective_ids,
            executor,
            tracer=tracer,
        )


@generate_ir_sub_network.register(Sort)
def _sort_rapidsmpf_network(ir: Sort, rec: SubNetGenerator) -> tuple[dict, dict]:
    """Wire multi-partition ``Sort`` to ``sort_actor``; single-partition uses ``default_node_single``."""
    executor = rec.state["config_options"].executor
    partition_info = rec.state["partition_info"]
    dynamic = executor.dynamic_planning is not None
    ir_context = ir_context_for_node(rec, ir)

    if partition_info[ir].count == 1 and (
        not dynamic or isinstance(ir.children[0], Repartition)
    ):
        nodes, channels = process_children(ir, rec)
        channels[ir] = ChannelManager(rec.state["context"])
        nodes[ir] = [
            default_node_single(
                rec.state["context"],
                ir,
                ir_context,
                channels[ir].reserve_input_slot(),
                channels[ir.children[0]].reserve_output_slot(),
            )
        ]
        return nodes, channels

    (child,) = ir.children
    nodes, channels = rec(child)
    by = _sort_by_column_names(ir)

    collective_ids = list(rec.state["collective_id_map"][ir])
    expected_id_count = 3 if dynamic else 2
    assert len(collective_ids) == expected_id_count, (
        f"Sort must have {expected_id_count} collective IDs, got {len(collective_ids)}."
    )

    channels[ir] = ChannelManager(rec.state["context"])
    nodes[ir] = [
        sort_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context,
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            by=by,
            num_partitions=partition_info[ir].count,
            executor=executor,
            collective_ids=collective_ids,
        )
    ]
    return nodes, channels
