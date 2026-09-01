# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Window ``over()`` actor for the RapidsMPF streaming runtime.

Implements the ``group_to_rows`` ``WindowMapping`` only: each input row
receives the value computed for its group. Other mappings (``explode``,
``join``) are not supported.

The actor picks one of three strategies at runtime based on the incoming
channel metadata and the shape of the windowed expressions.

Chunkwise (already partitioned)
    If the channel is already hash-partitioned on the over-keys (or any
    prefix of them), every group is fully contained within one rank's
    chunks. The window expression is correct on each chunk in isolation
    and no cross-rank coordination is needed.

Scalar broadcast (decomposable aggregations)
    When every aggregation is decomposable, partial aggregates can be
    combined associatively across ranks. Each rank computes per-chunk
    partials, an AllGather collects them, a single reduction yields the
    global aggregate per group, and each input chunk has those results
    joined back onto its rows by the partition keys. Order is preserved
    naturally: input chunks are buffered in receive order and emitted in
    the same order after the global aggregate is known.

Forward + return shuffle (non-decomposable aggregations)
    For functions that need every row in a group visible at once, a hash
    shuffle on the partition keys co-locates each group on one rank for
    evaluation. After evaluation, a second shuffle routes each row back
    to the rank that originally received it (output channels are
    rank-local, so only the originating rank can emit), and the rows are
    reassembled in input order using stamps that travel with the data
    through both shuffles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl.expr import GroupedWindow
from cudf_polars.dsl.expressions.base import ExecutionContext
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.streaming.actor_graph.collectives.shuffle import (
    LocalRepartitioner,
    ShuffleManager,
)
from cudf_polars.streaming.actor_graph.dispatch import generate_ir_sub_network
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    ChunkStore,
    NormalizedPartitioning,
    _evaluate_chunk_sync,
    _sample_chunks,
    allgather_and_reduce,
    allgather_reduce,
    chunk_to_frame,
    chunkwise_evaluate,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    gather_in_task_group,
    maybe_remap_partitioning,
    process_children,
    recv_metadata,
    replay_buffered_channel,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.over import Over, _build_over_groupby_irs
from cudf_polars.streaming.utils import _contains_input_order_window_without_order_by
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.expr import Col
    from cudf_polars.dsl.ir import IR, GroupBy, IRExecutionContext, Select
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.utils import TableSizeStats


@dataclass(frozen=True)
class _ScalarOverPlan:
    """Pre-computed IR rewrites for the scalar Over path."""

    key_names: tuple[str, ...]
    piecewise_ir: GroupBy
    reduction_ir: GroupBy
    agg_select_ir: Select


def _build_scalar_over_plan(ir: Over) -> _ScalarOverPlan:
    """Pre-compute the IR rewrites needed by the scalar Over path."""
    gw_nodes = tuple(ne.value for ne in ir.exprs if isinstance(ne.value, GroupedWindow))
    # Lowering rejects non-Col partition-by keys, so every by-child here is a Col.
    by_children = gw_nodes[0].children[: gw_nodes[0].by_count]
    key_names = tuple(cast("Col", c).name for c in by_children)
    piecewise_ir, reduction_ir, agg_select_ir = _build_over_groupby_irs(
        gw_nodes, ir.children[0]
    )
    return _ScalarOverPlan(
        key_names=key_names,
        piecewise_ir=piecewise_ir,
        reduction_ir=reduction_ir,
        agg_select_ir=agg_select_ir,
    )


def _broadcast_gw_sync(
    gw: GroupedWindow,
    chunk_df: DataFrame,
    global_agg_df: DataFrame,
    key_names: tuple[str, ...],
    stream: Stream,
) -> Any:
    """Broadcast the global aggregate for one GroupedWindow back to row positions."""
    by_exprs = gw.children[: gw.by_count]
    by_cols = broadcast(
        *(b.evaluate(chunk_df) for b in by_exprs),
        target_length=chunk_df.num_rows,
        stream=stream,
    )
    by_tbl = plc.Table([c.obj for c in by_cols])
    group_keys_tbl = global_agg_df.select(key_names).table

    out_names, out_dtypes = zip(
        *((ne.name, ne.value.dtype) for ne in gw.named_aggs), strict=True
    )
    value_tbl = global_agg_df.select(out_names).table

    broadcasted_cols = gw._broadcast_agg_results(
        by_tbl, group_keys_tbl, value_tbl, out_names, out_dtypes, stream
    )
    temp_df = DataFrame(broadcasted_cols, stream=stream)
    return gw.post.value.evaluate(temp_df, context=ExecutionContext.FRAME)


def _evaluate_ir_broadcast_sync(
    chunk: TableChunk,
    ir: Over,
    global_agg_df: DataFrame,
    key_names: tuple[str, ...],
    br: BufferResource,
) -> TableChunk:
    """Map the per-group aggregate onto a chunk's rows to produce its Over output."""
    chunk_df = chunk_to_frame(chunk, ir.children[0])
    # global_agg_df and chunk_df may live on different streams. Since we do
    # an evaluation of values in chunk_df via Expr.evaluate, run the
    # broadcast on chunk_dfs stream, making sure the global_agg stream
    # waits.
    with stream_ordered_after(
        lambda: chunk_df.stream, upstreams=[global_agg_df.stream]
    ) as stream:
        result_cols = [
            _broadcast_gw_sync(
                ne.value, chunk_df, global_agg_df, key_names, stream
            ).rename(ne.name)
            if isinstance(ne.value, GroupedWindow)
            else ne.evaluate(chunk_df, context=ExecutionContext.FRAME)
            for ne in ir.exprs
        ]

        return TableChunk.from_pylibcudf_table(
            plc.Table([c.obj for c in result_cols]),
            stream,
            exclusive_view=True,
            br=br,
        )


async def _evaluate_broadcast_chunk(
    context: Context,
    chunk: TableChunk,
    ir: Over,
    global_agg_df: DataFrame,
    key_names: tuple[str, ...],
    ir_context: IRExecutionContext,
    global_agg_per_row_size: int,
) -> TableChunk:
    """Unspill the chunk and map the per-group aggregate onto its rows."""
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=global_agg_per_row_size * chunk.shape[0],
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        return await ir_context.to_thread(
            _evaluate_ir_broadcast_sync,
            chunk,
            ir,
            global_agg_df,
            key_names,
            context.br(),
        )


@dataclass(frozen=True)
class OriginStamps:
    """
    Stamp column names that ride both shuffles for output reassembly.

    Parameters
    ----------
    chunk_index
        Column name for a dense rank-local 0..N-1 counter identifying which
        input chunk a row came from.
    position
        Column name for the row's position within its input chunk.
    rank
        Column name for the originating rank.
    """

    chunk_index: str
    position: str
    rank: str

    dtype: ClassVar[DataType] = DataType(pl.Int32())

    @property
    def names(self) -> tuple[str, str, str]:
        """Stamp column names, in the order they are appended to the table."""
        return (self.chunk_index, self.position, self.rank)


def _origin_stamps_for(ir: Over) -> OriginStamps:
    """Pick three stamp column names that do not collide with the schema."""
    names = unique_names((*ir.children[0].schema.keys(), *ir.schema.keys()))
    return OriginStamps(next(names), next(names), next(names))


# _append_origin_stamps appends three int32 columns.
_ORIGIN_STAMP_BYTES_PER_ROW = 3 * 4


def _append_origin_stamps(
    chunk: TableChunk,
    chunk_index: int,
    origin_rank: int,
    stream: Stream,
    br: Any,
) -> TableChunk:
    """Append (chunk_index, position, rank) stamp columns to *chunk*."""
    table = chunk.table_view()
    n_rows = table.num_rows()
    int32 = plc.types.DataType(plc.TypeId.INT32)
    chunk_index_col = plc.Column.from_scalar(
        plc.Scalar.from_py(chunk_index, int32, stream=stream), n_rows, stream=stream
    )
    rank_col = plc.Column.from_scalar(
        plc.Scalar.from_py(origin_rank, int32, stream=stream), n_rows, stream=stream
    )
    position_col = plc.filling.sequence(
        n_rows,
        plc.Scalar.from_py(0, int32, stream=stream),
        plc.Scalar.from_py(1, int32, stream=stream),
        stream=stream,
    )
    return TableChunk.from_pylibcudf_table(
        plc.Table([*table.columns(), chunk_index_col, position_col, rank_col]),
        stream,
        exclusive_view=False,
        br=br,
    )


def _evaluate_window_with_stamps(
    chunk: TableChunk,
    ir: Over,
    ir_context: IRExecutionContext,
    stamps: OriginStamps,
    *,
    sort_by_input_order: bool,
) -> DataFrame:
    """Evaluate *ir* on the un-stamped portion of *chunk*; reattach stamps after."""
    child_schema = ir.children[0].schema
    stream = ir_context.get_cuda_stream()
    n_child = len(child_schema)
    table = chunk.table_view()
    if sort_by_input_order:
        columns = table.columns()
        table = plc.sorting.stable_sort_by_key(
            table,
            plc.Table(
                [
                    columns[n_child + 2],  # origin rank
                    columns[n_child],  # origin chunk index (local)
                    columns[n_child + 1],  # origin row index (in chunk)
                ]
            ),
            [plc.types.Order.ASCENDING] * 3,
            [plc.types.NullOrder.AFTER] * 3,
            stream=stream,
        )
    columns = table.columns()

    input_df = DataFrame.from_table(
        plc.Table(columns[:n_child]),
        list(child_schema.keys()),
        list(child_schema.values()),
        stream,
    )
    result = ir.do_evaluate(ir.exprs, input_df, context=ir_context)
    stamp_cols = [
        Column(col, dtype=stamps.dtype, name=name)
        for col, name in zip(columns[n_child:], stamps.names, strict=True)
    ]
    return result.with_columns(stamp_cols, stream=stream)


def _split_off_origin_rank(
    result: DataFrame,
    br: Any,
) -> tuple[TableChunk, TableChunk] | None:
    """
    Split the origin-rank stamp off the evaluated result.

    Returns the payload and the single-column rank map that routes each row
    back to its origin, or ``None`` when there is nothing to route. The
    rearrangement itself is left to ``insert_index``, which reserves for it.
    """
    if result.table.num_rows() == 0:
        return None

    stream = result.stream
    columns = result.table.columns()
    return (
        TableChunk.from_pylibcudf_table(
            plc.Table(columns[:-1]), stream, exclusive_view=True, br=br
        ),
        TableChunk.from_pylibcudf_table(
            plc.Table([columns[-1]]), stream, exclusive_view=True, br=br
        ),
    )


async def _allgather_and_broadcast(
    context: Context,
    comm: Communicator,
    ir: Over,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    tracer: Any,
    collective_id: int,
    plan: _ScalarOverPlan,
) -> None:
    """Compute partial aggregates per chunk, AllGather globally, then broadcast to each chunk."""
    piecewise_ir = plan.piecewise_ir
    reduction_ir = plan.reduction_ir
    agg_select_ir = plan.agg_select_ir

    buffer = ChunkStore(context)
    partial_aggs: list[TableChunk] = []

    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg, br=context.br())
        chunk, extra = await make_table_chunks_available_or_wait(
            context,
            chunk,
            reserve_extra=chunk.data_alloc_size(),
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            partial = await ir_context.to_thread(
                _evaluate_chunk_sync,
                chunk,
                piecewise_ir,
                ir_context,
                context.br(),
            )
        partial_aggs.append(partial)
        buffer.insert(Message(msg.sequence_number, chunk))

    if partial_aggs:
        local_agg = await evaluate_batch(
            partial_aggs, context, reduction_ir, ir_context=ir_context
        )
    else:
        local_agg = empty_table_chunk(
            reduction_ir, context, ir_context.get_cuda_stream()
        )

    # AllGather the locally-reduced partials (pre post-aggregation) so a
    # single global reduction combines them; the post-aggregation step
    # runs once after.
    if comm.nranks > 1 and not metadata_in.duplicated:
        global_agg = await allgather_and_reduce(
            context, comm, collective_id, local_agg, reduction_ir, ir_context
        )
    else:
        global_agg = local_agg

    global_agg = await evaluate_chunk(
        context, global_agg, agg_select_ir, ir_context=ir_context
    )
    global_agg_df = chunk_to_frame(global_agg, agg_select_ir)
    global_agg_per_row_size = global_agg.data_alloc_size() // max(
        1, global_agg_df.num_rows
    )

    metadata_out = ChannelMetadata(
        local_count=metadata_in.local_count,
        partitioning=maybe_remap_partitioning(
            ir, metadata_in.partitioning, context=context
        ),
        duplicated=metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    for msg in buffer:
        result = await _evaluate_broadcast_chunk(
            context,
            TableChunk.from_message(msg, br=context.br()),
            ir,
            global_agg_df,
            plan.key_names,
            ir_context,
            global_agg_per_row_size,
        )
        await send_chunk(context, ch_out, result, msg.sequence_number, tracer=tracer)

    await ch_out.drain(context)


async def _choose_modulus(
    context: Context,
    comm: Communicator,
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    collective_id: int,
    target_partition_size: int,
    sample_chunk_count: int,
) -> tuple[TableSizeStats, int]:
    """
    Sample input, AllGather size estimates, and derive the forward-shuffle modulus.

    Returns the sample (whose chunks must be replayed back to the consumer)
    and the chosen number of forward-shuffle partitions.
    """
    sample = await _sample_chunks(
        context,
        ch_in,
        sample_chunk_count,
        target_partition_size,
        metadata_in.local_count,
    )
    if comm.nranks > 1 and not metadata_in.duplicated:
        total_bytes, total_count = await allgather_reduce(
            context, comm, collective_id, sample.total_size, sample.total_chunks
        )
    else:
        total_bytes, total_count = sample.total_size, sample.total_chunks
    modulus = min(
        max(comm.nranks, total_bytes // max(1, target_partition_size)),
        max(1, total_count),
    )
    return sample, modulus


async def _distribute_by_group(
    context: Context,
    comm: Communicator,
    forward_shuffle: ShuffleManager,
    ch_in: Channel[TableChunk],
    key_indices: tuple[int, ...],
    ir_context: IRExecutionContext,
    skip_insert: bool,  # noqa: FBT001
) -> list[int]:
    """Stream chunks from *ch_in* into the forward shuffle with origin stamps."""
    # We already have the upstream metadata; signal we don't need the replay
    # channel's copy.
    await ch_in.shutdown_metadata(context)

    sequence_numbers: list[int] = []
    chunk_index = 0
    async with forward_shuffle.inserting() as inserter:
        while (msg := await ch_in.recv(context)) is not None:
            chunk = TableChunk.from_message(msg, br=context.br())
            stamp_bytes = (
                0 if skip_insert else _ORIGIN_STAMP_BYTES_PER_ROW * chunk.shape[0]
            )
            # The chunk's data moves into shuffler-owned packed buffers, so the
            # stamp columns _append_origin_stamps allocates below are both the
            # extra we need and the only lasting addition.
            chunk, extra = await make_table_chunks_available_or_wait(
                context,
                chunk,
                reserve_extra=stamp_bytes,
                net_memory_delta=stamp_bytes,
            )
            sequence_numbers.append(msg.sequence_number)
            if not skip_insert:
                # TODO: For duplicated input only rank 0 inserts here, and
                # every row is stamped with origin_rank=0, so the return
                # shuffle routes all output back to rank 0 and ranks
                # 1..nranks-1 sit idle on emit. Slice the duplicated input
                # across ranks (e.g. stripe by row index) and stamp each
                # slice with its target origin rank to distribute emit work.
                with opaque_memory_usage(extra):
                    stamped = await ir_context.to_thread(
                        _append_origin_stamps,
                        chunk,
                        chunk_index,
                        comm.rank,
                        ir_context.get_cuda_stream(),
                        context.br(),
                    )
                await inserter.insert_hash(stamped, key_indices)
            chunk_index += 1
    return sequence_numbers


async def _evaluate_and_route_to_origin(
    context: Context,
    ir: Over,
    ir_context: IRExecutionContext,
    forward_shuffle: ShuffleManager,
    return_shuffle: ShuffleManager,
    stamps: OriginStamps,
    *,
    sort_by_input_order: bool,
) -> None:
    """Window-evaluate each local forward partition, then ship rows back to their origin."""
    async with return_shuffle.inserting() as inserter:
        for partition_id in forward_shuffle.local_partitions():
            stream = ir_context.get_cuda_stream()
            extracted = await forward_shuffle.extract_chunk(partition_id, stream)
            if extracted.num_rows() == 0:
                continue
            partition = TableChunk.from_pylibcudf_table(
                extracted, stream, exclusive_view=True, br=context.br()
            )
            evaluated = await ir_context.to_thread(
                _evaluate_window_with_stamps,
                partition,
                ir,
                ir_context,
                stamps,
                sort_by_input_order=sort_by_input_order,
            )
            routed = await ir_context.to_thread(
                _split_off_origin_rank, evaluated, context.br()
            )
            if routed is not None:
                await inserter.insert_index(*routed)


async def _reassemble_input_chunks(
    context: Context,
    ch_out: Channel[TableChunk],
    ir_context: IRExecutionContext,
    return_shuffle: ShuffleManager,
    sequence_numbers: list[int],
    ir: Over,
    tracer: Any,
) -> None:
    """Emit one output chunk per input chunk, in original order."""
    n_chunks = len(sequence_numbers)
    if n_chunks == 0:
        return

    n_exprs = len(ir.exprs)
    chunk_index_column = n_exprs

    # TODO: thread ir_context through repartition_by_index so each
    # PackedData piece moves on its own pool stream rather than sharing one.
    local = LocalRepartitioner(return_shuffle, local_count=n_chunks)
    await local.repartition_by_index(
        partition_col=chunk_index_column, stream=ir_context.get_cuda_stream()
    )

    for chunk_index, sequence_number in zip(
        local.local_partitions(), sequence_numbers, strict=True
    ):
        # Distinct stream per chunk so downstream work on different
        # chunks can overlap on the GPU.
        stream = ir_context.get_cuda_stream()
        tbl = await local.extract_chunk(chunk_index, stream)
        if tbl.num_rows() == 0:
            chunk = empty_table_chunk(ir, context, stream)
        else:
            sorted_tbl = plc.sorting.stable_sort_by_key(
                tbl,
                plc.Table([tbl.columns()[n_exprs]]),
                [plc.types.Order.ASCENDING],
                [plc.types.NullOrder.AFTER],
                stream=stream,
            )
            chunk = TableChunk.from_pylibcudf_table(
                plc.Table(sorted_tbl.columns()[:n_exprs]),
                stream,
                exclusive_view=True,
                br=context.br(),
            )
        await send_chunk(context, ch_out, chunk, sequence_number, tracer=tracer)


async def _shuffle_and_reassemble(
    context: Context,
    comm: Communicator,
    ir: Over,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    tracer: Any,
    size_collective_id: int,
    forward_shuffle_collective_id: int,
    return_shuffle_collective_id: int,
    target_partition_size: int,
    sample_chunk_count: int,
) -> None:
    """Hash-shuffle by partition keys, evaluate, then route rows back to their origin rank."""
    stamps = _origin_stamps_for(ir)

    metadata_out = ChannelMetadata(
        local_count=metadata_in.local_count,
        partitioning=maybe_remap_partitioning(
            ir, metadata_in.partitioning, context=context
        ),
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    skip_insert = metadata_in.duplicated and comm.rank != 0

    sample, forward_modulus = await _choose_modulus(
        context,
        comm,
        ch_in,
        metadata_in,
        size_collective_id,
        target_partition_size,
        sample_chunk_count,
    )

    forward_shuffle = ShuffleManager(
        context, comm, forward_modulus, forward_shuffle_collective_id
    )
    return_shuffle = ShuffleManager(
        context,
        comm,
        comm.nranks,
        return_shuffle_collective_id,
        partition_assignment=PartitionAssignment.CONTIGUOUS,
    )

    ch_replay = context.create_channel()
    sort_by_input_order = _contains_input_order_window_without_order_by(
        [ne.value for ne in ir.exprs]
    )
    sequence_numbers, _ = await gather_in_task_group(
        _distribute_by_group(
            context,
            comm,
            forward_shuffle,
            ch_replay,
            ir.key_indices,
            ir_context,
            skip_insert,
        ),
        replay_buffered_channel(
            context, ch_replay, ch_in, sample.chunks, metadata_in, trace_ir=ir
        ),
    )

    await _evaluate_and_route_to_origin(
        context,
        ir,
        ir_context,
        forward_shuffle,
        return_shuffle,
        stamps,
        sort_by_input_order=sort_by_input_order,
    )
    await _reassemble_input_chunks(
        context, ch_out, ir_context, return_shuffle, sequence_numbers, ir, tracer
    )

    await ch_out.drain(context)


@define_actor()
async def over_actor(
    context: Context,
    comm: Communicator,
    ir: Over,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    collective_ids: list[int],
    target_partition_size: int,
    sample_chunk_count: int,
    scalar_plan: _ScalarOverPlan | None,
) -> None:
    """
    Streaming actor for window ``over()`` expressions.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Over IR node.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    collective_ids
        Collective IDs reserved for this operation. Scalar Over nodes receive
        one ID (AllGather); non-scalar nodes receive two (one shared by the
        size AllGather and forward Shuffle, plus a separate one for the
        return Shuffle which overlaps with the forward extract).
    target_partition_size
        Target output partition size in bytes, used to compute the shuffle
        modulus for the non-scalar path.
    sample_chunk_count
        Maximum number of input chunks to sample when estimating the shuffle
        modulus on the non-scalar path.
    scalar_plan
        Pre-computed IR rewrites for the scalar Over path, built at planning
        time. ``None`` for non-scalar Over nodes.
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        metadata_in = await recv_metadata(ch_in, context)

        partitioning = NormalizedPartitioning.from_keys(
            metadata_in.partitioning,
            comm.nranks,
            keys=ir.key_indices,
            allow_subset=True,
        )
        if partitioning.is_strictly_partitioned(
            level="local" if metadata_in.duplicated else "flat",
        ):
            metadata_out = ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=maybe_remap_partitioning(
                    ir, metadata_in.partitioning, context=context
                ),
                duplicated=metadata_in.duplicated,
            )
            await chunkwise_evaluate(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_out,
                input_metadata=metadata_in,
                tracer=tracer,
            )
            return

        if ir.is_scalar:
            assert scalar_plan is not None
            await _allgather_and_broadcast(
                context,
                comm,
                ir,
                ir_context,
                ch_in,
                ch_out,
                metadata_in,
                tracer,
                collective_ids[0],
                scalar_plan,
            )
        else:
            await _shuffle_and_reassemble(
                context,
                comm,
                ir,
                ir_context,
                ch_in,
                ch_out,
                metadata_in,
                tracer,
                # collective_ids[0] is reused for the size AllGather and the
                # forward shuffle (sequential, no overlap); collective_ids[1]
                # is the return shuffle, which overlaps with forward extract.
                size_collective_id=collective_ids[0],
                forward_shuffle_collective_id=collective_ids[0],
                return_shuffle_collective_id=collective_ids[1],
                target_partition_size=target_partition_size,
                sample_chunk_count=sample_chunk_count,
            )


@generate_ir_sub_network.register(Over)
def _(
    ir: Over, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    executor = rec.state["config_options"].executor
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    if not ir.is_scalar and executor.dynamic_planning is None:
        raise ValueError(
            "Non-scalar over() requires dynamic planning to size the forward "
            "shuffle. Enable it via StreamingExecutor(dynamic_planning=...) "
            "or the --dynamic-planning CLI flag."
        )
    sample_chunk_count = (
        executor.dynamic_planning.sample_chunk_count
        if executor.dynamic_planning is not None
        else 0
    )
    scalar_plan = _build_scalar_over_plan(ir) if ir.is_scalar else None
    actors[ir] = [
        over_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_ids,
            executor.target_partition_size,
            sample_chunk_count,
            scalar_plan,
        )
    ]
    return actors, channels
