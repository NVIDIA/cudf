# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sort logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    Partitioning,
    RowOrdering,
    RowOrderingScope,
)
from cudf_streaming.table_chunk import (
    TableChunk,
)
from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import Sort
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.external_sort import (
    global_sort_external,
)
from cudf_polars.streaming.actor_graph.collectives.shuffle import ShuffleManager
from cudf_polars.streaming.actor_graph.collectives.sort_utils import (
    append_row_sequence_column,
    build_order_scheme,
    build_post_sort_ir,
    compute_sort_boundaries,
    insert_chunk_with_sort_splits,
    select_local_split_candidates,
    sort_by_column_names,
    sort_to_order_keys,
)
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.nodes import (
    default_node_single,
    shutdown_on_error,
)
from cudf_polars.streaming.actor_graph.partitioning import (
    NormalizedPartitioning,
    maybe_remap_partitioning,
    replace_row_orderings,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    ChunkStore,
    _sample_chunks,
    allgather_reduce,
    chunk_to_frame,
    chunkwise_evaluate,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    replay_buffered_channel,
    send_metadata,
)
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.sort import _has_simple_zlice

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_streaming.channel_metadata import OrderKey, Ordering
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import StreamingExecutor


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
        ChannelMetadata(
            local_count=1,
            partitioning=Partitioning(
                row_orderings=(
                    RowOrdering(sort_to_order_keys(ir), RowOrderingScope.GLOBAL),
                )
            ),
            duplicated=True,
        ),
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


async def _sample_chunks_for_size_estimate(
    context: Context,
    comm: Communicator,
    ch_in: Channel[TableChunk],
    num_partitions: int,
    metadata_in: ChannelMetadata,
    executor: StreamingExecutor,
    collective_ids: list[int],
) -> tuple[ChunkStore, int]:
    """
    Sample chunks and estimate total data size to derive num_partitions dynamically.

    The sampled chunks are returned in replay order. The caller is
    responsible for replaying them into a channel via replay_buffered_channel.
    """
    if executor.dynamic_planning is None:
        return ChunkStore(context), num_partitions

    size_estimate_id = collective_ids.pop()
    target_partition_size = executor.target_partition_size
    sample_chunk_count = executor.dynamic_planning.sample_chunk_count

    sample = await _sample_chunks(
        context,
        ch_in,
        sample_chunk_count,
        target_partition_size,
        metadata_in.local_count,
    )

    # Allgather to get global size estimate across all ranks
    if comm.nranks > 1 and not metadata_in.duplicated:
        (global_size,) = await allgather_reduce(
            context, comm, size_estimate_id, sample.total_size
        )
    else:
        global_size = sample.total_size

    num_partitions = max(1, math.ceil(global_size / target_partition_size))
    return sample.local_sample.chunks, num_partitions


async def _receive_and_buffer_chunks(
    context: Context,
    ch_in: Channel[TableChunk],
    chunk_store: ChunkStore,
    ir: Sort,
    by: list[str],
    num_partitions: int,
    comm: Communicator,
    ir_context: IRExecutionContext,
) -> list[TableChunk]:
    """Receive input chunks, collect local split candidates, and buffer chunks for later insert."""
    await recv_metadata(ch_in, context)
    local_candidates_list: list[TableChunk] = []
    local_row_offset = 0

    while (msg := await ch_in.recv(context)) is not None:
        seq_num = msg.sequence_number
        df = chunk_to_frame(
            # Make sure chunks are pre-sorted
            await evaluate_chunk(
                context,
                TableChunk.from_message(msg, br=context.br()),
                ir,
                ir_context=ir_context,
            ),
            ir,
        )
        local_candidates_list.append(
            TableChunk.from_pylibcudf_table(
                select_local_split_candidates(
                    df.select(by), by, num_partitions, seq_num
                ).table,
                df.stream,
                exclusive_view=True,
                br=context.br(),
            )
        )
        if ir.stable:
            tbl = append_row_sequence_column(df, comm.rank, local_row_offset)
            local_row_offset += tbl.num_rows()
        else:
            tbl = df.table
        chunk_store.insert(
            Message(
                seq_num,
                TableChunk.from_pylibcudf_table(
                    tbl, df.stream, exclusive_view=True, br=context.br()
                ),
            )
        )
        del df

    return local_candidates_list


async def _forward_from_chunk_store(
    context: Context, ch_out: Channel[TableChunk], chunk_store: ChunkStore
) -> None:
    """Forward buffered messages from a ChunkStore into a channel."""
    for msg in chunk_store:
        await ch_out.send(context, msg)
    await ch_out.drain(context)


async def _insert_chunks_into_shuffle(
    context: Context,
    comm: Communicator,
    ir: Sort,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    num_partitions: int,
    collective_ids: list[int],
    metadata_in: ChannelMetadata,
    sort_boundaries_df: DataFrame,
    by: list[str],
) -> tuple[ShuffleManager, Sort]:
    """Create shuffle manager and insert each buffered chunk with sort-based splits."""
    by_indices = names_to_indices(tuple(by), ir.schema)

    skip_insert = metadata_in.duplicated and comm.rank != 0

    shuffle = ShuffleManager(
        context,
        comm,
        num_partitions,
        collective_ids.pop(),
        partition_assignment=PartitionAssignment.CONTIGUOUS,
    )
    async with shuffle.inserting() as inserter:
        while (msg := await ch_in.recv(context)) is not None:
            if skip_insert:
                continue
            # TODO: Pre-sort chunks if they do not originate from the ChunkStore.
            # (Not possible until we use _global_sort outside of sort_actor.)
            await insert_chunk_with_sort_splits(
                context,
                inserter,
                TableChunk.from_message(msg, br=context.br()),
                msg.sequence_number,
                ir,
                ir_context,
                sort_boundaries_df,
                by_indices,
            )

    return shuffle, build_post_sort_ir(ir)


async def _extract_partitions_and_send(
    context: Context,
    ch_out: Channel[TableChunk],
    shuffle: ShuffleManager,
    post_sort_ir: Sort,
    ir_context: IRExecutionContext,
    output_schema: Schema,
    *,
    tracer: ActorTracer | None,
) -> None:
    """Extract each local partition from the shuffle, sort if needed, and send."""
    ncols_out = len(output_schema)
    for partition_id in shuffle.local_partitions():
        stream = ir_context.get_cuda_stream()
        table = await shuffle.extract_chunk(partition_id, stream)
        if table.num_rows() > 0:
            table = post_sort_ir.do_evaluate(
                *post_sort_ir._non_child_args,
                DataFrame.from_table(
                    table,
                    list(post_sort_ir.schema.keys()),
                    list(post_sort_ir.schema.values()),
                    stream,
                ),
                context=ir_context,
            ).table
            if table.num_columns() > ncols_out:
                table = plc.Table(table.columns()[:ncols_out])
            chunk = TableChunk.from_pylibcudf_table(
                table, stream, exclusive_view=True, br=context.br()
            )
            await send_chunk(context, ch_out, chunk, partition_id, tracer=tracer)

    await ch_out.drain(context)


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


def _sort_partitioning(
    ir: Sort,
    child_metadatas: Sequence[ChannelMetadata],
    context: Context,
) -> Partitioning:
    """Preserve partition layout and replace row-order metadata after a sort."""
    partitioning = (
        maybe_remap_partitioning(
            ir,
            child_metadatas[0].partitioning,
            context=context,
            preserves_row_order=False,
        )
        or Partitioning()
    )
    if not all(isinstance(ne.value, Col) for ne in ir.by):
        return partitioning
    return replace_row_orderings(
        partitioning,
        (RowOrdering(sort_to_order_keys(ir), RowOrderingScope.GLOBAL),),
    )


async def _global_sort(
    context: Context,
    comm: Communicator,
    ir: Sort,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    by: list[str],
    num_partitions: int,
    sort_boundaries_df: DataFrame,
    collective_ids: list[int],
    *,
    tracer: ActorTracer | None,
) -> None:
    """Global sort."""
    order_keys = sort_to_order_keys(ir)
    output_metadata = ChannelMetadata(
        local_count=max(1, num_partitions // comm.nranks),
        partitioning=Partitioning(
            build_order_scheme(context, order_keys, sort_boundaries_df),
            "inherit",
            row_orderings=(RowOrdering(order_keys, RowOrderingScope.GLOBAL),),
        ),
    )
    await send_metadata(ch_out, context, output_metadata)

    shuffle, post_sort_ir = await _insert_chunks_into_shuffle(
        context,
        comm,
        ir,
        ir_context,
        ch_in,
        num_partitions,
        collective_ids,
        metadata_in,
        sort_boundaries_df,
        by,
    )
    await _extract_partitions_and_send(
        context,
        ch_out,
        shuffle,
        post_sort_ir,
        ir_context,
        ir.schema,
        tracer=tracer,
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
    ch_sample_replay = context.create_channel()
    ch_chunk_store = context.create_channel()
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        chs_aux=(ch_sample_replay, ch_chunk_store),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
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

        order_keys = sort_to_order_keys(ir)
        partitioning = NormalizedPartitioning.from_keys(
            metadata_in.partitioning, comm.nranks, keys=order_keys
        )
        ordering = partitioning.get_ordering(
            level="local" if metadata_in.duplicated else "flat"
        )
        if _can_sort_chunkwise(ordering, order_keys):
            if tracer is not None:
                tracer.decision = "already_sorted"
            metadata_out = ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=replace_row_orderings(
                    metadata_in.partitioning or Partitioning(),
                    (RowOrdering(order_keys, RowOrderingScope.GLOBAL),),
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
                tracer=tracer,
            )
            return

        if executor.sort_strategy == "external":
            await global_sort_external(
                context,
                comm,
                ir,
                ir_context,
                ch_in,
                ch_out,
                metadata_in,
                by,
                num_partitions,
                executor,
                collective_ids,
                tracer=tracer,
            )
            return

        sampled_chunks, num_partitions = await _sample_chunks_for_size_estimate(
            context, comm, ch_in, num_partitions, metadata_in, executor, collective_ids
        )

        chunk_store = ChunkStore(context)
        _, local_candidates_list = await gather_in_task_group(
            replay_buffered_channel(
                context,
                ch_sample_replay,
                ch_in,
                sampled_chunks,
                metadata_in,
                trace_ir=ir,
            ),
            _receive_and_buffer_chunks(
                context,
                ch_sample_replay,
                chunk_store,
                ir,
                by,
                num_partitions,
                comm,
                ir_context,
            ),
        )

        need_allgather = comm.nranks > 1 and not metadata_in.duplicated
        sort_boundaries_df = await compute_sort_boundaries(
            context,
            comm,
            ir_context,
            local_candidates_list,
            ir,
            by,
            num_partitions,
            collective_ids.pop() if need_allgather else None,
        )

        await gather_in_task_group(
            _forward_from_chunk_store(context, ch_chunk_store, chunk_store),
            _global_sort(
                context,
                comm,
                ir,
                ir_context,
                ch_out,
                ch_chunk_store,
                metadata_in,
                by,
                num_partitions,
                sort_boundaries_df,
                collective_ids,
                tracer=tracer,
            ),
        )


@generate_ir_sub_network.register(Sort)
def _sort_rapidsmpf_network(ir: Sort, rec: SubNetGenerator) -> tuple[dict, dict]:
    """Use ``sort_actor`` for multi-partition sorts and the default actor otherwise."""
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
                partitioning_callback=partial(_sort_partitioning, ir),
            )
        ]
        return nodes, channels

    (child,) = ir.children
    nodes, channels = rec(child)
    by = sort_by_column_names(ir)

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
