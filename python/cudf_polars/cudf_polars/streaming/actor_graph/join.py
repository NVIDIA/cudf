# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import pylibcudf as plc
from cudf_streaming.bloom_filter import BloomFilter
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    OrderKey,
    OrderScheme,
    Partitioning,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from pylibcudf.hashing import LIBCUDF_DEFAULT_HASH_SEED
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.memory_reserve_or_wait import (
    reserve_memory,
)
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import IR, Join, Sort
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import (
    AllGatherManager,
)
from cudf_polars.streaming.actor_graph.collectives.ordering import (
    _partition_range,
    adjust_ordering,
)
from cudf_polars.streaming.actor_graph.collectives.shuffle import (
    _global_shuffle,
    _key_column_indices,
)
from cudf_polars.streaming.actor_graph.collectives.sort import (
    _global_sort,
)
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.streaming.actor_graph.nodes import default_node_multi
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    CUDF_ROW_LIMIT,
    MAX_ROWS_PER_PARTITION,
    ChannelManager,
    ChunkStore,
    NormalizedPartitioning,
    TableSizeStats,
    _sample_chunks,
    allgather_reduce,
    chunk_to_frame,
    empty_table_chunk,
    gather_in_task_group,
    maybe_remap_partitioning,
    process_children,
    recv_metadata,
    replay_buffered_channel,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.utils import _concat

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterable, MutableMapping

    from cudf_streaming.bloom_filter import BloomFilterChunk
    from cudf_streaming.channel_metadata import Ordering
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.streaming.base import PartitionInfo
    from cudf_polars.utils.config import StreamingExecutor


@dataclass(frozen=True)
class JoinStrategy:
    """Summary of sampling and strategy selection for a dynamic join."""

    left_meta: ChannelMetadata | None = None
    """Metadata from left channel"""
    right_meta: ChannelMetadata | None = None
    """Metadata from right channel"""
    broadcast_side: Literal["left", "right"] | None = None
    """The side to broadcast. If None, the strategy is a shuffle join."""
    shuffle_modulus: int = 0
    """The shuffle modulus. Only used for shuffle joins."""
    output_indices: tuple[int, ...] = ()
    """The shuffle indices for the output. Only used for shuffle joins."""
    left_indices: tuple[int, ...] = ()
    """The shuffle indices for the left side. Only used for shuffle joins."""
    right_indices: tuple[int, ...] = ()
    """The shuffle indices for the right side. Only used for shuffle joins."""
    left_keys: tuple[NamedExpr, ...] = ()
    """The key expressions for the left side. Only used for shuffle joins."""
    right_keys: tuple[NamedExpr, ...] = ()
    """The key expressions for the right side. Only used for shuffle joins."""
    ordered: bool = False
    """Whether this strategy uses ordered boundary alignment."""
    left_input_ordering: Ordering | None = None
    """Input ordering for the left side. Only used for ordered joins."""
    right_input_ordering: Ordering | None = None
    """Input ordering for the right side. Only used for ordered joins."""
    left_output_ordering: Ordering | None = None
    """Aligned output ordering for the left side. Only used for ordered joins."""
    right_output_ordering: Ordering | None = None
    """Aligned output ordering for the right side. Only used for ordered joins."""
    output_ordering: Ordering | None = None
    """Join-output ordering metadata. Only used for ordered joins."""
    sort_side: Literal["left", "right"] | None = None
    """The side to sort before ordered join, if any."""


@dataclass(frozen=True)
class JoinPrefilterDecision:
    """Decision for an optional join-key prefilter stage."""

    left_rows: int
    right_rows: int
    threshold: float
    filter_side: Literal["left", "right"] | None = None
    build_indices: tuple[int, ...] = ()
    apply_indices: tuple[int, ...] = ()
    key_column_count: int = 0
    small_large_ratio: float | None = None
    reason_skipped: str | None = None

    @property
    def enabled(self) -> bool:
        """Whether this decision applies a prefilter."""
        return self.reason_skipped is None and self.filter_side is not None

    def trace_dict(self) -> dict[str, Any]:
        """Return structured trace metadata for this decision."""
        return asdict(self)


@define_actor()
async def broadcast_join_actor(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    broadcast_side: Literal["left", "right"],
    collective_id: int,
    target_partition_size: int,
) -> None:
    """
    Broadcast-join actor for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Join IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    ch_left
        The left input Channel[TableChunk].
    ch_right
        The right input Channel[TableChunk].
    broadcast_side
        The side to broadcast.
    collective_id
        Pre-allocated collective ID for this operation.
    target_partition_size
        The target partition size in bytes.
    """
    async with shutdown_on_error(
        context,
        ch_out,
        ch_left,
        ch_right,
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        await _broadcast_join(
            context,
            comm,
            ir,
            ir_context,
            ch_out,
            ch_left,
            ch_right,
            JoinStrategy(broadcast_side=broadcast_side),
            [collective_id],
            target_partition_size,
            tracer=tracer,
        )


async def _collect_small_side_for_broadcast(
    context: Context,
    comm: Communicator,
    ch: Channel[TableChunk],
    ir: IR,
    *,
    need_allgather: bool,
    collective_id: int,
    ir_context: IRExecutionContext,
    concat_size_limit: int | None,
) -> tuple[list[DataFrame], int]:
    """
    Drain small-side channel into chunks, then build DataFrame(s) for broadcast.

    Returns (list of DataFrames to join against, total byte size of small side).
    """
    size = 0
    chunks: list[TableChunk] = []
    while (msg := await ch.recv(context)) is not None:
        chunks.append(TableChunk.from_message(msg, br=context.br()))
        size += chunks[-1].data_alloc_size()
    row_count = sum(c.shape[0] for c in chunks)

    if (can_concatenate := row_count < CUDF_ROW_LIMIT) and concat_size_limit:
        can_concatenate = size <= concat_size_limit

    dfs: list[DataFrame] = []
    if need_allgather:
        allgather = AllGatherManager(context, comm, collective_id)
        with allgather.inserting() as inserter:
            for s_id in range(len(chunks)):
                inserter.insert(s_id, chunks.pop(0))
        stream = ir_context.get_cuda_stream()
        gathered = await allgather.extract_concatenated(stream, ir_context=ir_context)
        # When every rank inserted zero chunks, the AllGather has no schema
        # to infer and returns a 0 column table. Substitute a properly typed
        # empty table for the small side so downstream joins still match the
        # expected schema.
        table = (
            empty_table_chunk(ir, context, stream).table_view()
            if gathered.num_columns() == 0 and len(ir.schema) > 0
            else gathered
        )
        dfs = [
            DataFrame.from_table(
                table,
                list(ir.schema.keys()),
                list(ir.schema.values()),
                stream,
            )
        ]
    elif chunks:
        if can_concatenate:
            chunks, extra = await make_table_chunks_available_or_wait(
                context,
                chunks,
                reserve_extra=size,
                net_memory_delta=0,
            )
            with opaque_memory_usage(extra):
                dfs = [
                    _concat(
                        *[chunk_to_frame(chunk, ir) for chunk in chunks],
                        context=ir_context,
                    )
                ]
        else:
            chunks, _ = await make_table_chunks_available_or_wait(
                context, chunks, reserve_extra=0, net_memory_delta=0
            )
            dfs = [chunk_to_frame(c, ir) for c in chunks]

    return dfs, size


async def _broadcast_join_large_chunk(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    small_dfs: list[DataFrame],
    small_child: IR,
    large_chunk: TableChunk,
    large_child: IR,
    seq_num: int,
    small_size: int,
    broadcast_side: Literal["left", "right"],
    *,
    tracer: ActorTracer | None,
) -> None:
    """Join one large-side chunk with the small DataFrame(s) and send the result."""
    large_df = chunk_to_frame(large_chunk, large_child)
    large_chunk_size = large_chunk.data_alloc_size()

    dfs_to_join = small_dfs
    if not dfs_to_join:
        stream = ir_context.get_cuda_stream()
        empty_small = empty_table_chunk(small_child, context, stream)
        dfs_to_join = [chunk_to_frame(empty_small, small_child)]

    join_results: list[DataFrame] = []
    input_bytes = large_chunk_size + small_size
    with opaque_memory_usage(
        await reserve_memory(context, size=input_bytes, net_memory_delta=0)
    ):
        for sdf in dfs_to_join:
            result = await ir_context.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                *([large_df, sdf] if broadcast_side == "right" else [sdf, large_df]),
                context=ir_context,
            )
            join_results.append(result)

        df = _concat(*join_results, context=ir_context)
        del join_results

    output_chunk = TableChunk.from_pylibcudf_table(
        df.table, df.stream, exclusive_view=True, br=context.br()
    )
    await send_chunk(context, ch_out, output_chunk, seq_num, tracer=tracer)
    del df, large_df


async def _broadcast_join(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    strategy: JoinStrategy,
    collective_ids: list[int],
    target_partition_size: int,
    *,
    tracer: ActorTracer | None,
) -> None:
    """
    Execute a broadcast join after initial sampling.

    The small side is gathered (if not already duplicated) and concatenated
    into a single DataFrame, then joined with each chunk from the large side.
    Pops one collective ID from collective_ids for allgather when needed.
    """
    left_metadata, right_metadata = await gather_in_task_group(
        recv_metadata(ch_left, context),
        recv_metadata(ch_right, context),
    )

    collective_id = collective_ids.pop(0) if collective_ids else 0
    broadcast_side = strategy.broadcast_side
    assert broadcast_side is not None
    left, right = ir.children
    if tracer is not None:
        tracer.decision = f"broadcast_{broadcast_side}"

    if broadcast_side == "right":
        small_ch, large_ch = ch_right, ch_left
        small_child, large_child = right, left
        small_metadata, large_metadata = right_metadata, left_metadata
        local_count = left_metadata.local_count
        partitioning = maybe_remap_partitioning(
            ir,
            left_metadata.partitioning,
            child_ir=ir.children[0],
            context=context,
        )
    else:
        small_ch, large_ch = ch_left, ch_right
        small_child, large_child = left, right
        small_metadata, large_metadata = left_metadata, right_metadata
        local_count = right_metadata.local_count
        partitioning = (
            maybe_remap_partitioning(
                ir,
                right_metadata.partitioning,
                child_ir=ir.children[1],
                context=context,
            )
            if ir.options[0] == "Right"
            else None
        )

    small_duplicated = small_metadata.duplicated
    need_allgather = comm.nranks > 1 and not small_duplicated
    output_duplicated = (
        small_duplicated or need_allgather
    ) and large_metadata.duplicated

    metadata_out = ChannelMetadata(
        local_count=local_count,
        partitioning=partitioning,
        duplicated=output_duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    small_dfs, small_size = await _collect_small_side_for_broadcast(
        context,
        comm,
        small_ch,
        small_child,
        need_allgather=need_allgather,
        collective_id=collective_id,
        ir_context=ir_context,
        concat_size_limit=(target_partition_size if ir.options[0] == "Inner" else None),
    )

    while (msg := await large_ch.recv(context)) is not None:
        await _broadcast_join_large_chunk(
            context,
            ir,
            ir_context,
            ch_out,
            small_dfs,
            small_child,
            TableChunk.from_message(msg, br=context.br()).make_available_and_spill(
                context.br(), allow_overbooking=True
            ),
            large_child,
            msg.sequence_number,
            small_size,
            broadcast_side,
            tracer=tracer,
        )

    await ch_out.drain(context)


def _get_key_indices(
    ir: Join,
    n_partitioned_keys: int | None,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[NamedExpr, ...],
    tuple[NamedExpr, ...],
]:
    left, right = ir.children
    n_keys = n_partitioned_keys if n_partitioned_keys is not None else len(ir.left_on)
    left_keys = ir.left_on[:n_keys]
    right_keys = ir.right_on[:n_keys]
    left_key_indices = _key_column_indices(left_keys, left.schema) or ()
    right_key_indices = _key_column_indices(right_keys, right.schema) or ()
    if ir.options[0] == "Right":
        output_keys = right_keys
    else:
        output_keys = left_keys
    output_key_indices = (
        _key_column_indices(output_keys, ir.schema)
        if left_key_indices and right_key_indices
        else None
    )
    return (
        left_key_indices,
        right_key_indices,
        output_key_indices or (),
        left_keys,
        right_keys,
    )


def _ordering_with_column_indices(
    ordering: Ordering,
    column_indices: tuple[int, ...],
) -> Ordering:
    """Return an ordering with the same semantics on different column indices."""
    return ordering.with_keys(
        tuple(
            OrderKey(index, key.order, key.null_order)
            for index, key in zip(
                column_indices,
                ordering.keys[: len(column_indices)],
                strict=True,
            )
        )
    )


def _ordering_prefix_matches(
    ordering: Ordering,
    reference: Ordering,
    column_indices: tuple[int, ...],
) -> bool:
    """True when ordering has the same leading order semantics as reference."""
    if len(ordering.keys) < len(column_indices):
        return False
    expected = tuple(
        OrderKey(index, key.order, key.null_order)
        for index, key in zip(column_indices, reference.keys, strict=True)
    )
    return ordering.keys[: len(expected)] == expected


def _make_ordered_strategy(
    ir: Join,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
) -> JoinStrategy | None:
    """Make an ordered strategy when both sides can align to strict boundaries."""
    left_scheme = left_partitioning.inter_rank_scheme
    right_scheme = right_partitioning.inter_rank_scheme
    if not isinstance(left_scheme, OrderScheme) or not isinstance(
        right_scheme, OrderScheme
    ):
        return None

    left_ordering = left_scheme.orderings[0]
    right_ordering = right_scheme.orderings[0]
    n_ordered_keys = min(len(left_ordering.keys), len(right_ordering.keys))
    (
        left_key_indices,
        right_key_indices,
        output_key_indices,
        left_keys,
        right_keys,
    ) = _get_key_indices(ir, n_ordered_keys)
    if not output_key_indices:
        return None

    # adjust_ordering needs a strict target layout. Prefer the output-preserving
    # side when it is strict, then fall back to the other strict side.
    if ir.options[0] == "Right":
        reference = (
            right_ordering if right_ordering.strict_boundaries else left_ordering
        )
    else:
        reference = left_ordering if left_ordering.strict_boundaries else right_ordering
    if not reference.strict_boundaries:
        return None

    reference_key_count = min(len(reference.keys), len(output_key_indices))
    left_key_indices = left_key_indices[:reference_key_count]
    right_key_indices = right_key_indices[:reference_key_count]
    output_key_indices = output_key_indices[:reference_key_count]
    if (
        len(left_key_indices) != reference_key_count
        or len(right_key_indices) != reference_key_count
        or len(output_key_indices) != reference_key_count
    ):
        return None
    if not _ordering_prefix_matches(left_ordering, reference, left_key_indices):
        return None
    if not _ordering_prefix_matches(right_ordering, reference, right_key_indices):
        return None

    return JoinStrategy(
        left_meta=left_metadata,
        right_meta=right_metadata,
        output_indices=output_key_indices,
        left_indices=left_key_indices,
        right_indices=right_key_indices,
        left_keys=left_keys[:reference_key_count],
        right_keys=right_keys[:reference_key_count],
        ordered=True,
        left_input_ordering=left_ordering,
        right_input_ordering=right_ordering,
        left_output_ordering=_ordering_with_column_indices(reference, left_key_indices),
        right_output_ordering=_ordering_with_column_indices(
            reference, right_key_indices
        ),
        output_ordering=_ordering_with_column_indices(reference, output_key_indices),
    )


def _make_sort_ordered_strategy(
    ir: Join,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    *,
    left_total_size: int,
    right_total_size: int,
) -> JoinStrategy | None:
    """Make an ordered strategy that sorts only the smaller unordered side."""
    left_scheme = left_partitioning.inter_rank_scheme
    right_scheme = right_partitioning.inter_rank_scheme

    if isinstance(left_scheme, OrderScheme) and not isinstance(
        right_scheme, OrderScheme
    ):
        sort_side: Literal["left", "right"] = "right"
        ordered_side = "left"
        reference = left_scheme.orderings[0]
    elif isinstance(right_scheme, OrderScheme) and not isinstance(
        left_scheme, OrderScheme
    ):
        sort_side = "left"
        ordered_side = "right"
        reference = right_scheme.orderings[0]
    else:
        return None

    if not reference.strict_boundaries:
        return None
    if ordered_side == "left" and left_total_size < right_total_size:
        return None
    if ordered_side == "right" and right_total_size < left_total_size:
        return None

    (
        left_key_indices,
        right_key_indices,
        output_key_indices,
        left_keys,
        right_keys,
    ) = _get_key_indices(ir, len(reference.keys))
    reference_key_count = min(len(reference.keys), len(output_key_indices))
    left_key_indices = left_key_indices[:reference_key_count]
    right_key_indices = right_key_indices[:reference_key_count]
    output_key_indices = output_key_indices[:reference_key_count]
    if (
        reference_key_count == 0
        or len(left_key_indices) != reference_key_count
        or len(right_key_indices) != reference_key_count
        or len(output_key_indices) != reference_key_count
    ):
        return None
    ordered_indices = left_key_indices if ordered_side == "left" else right_key_indices
    if not _ordering_prefix_matches(reference, reference, ordered_indices):
        return None

    return JoinStrategy(
        left_meta=left_metadata,
        right_meta=right_metadata,
        output_indices=output_key_indices,
        left_indices=left_key_indices,
        right_indices=right_key_indices,
        left_keys=left_keys[:reference_key_count],
        right_keys=right_keys[:reference_key_count],
        ordered=True,
        sort_side=sort_side,
        left_input_ordering=reference if ordered_side == "left" else None,
        right_input_ordering=reference if ordered_side == "right" else None,
        left_output_ordering=_ordering_with_column_indices(reference, left_key_indices),
        right_output_ordering=_ordering_with_column_indices(
            reference, right_key_indices
        ),
        output_ordering=_ordering_with_column_indices(reference, output_key_indices),
    )


async def _join_chunks(
    context: Context,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    tracer: ActorTracer | None,
) -> None:
    # Consume metadata from both shuffle outputs before reading data
    await gather_in_task_group(
        recv_metadata(ch_left, context),
        recv_metadata(ch_right, context),
    )

    left, right = ir.children
    while True:
        left_msg, right_msg = await gather_in_task_group(
            ch_left.recv(context), ch_right.recv(context)
        )
        if left_msg is None or right_msg is None:
            assert left_msg is None, (
                "Mismatched chunk counts in shuffle join: left has unmatched chunk. "
                f"Seq num: {left_msg.sequence_number}"
            )
            assert right_msg is None, (
                "Mismatched chunk counts in shuffle join: right has unmatched chunk. "
                f"Seq num: {right_msg.sequence_number}"
            )
            break
        assert left_msg.sequence_number == right_msg.sequence_number, (
            "Mismatched chunk sequence numbers in shuffle join. "
            f"Left: {left_msg.sequence_number}, Right: {right_msg.sequence_number}"
        )

        left_chunk = TableChunk.from_message(
            left_msg, br=context.br()
        ).make_available_and_spill(context.br(), allow_overbooking=True)
        right_chunk = TableChunk.from_message(
            right_msg, br=context.br()
        ).make_available_and_spill(context.br(), allow_overbooking=True)

        input_bytes = sum(
            col.device_buffer_size()
            for col in (
                *left_chunk.table_view().columns(),
                *right_chunk.table_view().columns(),
            )
        )
        with opaque_memory_usage(
            await reserve_memory(context, size=input_bytes, net_memory_delta=0)
        ):
            df = await ir_context.to_thread(
                ir.do_evaluate,
                *ir._non_child_args,
                chunk_to_frame(left_chunk, left),
                chunk_to_frame(right_chunk, right),
                context=ir_context,
            )
            del left_chunk, right_chunk

        output_chunk = TableChunk.from_pylibcudf_table(
            df.table, df.stream, exclusive_view=True, br=context.br()
        )
        await send_chunk(
            context,
            ch_out,
            output_chunk,
            left_msg.sequence_number,
            tracer=tracer,
        )
        del df

    await ch_out.drain(context)


def _log_shuffle_strategy_decision(
    tracer: ActorTracer,
    strategy: JoinStrategy,
    partitioning_left: NormalizedPartitioning,
    partitioning_right: NormalizedPartitioning,
) -> None:
    left_scheme_desired = HashScheme(strategy.left_indices, strategy.shuffle_modulus)
    right_scheme_desired = HashScheme(strategy.right_indices, strategy.shuffle_modulus)
    left_partitioned = (
        partitioning_left.inter_rank_scheme == left_scheme_desired
        and partitioning_left.local_scheme == "inherit"
    )
    right_partitioned = (
        partitioning_right.inter_rank_scheme == right_scheme_desired
        and partitioning_right.local_scheme == "inherit"
    )
    if left_partitioned and right_partitioned:
        tracer.decision = "chunkwise"
    elif left_partitioned:
        tracer.decision = "shuffle_right"
    elif right_partitioned:
        tracer.decision = "shuffle_left"
    else:
        tracer.decision = "shuffle"


async def passthrough_split(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_split: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    *,
    indices: Iterable[int],
) -> None:
    """
    Pass all messages from ch_in to ch_out, copying key columns to ch_split.

    Parameters
    ----------
    context
         Streaming context
    ch_in
         Channel to consume
    ch_split
         Channel to send key columns to
    ch_out
         Channel to forward ch_in to
    indices
         Column indices of the input table to send to ch_split

    Notes
    -----
    This sends everything to ch_split before forwarding to ch_out, so the
    consumer must consume all of ch_split before consuming ch_out.
    """
    meta = await recv_metadata(ch_in, context)
    await send_metadata(ch_out, context, meta)
    buffer = context.spillable_messages()
    mids = []
    while (msg := await ch_in.recv(context)) is not None:
        chunk = await TableChunk.from_message(
            msg, br=context.br()
        ).make_available_or_wait(context, net_memory_delta=0)
        columns = chunk.table_view().columns()
        key_table = TableChunk.from_pylibcudf_table(
            plc.Table(
                [
                    columns[i].copy(chunk.stream, mr=context.br().device_mr)
                    for i in indices
                ]
            ),
            chunk.stream,
            exclusive_view=True,
            br=context.br(),
        )
        mids.append(buffer.insert(Message(msg.sequence_number, chunk)))
        await ch_split.send(context, Message(msg.sequence_number, key_table))
    await ch_split.drain(context)
    for mid in mids:
        await ch_out.send(context, buffer.extract(mid=mid))
    await ch_out.drain(context)


def _select_join_prefilter(
    join_type: Literal["Inner", "Left", "Right", "Full", "Semi", "Anti", "Cross"],
    left_rows: int,
    right_rows: int,
    left_key_indices: tuple[int, ...],
    right_key_indices: tuple[int, ...],
    *,
    threshold: float,
    max_key_columns: int | None,
) -> JoinPrefilterDecision:
    """
    Determine whether to apply a prefilter to a join.

    Parameters
    ----------
    join_type
        Type of join.
    left_rows
        Estimated number of rows in the left table.
    right_rows
        Estimated number of rows in the right table.
    left_key_indices
        Column indices of the join keys in the left table.
    right_key_indices
        Column indices of the join keys in the right table.
    threshold
        Small-to-large row-count ratio at or above which filtering is disabled.
    max_key_columns
        Maximum number of columns to use from the key prefix. ``None`` uses all
        join-key columns.

    Returns
    -------
    JoinPrefilterDecision
        The selected prefilter configuration, or the reason it was skipped.
    """
    if threshold == 0.0:
        return JoinPrefilterDecision(
            left_rows=left_rows,
            right_rows=right_rows,
            threshold=threshold,
            reason_skipped="disabled",
        )

    if join_type not in ("Inner", "Semi", "Left", "Anti", "Right"):
        return JoinPrefilterDecision(
            left_rows=left_rows,
            right_rows=right_rows,
            threshold=threshold,
            reason_skipped="unsupported_join_type",
        )

    if len(left_key_indices) != len(right_key_indices) or len(left_key_indices) == 0:
        return JoinPrefilterDecision(
            left_rows=left_rows,
            right_rows=right_rows,
            threshold=threshold,
            reason_skipped="expression_keys",
        )
    key_column_count = len(left_key_indices)
    if max_key_columns is not None:
        key_column_count = min(key_column_count, max_key_columns)

    small_rows, large_rows = sorted((left_rows, right_rows))
    ratio = small_rows / large_rows if large_rows > 0 else None
    filter_side: Literal["left", "right"] | None = None
    reason_skipped: str | None = None

    if join_type in ("Inner", "Semi"):
        filter_side = "right" if left_rows <= right_rows else "left"
    elif join_type in ("Left", "Anti"):
        if left_rows >= right_rows:
            reason_skipped = "no_legal_large_side"
        else:
            filter_side = "right"
    else:
        if right_rows >= left_rows:
            reason_skipped = "no_legal_large_side"
        else:
            filter_side = "left"

    if reason_skipped is None:
        if ratio is None:
            reason_skipped = "no_large_side"
        elif ratio >= threshold:
            reason_skipped = "ratio_above_threshold"

    if reason_skipped is not None:
        filter_side = None

    if filter_side == "right":
        build_indices = left_key_indices[:key_column_count]
        apply_indices = right_key_indices[:key_column_count]
    elif filter_side == "left":
        build_indices = right_key_indices[:key_column_count]
        apply_indices = left_key_indices[:key_column_count]
    else:
        build_indices = ()
        apply_indices = ()

    return JoinPrefilterDecision(
        left_rows=left_rows,
        right_rows=right_rows,
        threshold=threshold,
        filter_side=filter_side,
        build_indices=build_indices,
        apply_indices=apply_indices,
        key_column_count=key_column_count,
        small_large_ratio=ratio,
        reason_skipped=reason_skipped,
    )


async def trace_row_count_passthrough(
    context: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    trace_stats: dict[str, Any],
    *,
    row_count_key: str,
) -> None:
    """Forward a table-chunk channel while counting rows."""
    metadata = await recv_metadata(ch_in, context)
    await send_metadata(ch_out, context, metadata)
    row_count = 0
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg, br=context.br())
        row_count += chunk.shape[0]
        await ch_out.send(context, Message(msg.sequence_number, chunk))
    trace_stats[row_count_key] = row_count
    await ch_out.drain(context)


def make_filter_tasks(
    context: Context,
    comm: Communicator,
    *,
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    decision: JoinPrefilterDecision,
    tag: int,
    trace_stats: dict[str, Any] | None,
) -> tuple[
    Channel[TableChunk],
    Channel[TableChunk],
    list[Coroutine[Any, Any, None]],
    list[Channel],
]:
    """
    Create bloom filter tasks for a pair of channels participating in a shuffle join.

    Parameters
    ----------
    context
        Streaming context
    comm
        Communicator
    ch_left
        Left input channel
    ch_right
        Right input channel
    decision
        Selected prefilter decision
    tag
        Collective ID for combining partial filters across ranks
    trace_stats
        Mutable trace metadata to update with actual row counts, or None

    Returns
    -------
    tuple
       Of new left and right channels, coroutines to await, and new channels to shutdown on error.
    """
    assert decision.enabled
    assert decision.filter_side in ("left", "right")
    bloom_build_output: Channel[BloomFilterChunk] = context.create_channel()
    bloom_build_input: Channel[TableChunk] = context.create_channel()
    passthrough_output: Channel[TableChunk] = context.create_channel()
    if decision.filter_side == "right":
        passthrough_input = ch_left
        ch_left = passthrough_output
        build_indices = decision.build_indices
        bloom_apply_input = ch_right
        apply_indices = decision.apply_indices
        ch_right = context.create_channel()
        bloom_apply_output = ch_right
    else:
        passthrough_input = ch_right
        ch_right = passthrough_output
        build_indices = decision.build_indices
        bloom_apply_input = ch_left
        apply_indices = decision.apply_indices
        ch_left = context.create_channel()
        bloom_apply_output = ch_left

    # TODO: configure based on GPU L2 size
    filter_size = BloomFilter.aligned_size(32 * 1024 * 1024)
    filter = BloomFilter(context, comm, LIBCUDF_DEFAULT_HASH_SEED, filter_size)
    filter_tasks: list[Coroutine[Any, Any, None]] = []
    chs_to_shutdown = [
        bloom_build_output,
        bloom_build_input,
        passthrough_output,
    ]

    apply_input = bloom_apply_input
    apply_output = bloom_apply_output
    if trace_stats is not None:
        counted_apply_input: Channel[TableChunk] = context.create_channel()
        raw_apply_output: Channel[TableChunk] = context.create_channel()
        filter_tasks.extend(
            [
                trace_row_count_passthrough(
                    context,
                    bloom_apply_input,
                    counted_apply_input,
                    trace_stats,
                    row_count_key="input_rows",
                ),
                trace_row_count_passthrough(
                    context,
                    raw_apply_output,
                    bloom_apply_output,
                    trace_stats,
                    row_count_key="output_rows",
                ),
            ]
        )
        chs_to_shutdown.extend([counted_apply_input, raw_apply_output])
        apply_input = counted_apply_input
        apply_output = raw_apply_output

    filter_tasks = [
        *filter_tasks,
        passthrough_split(
            context,
            passthrough_input,
            bloom_build_input,
            passthrough_output,
            indices=build_indices,
        ),
        filter.build(
            context,
            bloom_build_input,
            bloom_build_output,
            tag,
        ),
        filter.apply(
            context,
            bloom_build_output,
            apply_input,
            apply_output,
            apply_indices,
        ),
    ]
    return ch_left, ch_right, filter_tasks, chs_to_shutdown


async def _shuffle_join(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    strategy: JoinStrategy,
    collective_ids: list[int],
    *,
    row_counts: tuple[int, int],
    tracer: ActorTracer | None,
    prefilter_threshold: float,
    prefilter_max_key_columns: int | None,
    prefilter_trace: bool,
) -> None:
    """Execute a shuffle (hash) join."""
    # Send output metadata
    shuffle_modulus = strategy.shuffle_modulus
    output_indices = strategy.output_indices
    nranks = comm.nranks
    metadata_out = ChannelMetadata(
        local_count=max(1, shuffle_modulus // nranks),
        partitioning=Partitioning(
            HashScheme(column_indices=output_indices, modulus=shuffle_modulus),
            local="inherit",
        )
        if output_indices
        else None,
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)
    left_rows, right_rows = row_counts
    bloom_tag = collective_ids.pop(0)
    prefilter_decision = _select_join_prefilter(
        ir.options[0],
        left_rows,
        right_rows,
        strategy.left_indices,
        strategy.right_indices,
        threshold=prefilter_threshold,
        max_key_columns=prefilter_max_key_columns,
    )
    prefilter_trace_stats = prefilter_decision.trace_dict()

    if tracer is not None:
        tracer.set_extra("join_prefilter", prefilter_trace_stats)

    if prefilter_decision.enabled:
        if tracer is not None:
            tracer.decision = f"{tracer.decision or 'shuffle'}_prefiltered"
        ch_left, ch_right, filter_tasks, chs_to_shutdown = make_filter_tasks(
            context,
            comm,
            ch_left=ch_left,
            ch_right=ch_right,
            decision=prefilter_decision,
            tag=bloom_tag,
            trace_stats=prefilter_trace_stats if prefilter_trace else None,
        )
    else:
        filter_tasks = []
        chs_to_shutdown = []
    # Construct a shuffle-shuffle-join pipeline.
    # The shuffle operations will pass chunks through unchanged
    # if the data is already partitioned correctly.
    ch_left_shuffle = context.create_channel()
    ch_right_shuffle = context.create_channel()
    # note: this is an actor inside of an actor. How should we log that in our traces?
    async with shutdown_on_error(
        context,
        *chs_to_shutdown,
        ch_left_shuffle,
        ch_right_shuffle,
        trace_ir=ir,
        ir_context=ir_context,
    ):
        actor_tasks = [
            *filter_tasks,
            _global_shuffle(
                context,
                comm,
                ir_context,
                ch_left_shuffle,
                ch_left,
                strategy.left_keys,
                ir.children[0].schema,
                strategy.shuffle_modulus,
                collective_ids.pop(0),
            ),
            _global_shuffle(
                context,
                comm,
                ir_context,
                ch_right_shuffle,
                ch_right,
                strategy.right_keys,
                ir.children[1].schema,
                strategy.shuffle_modulus,
                collective_ids.pop(0),
            ),
            _join_chunks(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left_shuffle,
                ch_right_shuffle,
                tracer=tracer,
            ),
        ]
        await gather_in_task_group(*actor_tasks)


def _local_count_for_ordering(comm: Communicator, ordering: Ordering) -> int:
    """Return this rank's local partition count for a contiguous Ordering."""
    npartitions = ordering.num_boundaries + 1
    start, stop = _partition_range(comm.rank, comm.nranks, npartitions)
    return stop - start


def _boundaries_dataframe(
    context: Context,
    ordering: Ordering,
    schema_ir: IR,
    column_indices: tuple[int, ...],
) -> DataFrame:
    """Return ordering boundaries as a DataFrame for sort partitioning."""
    boundary_chunk = ordering.get_boundaries(context.br())
    return DataFrame.from_table(
        boundary_chunk.table_view(),
        [list(schema_ir.schema.keys())[i] for i in column_indices],
        [list(schema_ir.schema.values())[i] for i in column_indices],
        boundary_chunk.stream,
    )


def _sort_ir_for_ordered_join(
    schema_ir: IR,
    sort_keys: tuple[NamedExpr, ...],
    output_ordering: Ordering,
) -> Sort:
    """Build a Sort IR matching output_ordering semantics."""
    return Sort(
        schema_ir.schema,
        sort_keys,
        tuple(key.order for key in output_ordering.keys),
        tuple(key.null_order for key in output_ordering.keys),
        stable=False,
        zlice=None,
        df=schema_ir,
    )


def _sort_key_column_names(sort_keys: tuple[NamedExpr, ...]) -> list[str]:
    """Return column names for concrete sort keys."""
    names = [key.value.name for key in sort_keys if isinstance(key.value, Col)]
    if len(names) != len(sort_keys):
        raise NotImplementedError("Ordered join side sorting requires column keys.")
    return names


async def _sort_join_side_to_ordering(
    context: Context,
    comm: Communicator,
    schema_ir: IR,
    sort_keys: tuple[NamedExpr, ...],
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    input_metadata: ChannelMetadata,
    output_ordering: Ordering,
    collective_ids: list[int],
    *,
    tracer: ActorTracer | None,
) -> None:
    """Sort one join side directly to an existing strict Ordering layout."""
    sort_ir = _sort_ir_for_ordered_join(schema_ir, sort_keys, output_ordering)
    sort_boundaries_df = _boundaries_dataframe(
        context,
        output_ordering,
        schema_ir,
        output_ordering.column_indices,
    )
    await _global_sort(
        context,
        comm,
        sort_ir,
        ir_context,
        ch_out,
        ch_in,
        input_metadata,
        _sort_key_column_names(sort_keys),
        output_ordering.num_boundaries + 1,
        sort_boundaries_df,
        collective_ids,
        tracer=tracer,
    )


async def _adjust_ordered_join_side(
    context: Context,
    comm: Communicator,
    schema_ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    input_ordering: Ordering,
    output_ordering: Ordering,
    *,
    collective_id: int,
) -> None:
    """Send metadata, then align one join side to output_ordering."""
    await send_metadata(
        ch_out,
        context,
        ChannelMetadata(
            local_count=_local_count_for_ordering(comm, output_ordering),
            partitioning=Partitioning(
                OrderScheme([output_ordering]),
                local="inherit",
            ),
            duplicated=False,
        ),
    )
    await adjust_ordering(
        context,
        comm,
        schema_ir,
        ir_context,
        ch_out,
        ch_in,
        input_ordering,
        output_ordering,
        collective_id=collective_id,
    )


async def _ordered_join(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    strategy: JoinStrategy,
    collective_ids: list[int],
    *,
    tracer: ActorTracer | None,
) -> None:
    """Align ordered inputs to common boundaries, then join partition-wise."""
    assert strategy.left_output_ordering is not None
    assert strategy.right_output_ordering is not None
    assert strategy.output_ordering is not None

    metadata_out = ChannelMetadata(
        local_count=_local_count_for_ordering(comm, strategy.output_ordering),
        partitioning=Partitioning(
            OrderScheme([strategy.output_ordering]),
            local="inherit",
        ),
        duplicated=False,
    )
    await send_metadata(ch_out, context, metadata_out)

    if strategy.sort_side is not None:
        ch_sorted = context.create_channel()
        async with shutdown_on_error(
            context,
            ch_sorted,
            trace_ir=ir,
            ir_context=ir_context,
        ):
            if strategy.sort_side == "left":
                assert strategy.left_meta is not None
                await gather_in_task_group(
                    _sort_join_side_to_ordering(
                        context,
                        comm,
                        ir.children[0],
                        strategy.left_keys,
                        ir_context,
                        ch_sorted,
                        ch_left,
                        strategy.left_meta,
                        strategy.left_output_ordering,
                        collective_ids,
                        tracer=tracer,
                    ),
                    _join_chunks(
                        context,
                        ir,
                        ir_context,
                        ch_out,
                        ch_sorted,
                        ch_right,
                        tracer=tracer,
                    ),
                )
            else:
                assert strategy.right_meta is not None
                await gather_in_task_group(
                    _sort_join_side_to_ordering(
                        context,
                        comm,
                        ir.children[1],
                        strategy.right_keys,
                        ir_context,
                        ch_sorted,
                        ch_right,
                        strategy.right_meta,
                        strategy.right_output_ordering,
                        collective_ids,
                        tracer=tracer,
                    ),
                    _join_chunks(
                        context,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_sorted,
                        tracer=tracer,
                    ),
                )
        return

    await gather_in_task_group(
        recv_metadata(ch_left, context),
        recv_metadata(ch_right, context),
    )
    assert strategy.left_input_ordering is not None
    assert strategy.right_input_ordering is not None

    ch_left_adjusted = context.create_channel()
    ch_right_adjusted = context.create_channel()
    async with shutdown_on_error(
        context,
        ch_left_adjusted,
        ch_right_adjusted,
        trace_ir=ir,
        ir_context=ir_context,
    ):
        await gather_in_task_group(
            _adjust_ordered_join_side(
                context,
                comm,
                ir.children[0],
                ir_context,
                ch_left_adjusted,
                ch_left,
                strategy.left_input_ordering,
                strategy.left_output_ordering,
                collective_id=collective_ids.pop(0),
            ),
            _adjust_ordered_join_side(
                context,
                comm,
                ir.children[1],
                ir_context,
                ch_right_adjusted,
                ch_right,
                strategy.right_input_ordering,
                strategy.right_output_ordering,
                collective_id=collective_ids.pop(0),
            ),
            _join_chunks(
                context,
                ir,
                ir_context,
                ch_out,
                ch_left_adjusted,
                ch_right_adjusted,
                tracer=tracer,
            ),
        )


def _make_shuffle_strategy(
    ir: Join,
    shuffle_modulus: int,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
) -> JoinStrategy:
    """Make a shuffle strategy."""

    # Use the coarsest prefix so we only shuffle on keys one side may already have
    def _num_indices(partitioning: NormalizedPartitioning) -> int:
        return (
            len(partitioning.inter_rank_scheme.column_indices)
            if isinstance(partitioning.inter_rank_scheme, HashScheme)
            else 0
        )

    n_left = _num_indices(left_partitioning)
    n_right = _num_indices(right_partitioning)
    if n_left and n_right:
        n_partitioned_keys = min(n_left, n_right)
    elif n_left or n_right:
        n_partitioned_keys = max(n_left, n_right)
    else:
        n_partitioned_keys = None  # both unpartitioned: shuffle on all join keys

    (
        left_key_indices,
        right_key_indices,
        output_key_indices,
        left_keys,
        right_keys,
    ) = _get_key_indices(ir, n_partitioned_keys)

    return JoinStrategy(
        left_meta=left_metadata,
        right_meta=right_metadata,
        shuffle_modulus=shuffle_modulus,
        output_indices=output_key_indices,
        left_indices=left_key_indices,
        right_indices=right_key_indices,
        left_keys=left_keys,
        right_keys=right_keys,
    )


async def _aggregate_estimates(
    context: Context,
    comm: Communicator,
    left_sample: TableSizeStats,
    right_sample: TableSizeStats,
    collective_ids: list[int],
) -> tuple[TableSizeStats, TableSizeStats]:
    """Aggregate table-size and row estimates across ranks."""
    # AllGather size, row, and chunk count estimates across ranks
    (
        left_total,
        right_total,
        left_total_rows,
        right_total_rows,
        left_total_chunks,
        right_total_chunks,
    ) = await allgather_reduce(
        context,
        comm,
        collective_ids.pop(0),
        left_sample.total_size,
        right_sample.total_size,
        left_sample.total_rows,
        right_sample.total_rows,
        left_sample.total_chunks,
        right_sample.total_chunks,
    )

    new_left_sample = TableSizeStats(
        chunks=left_sample.chunks,
        total_size=left_total,
        total_rows=left_total_rows,
        total_chunks=left_total_chunks,
    )
    new_right_sample = TableSizeStats(
        chunks=right_sample.chunks,
        total_size=right_total,
        total_rows=right_total_rows,
        total_chunks=right_total_chunks,
    )
    return new_left_sample, new_right_sample


async def _choose_strategy_from_samples(
    comm: Communicator,
    ir: Join,
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    executor: StreamingExecutor,
    *,
    left_sample: TableSizeStats,
    right_sample: TableSizeStats,
    chunkwise: bool,
    tracer: ActorTracer | None,
) -> JoinStrategy:
    """Choose potential broadcast side and minimum shuffle modulus."""
    if chunkwise:
        if tracer is not None:
            tracer.decision = "chunkwise"
        # TODO: Ensure this emits a "dynamic planning" decision of "chunkwise"
        # Or push it up a level to the caller?
        assert isinstance(left_partitioning.inter_rank_scheme, HashScheme)
        return _make_shuffle_strategy(
            ir,
            left_partitioning.inter_rank_scheme.modulus,
            left_partitioning,
            right_partitioning,
            left_metadata,
            right_metadata,
        )

    left_total, right_total = left_sample.total_size, right_sample.total_size
    left_total_rows, right_total_rows = left_sample.total_rows, right_sample.total_rows
    left_total_chunks, right_total_chunks = (
        left_sample.total_chunks,
        right_sample.total_chunks,
    )

    # =====================================================================
    # Broadcast-Join Strategy Selection
    # =====================================================================
    # - Inner: can broadcast either side
    # - Left/Semi/Anti: must broadcast right (stream left to preserve all left rows)
    # - Right: must broadcast left (stream right to preserve all right rows)
    # - Full: cannot broadcast (must shuffle both to preserve both sides)

    # Determine which sides may be broadcasted
    broadcast_threshold = executor.broadcast_limit
    left_size_ok = left_total < broadcast_threshold and (
        left_total_rows < MAX_ROWS_PER_PARTITION or left_metadata.duplicated
    )
    right_size_ok = right_total < broadcast_threshold and (
        right_total_rows < MAX_ROWS_PER_PARTITION or right_metadata.duplicated
    )
    can_broadcast_left = left_size_ok and ir.options[0] in ("Inner", "Right")
    can_broadcast_right = right_size_ok and ir.options[0] in (
        "Inner",
        "Left",
        "Semi",
        "Anti",
    )

    broadcast_side: Literal["left", "right"] | None = None
    if can_broadcast_left and can_broadcast_right:
        # Choose side that is already duplicated.
        # If both or neither are duplicated, choose the side with fewer rows.
        if left_metadata.duplicated == right_metadata.duplicated:
            broadcast_side = "right" if right_total_rows <= left_total_rows else "left"
        elif left_metadata.duplicated:
            broadcast_side = "left"
        else:
            broadcast_side = "right"
    elif can_broadcast_left:
        broadcast_side = "left"
    elif can_broadcast_right:
        broadcast_side = "right"
    if broadcast_side is not None:
        return JoinStrategy(broadcast_side=broadcast_side)

    if not left_metadata.duplicated and not right_metadata.duplicated:
        ordered_strategy = _make_sort_ordered_strategy(
            ir,
            left_partitioning,
            right_partitioning,
            left_metadata,
            right_metadata,
            left_total_size=left_total,
            right_total_size=right_total,
        )
        if ordered_strategy is not None:
            if tracer is not None:
                tracer.decision = f"sort_{ordered_strategy.sort_side}_ordered"
            return ordered_strategy

    # Couldn't broadcast - Use a shuffle join instead.
    estimated_output_size = max(left_total, right_total)
    ideal_output_count = max(1, estimated_output_size // executor.target_partition_size)
    # Limit the output count to 10x the larger input side.
    # This is an arbitrary limit to prevent an oversized sample
    # from blowing up the chunk count.
    max_output_chunks = 10 * max(left_total_chunks, right_total_chunks)
    min_shuffle_modulus = min(ideal_output_count, max_output_chunks)

    # Stay away from cuDF's row limit
    if (estimated_rows_count := max(left_total_rows, right_total_rows)) > 0:
        min_partitions_for_row_limit = (
            estimated_rows_count + MAX_ROWS_PER_PARTITION - 1
        ) // MAX_ROWS_PER_PARTITION
        min_shuffle_modulus = max(min_shuffle_modulus, min_partitions_for_row_limit)

    shuffle_modulus = _choose_shuffle_modulus(
        comm,
        left_partitioning,
        right_partitioning,
        min_shuffle_modulus,
    )  # Global modulus

    strategy = _make_shuffle_strategy(
        ir,
        shuffle_modulus,
        left_partitioning,
        right_partitioning,
        left_metadata,
        right_metadata,
    )

    if tracer is not None:
        _log_shuffle_strategy_decision(
            tracer,
            strategy,
            left_partitioning,
            right_partitioning,
        )
    return strategy


def _choose_shuffle_modulus(
    comm: Communicator,
    left_partitioning: NormalizedPartitioning,
    right_partitioning: NormalizedPartitioning,
    min_shuffle_modulus: int,
) -> int:
    """Choose an appropriate modulus for a shuffle join."""

    def _modulus(partitioning: NormalizedPartitioning) -> int | None:
        return (
            partitioning.inter_rank_scheme.modulus
            if isinstance(partitioning.inter_rank_scheme, HashScheme)
            else None
        )

    left_modulus = _modulus(left_partitioning)
    right_modulus = _modulus(right_partitioning)
    default_modulus = max(comm.nranks, min_shuffle_modulus)
    small, large = sorted(
        [left_modulus or default_modulus, right_modulus or default_modulus]
    )
    if large % small == 0 and small >= min_shuffle_modulus:
        return small
    else:
        return max(large, min_shuffle_modulus)


async def _choose_strategy(
    context: Context,
    comm: Communicator,
    ir: Join,
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    left_metadata: ChannelMetadata,
    right_metadata: ChannelMetadata,
    executor: StreamingExecutor,
    collective_ids: list[int],
    *,
    tracer: ActorTracer | None,
) -> tuple[TableSizeStats, TableSizeStats, JoinStrategy]:
    """Sample both sides, aggregate estimates, and choose broadcast vs shuffle."""
    nranks = comm.nranks
    left_partitioning = NormalizedPartitioning.from_keys(
        left_metadata.partitioning,
        nranks,
        keys=names_to_indices(ir.left_on, ir.children[0].schema, concrete_prefix=True),
    )
    right_partitioning = NormalizedPartitioning.from_keys(
        right_metadata.partitioning,
        nranks,
        keys=names_to_indices(ir.right_on, ir.children[1].schema, concrete_prefix=True),
    )

    hash_chunkwise = isinstance(
        left_partitioning.inter_rank_scheme, HashScheme
    ) and isinstance(right_partitioning.inter_rank_scheme, HashScheme)
    if hash_chunkwise and left_partitioning.is_aligned_with(
        right_partitioning, context.br()
    ):
        # We can use a chunkwise join
        chunkwise = True
        left_sample = TableSizeStats(
            chunks=ChunkStore(context),
            total_chunks=left_metadata.local_count,
        )
        right_sample = TableSizeStats(
            chunks=ChunkStore(context),
            total_chunks=right_metadata.local_count,
        )
    elif (
        not left_metadata.duplicated
        and not right_metadata.duplicated
        and (
            ordered_strategy := _make_ordered_strategy(
                ir,
                left_partitioning,
                right_partitioning,
                left_metadata,
                right_metadata,
            )
        )
        is not None
    ):
        if tracer is not None:
            tracer.decision = "ordered"
        left_sample = TableSizeStats(
            chunks=ChunkStore(context),
            total_chunks=left_metadata.local_count,
        )
        right_sample = TableSizeStats(
            chunks=ChunkStore(context),
            total_chunks=right_metadata.local_count,
        )
        return left_sample, right_sample, ordered_strategy
    else:
        # Need to shuffle or broadcast - Use sampled data to choose a strategy
        chunkwise = False
        assert executor.dynamic_planning is not None
        sample_chunk_count = executor.dynamic_planning.sample_chunk_count
        target_partition_size = executor.target_partition_size
        left_sample, right_sample = await gather_in_task_group(
            _sample_chunks(
                context,
                ch_left,
                sample_chunk_count,
                target_partition_size,
                left_metadata.local_count,
            ),
            _sample_chunks(
                context,
                ch_right,
                sample_chunk_count,
                target_partition_size,
                right_metadata.local_count,
            ),
        )
        left_sample, right_sample = await _aggregate_estimates(
            context,
            comm,
            left_sample,
            right_sample,
            collective_ids,
        )

    strategy = await _choose_strategy_from_samples(
        comm,
        ir,
        left_metadata,
        right_metadata,
        left_partitioning,
        right_partitioning,
        executor,
        left_sample=left_sample,
        right_sample=right_sample,
        chunkwise=chunkwise,
        tracer=tracer,
    )

    return left_sample, right_sample, strategy


@define_actor()
async def join_actor(
    context: Context,
    comm: Communicator,
    ir: Join,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    executor: StreamingExecutor,
    collective_ids: list[int],
) -> None:
    """
    Dynamic Join actor that selects the best strategy at runtime.

    Receives metadata from the left and right channels, then either
    executes a shuffle join or a broadcast join. Strategy is chosen
    at runtime from sampled chunks when partitioning is not aligned.

    Parameters
    ----------
    context
        RapidsMPF context (communicator, etc.).
    comm
        The communicator.
    ir
        The Join IR node.
    ir_context
        Execution context for the plan.
    ch_out
        Output channel for the join result.
    ch_left
        Input channel for the left side.
    ch_right
        Input channel for the right side.
    executor
        Streaming executor configuration.
    collective_ids
        List of collective IDs for shuffle/broadcast; consumed as needed.
    """
    async with shutdown_on_error(
        context,
        ch_out,
        ch_left,
        ch_right,
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        left_metadata, right_metadata = await gather_in_task_group(
            recv_metadata(ch_left, context),
            recv_metadata(ch_right, context),
        )

        left_sample, right_sample, strategy = await _choose_strategy(
            context,
            comm,
            ir,
            ch_left,
            ch_right,
            left_metadata,
            right_metadata,
            executor,
            collective_ids,
            tracer=tracer,
        )
        ch_left_replay = context.create_channel()
        ch_right_replay = context.create_channel()
        async with shutdown_on_error(
            context,
            ch_left_replay,
            ch_right_replay,
            trace_ir=ir,
            ir_context=ir_context,
        ):
            actor_tasks = [
                replay_buffered_channel(
                    context,
                    ch_left_replay,
                    ch_left,
                    left_sample.chunks,
                    left_metadata,
                    trace_ir=ir,
                ),
                replay_buffered_channel(
                    context,
                    ch_right_replay,
                    ch_right,
                    right_sample.chunks,
                    right_metadata,
                    trace_ir=ir,
                ),
            ]
            ch_left = ch_left_replay
            ch_right = ch_right_replay

            if strategy.broadcast_side is not None:
                actor_tasks.append(
                    _broadcast_join(
                        context,
                        comm,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        strategy,
                        collective_ids,
                        executor.target_partition_size,
                        tracer=tracer,
                    )
                )
            elif strategy.ordered:
                actor_tasks.append(
                    _ordered_join(
                        context,
                        comm,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        strategy,
                        collective_ids,
                        tracer=tracer,
                    )
                )
            else:
                dynamic_options = executor.dynamic_planning
                prefilter_threshold = (
                    dynamic_options.join_prefilter_threshold
                    if dynamic_options is not None
                    else 0.0
                )
                prefilter_max_key_columns = (
                    dynamic_options.join_prefilter_max_key_columns
                    if dynamic_options is not None
                    else 1
                )
                prefilter_trace = (
                    dynamic_options.join_prefilter_trace
                    if dynamic_options is not None
                    else False
                )
                actor_tasks.append(
                    _shuffle_join(
                        context,
                        comm,
                        ir,
                        ir_context,
                        ch_out,
                        ch_left,
                        ch_right,
                        strategy,
                        collective_ids,
                        row_counts=(
                            left_sample.total_rows,
                            right_sample.total_rows,
                        ),
                        tracer=tracer,
                        prefilter_threshold=prefilter_threshold,
                        prefilter_max_key_columns=prefilter_max_key_columns,
                        prefilter_trace=prefilter_trace,
                    )
                )
            await gather_in_task_group(*actor_tasks)


def _use_pwise_join(
    executor: StreamingExecutor,
    partition_info: MutableMapping[IR, PartitionInfo],
    ir: Join,
) -> bool:
    """Whether to use a static-planning partition-wise join."""
    left, right = ir.children
    output_count = partition_info[ir].count
    if (
        output_count == 1
        and isinstance(left, Repartition)
        and isinstance(right, Repartition)
    ):
        # We fell back to single-partition behavior at lowering time
        return True

    if executor.name == "streaming" and executor.dynamic_planning is not None:
        return False

    left_count = partition_info[left].count
    right_count = partition_info[right].count
    left_partitioned = (
        partition_info[left].partitioned_on == ir.left_on and left_count == output_count
    )
    right_partitioned = (
        partition_info[right].partitioned_on == ir.right_on
        and right_count == output_count
    )
    return left_partitioned and right_partitioned


@generate_ir_sub_network.register(Join)
def _(
    ir: Join, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Join operation.
    left, right = ir.children
    partition_info = rec.state["partition_info"]
    left_count = partition_info[left].count
    right_count = partition_info[right].count
    executor = rec.state["config_options"].executor
    pwise_join = _use_pwise_join(executor, partition_info, ir)

    # Process children
    actors, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    if pwise_join:
        # Partition-wise join (use default_node_multi)
        partitioning_index = 1 if ir.options[0] == "Right" else 0
        actors[ir] = [
            default_node_multi(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                (
                    channels[left].reserve_output_slot(),
                    channels[right].reserve_output_slot(),
                ),
                partitioning_index=partitioning_index,
            )
        ]
        return actors, channels

    elif (
        executor.name == "streaming"
        and executor.dynamic_planning is not None
        and ir.options[0] in ("Inner", "Left", "Right", "Full", "Semi", "Anti")
    ):
        # Dynamic join - decide strategy at runtime
        collective_ids = list(rec.state["collective_id_map"].get(ir, []))
        # Join uses up to 4 collective IDs: allgather, left shuffle, right
        # shuffle, and bloom filter.
        if len(collective_ids) < 4:
            raise ValueError(
                "Dynamic join requires 4 reserved collective IDs "
                "(allgather + left shuffle + right shuffle + bloom filter); got "
                f"{len(collective_ids)} for this Join. "
                "Ensure ReserveOpIDs is run with dynamic_planning enabled."
            )
        actors[ir] = [
            join_actor(
                rec.state["context"],
                rec.state["comm"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                executor,
                collective_ids,
            )
        ]
        return actors, channels

    else:
        # Broadcast join (use broadcast_join_actor)
        broadcast_side: Literal["left", "right"]
        if left_count >= right_count:
            # Broadcast right, stream left
            broadcast_side = "right"
        else:
            broadcast_side = "left"
        actors[ir] = [
            broadcast_join_actor(
                rec.state["context"],
                rec.state["comm"],
                ir,
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                channels[left].reserve_output_slot(),
                channels[right].reserve_output_slot(),
                broadcast_side=broadcast_side,
                collective_id=rec.state["collective_id_map"][ir][0],
                target_partition_size=executor.target_partition_size,
            )
        ]
        return actors, channels
