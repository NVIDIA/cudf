# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core node definitions for the RapidsMPF streaming runtime."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, cast

from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor, when_all
from rapidsmpf.streaming.core.fanout import FanoutPolicy, fanout

from cudf_polars.dsl.ir import IR, Empty, HConcat
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.hconcat import build_hconcat_partitioning
from cudf_polars.streaming.actor_graph.partitioning import maybe_remap_partitioning
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    _leading_order_keys,
    chunk_to_frame,
    chunkwise_evaluate,
    empty_table_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cudf_streaming.channel_metadata import Partitioning
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator

    type PartitioningCallback = Callable[
        [Sequence[ChannelMetadata], Context],
        Partitioning | None,
    ]


@define_actor()
async def default_node_single(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    *,
    partitioning_callback: PartitioningCallback | None = None,
) -> None:
    """
    Single-channel default node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    ch_in
        The input Channel[TableChunk].
    partitioning_callback
        Optional builder for output partitioning metadata.

    Notes
    -----
    Chunks are processed in the order they are received.
    """
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        # Recv metadata and prepare output metadata
        metadata_in = await recv_metadata(ch_in, context)
        partitioning = (
            partitioning_callback((metadata_in,), context)
            if partitioning_callback is not None
            else maybe_remap_partitioning(ir, metadata_in.partitioning, context=context)
        )
        metadata_out = ChannelMetadata(
            local_count=metadata_in.local_count,
            partitioning=partitioning,
            duplicated=metadata_in.duplicated,
        )
        # Process chunks (handle empty input for aggregation-like operations)
        await chunkwise_evaluate(
            context,
            ir,
            ir_context,
            ch_out,
            ch_in,
            metadata_out,
            input_metadata=metadata_in,
            handle_empty_input=True,
            tracer=tracer,
        )


@define_actor()
async def default_node_multi(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    chs_in: tuple[Channel[TableChunk], ...],
    *,
    partitioning_callback: PartitioningCallback | None = None,
) -> None:
    """
    Pointwise node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    chs_in
        Tuple of input Channel[TableChunk]s.
    partitioning_callback
        Optional callback for building output partitioning from child metadata.
        When None, no partitioning metadata is preserved.
    """
    async with shutdown_on_error(
        context,
        chs_in=chs_in,
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        # Merge and forward basic metadata.
        local_count = 1
        duplicated = True
        partitioning = None
        child_metadatas = await gather_in_task_group(
            *(recv_metadata(ch, context) for ch in chs_in)
        )
        child_ordering_metadatas = [
            _leading_order_keys(md_child) for md_child in child_metadatas
        ]
        for md_child in child_metadatas:
            # Use simple "max" rule to determine counts.
            local_count = max(md_child.local_count, local_count)
            # Set "duplicated" to False as soon as we
            # find a non-duplicated child.
            duplicated = duplicated and md_child.duplicated
        if partitioning_callback is not None:
            partitioning = partitioning_callback(child_metadatas, context)
        metadata = ChannelMetadata(
            local_count=local_count,
            partitioning=partitioning,
            duplicated=duplicated,
        )
        await send_metadata(ch_out, context, metadata)
        if tracer is not None and duplicated:
            tracer.set_duplicated()

        seq_num = 0
        n_children = len(chs_in)
        finished_channels: set[int] = set()
        # Store TableChunk objects to keep data alive and prevent use-after-free
        # with stream-ordered allocations
        ready_chunks: list[TableChunk | None] = [None] * n_children
        chunk_count: list[int] = [0] * n_children

        # Recv/send data.
        while True:
            # Receive from all non-finished channels
            for ch_idx, ch_in in enumerate(chs_in):
                if ch_idx in finished_channels:
                    continue  # This channel already finished, reuse its data

                msg = await ch_in.recv(context)
                if msg is None:
                    # Channel finished - keep its last chunk for reuse
                    finished_channels.add(ch_idx)
                else:
                    # Store the new chunk (replacing previous if any)
                    ready_chunks[ch_idx] = TableChunk.from_message(msg, br=context.br())
                    chunk_count[ch_idx] += 1
                del msg

            # If all channels finished, we're done
            if len(finished_channels) == n_children:
                break

            # Check if any channel drained without providing data.
            # If so, create an empty chunk for that channel.
            for ch_idx, child in enumerate(ir.children):
                if ready_chunks[ch_idx] is None:
                    # Channel drained without data - create empty chunk
                    stream = ir_context.get_cuda_stream()
                    ready_chunks[ch_idx] = empty_table_chunk(child, context, stream)

            # Ensure all table chunks are unspilled and available.
            ready_chunks, extra = await make_table_chunks_available_or_wait(
                context,
                ready_chunks,
                reserve_extra=sum(
                    chunk.data_alloc_size()
                    for chunk in cast("list[TableChunk]", ready_chunks)
                ),
                net_memory_delta=0,
            )
            dfs = [
                chunk_to_frame(
                    cast("TableChunk", chunk),
                    child,
                    ordering_metadata=child_ordering_metadata,
                )
                for chunk, child, child_ordering_metadata in zip(
                    ready_chunks,
                    ir.children,
                    child_ordering_metadatas,
                    strict=True,
                )
            ]
            with opaque_memory_usage(extra):
                df = await ir_context.to_thread(
                    ir.do_evaluate,
                    *ir._non_child_args,
                    *dfs,
                    context=ir_context,
                )
                del dfs
            output_chunk = TableChunk.from_pylibcudf_table(
                df.table,
                df.stream,
                exclusive_view=True,
                br=context.br(),
            )
            await send_chunk(context, ch_out, output_chunk, seq_num, tracer=tracer)
            seq_num += 1
            del df

        # Drain the output channel
        del ready_chunks
        await ch_out.drain(context)


async def fanout_node(
    context: Context,
    ch_in: Channel[TableChunk],
    chs_out: Sequence[Channel[TableChunk]],
    policy: FanoutPolicy,
    *,
    trace_ir: IR,
    ir_context: IRExecutionContext,
) -> None:
    """
    Broadcast a channel to several consumers, metadata included.

    The rapidsmpf C++ fanout actor only broadcasts data messages, so the
    metadata is forwarded here first. The C++ actor is started only after
    that, since it drains ``chs_out`` and shuts down ``ch_in`` (including
    their metadata side-channels) when it finishes.

    This is deliberately *not* wrapped in ``define_actor``: channels are
    only shut down on error, by ``shutdown_on_error``.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ch_in
        The input Channel[TableChunk].
    chs_out
        The output Channel[TableChunk]s.
    policy
        The fanout policy for the data messages.
    trace_ir
        The IR node to trace. Passed through to shutdown_on_error.
    ir_context
        The execution context for the IR node.
    """
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=chs_out,
        trace_ir=trace_ir,
        ir_context=ir_context,
    ):
        metadata = await recv_metadata(ch_in, context)
        await gather_in_task_group(
            *(send_metadata(ch, context, metadata) for ch in chs_out)
        )
        await when_all(context, [fanout(context, ch_in, list(chs_out), policy)])


@generate_ir_sub_network.register(IR)
def _(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Default generate_ir_sub_network logic.
    # Use simple pointwise node.

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])
    ir_context = ir_context_for_node(rec, ir)

    if len(ir.children) == 1:
        # Single-channel default node
        nodes[ir] = [
            default_node_single(
                rec.state["context"],
                ir,
                ir_context,
                channels[ir].reserve_input_slot(),
                channels[ir.children[0]].reserve_output_slot(),
            )
        ]
    else:
        # Multi-channel default node
        nodes[ir] = [
            default_node_multi(
                rec.state["context"],
                ir,
                ir_context,
                channels[ir].reserve_input_slot(),
                tuple(channels[c].reserve_output_slot() for c in ir.children),
                partitioning_callback=(
                    partial(build_hconcat_partitioning, ir)
                    if isinstance(ir, HConcat) and ir.streaming_safe
                    else None
                ),
            )
        ]

    return nodes, channels


@define_actor()
async def empty_node(
    context: Context,
    ir: Empty,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
) -> None:
    """
    Empty node for rapidsmpf - produces a single empty chunk.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Empty node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    """
    async with shutdown_on_error(
        context,
        chs_out=(ch_out,),
        ir_context=ir_context,
        trace_ir=ir,
    ) as actor_scope:
        ir_context = actor_scope.require_ir_context()
        # Send metadata indicating a single empty chunk
        await send_metadata(
            ch_out,
            context,
            # All ranks generate the same "empty" data.
            ChannelMetadata(local_count=1, duplicated=True),
        )

        # Evaluate the IR node to create an empty DataFrame
        df: DataFrame = ir.do_evaluate(*ir._non_child_args, context=ir_context)

        # Return the output chunk (empty but with correct schema)
        chunk = TableChunk.from_pylibcudf_table(
            df.table, df.stream, exclusive_view=True, br=context.br()
        )
        await send_chunk(context, ch_out, chunk, 0, tracer=actor_scope.tracer)

        await ch_out.drain(context)


@generate_ir_sub_network.register(Empty)
def _(
    ir: Empty, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate network for Empty node - produces one empty chunk."""
    context = rec.state["context"]
    ir_context = ir_context_for_node(rec, ir)
    channels: dict[IR, ChannelManager] = {ir: ChannelManager(rec.state["context"])}
    nodes: dict[IR, list[Any]] = {
        ir: [empty_node(context, ir, ir_context, channels[ir].reserve_input_slot())]
    }
    return nodes, channels


def generate_ir_sub_network_wrapper(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """
    Generate a sub-network for the RapidsMPF streaming runtime.

    Parameters
    ----------
    ir
        The IR node.
    rec
        Recursive SubNetGenerator callable.

    Returns
    -------
    nodes
        Dictionary mapping each IR node to its list of streaming-network node(s).
    channels
        Dictionary mapping between each IR node and its
        corresponding streaming-network output ChannelManager.
    """
    nodes, channels = generate_ir_sub_network(ir, rec)

    # Check if this node needs fanout
    if (fanout_info := rec.state["fanout_nodes"].get(ir)) is not None:
        count = fanout_info.num_consumers
        manager = ChannelManager(rec.state["context"], count=count)
        policy = (
            FanoutPolicy.UNBOUNDED if fanout_info.unbounded else FanoutPolicy.BOUNDED
        )
        ch_in = channels[ir].reserve_output_slot()
        chs_out = [manager.reserve_input_slot() for _ in range(count)]
        nodes[ir].append(
            fanout_node(
                rec.state["context"],
                ch_in,
                chs_out,
                policy,
                trace_ir=ir,
                ir_context=ir_context_for_node(rec, ir),
            )
        )
        channels[ir] = manager
    return nodes, channels


@define_actor()
async def metadata_feeder_node(
    context: Context,
    ir: IR,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    metadata: ChannelMetadata,
    ir_context: IRExecutionContext,
) -> None:
    """
    Forward data with new metadata.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The IR node (for tracing).
    ch_in
        The input channel to pull data from.
    ch_out
        The output channel to forward data to and add metadata to.
    metadata
        The metadata to add to the output channel.
    ir_context
        The execution context for the IR node.
    """
    # TODO: Use ir_context
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ):
        await send_metadata(ch_out, context, metadata)
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
        await ch_out.drain(context)


@define_actor()
async def metadata_drain_node(
    context: Context,
    comm: Communicator,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Any,
    metadata_collector: list[ChannelMetadata] | None,
) -> None:
    """
    Drain metadata and forward data to a single channel.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The IR node.
    ir_context
        The execution context for the IR node.
    ch_in
        The input Channel[TableChunk] (with metadata and data channels).
    ch_out
        The output data channel.
    metadata_collector
        The list to collect the final metadata.
        This list will be mutated when the network is executed.
        If None, metadata will not be collected.
    """
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        ir_context=ir_context,
        trace_ir=ir,
    ):
        # Drain metadata channel (we don't need it after this point)
        msg = await ch_in.recv_metadata(context)
        if msg is None:
            # An upstream actor failed/was cancelled and shut down the channel
            return
        metadata = ChannelMetadata.from_message(msg)
        if metadata_collector is not None:
            metadata_collector.append(metadata)

        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)

        await ch_out.drain(context)
