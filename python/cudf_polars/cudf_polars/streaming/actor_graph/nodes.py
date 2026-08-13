# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core node definitions for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.spillable_messages import SpillableMessages

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, Empty
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    chunkwise_evaluate,
    empty_table_chunk,
    gather_in_task_group,
    make_spill_function,
    maybe_remap_partitioning,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator


@define_actor()
async def default_node_single(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
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

    Notes
    -----
    Chunks are processed in the order they are received.
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        # Recv metadata and prepare output metadata
        metadata_in = await recv_metadata(ch_in, context)
        metadata_out = ChannelMetadata(
            local_count=metadata_in.local_count,
            partitioning=maybe_remap_partitioning(
                ir, metadata_in.partitioning, context=context
            ),
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
    partitioning_index: int | None = None,
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
    partitioning_index
        Index of the input channel to preserve partitioning information for.
        If None, no partitioning information is preserved.
    """
    async with shutdown_on_error(
        context, *chs_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        # Merge and forward basic metadata.
        local_count = 1
        duplicated = True
        partitioning = None
        for idx, md_child in enumerate(
            await gather_in_task_group(*(recv_metadata(ch, context) for ch in chs_in))
        ):
            # Use simple "max" rule to determine counts.
            local_count = max(md_child.local_count, local_count)
            # Set "duplicated" to False as soon as we
            # find a non-duplicated child.
            duplicated = duplicated and md_child.duplicated
            if idx == partitioning_index:
                # Remap partitioning from child schema to output schema
                partitioning = maybe_remap_partitioning(
                    ir,
                    md_child.partitioning,
                    child_ir=ir.children[idx],
                    context=context,
                )
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
                DataFrame.from_table(
                    chunk.table_view(),  # type: ignore[union-attr]
                    list(child.schema.keys()),
                    list(child.schema.values()),
                    chunk.stream,  # type: ignore[union-attr]
                )
                for chunk, child in zip(ready_chunks, ir.children, strict=True)
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


@define_actor()
async def fanout_node_bounded(
    context: Context,
    ch_in: Channel[TableChunk],
    *chs_out: Channel[TableChunk],
    trace_ir: IR,
    ir_context: IRExecutionContext,
) -> None:
    """
    Bounded fanout node for rapidsmpf.

    Each chunk is broadcasted to all output channels
    as it arrives.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ch_in
        The input Channel[TableChunk].
    chs_out
        The output Channel[TableChunk]s.
    trace_ir
        The IR node to trace. Passed through to shutdown_on_error.
    ir_context
        The execution context for the IR node.
    """
    # TODO: Use rapidsmpf fanout node once available.
    # See: https://github.com/rapidsai/rapidsmpf/issues/560
    # TODO: Use ir_context
    async with shutdown_on_error(
        context, ch_in, *chs_out, trace_ir=trace_ir, ir_context=ir_context
    ):
        # Forward metadata to all outputs.
        metadata = await recv_metadata(ch_in, context)
        await gather_in_task_group(
            *(send_metadata(ch, context, metadata) for ch in chs_out)
        )

        while (msg := await ch_in.recv(context)) is not None:
            table_chunk = TableChunk.from_message(
                msg, br=context.br()
            ).make_available_and_spill(context.br(), allow_overbooking=True)
            seq_num = msg.sequence_number
            del msg
            for ch_out in chs_out:
                await ch_out.send(
                    context,
                    Message(
                        seq_num,
                        TableChunk.from_pylibcudf_table(
                            table_chunk.table_view(),
                            table_chunk.stream,
                            exclusive_view=False,
                            br=context.br(),
                        ),
                    ),
                )
            del table_chunk

        await gather_in_task_group(*(ch.drain(context) for ch in chs_out))


@define_actor()
async def fanout_node_unbounded(
    context: Context,
    ch_in: Channel[TableChunk],
    *chs_out: Channel[TableChunk],
    trace_ir: IR,
    ir_context: IRExecutionContext,
) -> None:
    """
    Unbounded fanout node for rapidsmpf with spilling support.

    Broadcasts chunks from input to all output channels. This is called
    "unbounded" because it handles the case where one channel may consume
    all data before another channel consumes any data.

    The implementation uses adaptive sending with spillable buffers:
    - Maintains a spillable FIFO buffer for each output channel
    - Messages are buffered in host memory (spillable to disk)
    - Sends to all channels concurrently
    - Receives next chunk as soon as any channel makes progress
    - Efficient for both balanced and imbalanced consumption patterns

    Parameters
    ----------
    context
        The rapidsmpf context.
    ch_in
        The input Channel[TableChunk].
    chs_out
        The output Channel[TableChunk]s.
    trace_ir
        The IR node to trace. Passed through to shutdown_on_error.
    ir_context
        The execution context for the IR node.
    """
    # TODO: Use rapidsmpf fanout node once available.
    # See: https://github.com/rapidsai/rapidsmpf/issues/560
    # TODO: Use ir_context
    async with shutdown_on_error(
        context, ch_in, *chs_out, trace_ir=trace_ir, ir_context=ir_context
    ):
        # Forward metadata to all outputs.
        metadata = await recv_metadata(ch_in, context)
        await gather_in_task_group(
            *(send_metadata(ch, context, metadata) for ch in chs_out)
        )

        # Spillable FIFO buffer for each output channel
        output_buffers: list[SpillableMessages] = [
            SpillableMessages(context.br()) for _ in chs_out
        ]
        num_outputs = len(chs_out)

        # Track message IDs in FIFO order for each output buffer
        buffer_ids: list[list[int]] = [[] for _ in chs_out]

        # Register a single spill function for all buffers
        # This ensures global FIFO ordering when spilling across all outputs
        spill_func_id = context.br().spill_manager.add_spill_function(
            make_spill_function(output_buffers, context), priority=0
        )

        # Track active send/drain tasks for each output
        active_tasks: dict[int, asyncio.Task] = {}

        # Track which outputs need to be drained (set when no more input)
        needs_drain: set[int] = set()

        # Receive task
        recv_task: asyncio.Task | None = asyncio.create_task(ch_in.recv(context))

        # Flag to indicate we should start a new receive (for backpressure)
        can_receive: bool = True

        try:

            async def send_one_from_buffer(idx: int) -> None:
                """
                Send one buffered message for output idx.

                The message remains in host memory (spillable) through the channel.
                The downstream consumer will call make_available() when needed.
                """
                if buffer_ids[idx]:
                    mid = buffer_ids[idx].pop(0)
                    msg = output_buffers[idx].extract(mid=mid)
                    await chs_out[idx].send(context, msg)

            async def drain_output(idx: int) -> None:
                """Drain output channel idx."""
                await chs_out[idx].drain(context)

            # Main loop: coordinate receiving, sending, and draining
            while (
                recv_task is not None or active_tasks or any(buffer_ids) or needs_drain
            ):
                # Collect all currently active tasks
                tasks_to_wait = list(active_tasks.values())
                # Only include recv_task if we're allowed to receive
                if recv_task is not None and can_receive:
                    tasks_to_wait.append(recv_task)

                # Start new tasks for outputs with work to do
                for idx in range(len(chs_out)):
                    if idx not in active_tasks:
                        if buffer_ids[idx]:
                            # Send next buffered message
                            task = asyncio.create_task(send_one_from_buffer(idx))
                            active_tasks[idx] = task
                            tasks_to_wait.append(task)
                        elif idx in needs_drain:
                            # Buffer empty and no more input - drain this output
                            task = asyncio.create_task(drain_output(idx))
                            active_tasks[idx] = task
                            tasks_to_wait.append(task)
                            needs_drain.discard(idx)

                # If nothing to wait for, we're done
                if not tasks_to_wait:
                    break

                # Wait for ANY task to complete
                done, _ = await asyncio.wait(
                    tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
                )

                # Process completed tasks
                for task in done:
                    if task is recv_task:
                        # Receive completed
                        msg = task.result()
                        if msg is None:
                            # End of input - mark all outputs as needing drain
                            recv_task = None
                            needs_drain.update(range(len(chs_out)))
                        else:
                            # Determine where to copy based on:
                            # 1. Current message location (avoid unnecessary transfers)
                            # 2. Available memory (avoid OOM)
                            content_desc = msg.get_content_description()
                            device_size = content_desc.content_sizes.get(
                                MemoryType.DEVICE, 0
                            )

                            # Check if we have enough device memory for all copies
                            # We need (num_outputs - 1) copies since last one reuses original
                            num_copies = num_outputs - 1
                            total_copy_cost = msg.copy_cost() * num_copies
                            available_device_mem = context.br().memory_available(
                                MemoryType.DEVICE
                            )

                            # Decide target memory:
                            # Use device ONLY if message is in device AND we have sufficient headroom.
                            if (
                                device_size > 0
                                and available_device_mem >= total_copy_cost
                            ):
                                # Use reserve_device_memory_and_spill to automatically trigger spilling
                                # if needed to make room for the copy
                                memory_reservation = (
                                    context.br().reserve_device_memory_and_spill(
                                        total_copy_cost,
                                        allow_overbooking=True,
                                    )
                                )
                            else:
                                # Use host memory for buffering - much safer
                                # Downstream consumers will make_available() when they need device memory
                                memory_reservation = context.br().reserve_or_fail(
                                    total_copy_cost,
                                    [MemoryType.PINNED_HOST, MemoryType.HOST],
                                )

                            # Copy message for each output buffer
                            # Copies are spillable and allow downstream consumers
                            # to control device memory allocation
                            for idx, sm in enumerate(output_buffers):
                                if idx < num_outputs - 1:
                                    # Copy to target memory and insert into spillable buffer
                                    mid = sm.insert(msg.copy(memory_reservation))
                                else:
                                    # Optimization: reuse the original message for last output
                                    # (no copy needed)
                                    mid = sm.insert(msg)
                                buffer_ids[idx].append(mid)

                            # Don't receive next chunk until at least one send completes
                            can_receive = False
                            recv_task = asyncio.create_task(ch_in.recv(context))
                    else:
                        # Must be a send or drain task - find which output and remove it
                        for idx, at in list(active_tasks.items()):
                            if at is task:
                                del active_tasks[idx]
                                # A send completed - allow receiving again
                                can_receive = True
                                break

        finally:
            # Cancel any outstanding tasks
            if recv_task is not None and not recv_task.done():
                recv_task.cancel()
            for task in active_tasks.values():
                if not task.done():
                    task.cancel()
            # Clean up spill function registration
            context.br().spill_manager.remove_spill_function(spill_func_id)


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

    if len(ir.children) == 1:
        # Single-channel default node
        nodes[ir] = [
            default_node_single(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
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
                rec.state["ir_context"],
                channels[ir].reserve_input_slot(),
                tuple(channels[c].reserve_output_slot() for c in ir.children),
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
    async with shutdown_on_error(context, ch_out, ir_context=ir_context, trace_ir=ir):
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
        await ch_out.send(context, Message(0, chunk))

        await ch_out.drain(context)


@generate_ir_sub_network.register(Empty)
def _(
    ir: Empty, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate network for Empty node - produces one empty chunk."""
    context = rec.state["context"]
    ir_context = rec.state["ir_context"]
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
        fanout_node: Any
        if fanout_info.unbounded:
            fanout_node = fanout_node_unbounded(
                rec.state["context"],
                channels[ir].reserve_output_slot(),
                *[manager.reserve_input_slot() for _ in range(count)],
                trace_ir=ir,
                ir_context=rec.state["ir_context"],
            )
        else:  # "bounded"
            fanout_node = fanout_node_bounded(
                rec.state["context"],
                channels[ir].reserve_output_slot(),
                *[manager.reserve_input_slot() for _ in range(count)],
                trace_ir=ir,
                ir_context=rec.state["ir_context"],
            )
        nodes[ir].append(fanout_node)
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
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
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
        context, ch_in, ch_out, ir_context=ir_context, trace_ir=ir
    ):
        # Drain metadata channel (we don't need it after this point)
        metadata = await recv_metadata(ch_in, context)
        if metadata_collector is not None:
            metadata_collector.append(metadata)

        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)

        await ch_out.drain(context)
