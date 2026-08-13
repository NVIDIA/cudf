# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Union logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.ir import Union
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.streaming.actor_graph.nodes import define_actor, shutdown_on_error
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    empty_table_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    send_metadata,
)

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.core import SubNetGenerator


@define_actor()
async def union_node(
    context: Context,
    comm: Communicator,
    ir: Union,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    *chs_in: Channel[TableChunk],
) -> None:
    """
    Union node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Union IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    chs_in
        The input Channel[TableChunk]s.
    """
    async with shutdown_on_error(
        context, *chs_in, ch_out, trace_ir=ir, ir_context=ir_context
    ):
        # Merge and forward metadata.
        # Union loses partitioning/ordering info since sources may differ.
        # TODO: Warn users that Union does NOT preserve order?
        metadata = await gather_in_task_group(
            *(recv_metadata(ch, context) for ch in chs_in)
        )
        # Chunk counts on the wire are uniform across ranks, so report the
        # full sum.
        total_local_count = sum(meta.local_count for meta in metadata)
        duplicated = all(meta.duplicated for meta in metadata)
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(
                local_count=total_local_count,
                duplicated=duplicated,
            ),
        )

        # When a child has duplicated=True, every rank has produced the same
        # rows, so we drop them everywhere except rank 0 to avoid N counting.
        suppress = tuple(meta.duplicated and comm.rank != 0 for meta in metadata)

        seq_num_offset = 0
        for ch_in, drop in zip(chs_in, suppress, strict=True):
            num_ch_chunks = 0
            while (msg := await ch_in.recv(context)) is not None:
                if drop:
                    stream = ir_context.get_cuda_stream()
                    out_chunk = empty_table_chunk(ir, context, stream)
                else:
                    out_chunk = TableChunk.from_message(
                        msg, br=context.br()
                    ).make_available_and_spill(context.br(), allow_overbooking=True)
                await ch_out.send(
                    context,
                    Message(
                        msg.sequence_number + seq_num_offset,
                        out_chunk,
                    ),
                )
                num_ch_chunks += 1
            seq_num_offset += num_ch_chunks

        await ch_out.drain(context)


@generate_ir_sub_network.register(Union)
def _(
    ir: Union, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Union operation.
    # Pass-through all child chunks in channel order.

    # Process children
    nodes, channels = process_children(ir, rec)

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Add simple python node
    nodes[ir] = [
        union_node(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            *[channels[c].reserve_output_slot() for c in ir.children],
        )
    ]
    return nodes, channels
