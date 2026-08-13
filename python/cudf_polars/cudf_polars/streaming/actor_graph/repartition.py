# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-chunking logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.containers import DataFrame
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.dispatch import generate_ir_sub_network
from cudf_polars.streaming.actor_graph.nodes import shutdown_on_error
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    empty_table_chunk,
    recv_metadata,
    send_metadata,
)
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.utils import _concat

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator


@define_actor()
async def concatenate_node(
    context: Context,
    comm: Communicator,
    ir: Repartition,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    *,
    output_count: int,
    collective_id: int,
) -> None:
    """
    Concatenate node for rapidsmpf.

    This node reduces the number of chunks via tree-like concatenation.
    The interpretation of output_count depends on whether input is duplicated:

    - Duplicated input: Each rank reduces locally to output_count chunks.
      Output remains duplicated.
    - Non-duplicated, output_count=1: AllGather to produce single duplicated
      chunk across all ranks.
    - Non-duplicated, output_count>1: Local reduction, distribute chunks
      across ranks (local_count = ceil(output_count / nranks)).

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Repartition IR node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output Channel[TableChunk].
    ch_in
        The input Channel[TableChunk].
    output_count
        The expected global number of output chunks.
    collective_id
        Pre-allocated collective ID for this operation.
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        # Receive metadata.
        input_metadata = await recv_metadata(ch_in, context)
        nranks = comm.nranks

        # Interpret output_count as the GLOBAL target chunk count.
        # Calculate local target based on whether data is duplicated.
        if input_metadata.duplicated:
            # Duplicated input: each rank reduces locally to output_count chunks.
            # Output remains duplicated (identical on all ranks).
            local_output_count = output_count
            output_duplicated = True
        elif output_count == 1 and nranks > 1:
            # Special case: non-duplicated input reducing to 1 global chunk.
            # Requires AllGather, output becomes duplicated.
            local_output_count = 1
            output_duplicated = True
        else:
            # Non-duplicated input with output_count > 1 (or single rank).
            # Distribute chunks across ranks.
            local_output_count = max(1, math.ceil(output_count / nranks))
            output_duplicated = False

        # NOTE: For now, Repartiton (e.g. concatenate_node) always destroys
        # partitioning metadata. However, this may change when we support
        # partitioning types other than HashPartitioned. For example, when
        # we adopt multi-stage shuffling (a global shuffle between ranks,
        # followed by a local shuffle within each rank), some cases will
        # preserve global partitioning.

        # max_chunks corresponds to the number of input chunks we can
        # concatenate together per output chunk.
        # If None, we must concatenate everything into a single chunk.
        max_chunks: int | None = None
        if local_output_count > 1:
            # Make sure max_chunks is at least 2.
            max_chunks = max(
                2, math.ceil(input_metadata.local_count / local_output_count)
            )

        # Check if we need global communication (AllGather).
        need_global_repartition = (
            nranks > 1 and not input_metadata.duplicated and output_count == 1
        )

        chunks: list[TableChunk]
        msg: TableChunk | None
        if need_global_repartition:
            # Global repartitioning via AllGather to single duplicated chunk.

            # Send metadata.
            metadata = ChannelMetadata(
                local_count=local_output_count,
                duplicated=output_duplicated,
            )
            await send_metadata(ch_out, context, metadata)
            if tracer is not None and output_duplicated:
                tracer.set_duplicated()

            stream = context.br().stream_pool.get_stream()
            seq_num = 0
            allgather = AllGatherManager(context, comm, collective_id)
            with allgather.inserting() as inserter:
                while (msg := await ch_in.recv(context)) is not None:
                    inserter.insert(
                        seq_num, TableChunk.from_message(msg, br=context.br())
                    )
                    seq_num += 1
                    del msg

            # Extract concatenated result
            result_table = await allgather.extract_concatenated(
                stream, ir_context=ir_context
            )

            # If no chunks were gathered, result_table has 0 columns.
            # We need to create an empty table with the correct schema.
            if result_table.num_columns() == 0 and len(ir.schema) > 0:
                output_chunk = empty_table_chunk(ir, context, stream)
            else:
                output_chunk = TableChunk.from_pylibcudf_table(
                    result_table, stream, exclusive_view=True, br=context.br()
                )

            await send_chunk(context, ch_out, output_chunk, 0, tracer=tracer)
        else:
            # Local repartitioning (tree reduction).

            # Send metadata.
            metadata = ChannelMetadata(
                local_count=local_output_count,
                duplicated=output_duplicated,
            )
            await send_metadata(ch_out, context, metadata)
            if tracer is not None and output_duplicated:
                tracer.set_duplicated()

            # Local repartitioning
            seq_num = 0
            while True:
                chunks = []
                done_receiving = False

                # Collect chunks up to max_chunks or until end of stream
                while len(chunks) < (max_chunks or float("inf")):
                    msg = await ch_in.recv(context)
                    if msg is None:
                        done_receiving = True
                        break
                    chunks.append(TableChunk.from_message(msg, br=context.br()))

                if chunks:
                    chunks, extra = await make_table_chunks_available_or_wait(
                        context,
                        chunks,
                        reserve_extra=sum(chunk.data_alloc_size() for chunk in chunks),
                        net_memory_delta=0,
                    )
                    with opaque_memory_usage(extra):
                        df = _concat(
                            *(
                                DataFrame.from_table(
                                    chunk.table_view(),
                                    list(ir.schema.keys()),
                                    list(ir.schema.values()),
                                    chunk.stream,
                                )
                                for chunk in chunks
                            ),
                            context=ir_context,
                        )
                        if len(chunks) > 1:
                            # _concat reuses chunks[0] if len(chunks) == 1
                            del chunks
                    output_chunk = TableChunk.from_pylibcudf_table(
                        df.table,
                        df.stream,
                        exclusive_view=True,
                        br=context.br(),
                    )
                    await send_chunk(
                        context, ch_out, output_chunk, seq_num, tracer=tracer
                    )
                    seq_num += 1
                    del df

                # Break if we reached end of stream
                if done_receiving:
                    break

        await ch_out.drain(context)


@generate_ir_sub_network.register(Repartition)
def _(
    ir: Repartition, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Repartition node.

    partition_info = rec.state["partition_info"]
    if partition_info[ir].count > 1:
        count_output = partition_info[ir].count
        count_input = partition_info[ir.children[0]].count
        if count_input < count_output:
            raise ValueError("Repartitioning to more chunks is not supported.")

    # Process children
    nodes, channels = rec(ir.children[0])

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Look up the reserved shuffle ID for this operation
    collective_id = rec.state["collective_id_map"][ir][0]

    # Add python node
    nodes[ir] = [
        concatenate_node(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            output_count=partition_info[ir].count,
            collective_id=collective_id,
        )
    ]
    return nodes, channels
