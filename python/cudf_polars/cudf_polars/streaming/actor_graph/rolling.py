# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.dsl.ir import IR, Rolling
from cudf_polars.streaming.actor_graph.collectives.overlap import (
    BufferedChunk,
    OverlapInput,
    RangeOverlapPolicy,
    RowCountOverlapPolicy,
)
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    evaluate_chunk,
    maybe_remap_partitioning,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.rolling import FixedSizeRolling
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.collectives.overlap import (
        OverlapPolicy,
        StagedChunk,
    )
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer


@dataclass
class RollingManager[BufferedChunkT: BufferedChunk]:
    """
    Rolling policy and rank-agnostic local executor.

    Owned chunks produce output. Ghost chunks provide history or future
    context only.
    """

    context: Context
    ir: Rolling | FixedSizeRolling
    ir_context: IRExecutionContext
    policy: OverlapPolicy[BufferedChunkT]

    def __enter__(self) -> RollingManager[BufferedChunkT]:
        """Enter the rolling manager context."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release policy-owned resources."""
        self.policy.close()

    @staticmethod
    def _latest_nonempty_chunk(
        current: BufferedChunkT,
        future: deque[StagedChunk[BufferedChunkT]],
    ) -> BufferedChunkT:
        """Return the latest nonempty chunk available to the cursor."""
        for staged in reversed(future):
            if staged.chunk.num_rows != 0:
                return staged.chunk
        return current

    async def _fill_future(
        self,
        chunks: AsyncIterator[StagedChunk[BufferedChunkT]],
        current: BufferedChunkT,
        future: deque[StagedChunk[BufferedChunkT]],
    ) -> bool:
        """Read chunks until current has enough following context."""
        while not self.policy.has_complete_future(
            self._latest_nonempty_chunk(current, future),
            current,
        ):
            chunk = await anext(chunks, None)
            if chunk is None:
                return True
            future.append(chunk)
        return False

    async def _evaluate_cursor(
        self,
        cursor: BufferedChunkT,
        chunks: Sequence[BufferedChunkT],
    ) -> TableChunk:
        """
        Evaluate one owned cursor with its required overlap.

        ``chunks`` use one row coordinate system for this rank. The policy
        returns a half-open span in that system. The result is trimmed back to
        rows owned by ``cursor``.
        """
        start, stop = await self.policy.overlap_bounds(self.context, cursor, chunks)
        if (start, stop) == (cursor.row_start, cursor.row_stop):
            return await evaluate_chunk(
                self.context,
                cursor.chunk,
                self.ir,
                ir_context=self.ir_context,
                already_available=True,
            )
        pieces = []
        for item in chunks:
            begin, end = max(start, item.row_start), min(stop, item.row_stop)
            if begin < end:
                pieces.append(
                    TableChunk.from_pylibcudf_table(
                        plc.copying.slice(
                            item.chunk.table_view(),
                            [begin - item.row_start, end - item.row_start],
                            stream=item.chunk.stream,
                        )[0],
                        item.chunk.stream,
                        exclusive_view=False,
                        br=self.context.br(),
                    )
                )
        assert pieces
        if len(pieces) == 1:
            evaluation_chunk = pieces[0]
        else:
            reservation = await self.context.memory(MemoryType.DEVICE).reserve_or_wait(
                sum(piece.data_alloc_size() for piece in pieces),
                net_memory_delta=0,
            )
            with (
                opaque_memory_usage(reservation),
                stream_ordered_after(
                    self.ir_context.get_cuda_stream,
                    [piece.stream for piece in pieces],
                ) as stream,
            ):
                table = plc.concatenate.concatenate(
                    [piece.table_view() for piece in pieces],
                    stream=stream,
                    mr=self.context.br().device_mr,
                )
                evaluation_chunk = TableChunk.from_pylibcudf_table(
                    table,
                    stream,
                    exclusive_view=True,
                    br=self.context.br(),
                )
        result = await evaluate_chunk(
            self.context,
            evaluation_chunk,
            self.ir,
            ir_context=self.ir_context,
            already_available=True,
        )
        (table,) = plc.copying.slice(
            result.table_view(),
            [cursor.row_start - start, cursor.row_stop - start],
            stream=result.stream,
        )
        reservation = await self.context.memory(MemoryType.DEVICE).reserve_or_wait(
            result.data_alloc_size(), net_memory_delta=0
        )
        with opaque_memory_usage(reservation):
            table = table.copy(result.stream, self.context.br().device_mr)
        return TableChunk.from_pylibcudf_table(
            table,
            result.stream,
            exclusive_view=True,
            br=self.context.br(),
        )

    async def execute(
        self,
        ch_out: Channel[TableChunk],
        tracer: ActorTracer | None,
        rolling_input: OverlapInput[BufferedChunkT],
    ) -> None:
        """Evaluate owned chunks. Ghost chunks provide context but no output."""
        history: list[BufferedChunkT] = []
        future: deque[StagedChunk[BufferedChunkT]] = deque()
        input_exhausted = False
        output_sequence_number = 0
        chunks = aiter(rolling_input)
        staged = await anext(chunks, None)
        while staged is not None:
            chunk = staged.chunk
            self.policy.validate_chunk(chunk)
            if not staged.owned:
                history.append(chunk)
            elif chunk.num_rows == 0:
                result = await evaluate_chunk(
                    self.context,
                    chunk.chunk,
                    self.ir,
                    ir_context=self.ir_context,
                    already_available=True,
                )
                await send_chunk(
                    self.context,
                    ch_out,
                    result,
                    output_sequence_number,
                    tracer=tracer,
                )
                output_sequence_number += 1
            else:
                cursor = chunk
                history = self.policy.evict_history(history, cursor)
                if not input_exhausted:
                    input_exhausted = await self._fill_future(
                        chunks,
                        cursor,
                        future,
                    )
                result = await self._evaluate_cursor(
                    cursor,
                    [*history, cursor, *(item.chunk for item in future)],
                )
                retained = await self.policy.history_chunk(
                    self.context,
                    cursor,
                )
                if retained is not None:
                    history.append(retained)
                await send_chunk(
                    self.context,
                    ch_out,
                    result,
                    output_sequence_number,
                    tracer=tracer,
                )
                output_sequence_number += 1
            if future:
                staged = future.popleft()
            else:
                staged = await anext(chunks, None)
        await ch_out.drain(self.context)


@define_actor()
async def rolling_actor(
    context: Context,
    comm: Communicator,
    ir: Rolling | FixedSizeRolling,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    *,
    collective_id: int,
) -> None:
    """Streaming actor for rolling operations."""
    async with shutdown_on_error(
        context, chs_in=(ch_in,), chs_out=(ch_out,), trace_ir=ir, ir_context=ir_context
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        metadata_in = await recv_metadata(ch_in, context)
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=maybe_remap_partitioning(
                    ir,
                    metadata_in.partitioning,
                    child_ir=ir.children[0],
                    context=context,
                ),
                duplicated=metadata_in.duplicated,
            ),
        )
        if tracer is not None and metadata_in.duplicated:
            tracer.set_duplicated()

        async def _execute_policy[BufferedChunkT: BufferedChunk](
            policy: OverlapPolicy[BufferedChunkT],
        ) -> None:
            """Prepare input and execute one concrete overlap policy."""
            with RollingManager(context, ir, ir_context, policy) as manager:
                async with OverlapInput(
                    context,
                    comm,
                    ch_in,
                    policy,
                    duplicated=metadata_in.duplicated,
                    collective_id=collective_id,
                ) as rolling_input:
                    await manager.execute(ch_out, tracer, rolling_input)

        if isinstance(ir, Rolling):
            await _execute_policy(
                RangeOverlapPolicy.from_ir(ir, ir_context.get_cuda_stream())
            )
        else:
            await _execute_policy(
                RowCountOverlapPolicy(ir.preceding_overlap, ir.following_overlap)
            )


def generate_rolling_sub_network(
    ir: Rolling | FixedSizeRolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for rolling operations."""
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    actors[ir] = [
        rolling_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context_for_node(rec, ir),
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_id=collective_id,
        )
    ]
    return actors, channels


@generate_ir_sub_network.register(FixedSizeRolling)
def _(
    ir: FixedSizeRolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for fixed-size rolling expressions."""
    return generate_rolling_sub_network(ir, rec)


@generate_ir_sub_network.register(Rolling)
def _(
    ir: Rolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for a Rolling operation."""
    if len(ir.keys) > 0 or ir.zlice is not None:
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    return generate_rolling_sub_network(ir, rec)
