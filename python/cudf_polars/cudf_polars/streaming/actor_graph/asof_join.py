# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming as-of join actor and cursor."""

from __future__ import annotations

from collections import deque
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import TableChunk, make_table_chunks_available_or_wait
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, AsofJoin
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.partitioning import maybe_remap_partitioning
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    chunk_to_frame,
    concat_batch,
    empty_table_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.join import _has_direct_asof_on
from cudf_polars.streaming.utils import _concat
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator


async def _iter_right_chunks(
    context: Context, channel: Channel[TableChunk]
) -> AsyncGenerator[TableChunk, None]:
    while (msg := await channel.recv(context)) is not None:
        yield TableChunk.from_message(msg, br=context.br())


async def _iter_buffered_chunks(
    chunks: list[TableChunk],
) -> AsyncGenerator[TableChunk, None]:
    pending = deque(chunks)
    chunks.clear()
    while pending:
        yield pending.popleft()


async def _gather_right(
    context: Context,
    comm: Communicator,
    channel: Channel[TableChunk],
    *,
    collective_id: int,
) -> list[TableChunk]:
    allgather = AllGatherManager(context, comm, collective_id)
    with allgather.inserting() as inserter:
        while (msg := await channel.recv(context)) is not None:
            await inserter.insert(
                msg.sequence_number, TableChunk.from_message(msg, br=context.br())
            )
    return await allgather.extract_chunks(ordered=True)


class _TimeOrder:
    """Validate ascending chunks and infer null placement as rows arrive."""

    def __init__(self, ir_context: IRExecutionContext) -> None:
        self.ir_context = ir_context
        self.previous: tuple[plc.Table, Any] | None = None
        self.null_orders: tuple[plc.types.NullOrder, ...] = (
            plc.types.NullOrder.BEFORE,
            plc.types.NullOrder.AFTER,
        )

    @property
    def null_order(self) -> plc.types.NullOrder:
        return self.null_orders[0]

    @staticmethod
    def endpoint(column: plc.Column, row: int, stream: Any) -> plc.Table:
        return plc.copying.slice(plc.Table([column]), [row, row + 1], stream=stream)[
            0
        ].copy(stream=stream)

    def check(self, column: plc.Column, upstream: Any) -> None:
        """Check this chunk and its transition from the previous chunk."""
        if not column.size():
            return
        upstreams = (
            (upstream,) if self.previous is None else (self.previous[1], upstream)
        )
        with stream_ordered_after(
            self.ir_context.get_cuda_stream, upstreams=upstreams
        ) as stream:
            table = plc.Table([column])
            pair = (
                None
                if self.previous is None
                else plc.concatenate.concatenate(
                    [self.previous[0], self.endpoint(column, 0, stream)],
                    stream=stream,
                )
            )
            has_null = column.null_count() or (
                self.previous is not None and self.previous[0].columns()[0].null_count()
            )
            candidates = self.null_orders if has_null else self.null_orders[:1]
            valid = tuple(
                null_order
                for null_order in candidates
                if plc.sorting.is_sorted(
                    table,
                    [plc.types.Order.ASCENDING],
                    [null_order],
                    stream=stream,
                )
                and (
                    pair is None
                    or plc.sorting.is_sorted(
                        pair,
                        [plc.types.Order.ASCENDING],
                        [null_order],
                        stream=stream,
                    )
                )
            )
            if not valid:
                raise pl.exceptions.InvalidOperationError(
                    "argument in operation 'asof_join' is not sorted, "
                    "please sort the 'expr/series/column' first"
                )
            if has_null:
                self.null_orders = valid
            self.previous = self.endpoint(column, column.size() - 1, stream), stream


class _AsofCursor:
    """
    Probe left chunks against successive time-ordered right windows.

    Notes
    -----
    Each window includes the last right row per group from earlier windows.
    Left rows equal to a window's maximum time wait for the next window, so
    equal-time matches select the last physical right row even across chunks.
    """

    def __init__(
        self,
        context: Context,
        ir_context: IRExecutionContext,
        ir: AsofJoin,
        right_chunks: AsyncIterator[TableChunk],
    ) -> None:
        self.context = context
        self.ir_context = ir_context
        self.ir = ir
        self.left_time = ir.left_on.name
        self.right_time = ir.right_on.name
        self.left_index = names_to_indices((self.left_time,), ir.children[0].schema)[0]
        self.right_index = names_to_indices((self.right_time,), ir.children[1].schema)[
            0
        ]
        self.right_by_indices = names_to_indices(ir.right_by, ir.children[1].schema)
        self.right_chunks = right_chunks
        self.predecessor: DataFrame | None = None
        self.current: DataFrame | None = None
        self.right_order = _TimeOrder(ir_context)
        self.maximum: plc.Table | None = None
        self.maximum_stream: Any = None
        self.joiner: plc.join.AsofJoin | None = None
        self.payload: DataFrame | None = None
        self.at_eof = False
        self.left_order = _TimeOrder(ir_context)

    async def _reserved(
        self, size: int, func: Any, /, *args: Any, **kwargs: Any
    ) -> Any:
        reservation = await reserve_memory(self.context, size=size, net_memory_delta=0)
        with opaque_memory_usage(reservation):
            return await self.ir_context.to_thread(func, *args, **kwargs)

    def _prepare(self, right: DataFrame) -> None:
        ir = self.ir
        discarded = set(ir.right_by)
        if ir.options[2] and ir.left_on.name == ir.right_on.name:
            discarded.add(ir.right_on.name)
        self.joiner, self.payload = AsofJoin._prepare_right(
            right,
            right.column_map[self.right_time],
            ir.right_by,
            discarded,
            already_sorted=not ir.right_by,
        )

    async def _non_null_right_chunk(self, chunk: TableChunk) -> TableChunk | None:
        """Make one right chunk available and discard rows with null time."""
        chunk, _ = await make_table_chunks_available_or_wait(
            self.context, chunk, reserve_extra=0, net_memory_delta=0
        )
        if not chunk.shape[0]:
            return None
        if not chunk.table_view().columns()[self.right_index].null_count():
            return chunk
        non_null = await self._reserved(
            chunk.data_alloc_size(),
            plc.stream_compaction.drop_nulls,
            chunk.table_view(),
            [self.right_index],
            1,
            stream=chunk.stream,
        )
        if non_null.num_rows() == 0:
            return None
        return TableChunk.from_pylibcudf_table(
            non_null, chunk.stream, exclusive_view=True, br=self.context.br()
        )

    async def _advance(self) -> None:
        """Retain each group's last right row before loading the next window."""
        if self.current is not None:
            if self.ir.right_by:
                last_rows = await self._reserved(
                    self.current._size_bytes,
                    plc.stream_compaction.stable_distinct,
                    self.current.table,
                    list(self.right_by_indices),
                    plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
                    plc.types.NullEquality.EQUAL,
                    plc.types.NanEquality.ALL_EQUAL,
                    self.current.stream,
                )
                self.predecessor = DataFrame.from_table(
                    last_rows,
                    self.current.column_names,
                    self.current.dtypes,
                    stream=self.current.stream,
                )
            else:
                tail = self.current.slice((-1, 1))
                last_row = await self._reserved(
                    tail._size_bytes, tail.table.copy, stream=tail.stream
                )
                self.predecessor = DataFrame.from_table(
                    last_row, tail.column_names, tail.dtypes, stream=tail.stream
                )
            self.current = None
            self.joiner = None
            self.payload = None
        while True:
            chunk = await anext(self.right_chunks, None)
            if chunk is None:
                self.at_eof = True
                self.maximum = None
                return
            right_chunk = await self._non_null_right_chunk(chunk)
            if right_chunk is None:
                continue
            await self.ir_context.to_thread(
                self.right_order.check,
                right_chunk.table_view().columns()[self.right_index],
                right_chunk.stream,
            )
            time_col = right_chunk.table_view().columns()[self.right_index]
            self.maximum = await self.ir_context.to_thread(
                self.right_order.endpoint,
                time_col,
                right_chunk.shape[0] - 1,
                right_chunk.stream,
            )
            self.maximum_stream = right_chunk.stream
            right = chunk_to_frame(right_chunk, self.ir.children[1])
            if self.predecessor is not None:
                right = await self._reserved(
                    self.predecessor._size_bytes + right._size_bytes,
                    _concat,
                    self.predecessor,
                    right,
                    context=self.ir_context,
                )
                self.predecessor = None
            self.current = right
            return

    async def _ensure_prepared(self) -> None:
        if self.joiner is None:
            assert self.current is not None
            await self._reserved(
                self.current._size_bytes * (2 if self.ir.right_by else 1),
                self._prepare,
                self.current,
            )

    def _lower_bound(self, column: plc.Column, left_stream: Any) -> int:
        assert self.maximum is not None
        with stream_ordered_after(
            self.ir_context.get_cuda_stream,
            upstreams=(left_stream, self.maximum_stream),
        ) as stream:
            end = plc.search.lower_bound(
                plc.Table([column]),
                self.maximum,
                [plc.types.Order.ASCENDING],
                [self.left_order.null_order],
                stream=stream,
            )
            split = end.to_scalar(stream=stream).to_py(stream=stream)
            assert isinstance(split, int)
            return split

    async def probe(self, left_chunk: TableChunk) -> TableChunk:
        """
        Probe one left chunk, advancing right windows as needed.

        Keep the left chunk's row order and publish one owned output chunk.
        """
        ir = self.ir
        await self.ir_context.to_thread(
            self.left_order.check,
            left_chunk.table_view().columns()[self.left_index],
            left_chunk.stream,
        )
        left = chunk_to_frame(left_chunk, ir.children[0])
        fragments: list[DataFrame] = []
        offset = 0
        while offset < left.num_rows:
            if self.current is None and not self.at_eof:
                await self._advance()
            if self.at_eof:
                if self.current is None:
                    if self.predecessor is None:
                        empty = empty_table_chunk(
                            ir.children[1],
                            self.context,
                            self.ir_context.get_cuda_stream(),
                        )
                        self.current = chunk_to_frame(empty, ir.children[1])
                    else:
                        self.current = self.predecessor
                count = left.num_rows - offset
            else:
                assert self.maximum is not None
                end = await self.ir_context.to_thread(
                    self._lower_bound,
                    left.column_map[self.left_time].obj,
                    left.stream,
                )
                count = max(0, end - offset)
                if not count:
                    await self._advance()
                    continue
            part = left.slice((offset, count))
            await self._ensure_prepared()
            assert self.joiner is not None
            assert self.payload is not None
            # Estimate repeated payload rows and the gather map.
            payload_rows = max(self.payload.num_rows, 1)
            output_payload = (
                self.payload._size_bytes * part.num_rows + payload_rows - 1
            ) // payload_rows
            fragment = await self._reserved(
                part._size_bytes
                + self.payload._size_bytes
                + output_payload
                + 8 * part.num_rows,
                AsofJoin._probe_prepared,
                part,
                self.payload,
                self.joiner,
                part.column_map[self.left_time],
                ir.left_by,
                ir.options[1],
                context=self.ir_context,
            )
            fragments.append(fragment)
            offset += count
        if not fragments:
            return empty_table_chunk(
                ir, self.context, self.ir_context.get_cuda_stream()
            )
        if len(fragments) == 1:
            (result,) = fragments
            owned = await self._reserved(
                result._size_bytes, result.table.copy, stream=result.stream
            )
            return TableChunk.from_pylibcudf_table(
                owned, result.stream, exclusive_view=True, br=self.context.br()
            )
        return await concat_batch(
            [
                TableChunk.from_pylibcudf_table(
                    part.table,
                    part.stream,
                    exclusive_view=False,
                    br=self.context.br(),
                )
                for part in fragments
            ],
            self.context,
            ir.schema,
            self.ir_context,
        )

    async def validate_remaining_right(self) -> None:
        """Check the unused right tail for an ordering violation."""
        async for packed_chunk in self.right_chunks:
            chunk, _ = await make_table_chunks_available_or_wait(
                self.context, packed_chunk, reserve_extra=0, net_memory_delta=0
            )
            time_col = chunk.table_view().columns()[self.right_index]
            if time_col.null_count():
                non_null = await self._reserved(
                    time_col.device_buffer_size(),
                    plc.stream_compaction.drop_nulls,
                    plc.Table([time_col]),
                    [0],
                    1,
                    stream=chunk.stream,
                )
                time_col = non_null.columns()[0]
            await self.ir_context.to_thread(
                self.right_order.check, time_col, chunk.stream
            )


@define_actor()
async def asof_join_actor(
    context: Context,
    comm: Communicator,
    ir: AsofJoin,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_left: Channel[TableChunk],
    ch_right: Channel[TableChunk],
    right_gather_id: int,
) -> None:
    """
    Execute a streaming as-of join while preserving left-row ownership.

    Notes
    -----
    The current backward path broadcasts right chunks to every rank. Each
    rank probes its left chunks with a cursor and preserves their ownership.
    The cursor checks physical time order as chunks arrive. It does not need
    boundary metadata.
    """
    async with shutdown_on_error(
        context,
        chs_in=(ch_left, ch_right),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        left_metadata, right_metadata = await gather_in_task_group(
            recv_metadata(ch_left, context), recv_metadata(ch_right, context)
        )
        assert _has_direct_asof_on(ir), "The asof-join actor requires direct keys."
        if tracer is not None:
            tracer.decision = "asof_cursor"
            if left_metadata.duplicated:
                tracer.set_duplicated()
        if right_metadata.duplicated or comm.nranks == 1:
            right_source = _iter_right_chunks(context, ch_right)
        else:
            right_chunks = await _gather_right(
                context, comm, ch_right, collective_id=right_gather_id
            )
            right_source = _iter_buffered_chunks(right_chunks)
        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(
                local_count=left_metadata.local_count,
                partitioning=maybe_remap_partitioning(
                    ir,
                    left_metadata.partitioning,
                    child_index=0,
                    context=context,
                    preserves_row_order=True,
                ),
                duplicated=left_metadata.duplicated,
            ),
        )
        cursor = _AsofCursor(context, ir_context, ir, right_source)
        async with aclosing(right_source):
            while (msg := await ch_left.recv(context)) is not None:
                chunk = TableChunk.from_message(msg, br=context.br())
                chunk, _ = await make_table_chunks_available_or_wait(
                    context, chunk, reserve_extra=0, net_memory_delta=0
                )
                result = await cursor.probe(chunk)
                await send_chunk(
                    context,
                    ch_out,
                    result,
                    msg.sequence_number,
                    tracer=tracer,
                )
                del chunk, result
            # A later disorder cannot change a completed match, but must fail
            # the query rather than silently return a potentially wrong result.
            await cursor.validate_remaining_right()
        await ch_out.drain(context)


@generate_ir_sub_network.register(AsofJoin)
def generate_asof_join_sub_network(
    ir: AsofJoin, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Select the dedicated as-of actor or the single-partition default actor."""
    if not _has_direct_asof_on(ir):
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    right_gather_id = rec.state["collective_id_map"][ir][0]
    actors[ir] = [
        asof_join_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context_for_node(rec, ir),
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            channels[ir.children[1]].reserve_output_slot(),
            right_gather_id,
        )
    ]
    return actors, channels
