# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming as-of join actor and cursor."""

from __future__ import annotations

from collections import deque
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata, OrderKey, RowOrderingScope
from cudf_streaming.table_chunk import TableChunk, make_table_chunks_available_or_wait
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IR, AsofJoin
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.partitioning import (
    is_row_ordered_by,
    maybe_remap_partitioning,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    chunk_to_frame,
    concat_batch,
    empty_table_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    run_with_reservation,
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


async def _iter_channel_chunks(
    context: Context, channel: Channel[TableChunk]
) -> AsyncGenerator[TableChunk, None]:
    """Yield chunks from a channel until it drains."""
    while (msg := await channel.recv(context)) is not None:
        yield TableChunk.from_message(msg, br=context.br())


async def _iter_buffered_chunks(
    chunks: list[TableChunk],
) -> AsyncGenerator[TableChunk, None]:
    """Yield gathered chunks and clear the input list."""
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
    """Gather right chunks from all ranks in rank order."""
    async with AllGatherManager(context, comm, collective_id) as allgather:
        with allgather.inserting() as inserter:
            sequence_number = 0
            while (msg := await channel.recv(context)) is not None:
                await inserter.insert(
                    sequence_number, TableChunk.from_message(msg, br=context.br())
                )
                sequence_number += 1
        return await allgather.extract_chunks(ordered=True)


class _AscendingOrder:
    """Validate ascending keys across chunks and infer null placement if needed."""

    def __init__(
        self,
        ir_context: IRExecutionContext,
        candidates: tuple[tuple[plc.types.NullOrder, ...], ...],
    ) -> None:
        """Initialize order checks across successive chunks."""
        self.ir_context = ir_context
        self.previous: tuple[plc.Table, Any] | None = None
        self.candidates = candidates

    @property
    def null_orders(self) -> tuple[plc.types.NullOrder, ...]:
        """Return null placements consistent with the chunks seen so far."""
        return self.candidates[0]

    @staticmethod
    def endpoint(table: plc.Table, row: int, stream: Any) -> plc.Table:
        """Copy one key row for cross-chunk comparisons."""
        return plc.copying.slice(table, [row, row + 1], stream=stream)[0].copy(
            stream=stream
        )

    def check(self, table: plc.Table, upstream: Any) -> None:
        """Check this chunk and its transition from the previous chunk."""
        if not table.num_rows():
            return
        upstreams = (
            (upstream,) if self.previous is None else (self.previous[1], upstream)
        )
        with stream_ordered_after(
            self.ir_context.get_cuda_stream, upstreams=upstreams
        ) as stream:
            pair = (
                None
                if self.previous is None
                else plc.concatenate.concatenate(
                    [self.previous[0], self.endpoint(table, 0, stream)],
                    stream=stream,
                )
            )
            has_null = any(column.null_count() for column in table.columns()) or (
                self.previous is not None
                and any(column.null_count() for column in self.previous[0].columns())
            )
            candidates = self.candidates if has_null else self.candidates[:1]
            orders = [plc.types.Order.ASCENDING] * table.num_columns()
            valid = []
            for null_orders in candidates:
                null_order_list = list(null_orders)
                if not plc.sorting.is_sorted(
                    table, orders, null_order_list, stream=stream
                ):
                    continue
                if pair is not None and not plc.sorting.is_sorted(
                    pair, orders, null_order_list, stream=stream
                ):
                    continue
                valid.append(null_orders)
            if not valid:
                raise pl.exceptions.InvalidOperationError(
                    "argument in operation 'asof_join' is not sorted, "
                    "please sort the 'expr/series/column' first"
                )
            if has_null:
                self.candidates = tuple(valid)
            self.previous = self.endpoint(table, table.num_rows() - 1, stream), stream


class _AsofCursor:
    """
    Probe left chunks against successive right windows.

    Notes
    -----
    Time-major windows retain the last right row per group. Key-major windows
    retain only their final row, since earlier keys cannot recur. Left rows
    equal to a window's maximum key wait for the next window, so ties select
    the last physical right row even across chunks. Grouped input with neither
    ordering uses one prepared right table.
    """

    def __init__(
        self,
        context: Context,
        ir_context: IRExecutionContext,
        ir: AsofJoin,
        right_chunks: AsyncIterator[TableChunk],
        *,
        key_major: bool,
    ) -> None:
        """Set up the right stream and state for probing left chunks."""
        self.context = context
        self.ir_context = ir_context
        self.ir = ir
        self.left_time = ir.left_on.name
        self.right_time = ir.right_on.name
        self.key_major = key_major
        self.left_indices = names_to_indices(
            (*ir.left_by, self.left_time) if key_major else (self.left_time,),
            ir.children[0].schema,
        )
        self.right_indices = names_to_indices(
            (*ir.right_by, self.right_time) if key_major else (self.right_time,),
            ir.children[1].schema,
        )
        self.right_index = self.right_indices[-1]
        self.right_by_indices = names_to_indices(ir.right_by, ir.children[1].schema)
        self.right_chunks = right_chunks
        self.predecessor: DataFrame | None = None
        self.current: DataFrame | None = None
        candidates = (
            ((plc.types.NullOrder.BEFORE,) * len(self.right_indices),)
            if key_major
            else ((plc.types.NullOrder.BEFORE,), (plc.types.NullOrder.AFTER,))
        )
        self.right_order = _AscendingOrder(ir_context, candidates)
        self.maximum: plc.Table | None = None
        self.maximum_stream: Any = None
        self.joiner: plc.join.AsofJoin | None = None
        self.payload: DataFrame | None = None
        self.at_eof = False
        self.left_order = _AscendingOrder(ir_context, candidates)
        self.check_left_order = True

    @staticmethod
    def _keys(table: plc.Table, indices: tuple[int, ...]) -> plc.Table:
        """Select the columns that define cursor progress."""
        return plc.Table([table.columns()[index] for index in indices])

    @staticmethod
    def has_time_order(
        metadata: ChannelMetadata, index: int, scope: RowOrderingScope
    ) -> bool:
        """Return whether metadata promises ascending time order."""
        return any(
            is_row_ordered_by(
                metadata.partitioning,
                (OrderKey(index, plc.types.Order.ASCENDING, null_order),),
                scope=scope,
            )
            for null_order in (
                plc.types.NullOrder.BEFORE,
                plc.types.NullOrder.AFTER,
            )
        )

    @staticmethod
    def has_key_major_order(
        ir: AsofJoin,
        left: ChannelMetadata,
        right: ChannelMetadata,
        right_scope: RowOrderingScope,
    ) -> bool:
        """Check for libcudf-compatible ``(by..., on)`` ordering."""
        left_keys = (
            *names_to_indices(ir.left_by, ir.children[0].schema),
            names_to_indices((ir.left_on.name,), ir.children[0].schema)[0],
        )
        right_keys = (
            *names_to_indices(ir.right_by, ir.children[1].schema),
            names_to_indices((ir.right_on.name,), ir.children[1].schema)[0],
        )
        order = plc.types.Order.ASCENDING
        # libcudf's grouped as-of join requires null grouping keys first.
        null_order = plc.types.NullOrder.BEFORE
        return is_row_ordered_by(
            left.partitioning,
            tuple(OrderKey(index, order, null_order) for index in left_keys),
            scope=RowOrderingScope.RANK_LOCAL,
        ) and is_row_ordered_by(
            right.partitioning,
            tuple(OrderKey(index, order, null_order) for index in right_keys),
            scope=right_scope,
        )

    @classmethod
    def select_order(
        cls,
        ir: AsofJoin,
        left: ChannelMetadata,
        right: ChannelMetadata,
        nranks: int,
    ) -> Literal["time", "key", "prepared"]:
        """Choose a streaming order or full-right grouped preparation."""
        if not ir.right_by:
            return "time"
        right_scope = (
            RowOrderingScope.RANK_LOCAL
            if right.duplicated or nranks == 1
            else RowOrderingScope.GLOBAL
        )
        if cls.has_time_order(
            left,
            names_to_indices((ir.left_on.name,), ir.children[0].schema)[0],
            RowOrderingScope.RANK_LOCAL,
        ) and cls.has_time_order(
            right,
            names_to_indices((ir.right_on.name,), ir.children[1].schema)[0],
            right_scope,
        ):
            return "time"
        if cls.has_key_major_order(ir, left, right, right_scope):
            return "key"
        return "prepared"

    async def prepare_grouped_right(self) -> None:
        """Prepare a full right table when grouped time order is unproven."""
        chunks = [chunk async for chunk in self.right_chunks]
        right = (
            await concat_batch(
                chunks, self.context, self.ir.children[1].schema, self.ir_context
            )
            if chunks
            else empty_table_chunk(
                self.ir.children[1], self.context, self.ir_context.get_cuda_stream()
            )
        )
        self.current = chunk_to_frame(right, self.ir.children[1])
        self.at_eof = True
        self.check_left_order = False

    def _prepare(self, right: DataFrame) -> None:
        """Prepare the current right window for repeated probes."""
        ir = self.ir
        discarded = set(ir.right_by)
        if ir.options[2] and ir.left_on.name == ir.right_on.name:
            discarded.add(ir.right_on.name)
        self.joiner, self.payload = AsofJoin._prepare_right(
            right,
            right.column_map[self.right_time],
            ir.right_by,
            discarded,
            already_sorted=not ir.right_by or self.key_major,
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
        non_null = await run_with_reservation(
            self.context,
            self.ir_context,
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

    async def _last_right_rows(self, current: DataFrame) -> DataFrame:
        """Retain the latest right row per group, or the final row."""
        if self.ir.right_by and not self.key_major:
            last_rows = await run_with_reservation(
                self.context,
                self.ir_context,
                current._size_bytes,
                plc.stream_compaction.stable_distinct,
                current.table,
                list(self.right_by_indices),
                plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
                plc.types.NullEquality.EQUAL,
                plc.types.NanEquality.ALL_EQUAL,
                current.stream,
            )
        else:
            tail = current.slice((-1, 1))
            last_rows = await run_with_reservation(
                self.context,
                self.ir_context,
                tail._size_bytes,
                tail.table.copy,
                stream=tail.stream,
            )
        return DataFrame.from_table(
            last_rows, current.column_names, current.dtypes, stream=current.stream
        )

    async def _advance(self) -> None:
        """Retain right rows that may match a later left window."""
        if self.current is not None:
            self.predecessor = await self._last_right_rows(self.current)
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
            keys = self._keys(right_chunk.table_view(), self.right_indices)
            await self.ir_context.to_thread(
                self.right_order.check,
                keys,
                right_chunk.stream,
            )
            self.maximum = await self.ir_context.to_thread(
                self.right_order.endpoint,
                keys,
                right_chunk.shape[0] - 1,
                right_chunk.stream,
            )
            self.maximum_stream = right_chunk.stream
            right = chunk_to_frame(right_chunk, self.ir.children[1])
            if self.predecessor is not None:
                right = await run_with_reservation(
                    self.context,
                    self.ir_context,
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
        """Prepare the current right window only when needed."""
        if self.joiner is None:
            assert self.current is not None
            await run_with_reservation(
                self.context,
                self.ir_context,
                self.current._size_bytes
                * (2 if self.ir.right_by and not self.key_major else 1),
                self._prepare,
                self.current,
            )

    def _lower_bound(self, left: DataFrame) -> int:
        """Find where left keys reach the current right-window maximum."""
        assert self.maximum is not None
        with stream_ordered_after(
            self.ir_context.get_cuda_stream,
            upstreams=(left.stream, self.maximum_stream),
        ) as stream:
            end = plc.search.lower_bound(
                self._keys(left.table, self.left_indices),
                self.maximum,
                [plc.types.Order.ASCENDING] * len(self.left_indices),
                list(self.left_order.null_orders),
                stream=stream,
            )
            split = end.to_scalar(stream=stream).to_py(stream=stream)
            assert isinstance(split, int)
            return split

    async def _probe_part(self, part: DataFrame) -> DataFrame:
        """Probe a left slice against the current right window."""
        await self._ensure_prepared()
        assert self.joiner is not None
        assert self.payload is not None
        # Estimate repeated payload rows and the gather map.
        payload_rows = max(self.payload.num_rows, 1)
        output_payload = (
            self.payload._size_bytes * part.num_rows + payload_rows - 1
        ) // payload_rows
        return await run_with_reservation(
            self.context,
            self.ir_context,
            part._size_bytes
            + self.payload._size_bytes
            + output_payload
            + 8 * part.num_rows,
            AsofJoin._probe_prepared,
            part,
            self.payload,
            self.joiner,
            part.column_map[self.left_time],
            self.ir.left_by,
            self.ir.options[1],
            context=self.ir_context,
        )

    async def _owned_result(self, fragments: list[DataFrame]) -> TableChunk:
        """Assemble an output chunk that does not borrow the left input."""
        if not fragments:
            return empty_table_chunk(
                self.ir, self.context, self.ir_context.get_cuda_stream()
            )
        if len(fragments) == 1:
            (result,) = fragments
            owned = await run_with_reservation(
                self.context,
                self.ir_context,
                result._size_bytes,
                result.table.copy,
                stream=result.stream,
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
            self.ir.schema,
            self.ir_context,
        )

    async def probe(self, left_chunk: TableChunk) -> TableChunk:
        """
        Probe one left chunk, advancing right windows as needed.

        Keep the left chunk's row order and publish one owned output chunk.
        """
        ir = self.ir
        if self.check_left_order:
            await self.ir_context.to_thread(
                self.left_order.check,
                self._keys(left_chunk.table_view(), self.left_indices),
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
                    left,
                )
                count = max(0, end - offset)
                if not count:
                    await self._advance()
                    continue
            part = left.slice((offset, count))
            fragments.append(await self._probe_part(part))
            offset += count
        return await self._owned_result(fragments)

    async def validate_remaining_right(self) -> None:
        """Check the unused right tail for an ordering violation."""
        async for packed_chunk in self.right_chunks:
            chunk, _ = await make_table_chunks_available_or_wait(
                self.context, packed_chunk, reserve_extra=0, net_memory_delta=0
            )
            keys = self._keys(chunk.table_view(), self.right_indices)
            if keys.columns()[-1].null_count():
                keys = await run_with_reservation(
                    self.context,
                    self.ir_context,
                    sum(column.device_buffer_size() for column in keys.columns()),
                    plc.stream_compaction.drop_nulls,
                    keys,
                    [len(self.right_indices) - 1],
                    1,
                    stream=chunk.stream,
                )
            await self.ir_context.to_thread(
                self.right_order.check,
                keys,
                chunk.stream,
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
    rank probes its left chunks in time-major or key-major order when metadata
    proves either layout. Other grouped layouts prepare the full right table
    once before probing left chunks. Left ownership is preserved in all cases.
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
        _, _, coalesce, check_sortedness = ir.options
        discarded = set(ir.right_by)
        if coalesce and ir.left_on.name == ir.right_on.name:
            discarded.add(ir.right_on.name)
        has_payload = bool(ir.children[1].schema.keys() - discarded)
        skip_probe = not has_payload and (bool(ir.right_by) or not check_sortedness)
        mode = _AsofCursor.select_order(ir, left_metadata, right_metadata, comm.nranks)
        if tracer is not None:
            tracer.decision = "passthrough" if skip_probe else f"asof_{mode}"
            if left_metadata.duplicated:
                tracer.set_duplicated()
        if skip_probe:
            while await ch_right.recv(context) is not None:
                pass
        elif right_metadata.duplicated or comm.nranks == 1:
            right_source = _iter_channel_chunks(context, ch_right)
        else:
            # TODO: For large right inputs, use sparse exchange to send each rank
            # its needed right rows and predecessor state instead of broadcasting.
            right_chunks = await _gather_right(
                context, comm, ch_right, collective_id=right_gather_id
            )
            right_source = _iter_buffered_chunks(right_chunks)
        output_metadata = ChannelMetadata(
            local_count=left_metadata.local_count,
            partitioning=maybe_remap_partitioning(
                ir,
                left_metadata.partitioning,
                child_index=0,
                context=context,
                preserves_row_order=True,
            ),
            duplicated=left_metadata.duplicated,
        )
        await send_metadata(ch_out, context, output_metadata)
        if skip_probe:
            while (msg := await ch_left.recv(context)) is not None:
                await send_chunk(
                    context,
                    ch_out,
                    TableChunk.from_message(msg, br=context.br()),
                    msg.sequence_number,
                    tracer=tracer,
                )
        else:
            cursor = _AsofCursor(
                context, ir_context, ir, right_source, key_major=mode == "key"
            )
            async with aclosing(right_source):
                if mode == "prepared":
                    await cursor.prepare_grouped_right()
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
