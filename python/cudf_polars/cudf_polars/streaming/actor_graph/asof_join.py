# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming as-of join actor and cursor."""

from __future__ import annotations

import warnings
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata, OrderKey, RowOrderingScope
from cudf_streaming.table_chunk import TableChunk, make_table_chunks_available_or_wait
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.message import Message

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
    CUDF_ROW_LIMIT,
    ChannelManager,
    ChunkStore,
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
from cudf_polars.utils.cuda_stream import join_cuda_streams, stream_ordered_after
from cudf_polars.utils.dtypes import make_empty_column

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
    context: Context, chunks: ChunkStore
) -> AsyncGenerator[TableChunk, None]:
    """Yield gathered chunks from spillable storage."""
    for msg in chunks:
        yield TableChunk.from_message(msg, br=context.br())


async def _gather_right(
    context: Context,
    comm: Communicator,
    channel: Channel[TableChunk],
    *,
    collective_id: int,
) -> ChunkStore:
    """Gather right chunks into registered spillable storage in rank order."""
    store = ChunkStore(context)
    try:
        async with AllGatherManager(context, comm, collective_id) as allgather:
            with allgather.inserting() as inserter:
                sequence_number = 0
                while (msg := await channel.recv(context)) is not None:
                    await inserter.insert(
                        sequence_number, TableChunk.from_message(msg, br=context.br())
                    )
                    sequence_number += 1
            for sequence_number, chunk in enumerate(
                await allgather.extract_chunks(ordered=True)
            ):
                store.insert(Message(sequence_number, chunk))
    except BaseException:
        store.clear()
        raise
    return store


class _AscendingOrder:
    """Optionally validate ascending keys across chunks."""

    def __init__(
        self,
        ir_context: IRExecutionContext,
        *,
        validate: bool = True,
        track_first: bool = False,
    ) -> None:
        """Initialize order checks across successive chunks."""
        self.ir_context = ir_context
        self.first: tuple[plc.Table, Any] | None = None
        self.previous: tuple[plc.Table, Any] | None = None
        self.validate = validate
        self.track_first = track_first

    @staticmethod
    def endpoint(table: plc.Table, row: int, stream: Any) -> plc.Table:
        """Copy one key row for cross-chunk comparisons."""
        return plc.copying.slice(table, [row, row + 1], stream=stream)[0].copy(
            stream=stream
        )

    @staticmethod
    def reservation_size(table: plc.Table) -> int:
        """Estimate temporary storage for a few copied key rows."""
        row_bytes = sum(
            column.device_buffer_size() for column in table.columns()
        ) // max(table.num_rows(), 1)
        return max(1, 4 * row_bytes)

    def check(self, table: plc.Table, upstream: Any) -> None:
        """Observe this chunk and optionally validate its ascending order."""
        if not table.num_rows():
            return None
        upstreams = (
            (upstream,) if self.previous is None else (self.previous[1], upstream)
        )
        with stream_ordered_after(
            self.ir_context.get_cuda_stream, upstreams=upstreams
        ) as stream:
            if self.validate:
                pair = (
                    None
                    if self.previous is None
                    else plc.concatenate.concatenate(
                        [self.previous[0], self.endpoint(table, 0, stream)],
                        stream=stream,
                    )
                )
                orders = [plc.types.Order.ASCENDING] * table.num_columns()
                null_orders = [plc.types.NullOrder.BEFORE] * table.num_columns()
                if not plc.sorting.is_sorted(
                    table, orders, null_orders, stream=stream
                ) or (
                    pair is not None
                    and not plc.sorting.is_sorted(
                        pair, orders, null_orders, stream=stream
                    )
                ):
                    raise pl.exceptions.InvalidOperationError(
                        "argument in operation 'asof_join' is not sorted, "
                        "please sort the 'expr/series/column' first"
                    )
            if self.track_first and self.first is None:
                self.first = self.endpoint(table, 0, stream), stream
            self.previous = self.endpoint(table, table.num_rows() - 1, stream), stream


class _AsofCursor:
    """
    Probe left chunks against successive right windows.

    Notes
    -----
    Time-major windows retain the last right row per group. Key-major windows
    retain only their final row, since earlier keys cannot recur. Left rows
    equal to a window's maximum key wait for the next window, so ties select
    the last physical right row even across chunks. Grouped input without
    compatible ordering uses one prepared right table.
    """

    def __init__(
        self,
        context: Context,
        ir_context: IRExecutionContext,
        ir: AsofJoin,
        right_chunks: AsyncIterator[TableChunk],
        *,
        mode: Literal["time", "key", "prepared"],
        left_order_promised: bool,
        right_order_promised: bool,
        check_global_left_order: bool,
    ) -> None:
        """Set up right chunks and state for probing left chunks."""
        self.context = context
        self.ir_context = ir_context
        self.ir = ir
        self.left_time = ir.left_on.name
        self.right_time = ir.right_on.name
        self.mode = mode
        self.check_global_left_order = check_global_left_order
        self.left_indices = names_to_indices(
            (*ir.left_by, self.left_time) if mode == "key" else (self.left_time,),
            ir.children[0].schema,
        )
        self.right_indices = names_to_indices(
            (*ir.right_by, self.right_time) if mode == "key" else (self.right_time,),
            ir.children[1].schema,
        )
        self.right_index = self.right_indices[-1]
        self.right_by_indices = names_to_indices(ir.right_by, ir.children[1].schema)
        self.right_chunks = right_chunks
        self.predecessor: DataFrame | None = None
        self.current: DataFrame | None = None
        self.right_order = _AscendingOrder(
            ir_context, validate=ir.options[3] and not right_order_promised
        )
        self.joiner: plc.join.AsofJoin | None = None
        self.payload: DataFrame | None = None
        self.at_eof = False
        self.left_order = _AscendingOrder(
            ir_context,
            validate=ir.options[3] and not left_order_promised,
            track_first=check_global_left_order,
        )

    @staticmethod
    def _keys(table: plc.Table, indices: tuple[int, ...]) -> plc.Table:
        """Select the columns that define cursor progress."""
        return plc.Table([table.columns()[index] for index in indices])

    @staticmethod
    def _searchable_left_keys(
        keys: plc.Table, stream: Any
    ) -> tuple[plc.Table, plc.Column]:
        """Drop null as-of keys while retaining their original row positions."""
        row_positions = plc.filling.sequence(
            keys.num_rows(),
            plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=stream),
            plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=stream),
            stream=stream,
        )
        filtered = plc.stream_compaction.drop_nulls(
            plc.Table([*keys.columns(), row_positions]),
            [keys.num_columns() - 1],
            1,
            stream=stream,
        )
        return plc.Table(filtered.columns()[:-1]), filtered.columns()[-1]

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
    ) -> tuple[Literal["time", "key", "prepared"], bool, bool]:
        """Choose the join layout and identify trusted input ordering."""
        right_scope = (
            RowOrderingScope.RANK_LOCAL
            if right.duplicated or nranks == 1
            else RowOrderingScope.GLOBAL
        )
        left_time_ordered = cls.has_time_order(
            left,
            names_to_indices((ir.left_on.name,), ir.children[0].schema)[0],
            RowOrderingScope.RANK_LOCAL,
        )
        right_time_ordered = cls.has_time_order(
            right,
            names_to_indices((ir.right_on.name,), ir.children[1].schema)[0],
            right_scope,
        )
        if not ir.right_by or (left_time_ordered and right_time_ordered):
            return "time", left_time_ordered, right_time_ordered
        if cls.has_key_major_order(ir, left, right, right_scope):
            return "key", True, True
        return "prepared", False, False

    async def prepare_grouped_right(self) -> None:
        """Prepare a full right table when grouped time order is unproven."""
        chunks = []
        total_rows = 0
        async for chunk in self.right_chunks:
            total_rows += chunk.shape[0]
            if total_rows > CUDF_ROW_LIMIT:
                raise NotImplementedError(
                    "Streaming grouped as-of join needs a right table within "
                    "the cuDF row limit when input order is unknown"
                )
            chunks.append(chunk)
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
            already_sorted=not ir.right_by or self.mode == "key",
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
        if self.ir.right_by and self.mode != "key":
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
            await self.aclose()
        while True:
            chunk = await anext(self.right_chunks, None)
            if chunk is None:
                self.at_eof = True
                return
            right_chunk = await self._non_null_right_chunk(chunk)
            if right_chunk is None:
                continue
            keys = self._keys(right_chunk.table_view(), self.right_indices)
            await run_with_reservation(
                self.context,
                self.ir_context,
                self.right_order.reservation_size(keys),
                self.right_order.check,
                keys,
                right_chunk.stream,
            )
            right = chunk_to_frame(right_chunk, self.ir.children[1])
            if self.predecessor is not None:
                if self.predecessor.num_rows + right.num_rows > CUDF_ROW_LIMIT:
                    raise NotImplementedError(
                        "Streaming as-of join predecessor state exceeds "
                        "the cuDF row limit"
                    )
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
                * (2 if self.ir.right_by and self.mode != "key" else 1),
                self._prepare,
                self.current,
            )

    def _clear_prepared(self) -> None:
        """Release the native joiner on a worker thread."""
        self.joiner = None
        self.payload = None

    async def aclose(self) -> None:
        """Release a prepared joiner, including after actor cancellation."""
        if self.joiner is not None:
            await self.ir_context.to_thread(self._clear_prepared)

    def _lower_bound(
        self, left: DataFrame, keys: plc.Table, positions: plc.Column | None
    ) -> int:
        """Find the original row where left keys reach the right-window maximum."""
        assert self.right_order.previous is not None
        maximum, maximum_stream = self.right_order.previous
        if not keys.num_rows():
            return left.num_rows
        with stream_ordered_after(
            self.ir_context.get_cuda_stream,
            upstreams=(left.stream, maximum_stream),
        ) as stream:
            end = plc.search.lower_bound(
                keys,
                maximum,
                [plc.types.Order.ASCENDING] * len(self.left_indices),
                [plc.types.NullOrder.BEFORE] * len(self.left_indices),
                stream=stream,
            )
            if positions is not None:
                (end,) = plc.copying.gather(
                    plc.Table([positions]),
                    end,
                    plc.copying.OutOfBoundsPolicy.NULLIFY,
                    stream=stream,
                ).columns()
            split = end.to_scalar(stream=stream).to_py(stream=stream)
            if split is None:
                return left.num_rows
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

    async def observe_left(
        self, left_chunk: TableChunk
    ) -> tuple[plc.Table | None, plc.Column | None]:
        """Track non-null left keys for local and rank-boundary checks."""
        if self.mode == "prepared":
            return None, None
        keys = self._keys(left_chunk.table_view(), self.left_indices)
        positions: plc.Column | None = None
        if keys.columns()[-1].null_count():
            positions_size = (
                2 * plc.types.size_of(plc.types.SIZE_TYPE) * left_chunk.shape[0]
            )
            keys, positions = await run_with_reservation(
                self.context,
                self.ir_context,
                sum(column.device_buffer_size() for column in keys.columns())
                + positions_size,
                self._searchable_left_keys,
                keys,
                left_chunk.stream,
            )
        if self.left_order.validate or self.check_global_left_order:
            await run_with_reservation(
                self.context,
                self.ir_context,
                self.left_order.reservation_size(keys),
                self.left_order.check,
                keys,
                left_chunk.stream,
            )
        return keys, positions

    def _left_endpoint_chunk(self) -> TableChunk:
        """Copy this rank's first and last non-null left keys."""
        stream = self.ir_context.get_cuda_stream()
        if self.left_order.first is None:
            table = plc.Table(
                [make_empty_column(self.ir.children[0].schema[self.left_time], stream)]
            )
        else:
            first, first_stream = self.left_order.first
            assert self.left_order.previous is not None
            last, last_stream = self.left_order.previous
            with stream_ordered_after(
                self.ir_context.get_cuda_stream,
                upstreams=(first_stream, last_stream),
            ) as stream:
                table = plc.concatenate.concatenate([first, last], stream=stream)
        return TableChunk.from_pylibcudf_table(
            table, stream, exclusive_view=True, br=self.context.br()
        )

    async def validate_global_left_order(
        self, comm: Communicator, collective_id: int
    ) -> None:
        """Check rank endpoints after the left chunks have been emitted."""
        if not self.check_global_left_order:
            return
        first = self.left_order.first
        endpoint_bytes = (
            self.left_order.reservation_size(first[0]) if first is not None else 1
        )
        local = await run_with_reservation(
            self.context, self.ir_context, endpoint_bytes, self._left_endpoint_chunk
        )
        stream = self.ir_context.get_cuda_stream()
        async with AllGatherManager(self.context, comm, collective_id) as allgather:
            with allgather.inserting() as inserter:
                await inserter.insert(comm.rank, local)
            gathered = await allgather.extract_concatenated(
                stream, ordered=True, ir_context=self.ir_context
            )
        if not await run_with_reservation(
            self.context,
            self.ir_context,
            max(1, sum(column.device_buffer_size() for column in gathered.columns())),
            plc.sorting.is_sorted,
            gathered,
            [plc.types.Order.ASCENDING],
            [plc.types.NullOrder.BEFORE],
            stream=stream,
        ):
            raise pl.exceptions.InvalidOperationError(
                "argument in operation 'asof_join' is not sorted, "
                "please sort the 'expr/series/column' first"
            )

    async def probe(self, left_chunk: TableChunk) -> TableChunk:
        """
        Probe one left chunk, advancing right windows as needed.

        Keep the left chunk's row order and publish one owned output chunk.
        """
        ir = self.ir
        search_keys, positions = await self.observe_left(left_chunk)
        left = chunk_to_frame(left_chunk, ir.children[0])
        fragments: list[DataFrame] = []
        offset = 0
        try:
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
                    assert search_keys is not None
                    end = await run_with_reservation(
                        self.context,
                        self.ir_context,
                        4 * left.num_rows + 32,
                        self._lower_bound,
                        left,
                        search_keys,
                        positions,
                    )
                    count = max(0, end - offset)
                    if not count:
                        await self._advance()
                        continue
                part = left.slice((offset, count))
                fragments.append(await self._probe_part(part))
                offset += count
            return await self._owned_result(fragments)
        finally:
            if fragments:
                join_cuda_streams(
                    downstreams=(left.stream,),
                    upstreams=tuple(part.stream for part in fragments),
                )

    async def validate_remaining_right(self) -> None:
        """Drain unused right chunks and check their order if requested."""
        async for packed_chunk in self.right_chunks:
            if not self.right_order.validate:
                continue
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
            await run_with_reservation(
                self.context,
                self.ir_context,
                self.right_order.reservation_size(keys),
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
    once before probing left chunks and must fit within the cuDF row limit.
    Left ownership is preserved in all cases. With sortedness checking disabled,
    the cursor trusts input order.
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
        if ir.right_by and check_sortedness and comm.rank == 0:
            warnings.warn(
                "Sortedness of columns cannot be checked when 'by' groups provided",
                UserWarning,
                stacklevel=2,
            )
        discarded = set(ir.right_by)
        if coalesce and ir.left_on.name == ir.right_on.name:
            discarded.add(ir.right_on.name)
        has_payload = bool(ir.children[1].schema.keys() - discarded)
        skip_probe = not has_payload and (bool(ir.right_by) or not check_sortedness)
        mode, left_order_promised, right_order_promised = _AsofCursor.select_order(
            ir, left_metadata, right_metadata, comm.nranks
        )
        check_global_left_order = (
            check_sortedness
            and not ir.right_by
            and comm.nranks > 1
            and not left_metadata.duplicated
            and not _AsofCursor.has_time_order(
                left_metadata,
                names_to_indices((ir.left_on.name,), ir.children[0].schema)[0],
                RowOrderingScope.GLOBAL,
            )
        )
        if tracer is not None:
            tracer.decision = "passthrough" if skip_probe else f"asof_{mode}"
            if left_metadata.duplicated:
                tracer.set_duplicated()
        right_store: ChunkStore | None = None
        cursor: _AsofCursor | None = None
        try:
            if skip_probe:
                while await ch_right.recv(context) is not None:
                    pass
            elif right_metadata.duplicated or comm.nranks == 1:
                right_source = _iter_channel_chunks(context, ch_right)
            else:
                # TODO: For large right inputs, use sparse exchange to send each rank
                # its needed right rows and predecessor state instead of broadcasting.
                right_store = await _gather_right(
                    context, comm, ch_right, collective_id=right_gather_id
                )
                right_source = _iter_buffered_chunks(context, right_store)
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
                    context,
                    ir_context,
                    ir,
                    right_source,
                    mode=mode,
                    left_order_promised=left_order_promised,
                    right_order_promised=right_order_promised,
                    check_global_left_order=check_global_left_order,
                )
                async with aclosing(right_source):
                    if mode == "prepared" and has_payload:
                        await cursor.prepare_grouped_right()
                    while (msg := await ch_left.recv(context)) is not None:
                        chunk = TableChunk.from_message(msg, br=context.br())
                        chunk, _ = await make_table_chunks_available_or_wait(
                            context, chunk, reserve_extra=0, net_memory_delta=0
                        )
                        if has_payload:
                            result = await cursor.probe(chunk)
                        else:
                            await cursor.observe_left(chunk)
                            result = chunk
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
                    await cursor.validate_global_left_order(comm, right_gather_id)
            await ch_out.drain(context)
        finally:
            try:
                if cursor is not None:
                    await cursor.aclose()
            finally:
                if right_store is not None:
                    right_store.clear()


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
