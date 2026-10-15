# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pylibcudf as plc
from cudf_streaming.channel_metadata import ChannelMetadata
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.dsl.ir import IR, Rolling
from cudf_polars.dsl.utils.windows import duration_to_scalar
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    empty_table_chunk,
    evaluate_chunk,
    maybe_remap_partitioning,
    names_to_indices,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.rolling import FixedSizeRolling
from cudf_polars.streaming.utils import _fallback_inform
from cudf_polars.utils.cuda_stream import join_cuda_streams, stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.message import Message
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


@dataclass
class BufferedChunk:
    """An input chunk with its global row span."""

    sequence_number: int
    chunk: TableChunk
    row_start: int
    num_rows: int

    @property
    def row_stop(self) -> int:
        """Global row offset immediately after this chunk."""
        return self.row_start + self.num_rows


@dataclass
class RangeOverlap:
    """Range-window metadata for one buffered input chunk."""

    index_column: plc.Column
    lower_bound_column: plc.Column
    upper_bound_column: plc.Column
    first: Any | None
    last: Any | None
    lower_bound: Any | None
    upper_bound: Any | None


@dataclass
class RangeBufferedChunk(BufferedChunk):
    """A buffered chunk with range-window overlap metadata."""

    overlap: RangeOverlap


@dataclass
class RollingManager:
    """Prepared rolling policy and input stream."""

    context: Context
    ir: Rolling | FixedSizeRolling
    ir_context: IRExecutionContext
    ch_in: Channel[TableChunk]
    policy: _RollingPolicy[Any] = field(init=False)

    def __post_init__(self) -> None:
        """Choose the policy for this rolling node."""
        if isinstance(self.ir, Rolling):
            self.policy = RangeOverlapPolicy.from_ir(
                self.ir, self.ir_context.get_cuda_stream()
            )
        else:
            self.policy = RowCountOverlapPolicy(
                self.ir.preceding_overlap, self.ir.following_overlap
            )

    def __enter__(self) -> RollingManager:
        """Enter the rolling manager context."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release policy-owned resources."""
        self.policy.close()

    async def input_chunks(self) -> AsyncIterator[Any]:
        """Yield input chunks prepared for rolling execution."""
        row_offset = 0
        while (msg := await self.ch_in.recv(self.context)) is not None:
            chunk = await self.policy.prepare_chunk(
                self.context, msg, row_offset=row_offset
            )
            self.policy.validate_cursor(self.context, chunk)
            self.policy.observe(chunk)
            row_offset = chunk.row_stop
            yield chunk

    @staticmethod
    async def _next_chunk(chunks: AsyncIterator[Any]) -> Any | None:
        try:
            return await anext(chunks)
        except StopAsyncIteration:
            return None

    @staticmethod
    def _latest_nonempty_chunk(current: Any, future: list[Any]) -> Any:
        for chunk in reversed(future):
            if chunk.num_rows != 0:
                return chunk
        return current

    async def _fill_future(
        self,
        chunks: AsyncIterator[Any],
        current: Any,
        future: list[Any],
    ) -> bool:
        """Read chunks until current has enough leading context."""
        while not self.policy.has_complete_future(
            self._latest_nonempty_chunk(current, future),
            current,
            self.context,
        ):
            chunk = await self._next_chunk(chunks)
            if chunk is None:
                return True
            future.append(chunk)
        return False

    async def execute(
        self, ch_out: Channel[TableChunk], tracer: ActorTracer | None
    ) -> None:
        """Evaluate this rolling node over its prepared input stream."""
        history: list[Any] = []
        future: list[Any] = []
        input_exhausted = False
        chunks = self.input_chunks()
        cursor = await self._next_chunk(chunks)
        while cursor is not None:
            if cursor.num_rows == 0:
                result = await evaluate_chunk(
                    self.context,
                    cursor.chunk,
                    self.ir,
                    ir_context=self.ir_context,
                    already_available=True,
                )
            else:
                history = self.policy.evict_history(history, cursor, self.context)
                if not input_exhausted:
                    input_exhausted = await self._fill_future(
                        chunks,
                        cursor,
                        future,
                    )
                result = await self.policy.evaluate_cursor(
                    self.context,
                    self.ir,
                    self.ir_context,
                    cursor,
                    history=history,
                    future=future,
                )
                retained = await self.policy.history_chunk(
                    self.context,
                    cursor,
                    ir_context=self.ir_context,
                )
                if retained is not None:
                    history.append(retained)

            await send_chunk(
                self.context, ch_out, result, cursor.sequence_number, tracer=tracer
            )
            if future:
                cursor = future.pop(0)
            else:
                cursor = await self._next_chunk(chunks)
        await ch_out.drain(self.context)


class _RollingPolicy[BufferedChunkT: BufferedChunk]:
    """Base policy for evaluating ghost-expanded cursor chunks."""

    def observe(self, chunk: BufferedChunkT) -> None:
        del chunk

    def close(self) -> None:
        pass

    def validate_cursor(self, context: Context, cursor: BufferedChunkT) -> None:
        del context, cursor

    def evict_history(
        self,
        history: list[BufferedChunkT],
        cursor: BufferedChunkT,
        context: Context,
    ) -> list[BufferedChunkT]:
        raise NotImplementedError

    def has_complete_future(
        self,
        latest: BufferedChunkT,
        current: BufferedChunkT,
        context: Context,
    ) -> bool:
        raise NotImplementedError

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunkT,
        *,
        ir_context: IRExecutionContext,
    ) -> BufferedChunkT | None:
        """Return the cursor rows that future cursors may need."""
        del context, ir_context
        return cursor

    async def evaluate_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: BufferedChunkT,
        *,
        history: list[BufferedChunkT],
        future: list[BufferedChunkT],
    ) -> TableChunk:
        raise NotImplementedError

    async def prepare_chunk(
        self,
        context: Context,
        msg: Message,
        *,
        row_offset: int,
    ) -> BufferedChunkT:
        raise NotImplementedError

    async def _extract_region(
        self,
        context: Context,
        input_chunks: Sequence[BufferedChunk],
        row_start: int,
        row_stop: int,
        *,
        ir_context: IRExecutionContext,
    ) -> TableChunk:
        """Slice all buffered chunks intersecting a global row range."""
        chunks: list[TableChunk] = []
        for buf in input_chunks:
            if buf.row_start >= row_stop:
                break
            start = max(row_start, buf.row_start)
            stop = min(row_stop, buf.row_stop)
            if start < stop:
                if start == buf.row_start and stop == buf.row_stop:
                    chunks.append(buf.chunk)
                else:
                    chunks.append(
                        TableChunk.from_pylibcudf_table(
                            plc.copying.slice(
                                buf.chunk.table_view(),
                                # Translate to chunk-local row coordinates
                                [
                                    start - buf.row_start,
                                    stop - buf.row_start,
                                ],
                                stream=buf.chunk.stream,
                            )[0],
                            buf.chunk.stream,
                            exclusive_view=False,
                            br=context.br(),
                        )
                    )
        assert chunks, "Should have found at least one chunk to extract from"
        if len(chunks) == 1:
            return chunks[0]
        reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
            sum(chunk.data_alloc_size() for chunk in chunks), net_memory_delta=0
        )
        chunk_streams = [chunk.stream for chunk in chunks]
        with (
            opaque_memory_usage(reservation),
            stream_ordered_after(ir_context.get_cuda_stream, chunk_streams) as stream,
        ):
            table = plc.concatenate.concatenate(
                [chunk.table_view() for chunk in chunks],
                stream=stream,
                mr=context.br().device_mr,
            )
            return TableChunk.from_pylibcudf_table(
                table, stream=stream, exclusive_view=True, br=context.br()
            )

    async def _evaluate_ghosted_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: BufferedChunk,
        *,
        chunks: Sequence[BufferedChunk],
        ghost_start: int,
        ghost_stop: int,
    ) -> TableChunk:
        """Evaluate a cursor chunk with surrounding ghost rows."""
        if ghost_start == cursor.row_start and ghost_stop == cursor.row_stop:
            return await evaluate_chunk(
                context,
                cursor.chunk,
                ir,
                ir_context=ir_context,
                already_available=True,
            )
        ghosted_chunk = await self._extract_region(
            context, chunks, ghost_start, ghost_stop, ir_context=ir_context
        )
        result = await evaluate_chunk(
            context,
            ghosted_chunk,
            ir,
            ir_context=ir_context,
            already_available=True,
        )
        (table,) = plc.copying.slice(
            result.table_view(),
            [cursor.row_start - ghost_start, cursor.row_stop - ghost_start],
            stream=result.stream,
        )
        reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
            result.data_alloc_size(), net_memory_delta=0
        )
        with opaque_memory_usage(reservation):
            table = table.copy(result.stream, context.br().device_mr)
        return TableChunk.from_pylibcudf_table(
            table,
            result.stream,
            exclusive_view=True,
            br=context.br(),
        )


@dataclass
class RangeOverlapPolicy(_RollingPolicy[RangeBufferedChunk]):
    """Overlap policy for range-based rolling windows."""

    lower: plc.Scalar
    upper: plc.Scalar
    index: int
    index_dtype: plc.DataType
    find_start: Callable[..., plc.Column]
    find_end: Callable[..., plc.Column]
    start_closed: bool
    end_closed: bool
    stream: Stream
    index_name: str
    observed_streams: set[Stream] = field(default_factory=set)
    previous_index_value: Any | None = None

    @classmethod
    def from_ir(cls, ir: Rolling, stream: Stream) -> RangeOverlapPolicy:
        """Create reusable range-bound state for the rolling actor."""
        (index,) = names_to_indices([ir.index.name], ir.children[0].schema)
        side = ir.closed_window
        start_closed = side in ("both", "left")
        end_closed = side in ("both", "right")
        dtype = ir.index_dtype
        return cls(
            # Note: not using windows_to_offsets because that flips the sign of
            # preceding_ordinal.
            duration_to_scalar(dtype, ir.preceding_ordinal, stream=stream),
            duration_to_scalar(
                dtype,
                ir.preceding_ordinal + ir.following_ordinal,
                stream=stream,
            ),
            index,
            dtype,
            plc.search.lower_bound if start_closed else plc.search.upper_bound,
            plc.search.upper_bound if end_closed else plc.search.lower_bound,
            start_closed,
            end_closed,
            stream,
            ir.index.name,
        )

    def observe(self, chunk: RangeBufferedChunk) -> None:
        """Track streams that own range-bound columns."""
        self.observed_streams.add(chunk.chunk.stream)

    def close(self) -> None:
        """Keep offset scalars alive until all observed work is ordered."""
        if self.observed_streams:
            join_cuda_streams(
                downstreams=(self.stream,),
                upstreams=tuple(self.observed_streams),
            )

    @staticmethod
    def _index_with_offset(
        index: plc.Column,
        row: int,
        offset: plc.Scalar,
        dtype: plc.DataType,
        stream: Stream,
        br: BufferResource,
    ) -> plc.Column:
        """Return ``index[row] + offset`` as a single-row device column."""
        (endpoint,) = plc.copying.slice(index, [row, row + 1], stream=stream)
        if endpoint.type() != dtype:
            endpoint = plc.unary.cast(endpoint, dtype, stream=stream)
        return plc.binaryop.binary_operation(
            endpoint,
            offset,
            plc.binaryop.BinaryOperator.ADD,
            dtype,
            stream=stream,
            mr=br.device_mr,
        )

    @staticmethod
    def _chrono_storage_dtype(dtype: plc.DataType) -> plc.DataType:
        """Return the integer storage type for a chrono dtype."""
        if dtype.id() in (plc.TypeId.TIMESTAMP_DAYS, plc.TypeId.DURATION_DAYS):
            return plc.DataType(plc.TypeId.INT32)
        return plc.DataType(plc.TypeId.INT64)

    @classmethod
    def _host_ordering_value(
        cls,
        value: plc.Column,
        *,
        stream: Stream,
        br: BufferResource,
    ) -> Any:
        """Copy a single-row ordering value to host."""
        if plc.traits.is_chrono(value.type()):
            value = plc.unary.bit_cast(
                value,
                cls._chrono_storage_dtype(value.type()),
                stream=stream,
                mr=br.device_mr,
            )
        return value.to_scalar(stream=stream).to_py(stream=stream)

    @classmethod
    def _host_ordering_value_at(
        cls,
        column: plc.Column,
        row: int,
        *,
        stream: Stream,
        br: BufferResource,
    ) -> Any:
        """Copy one ordering value from a device column to host."""
        (value,) = plc.copying.slice(column, [row, row + 1], stream=stream)
        return cls._host_ordering_value(value, stream=stream, br=br)

    @staticmethod
    async def _global_insertion_row(
        chunks: Sequence[RangeBufferedChunk],
        needle: plc.Column,
        needle_value: Any,
        find: Callable[..., plc.Column],
        *,
        upper_bound: bool,
        dtype: plc.DataType,
        context: Context,
        needle_stream: Stream,
    ) -> int:
        """Return the globally indexed insertion row of a needle in some chunks."""
        assert len(chunks) > 0
        for chunk in chunks:
            if chunk.num_rows == 0:
                continue
            assert chunk.overlap.last is not None
            if (
                chunk.overlap.last <= needle_value
                if upper_bound
                else chunk.overlap.last < needle_value
            ):
                continue
            stream = chunk.chunk.stream
            join_cuda_streams(downstreams=[stream], upstreams=[needle_stream])
            index_column = chunk.overlap.index_column
            needs_cast = index_column.type() != dtype
            memory_scope = nullcontext()
            if needs_cast:
                reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
                    chunk.num_rows * 8, net_memory_delta=0
                )
                memory_scope = opaque_memory_usage(reservation)
            with memory_scope:
                if needs_cast:
                    index_column = plc.unary.cast(index_column, dtype, stream=stream)
                # Since this returns a python integer, the work queued on
                # search stream is complete, so we don't need to join back
                # to the search and needle streams.
                insertion_value = (
                    find(
                        plc.Table([index_column]),
                        plc.Table([needle]),
                        [plc.types.Order.ASCENDING],
                        [plc.types.NullOrder.AFTER],
                        stream=stream,
                        mr=context.br().device_mr,
                    )
                    .to_scalar(stream=stream)
                    .to_py(stream=stream)
                )
                assert isinstance(insertion_value, int)
                insertion_point = insertion_value
            if insertion_point < chunk.num_rows:
                return chunk.row_start + insertion_point
        # Needle is later than all the chunks we know about.
        return chunks[-1].row_stop

    def validate_index_column(self, index_column: plc.Column, stream: Stream) -> None:
        """Validate chunk-local ordering before using search primitives."""
        if index_column.null_count() != 0:
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling may not contain nulls"
            )
        if index_column.size() > 1 and not plc.sorting.is_sorted(
            plc.Table([index_column]),
            [plc.types.Order.ASCENDING],
            [plc.types.NullOrder.BEFORE],
            stream=stream,
        ):
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling is not sorted, "
                "please sort first"
            )

    def validate_cursor(self, context: Context, cursor: RangeBufferedChunk) -> None:
        """Check that the index remains sorted across chunk boundaries."""
        del context
        if cursor.num_rows == 0:
            return
        assert cursor.overlap.first is not None
        assert cursor.overlap.last is not None
        if (
            self.previous_index_value is not None
            and cursor.overlap.first < self.previous_index_value
        ):
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling is not sorted, "
                "please sort first"
            )
        self.previous_index_value = cursor.overlap.last

    async def prepare_chunk(
        self,
        context: Context,
        msg: Message,
        *,
        row_offset: int,
    ) -> RangeBufferedChunk:
        """Convert a message to a staged chunk and extract range metadata."""
        chunk = TableChunk.from_message(msg, br=context.br())
        nrows, _ = chunk.shape
        chunk, extra = await make_table_chunks_available_or_wait(
            context,
            chunk,
            reserve_extra=32,
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            index_column = chunk.table_view().columns()[self.index]
            self.validate_index_column(index_column, chunk.stream)
            if nrows == 0:
                overlap = RangeOverlap(
                    index_column, index_column, index_column, None, None, None, None
                )
            else:
                join_cuda_streams(downstreams=(chunk.stream,), upstreams=(self.stream,))
                lower_bound = self._index_with_offset(
                    index_column,
                    0,
                    self.lower,
                    self.index_dtype,
                    chunk.stream,
                    context.br(),
                )
                upper_bound = self._index_with_offset(
                    index_column,
                    nrows - 1,
                    self.upper,
                    self.index_dtype,
                    chunk.stream,
                    context.br(),
                )
                overlap = RangeOverlap(
                    index_column,
                    lower_bound,
                    upper_bound,
                    self._host_ordering_value_at(
                        index_column, 0, stream=chunk.stream, br=context.br()
                    ),
                    self._host_ordering_value_at(
                        index_column,
                        nrows - 1,
                        stream=chunk.stream,
                        br=context.br(),
                    ),
                    self._host_ordering_value(
                        lower_bound, stream=chunk.stream, br=context.br()
                    ),
                    self._host_ordering_value(
                        upper_bound, stream=chunk.stream, br=context.br()
                    ),
                )
        return RangeBufferedChunk(
            msg.sequence_number, chunk, row_offset, nrows, overlap
        )

    def evict_history(
        self,
        history: list[RangeBufferedChunk],
        cursor: RangeBufferedChunk,
        context: Context,
    ) -> list[RangeBufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        del context
        if not history:
            return []
        assert cursor.overlap.lower_bound is not None
        if self.start_closed:
            return [
                chunk
                for chunk in history
                if chunk.overlap.last is not None
                and chunk.overlap.last >= cursor.overlap.lower_bound
            ]
        return [
            chunk
            for chunk in history
            if chunk.overlap.last is not None
            and chunk.overlap.last > cursor.overlap.lower_bound
        ]

    def has_complete_future(
        self,
        latest: RangeBufferedChunk,
        current: RangeBufferedChunk,
        context: Context,
    ) -> bool:
        """Return whether latest contains current's upper insertion point."""
        del context
        if latest.overlap.last is None:
            return False
        assert current.overlap.upper_bound is not None
        if self.end_closed:
            return latest.overlap.last > current.overlap.upper_bound
        return latest.overlap.last >= current.overlap.upper_bound

    async def evaluate_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: RangeBufferedChunk,
        *,
        history: list[RangeBufferedChunk],
        future: list[RangeBufferedChunk],
    ) -> TableChunk:
        """Evaluate the rolling aggregation for the cursor chunk."""
        chunks = [*history, cursor, *future]
        assert cursor.overlap.lower_bound is not None
        assert cursor.overlap.upper_bound is not None
        ghost_start = await self._global_insertion_row(
            chunks,
            cursor.overlap.lower_bound_column,
            cursor.overlap.lower_bound,
            self.find_start,
            upper_bound=not self.start_closed,
            dtype=self.index_dtype,
            context=context,
            needle_stream=cursor.chunk.stream,
        )
        ghost_stop = await self._global_insertion_row(
            chunks,
            cursor.overlap.upper_bound_column,
            cursor.overlap.upper_bound,
            self.find_end,
            upper_bound=self.end_closed,
            dtype=self.index_dtype,
            context=context,
            needle_stream=cursor.chunk.stream,
        )
        # We must extract at least the whole of the current cursor chunk.
        ghost_start = min(cursor.row_start, ghost_start)
        ghost_stop = max(cursor.row_stop, ghost_stop)
        return await self._evaluate_ghosted_cursor(
            context,
            ir,
            ir_context,
            cursor,
            chunks=chunks,
            ghost_start=ghost_start,
            ghost_stop=ghost_stop,
        )


@dataclass
class RowCountOverlapPolicy(_RollingPolicy[BufferedChunk]):
    """Overlap policy for fixed-size rolling expressions."""

    preceding: int
    following: int

    async def prepare_chunk(
        self,
        context: Context,
        msg: Message,
        *,
        row_offset: int,
    ) -> BufferedChunk:
        """Convert a message to an available chunk with row-span metadata."""
        chunk = TableChunk.from_message(msg, br=context.br())
        nrows, _ = chunk.shape
        chunk, _ = await make_table_chunks_available_or_wait(
            context, chunk, reserve_extra=0, net_memory_delta=0
        )
        return BufferedChunk(msg.sequence_number, chunk, row_offset, nrows)

    def evict_history(
        self,
        history: list[BufferedChunk],
        cursor: BufferedChunk,
        context: Context,
    ) -> list[BufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        del context
        ghost_start = max(0, cursor.row_start - self.preceding)
        return [chunk for chunk in history if chunk.row_stop > ghost_start]

    def has_complete_future(
        self,
        latest: BufferedChunk,
        current: BufferedChunk,
        context: Context,
    ) -> bool:
        """Return whether latest contains enough leading rows."""
        del context
        return latest.row_stop >= current.row_stop + self.following

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunk,
        *,
        ir_context: IRExecutionContext,
    ) -> BufferedChunk | None:
        """Keep only emitted rows that a future fixed-size window can use."""
        del ir_context
        if self.preceding == 0 or cursor.num_rows == 0:
            return None
        start = max(cursor.row_start, cursor.row_stop - self.preceding)
        if start == cursor.row_start:
            return cursor

        stream = cursor.chunk.stream
        (table,) = plc.copying.slice(
            cursor.chunk.table_view(),
            [start - cursor.row_start, cursor.num_rows],
            stream=stream,
        )
        reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
            cursor.chunk.data_alloc_size(), net_memory_delta=0
        )
        with opaque_memory_usage(reservation):
            table = table.copy(stream=stream, mr=context.br().device_mr)
        return BufferedChunk(
            cursor.sequence_number,
            TableChunk.from_pylibcudf_table(
                table,
                stream,
                exclusive_view=True,
                br=context.br(),
            ),
            start,
            cursor.row_stop - start,
        )

    async def evaluate_cursor(
        self,
        context: Context,
        ir: IR,
        ir_context: IRExecutionContext,
        cursor: BufferedChunk,
        *,
        history: list[BufferedChunk],
        future: list[BufferedChunk],
    ) -> TableChunk:
        """Evaluate fixed-size rolling expressions for the cursor chunk."""
        chunks = [*history, cursor, *future]
        ghost_start = max(0, cursor.row_start - self.preceding)
        ghost_stop = min(cursor.row_stop + self.following, chunks[-1].row_stop)
        return await self._evaluate_ghosted_cursor(
            context,
            ir,
            ir_context,
            cursor,
            chunks=chunks,
            ghost_start=ghost_start,
            ghost_stop=ghost_stop,
        )


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
    config_options: ConfigOptions[StreamingExecutor],
) -> None:
    """
    Streaming actor for rolling operations.

    Multi-rank input currently falls back to all-gather evaluation. The
    rank-local path stages input chunks, evaluates each output chunk with enough
    ghost rows to satisfy the requested window, and emits chunks in order.
    """
    async with shutdown_on_error(
        context, chs_in=(ch_in,), chs_out=(ch_out,), trace_ir=ir, ir_context=ir_context
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        metadata_in = await recv_metadata(ch_in, context)
        if comm.nranks != 1 and not metadata_in.duplicated:
            _fallback_inform(
                "Rolling does not support multi-rank inputs. "
                "Falling back to all-gather evaluation.",
                config_options,
            )
            metadata = ChannelMetadata(
                local_count=1, partitioning=None, duplicated=True
            )
            await send_metadata(ch_out, context, metadata)
            if tracer is not None:
                tracer.set_duplicated()

            stream = ir_context.get_cuda_stream()
            ag = AllGatherManager(context, comm, collective_id)
            with ag.inserting() as inserter:
                while (msg := await ch_in.recv(context)) is not None:
                    chunk = TableChunk.from_message(msg, context.br())
                    await inserter.insert(msg.sequence_number, chunk)
            table = await ag.extract_concatenated(
                stream, ordered=True, ir_context=ir_context
            )
            if table.num_columns() == 0 and len(ir.children[0].schema) > 0:
                chunk = empty_table_chunk(ir.children[0], context, stream)
            else:
                chunk = TableChunk.from_pylibcudf_table(
                    table, stream, exclusive_view=True, br=context.br()
                )
            result = await evaluate_chunk(
                context,
                chunk,
                ir,
                ir_context=ir_context,
                already_available=True,
            )
            await send_chunk(context, ch_out, result, 0, tracer=tracer)
            await ch_out.drain(context)
            return

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

        with RollingManager(context, ir, ir_context, ch_in) as manager:
            await manager.execute(ch_out, tracer)


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
            config_options=rec.state["config_options"],
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
