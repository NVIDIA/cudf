# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass
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
from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.ir import IR, Rolling
from cudf_polars.dsl.utils.windows import duration_to_scalar
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.dispatch import generate_ir_sub_network
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    _evaluate_chunk_sync,
    empty_table_chunk,
    names_to_indices,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.utils import _fallback_inform
from cudf_polars.utils.cuda_stream import join_cuda_streams, stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import Callable

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


@dataclass
class BufferedChunk:
    """
    An input chunk, with additional metadata about location in the global frame.

    Consider a stream of chunks representing a frame with N total rows.
    Each BufferedChunk represents a non-overlapping slice of that frame,
    corresponding to the rows [row_start, row_stop)
    """

    sequence_number: int
    chunk: TableChunk
    index_column: plc.Column
    row_start: int
    num_rows: int
    # Single row columns describing the lower and upper bound of values in
    # the index column that can contribute to an aggregation over this
    # chunk.
    lower_bound: plc.Column
    upper_bound: plc.Column

    @property
    def row_stop(self) -> int:
        """Global row offset immediately after this chunk."""
        return self.row_start + self.num_rows


@dataclass
class Window:
    """
    Chunk-independent portion of the window description.

    The lower and upper scalars are the values added to the endpoints of a
    chunk's index column to obtain the bounding box for the aggregation.
    """

    lower: plc.Scalar
    upper: plc.Scalar
    index: int
    index_dtype: plc.DataType
    find_start: Callable[..., plc.Column]
    find_end: Callable[..., plc.Column]
    stream: Stream


def index_with_offset(
    index: plc.Column,
    row: int,
    offset: plc.Scalar,
    stream: Stream,
    br: BufferResource,
) -> plc.Column:
    """Return ``index[row] + offset`` as a single-row device column."""
    (endpoint,) = plc.copying.slice(index, [row, row + 1], stream=stream)
    return plc.binaryop.binary_operation(
        endpoint,
        offset,
        plc.binaryop.BinaryOperator.ADD,
        index.type(),
        stream=stream,
        mr=br.device_mr,
    )


async def prepare_chunk(
    context: Context,
    msg: Message,
    *,
    row_offset: int,
    window: Window,
    observed_streams: set[Stream] | None = None,
) -> BufferedChunk:
    """Convert a message to a staged chunk and extract its physical index."""
    chunk = TableChunk.from_message(msg, br=context.br())
    nrows, _ = chunk.shape
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        # TODO: Only reserve if needing to cast index column
        reserve_extra=nrows * 8,
        net_memory_delta=0,
    )
    if observed_streams is not None:
        observed_streams.add(chunk.stream)
    with opaque_memory_usage(extra):
        index_column = chunk.table_view().columns()[window.index]
        if index_column.type() != window.index_dtype:
            index_column = plc.unary.cast(
                index_column, window.index_dtype, stream=chunk.stream
            )
    if nrows == 0:
        lower_bound = upper_bound = index_column
    else:
        lower_bound = index_with_offset(
            index_column, 0, window.lower, chunk.stream, context.br()
        )
        upper_bound = index_with_offset(
            index_column, nrows - 1, window.upper, chunk.stream, context.br()
        )
    return BufferedChunk(
        msg.sequence_number,
        chunk,
        index_column,
        row_offset,
        nrows,
        lower_bound,
        upper_bound,
    )


def make_window(
    ir: Rolling,
    stream: Stream,
) -> Window:
    """Create reusable offset scalars for the rolling actor."""
    (index,) = names_to_indices([ir.index.name], ir.children[0].schema)
    side = ir.closed_window
    find_start = (
        plc.search.lower_bound if side in ("both", "left") else plc.search.upper_bound
    )
    find_end = (
        plc.search.upper_bound
        if ir.closed_window in ("both", "right")
        else plc.search.lower_bound
    )
    dtype = ir.index_dtype
    offsets = Window(
        # Note: not using windows_to_offsets because that flips the sign of preceding_ordinal.
        duration_to_scalar(dtype, ir.preceding_ordinal, stream=stream),
        duration_to_scalar(
            dtype, ir.preceding_ordinal + ir.following_ordinal, stream=stream
        ),
        index,
        dtype,
        find_start,
        find_end,
        stream,
    )
    # Simpler and probably more efficient that inducing cross-stream deps
    # to read the windows every time we add them to a chunk.
    stream.synchronize()
    return offsets


def _chrono_storage_dtype(dtype: plc.DataType) -> plc.DataType:
    """Return the integer storage type for a chrono dtype."""
    if dtype.id() in (plc.TypeId.TIMESTAMP_DAYS, plc.TypeId.DURATION_DAYS):
        return plc.DataType(plc.TypeId.INT32)
    return plc.DataType(plc.TypeId.INT64)


def _host_ordering_value(
    value: plc.Column,
    *,
    stream: Stream,
    br: BufferResource,
) -> Any:
    """Copy a single-row ordering value to host."""
    if plc.traits.is_chrono(value.type()):
        value = plc.unary.bit_cast(
            value, _chrono_storage_dtype(value.type()), stream=stream, mr=br.device_mr
        )
    return value.to_scalar(stream=stream).to_py(stream=stream)


def global_insertion_row(
    chunks: list[BufferedChunk],
    needle: plc.Column,
    find: Callable[..., plc.Column],
    *,
    needle_stream: Stream,
    br: BufferResource,
) -> int:
    """Return the globally indexed insertion row of a needle in some chunks."""
    assert len(chunks) > 0
    for chunk in chunks:
        if chunk.num_rows == 0:
            continue
        stream = chunk.chunk.stream
        join_cuda_streams(downstreams=[stream], upstreams=[needle_stream])
        # Since this returns a python integer, the work queued on search stream
        # is complete, so we don't need to join back to the search and needle
        # streams.
        insertion_point: int = (
            find(  # type: ignore[assignment]
                plc.Table([chunk.index_column]),
                plc.Table([needle]),
                [plc.types.Order.ASCENDING],
                [plc.types.NullOrder.AFTER],
                stream=stream,
                mr=br.device_mr,
            )
            .to_scalar(stream=stream)
            .to_py(stream=stream)
        )
        if insertion_point < chunk.num_rows:
            return chunk.row_start + insertion_point
    # Needle is later than all the chunks we know about.
    return chunks[-1].row_stop


def chunk_contains_upper_bound(
    chunk: BufferedChunk,
    needle: plc.Column,
    find: Callable[..., plc.Column],
    *,
    needle_stream: Stream,
    br: BufferResource,
) -> bool:
    """Return whether no future chunk can add rows to this upper bound."""
    insertion_row = global_insertion_row(
        [chunk], needle, find, needle_stream=needle_stream, br=br
    )
    return insertion_row < chunk.num_rows + chunk.row_start


def latest_nonempty_chunk(
    current: BufferedChunk, future: list[BufferedChunk]
) -> BufferedChunk:
    """Return the last non-empty chunk staged at or after current."""
    for chunk in reversed(future):
        if chunk.num_rows != 0:
            return chunk
    return current


async def extract_region(
    context: Context,
    input_chunks: list[BufferedChunk],
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


async def evaluate_available_chunk(
    context: Context,
    chunk: TableChunk,
    ir: IR,
    *,
    ir_context: IRExecutionContext,
) -> TableChunk:
    """Evaluate an already available chunk."""
    reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
        chunk.data_alloc_size(), net_memory_delta=0
    )
    with opaque_memory_usage(reservation):
        return await ir_context.to_thread(
            _evaluate_chunk_sync, chunk, ir, ir_context, context.br()
        )


def evict_history(
    history: list[BufferedChunk],
    cursor: BufferedChunk,
    window: Window,
    *,
    br: BufferResource,
) -> list[BufferedChunk]:
    """Drop history chunks that cannot contribute to the cursor chunk."""
    if not history:
        return []
    insertion_point = global_insertion_row(
        history,
        cursor.lower_bound,
        window.find_start,
        needle_stream=cursor.chunk.stream,
        br=br,
    )
    return [chunk for chunk in history if chunk.row_stop > insertion_point]


async def recv_buffered_chunk(
    context: Context,
    ch_in: Channel[TableChunk],
    *,
    row_offset: int,
    window: Window,
    observed_streams: set[Stream] | None = None,
) -> tuple[BufferedChunk | None, int]:
    """Receive and prepare one input chunk."""
    if (msg := await ch_in.recv(context)) is None:
        return None, row_offset
    chunk = await prepare_chunk(
        context,
        msg,
        row_offset=row_offset,
        window=window,
        observed_streams=observed_streams,
    )
    return chunk, chunk.row_stop


async def fill_future(
    context: Context,
    ch_in: Channel[TableChunk],
    current: BufferedChunk,
    future: list[BufferedChunk],
    row_offset: int,
    *,
    window: Window,
    observed_streams: set[Stream] | None = None,
) -> tuple[bool, int, list[BufferedChunk]]:
    """
    Read "leading" chunks from the input channel for the current chunk.

    Parameters
    ----------
    context
        Streaming context
    ch_in
        Input channel to read from
    current
        Current chunk we're trying to ghost-expand
    future
        Known leading ghost chunks
    row_offset
        Offset of the next chunk's rows in the logical "global" frame.
    window
        Window definition for finding bounding box
    observed_streams
        Streams that own staged chunk data or derived window-bound columns.

    Returns
    -------
    tuple
        Whether the input is exhausted, the new row_offset, and the newly
        updated future ghost region.
    """
    # As long as the current chunk's bounding box is past the end of the
    # most recently observed chunk, we need to grow the ghost region.
    while not chunk_contains_upper_bound(
        latest_nonempty_chunk(current, future),
        current.upper_bound,
        window.find_end,
        needle_stream=current.chunk.stream,
        br=context.br(),
    ):
        chunk, row_offset = await recv_buffered_chunk(
            context,
            ch_in,
            row_offset=row_offset,
            window=window,
            observed_streams=observed_streams,
        )
        if chunk is None:
            return True, row_offset, future
        future.append(chunk)
    return False, row_offset, future


async def evaluate_cursor(
    context: Context,
    ir: Rolling,
    ir_context: IRExecutionContext,
    cursor: BufferedChunk,
    *,
    history: list[BufferedChunk],
    future: list[BufferedChunk],
    window: Window,
) -> TableChunk:
    """Evaluate the rolling aggregation for the "cursor" chunk."""
    chunks = [*history, cursor, *future]
    ghost_start = global_insertion_row(
        chunks,
        cursor.lower_bound,
        window.find_start,
        needle_stream=cursor.chunk.stream,
        br=context.br(),
    )
    ghost_stop = global_insertion_row(
        chunks,
        cursor.upper_bound,
        window.find_end,
        needle_stream=cursor.chunk.stream,
        br=context.br(),
    )
    # We must extract at least the whole of the current cursor chunk.
    ghost_start = min(cursor.row_start, ghost_start)
    ghost_stop = max(cursor.row_stop, ghost_stop)
    ghosted_chunk = await extract_region(
        context, chunks, ghost_start, ghost_stop, ir_context=ir_context
    )
    result = await evaluate_available_chunk(
        context,
        ghosted_chunk,
        ir,
        ir_context=ir_context,
    )
    (table,) = plc.copying.slice(
        result.table_view(),
        [cursor.row_start - ghost_start, cursor.row_stop - ghost_start],
        stream=result.stream,
    )
    return TableChunk.from_pylibcudf_table(
        table.copy(result.stream, context.br().device_mr),
        result.stream,
        exclusive_view=True,
        br=context.br(),
    )


def index_value(chunk: BufferedChunk, row: int, *, br: BufferResource) -> Any:
    """Return one index value from a non-empty buffered chunk."""
    stream = chunk.chunk.stream
    (value,) = plc.copying.slice(chunk.index_column, [row, row + 1], stream=stream)
    return _host_ordering_value(value, stream=stream, br=br)


def check_cross_chunk_sorted(
    cursor: BufferedChunk, *, index_name: str, previous: Any | None, br: BufferResource
) -> Any | None:
    """Return the cursor's last index value after cross-chunk sortedness check."""
    if cursor.num_rows == 0:
        return previous
    first = index_value(cursor, 0, br=br)
    if previous is not None and first < previous:
        raise RuntimeError(
            f"Index column '{index_name}' in rolling is not sorted, please sort first"
        )
    return index_value(cursor, cursor.num_rows - 1, br=br)


@define_actor()
async def rolling_actor(
    context: Context,
    comm: Communicator,
    ir: Rolling,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    *,
    collective_id: int,
    config_options: ConfigOptions[StreamingExecutor],
) -> None:
    """
    Single-rank streaming actor for range-based rolling aggregations.

    Chunks are received in order. The actor stages rank-local input, evaluates
    each output chunk with enough ghost rows to satisfy its rolling windows, and
    emits chunks in the same order. Multi-rank boundary exchange is deliberately
    left for a later implementation.
    """
    async with shutdown_on_error(
        context, chs_in=(ch_in,), chs_out=(ch_out,), trace_ir=ir, ir_context=ir_context
    ) as actor_scope:
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
            if (tracer := actor_scope.tracer) is not None:
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
            result = await evaluate_available_chunk(
                context,
                chunk,
                ir,
                ir_context=ir_context,
            )
            if (tracer := actor_scope.tracer) is not None:
                tracer.add_chunk(chunk=result)
            await ch_out.send(context, Message(0, result))
            await ch_out.drain(context)
            return

        await send_metadata(
            ch_out,
            context,
            ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=None,
                duplicated=metadata_in.duplicated,
            ),
        )
        if (tracer := actor_scope.tracer) is not None and metadata_in.duplicated:
            tracer.set_duplicated()

        window = make_window(ir, ir_context.get_cuda_stream())
        # streams that window deallocation will be ordered after
        observed_streams: set[Stream] = set()
        # chunks we have already evaluated that might be needed to evaluate future chunks
        history: list[BufferedChunk] = []
        # chunks we have not yet evaluated that are needed to evaluate the current chunk
        future: list[BufferedChunk] = []
        try:
            input_exhausted = False
            previous_index_value = None
            cursor, row_offset = await recv_buffered_chunk(
                context,
                ch_in,
                row_offset=0,
                window=window,
                observed_streams=observed_streams,
            )
            while cursor is not None:
                if cursor.num_rows == 0:
                    result = await evaluate_available_chunk(
                        context,
                        cursor.chunk,
                        ir,
                        ir_context=ir_context,
                    )
                else:
                    previous_index_value = check_cross_chunk_sorted(
                        cursor,
                        index_name=ir.index.name,
                        previous=previous_index_value,
                        br=context.br(),
                    )
                    history = evict_history(history, cursor, window, br=context.br())
                    if not input_exhausted:
                        input_exhausted, row_offset, future = await fill_future(
                            context,
                            ch_in,
                            cursor,
                            future,
                            row_offset,
                            window=window,
                            observed_streams=observed_streams,
                        )
                    result = await evaluate_cursor(
                        context,
                        ir,
                        ir_context,
                        cursor,
                        history=history,
                        future=future,
                        window=window,
                    )
                    history.append(cursor)
                if (tracer := actor_scope.tracer) is not None:
                    actor_scope.tracer.add_chunk(chunk=result)
                await ch_out.send(context, Message(cursor.sequence_number, result))

                if future:
                    cursor, *future = future
                else:
                    cursor, row_offset = await recv_buffered_chunk(
                        context,
                        ch_in,
                        row_offset=row_offset,
                        window=window,
                        observed_streams=observed_streams,
                    )
        finally:
            join_cuda_streams(
                downstreams=(window.stream,),
                upstreams=tuple(observed_streams),
            )

        await ch_out.drain(context)


@generate_ir_sub_network.register(Rolling)
def _(
    ir: Rolling, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for a Rolling operation."""
    if len(ir.keys) > 0 or ir.zlice is not None:
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    actors[ir] = [
        rolling_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_id=collective_id,
            config_options=rec.state["config_options"],
        )
    ]
    return actors, channels
