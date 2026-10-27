# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-overlap planning and exchange for streaming operators."""

from __future__ import annotations

import operator
import struct
from bisect import bisect_left
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.utils.windows import duration_to_scalar
from cudf_polars.streaming.actor_graph.utils import ChunkStore, names_to_indices
from cudf_polars.utils.cuda_stream import join_cuda_streams

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Sequence

    from pylibcudf.utils import CudaStreamLike
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.memory_resource import DeviceMemoryResource
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import Rolling


type RowRange = tuple[int, int]
type GhostSend = tuple[int, int, int]


type _BoundSearch = Callable[
    [
        plc.Table,
        plc.Table,
        list[plc.types.Order],
        list[plc.types.NullOrder],
        CudaStreamLike | None,
        DeviceMemoryResource | None,
    ],
    plc.Column,
]


class RangeValidation(IntEnum):
    """Validation status for one rank's range index."""

    VALID = 0
    NULL = 1
    UNSORTED = 2
    OUT_OF_BOUNDS = 3


@dataclass
class BufferedChunk:
    """An available chunk with its span in the prepared input stream."""

    chunk: TableChunk
    row_start: int
    num_rows: int

    @property
    def row_stop(self) -> int:
        """Return the first row after this chunk."""
        return self.row_start + self.num_rows


@dataclass
class StagedChunk[BufferedChunkT: BufferedChunk]:
    """A prepared chunk marked as owned or ghost."""

    chunk: BufferedChunkT
    """Chunk and policy metadata used by local execution."""
    owned: bool
    """Whether this rank owns the chunk's output rows."""


@dataclass(frozen=True)
class RangeValues:
    """
    Host values describing one nonempty index range.

    The policy determines whether each needed boundary is included.
    """

    first_owned: int
    """First index value in the source-owned range."""
    last_owned: int
    """Last index value in the source-owned range."""
    first_needed: int
    """First boundary required to evaluate the chunk's rows."""
    last_needed: int
    """Last boundary required to evaluate the chunk's rows."""


type RangeRequest = tuple[int, RangeValues]


@dataclass
class RangeOverlap:
    """
    Device and host range metadata for one buffered chunk.

    Device columns preserve the index type and support exact libcudf searches.
    Host values support validation and overlap planning with ordinary Python
    comparisons.
    """

    index_column: plc.Column
    """Index values stored in the buffered chunk."""
    first_needed_boundary: plc.Column
    """Single-row device column containing the first needed boundary."""
    last_needed_boundary: plc.Column
    """Single-row device column containing the last needed boundary."""
    host_values: RangeValues | None
    """Host values for a nonempty chunk, or ``None`` for an empty chunk."""


@dataclass
class RangeBufferedChunk(BufferedChunk):
    """A buffered chunk with range-overlap metadata."""

    overlap: RangeOverlap


class _OverlapPolicy[BufferedChunkT: BufferedChunk]:
    """Base policy for finding the rows needed to evaluate a cursor."""

    def observe(self, chunk: BufferedChunkT) -> None:
        """Record resources that must remain valid while using a chunk."""

    def close(self) -> None:
        """Release resources owned by this policy."""

    def validate_chunk(self, chunk: BufferedChunkT) -> None:
        """Validate a chunk against chunks previously seen by this policy."""

    def evict_history(
        self,
        history: list[BufferedChunkT],
        cursor: BufferedChunkT,
    ) -> list[BufferedChunkT]:
        """Return history chunks that may overlap the cursor or a later chunk."""
        raise NotImplementedError

    def has_complete_future(
        self,
        latest: BufferedChunkT,
        current: BufferedChunkT,
    ) -> bool:
        """Return whether the buffered future covers the current cursor."""
        raise NotImplementedError

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunkT,
    ) -> BufferedChunkT | None:
        """Return the cursor rows that future cursors may need as history."""
        del context
        return cursor

    async def overlap_bounds(
        self,
        context: Context,
        cursor: BufferedChunkT,
        chunks: Sequence[BufferedChunkT],
    ) -> tuple[int, int]:
        """Return the needed span in prepared-stream coordinates."""
        raise NotImplementedError

    async def prepare_chunk(
        self,
        context: Context,
        chunk: TableChunk,
        *,
        row_offset: int,
        host_values: RangeValues | None = None,
    ) -> BufferedChunkT:
        """
        Attach policy metadata to a chunk at a prepared-stream row offset.

        ``row_offset`` is the chunk's first row in the prepared input stream.
        """
        raise NotImplementedError


@dataclass
class RangeOverlapPolicy(_OverlapPolicy[RangeBufferedChunk]):
    """
    Policy and local execution state for range-based overlap.

    The mutable fields track ordering across chunks and the streams using the
    policy's device scalars. Each consumer creates and closes one instance.
    """

    first_needed_offset: plc.Scalar
    """Offset from the first owned value to the first needed boundary."""
    last_needed_offset: plc.Scalar
    """Offset from the last owned value to the last needed boundary."""
    index_position: int
    """Position of the ordered index column in each input chunk."""
    index_dtype: plc.DataType
    find_start: _BoundSearch
    find_end: _BoundSearch
    start_closed: bool
    end_closed: bool
    stream: Stream
    index_name: str
    observed_streams: set[Stream] = field(default_factory=set, init=False, repr=False)
    previous_last_owned: int | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_ir(cls, ir: Rolling, stream: Stream) -> RangeOverlapPolicy:
        """Create actor-local range-bound state for a rolling node."""
        (index_position,) = names_to_indices([ir.index.name], ir.children[0].schema)
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
            index_position,
            dtype,
            plc.search.lower_bound if start_closed else plc.search.upper_bound,
            plc.search.upper_bound if end_closed else plc.search.lower_bound,
            start_closed,
            end_closed,
            stream,
            ir.index.name,
        )

    def observe(self, chunk: RangeBufferedChunk) -> None:
        """Track streams used by range-overlap metadata."""
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
    ) -> int:
        """Copy a single-row ordering value to host."""
        if plc.traits.is_chrono(value.type()):
            value = plc.unary.bit_cast(
                value,
                cls._chrono_storage_dtype(value.type()),
                stream=stream,
                mr=br.device_mr,
            )
        result = value.to_scalar(stream=stream).to_py(stream=stream)
        assert isinstance(result, int)
        return result

    @classmethod
    def _host_ordering_value_at(
        cls,
        column: plc.Column,
        row: int,
        *,
        stream: Stream,
        br: BufferResource,
    ) -> int:
        """Copy one ordering value from a device column to host."""
        (value,) = plc.copying.slice(column, [row, row + 1], stream=stream)
        return cls._host_ordering_value(value, stream=stream, br=br)

    @staticmethod
    async def _insertion_point(
        context: Context,
        index_column: plc.Column,
        needle: plc.Column,
        find: _BoundSearch,
        dtype: plc.DataType,
        stream: Stream,
    ) -> int:
        """Find a needle's insertion point in one available index column."""
        needs_cast = index_column.type() != dtype
        memory_scope = nullcontext()
        if needs_cast:
            reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
                index_column.size() * 8, net_memory_delta=0
            )
            memory_scope = opaque_memory_usage(reservation)
        with memory_scope:
            if needs_cast:
                index_column = plc.unary.cast(index_column, dtype, stream=stream)
            value = (
                find(
                    plc.Table([index_column]),
                    plc.Table([needle]),
                    [plc.types.Order.ASCENDING],
                    [plc.types.NullOrder.AFTER],
                    stream,
                    context.br().device_mr,
                )
                .to_scalar(stream=stream)
                .to_py(stream=stream)
            )
        assert isinstance(value, int)
        return value

    async def insertion_point(
        self,
        context: Context,
        index_column: plc.Column,
        value: int,
        find: _BoundSearch,
        stream: Stream,
    ) -> int:
        """Find a host value's insertion point in an index column."""
        scalar = plc.Scalar.from_py(
            value,
            self._chrono_storage_dtype(self.index_dtype)
            if plc.traits.is_chrono(self.index_dtype)
            else self.index_dtype,
            stream=stream,
        )
        needle = plc.Column.from_scalar(scalar, 1, stream=stream)
        if plc.traits.is_chrono(self.index_dtype):
            needle = plc.unary.bit_cast(
                needle,
                self.index_dtype,
                stream=stream,
                mr=context.br().device_mr,
            )
        return await self._insertion_point(
            context,
            index_column,
            needle,
            find,
            self.index_dtype,
            stream,
        )

    @staticmethod
    async def _insertion_row(
        chunks: Sequence[RangeBufferedChunk],
        needle: plc.Column,
        needle_value: int,
        find: _BoundSearch,
        *,
        upper_bound: bool,
        dtype: plc.DataType,
        context: Context,
        needle_stream: Stream,
    ) -> int:
        """Return a needle's insertion row in prepared-stream coordinates."""
        assert len(chunks) > 0
        for chunk in chunks:
            if chunk.num_rows == 0:
                continue
            assert chunk.overlap.host_values is not None
            if (
                chunk.overlap.host_values.last_owned <= needle_value
                if upper_bound
                else chunk.overlap.host_values.last_owned < needle_value
            ):
                continue
            stream = chunk.chunk.stream
            join_cuda_streams(downstreams=[stream], upstreams=[needle_stream])
            insertion_point = await RangeOverlapPolicy._insertion_point(
                context,
                chunk.overlap.index_column,
                needle,
                find,
                dtype,
                stream,
            )
            if insertion_point < chunk.num_rows:
                return chunk.row_start + insertion_point
        # Needle is later than all the chunks we know about.
        return chunks[-1].row_stop

    @staticmethod
    def index_validation(index_column: plc.Column, stream: Stream) -> RangeValidation:
        """Return the validation status for one index column."""
        if index_column.null_count() != 0:
            return RangeValidation.NULL
        if index_column.size() > 1 and not plc.sorting.is_sorted(
            plc.Table([index_column]),
            [plc.types.Order.ASCENDING],
            [plc.types.NullOrder.BEFORE],
            stream=stream,
        ):
            return RangeValidation.UNSORTED
        return RangeValidation.VALID

    def validate_index_column(self, index_column: plc.Column, stream: Stream) -> None:
        """Validate chunk-local ordering before using search primitives."""
        self.raise_validation(self.index_validation(index_column, stream))

    def raise_validation(self, validation: RangeValidation) -> None:
        """Raise the corresponding range-index validation error."""
        if validation is RangeValidation.NULL:
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling may not contain nulls"
            )
        if validation is RangeValidation.UNSORTED:
            raise RuntimeError(
                f"Index column '{self.index_name}' in rolling is not sorted, "
                "please sort first"
            )

    def validate_chunk(self, chunk: RangeBufferedChunk) -> None:
        """Check that the index remains sorted across chunk boundaries."""
        if chunk.num_rows == 0:
            return
        assert chunk.overlap.host_values is not None
        if (
            self.previous_last_owned is not None
            and chunk.overlap.host_values.first_owned < self.previous_last_owned
        ):
            self.raise_validation(RangeValidation.UNSORTED)
        self.previous_last_owned = chunk.overlap.host_values.last_owned

    async def prepare_chunk(
        self,
        context: Context,
        chunk: TableChunk,
        *,
        row_offset: int,
        host_values: RangeValues | None = None,
        validate_index: bool = True,
    ) -> RangeBufferedChunk:
        """Extract range metadata from a table chunk."""
        nrows, _ = chunk.shape
        chunk, extra = await make_table_chunks_available_or_wait(
            context,
            chunk,
            reserve_extra=32,
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            index_column = chunk.table_view().columns()[self.index_position]
            if host_values is None and validate_index:
                self.validate_index_column(index_column, chunk.stream)
            if nrows == 0:
                overlap = RangeOverlap(index_column, index_column, index_column, None)
            else:
                join_cuda_streams(downstreams=(chunk.stream,), upstreams=(self.stream,))
                first_needed_boundary = self._index_with_offset(
                    index_column,
                    0,
                    self.first_needed_offset,
                    self.index_dtype,
                    chunk.stream,
                    context.br(),
                )
                last_needed_boundary = self._index_with_offset(
                    index_column,
                    nrows - 1,
                    self.last_needed_offset,
                    self.index_dtype,
                    chunk.stream,
                    context.br(),
                )
                if host_values is None and (
                    validate_index or index_column.null_count() == 0
                ):
                    host_values = RangeValues(
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
                            first_needed_boundary,
                            stream=chunk.stream,
                            br=context.br(),
                        ),
                        self._host_ordering_value(
                            last_needed_boundary,
                            stream=chunk.stream,
                            br=context.br(),
                        ),
                    )
                overlap = RangeOverlap(
                    index_column,
                    first_needed_boundary,
                    last_needed_boundary,
                    host_values,
                )
        return RangeBufferedChunk(chunk, row_offset, nrows, overlap)

    def evict_history(
        self,
        history: list[RangeBufferedChunk],
        cursor: RangeBufferedChunk,
    ) -> list[RangeBufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        assert cursor.overlap.host_values is not None
        keep = operator.ge if self.start_closed else operator.gt
        return [
            chunk
            for chunk in history
            if chunk.overlap.host_values is not None
            and keep(
                chunk.overlap.host_values.last_owned,
                cursor.overlap.host_values.first_needed,
            )
        ]

    def has_complete_future(
        self,
        latest: RangeBufferedChunk,
        current: RangeBufferedChunk,
    ) -> bool:
        """Return whether buffered input reaches the last needed boundary."""
        if latest.overlap.host_values is None:
            return False
        assert current.overlap.host_values is not None
        complete = operator.gt if self.end_closed else operator.ge
        return complete(
            latest.overlap.host_values.last_owned,
            current.overlap.host_values.last_needed,
        )

    async def overlap_bounds(
        self,
        context: Context,
        cursor: RangeBufferedChunk,
        chunks: Sequence[RangeBufferedChunk],
    ) -> tuple[int, int]:
        """Return the rows needed to evaluate a range cursor."""
        assert cursor.overlap.host_values is not None
        overlap_start = await self._insertion_row(
            chunks,
            cursor.overlap.first_needed_boundary,
            cursor.overlap.host_values.first_needed,
            self.find_start,
            upper_bound=not self.start_closed,
            dtype=self.index_dtype,
            context=context,
            needle_stream=cursor.chunk.stream,
        )
        overlap_stop = await self._insertion_row(
            chunks,
            cursor.overlap.last_needed_boundary,
            cursor.overlap.host_values.last_needed,
            self.find_end,
            upper_bound=self.end_closed,
            dtype=self.index_dtype,
            context=context,
            needle_stream=cursor.chunk.stream,
        )
        # We must extract at least the whole of the current cursor chunk.
        overlap_start = min(cursor.row_start, overlap_start)
        overlap_stop = max(cursor.row_stop, overlap_stop)
        return overlap_start, overlap_stop


@dataclass
class RowCountOverlapPolicy(_OverlapPolicy[BufferedChunk]):
    """Overlap policy for fixed-size row windows."""

    preceding: int
    following: int

    async def prepare_chunk(
        self,
        context: Context,
        chunk: TableChunk,
        *,
        row_offset: int,
        host_values: RangeValues | None = None,
    ) -> BufferedChunk:
        """Attach row-span metadata to an available chunk."""
        del host_values
        nrows, _ = chunk.shape
        chunk, _ = await make_table_chunks_available_or_wait(
            context, chunk, reserve_extra=0, net_memory_delta=0
        )
        return BufferedChunk(chunk, row_offset, nrows)

    def evict_history(
        self,
        history: list[BufferedChunk],
        cursor: BufferedChunk,
    ) -> list[BufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        overlap_start = max(0, cursor.row_start - self.preceding)
        return [chunk for chunk in history if chunk.row_stop > overlap_start]

    def has_complete_future(
        self,
        latest: BufferedChunk,
        current: BufferedChunk,
    ) -> bool:
        """Return whether buffered input reaches the needed stop row."""
        return latest.row_stop >= current.row_stop + self.following

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunk,
    ) -> BufferedChunk | None:
        """Keep only cursor rows that a future fixed-size window can use."""
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
            TableChunk.from_pylibcudf_table(
                table,
                stream,
                exclusive_view=True,
                br=context.br(),
            ),
            start,
            cursor.row_stop - start,
        )

    async def overlap_bounds(
        self,
        context: Context,
        cursor: BufferedChunk,
        chunks: Sequence[BufferedChunk],
    ) -> tuple[int, int]:
        """Return the rows needed to evaluate a fixed-size cursor."""
        del context
        overlap_start = max(0, cursor.row_start - self.preceding)
        overlap_stop = min(cursor.row_stop + self.following, chunks[-1].row_stop)
        return overlap_start, overlap_stop


type OverlapPolicy[BufferedChunkT: BufferedChunk] = _OverlapPolicy[BufferedChunkT]


@dataclass(frozen=True)
class _CountSummary:
    """Number of rows owned by one rank."""

    rows: int


@dataclass(frozen=True)
class _RangeSummary:
    """Validation and host range values for one rank's input."""

    rows: int
    validation: RangeValidation
    # Host range values may be absent for empty or invalid input.
    range_values: RangeValues | None


@dataclass(frozen=True)
class _PreparedInput:
    """Buffered local input and ghost exchange state for one rank."""

    owned_buffer: ChunkStore
    """Spillable messages containing locally owned chunks."""
    ghost_exchange: SparseAlltoall | None
    """Exchange used to send or receive ghost chunks."""
    ghost_sources: tuple[int, ...]
    """Candidate source ranks for ghost chunks in input order."""
    owned_host_values: tuple[RangeValues | None, ...]
    """Cached host range values for locally owned chunks."""


class OverlapInput[BufferedChunkT: BufferedChunk]:
    """
    Yield owned input with the ghost chunks required by the overlap policy.

    A window near a rank boundary may need rows owned by another rank.
    For non-duplicated multi-rank input, entering this context buffers the full
    local input in spillable messages, all-gathers a compact count or range
    summary, and exchanges only the required slices. This preparation is a
    barrier, so no chunks are yielded until every rank has summarized its
    local input.

    Iteration yields left ghost chunks, owned chunks, and right ghost chunks in
    row order. Ghost chunks provide context but do not produce output rows.
    Single-rank and duplicated input streams directly from the channel.
    """

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        ch_in: Channel[TableChunk],
        policy: OverlapPolicy[BufferedChunkT],
        *,
        duplicated: bool,
        collective_id: int,
    ) -> None:
        """Create an input stream for one overlap policy and input channel."""
        self.context = context
        self.comm = comm
        self.ch_in = ch_in
        self.policy = policy
        self.multirank = comm.nranks != 1 and not duplicated
        self.collective_id = collective_id
        self._prepared: _PreparedInput | None = None

    async def __aenter__(self) -> OverlapInput[BufferedChunkT]:
        """Prepare non-duplicated multi-rank input and wait for all ranks."""
        if self.multirank:
            self._prepared = await self._prepare()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Release messages still owned by this input."""
        prepared, self._prepared = self._prepared, None
        if prepared is not None:
            prepared.owned_buffer.clear()

    def __aiter__(self) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
        """Yield prepared chunks in row order."""
        return self._chunks()

    async def _raw_owned_chunks(
        self,
    ) -> AsyncIterator[tuple[TableChunk, RangeValues | None]]:
        """Yield locally owned chunks and any cached host range values."""
        if not self.multirank:
            while (msg := await self.ch_in.recv(self.context)) is not None:
                yield TableChunk.from_message(msg, br=self.context.br()), None
        else:
            assert self._prepared is not None
            for position, msg in enumerate(self._prepared.owned_buffer):
                yield (
                    TableChunk.from_message(msg, br=self.context.br()),
                    self._prepared.owned_host_values[position]
                    if self._prepared.owned_host_values
                    else None,
                )

    async def _raw_ghost_chunks(
        self, sources: Iterable[int]
    ) -> AsyncIterator[tuple[TableChunk, None]]:
        """Yield exchanged chunks from each source rank in row order."""
        prepared = self._prepared
        for source in sources:
            assert prepared is not None
            assert prepared.ghost_exchange is not None
            for packed in prepared.ghost_exchange.extract(source):
                yield TableChunk.from_packed_data(packed, self.context.br()), None

    async def _chunks(self) -> AsyncIterator[StagedChunk[BufferedChunkT]]:
        """Prepare chunks in one row coordinate system for this rank."""
        if self.multirank and self._prepared is None:
            raise RuntimeError(
                "Multi-rank overlap input must be used as an async context manager"
            )
        row_offset = 0
        sources = self._prepared.ghost_sources if self._prepared else ()
        split = bisect_left(sources, self.comm.rank)
        inputs = (
            (False, self._raw_ghost_chunks(sources[:split])),
            (True, self._raw_owned_chunks()),
            (False, self._raw_ghost_chunks(sources[split:])),
        )
        for owned, values in inputs:
            async for raw_chunk, host_values in values:
                chunk = await self.policy.prepare_chunk(
                    self.context,
                    raw_chunk,
                    row_offset=row_offset,
                    host_values=host_values,
                )
                self.policy.observe(chunk)
                row_offset = chunk.row_stop
                yield StagedChunk(chunk, owned=owned)

    async def _prepare(self) -> _PreparedInput:
        """Buffer local input, agree on a plan, and exchange ghost chunks."""
        policy = self.policy
        if not isinstance(policy, (RangeOverlapPolicy, RowCountOverlapPolicy)):
            raise TypeError(f"Unsupported overlap policy: {type(policy).__name__}")
        buffer = ChunkStore(self.context)
        try:
            if isinstance(policy, RangeOverlapPolicy):
                summary, host_values = await self._buffer_range_input(buffer, policy)
                gathered = await self._gather_ranges(summary)
                self._validate_ranges(policy, gathered)
                sends, sources, destinations = self._range_plan(summary, gathered)
                row_offset = sum(item.rows for item in gathered[: self.comm.rank])
                return await self._exchange(
                    buffer,
                    policy,
                    sends,
                    sources,
                    destinations,
                    row_offset,
                    host_values=host_values,
                    summaries=gathered,
                )

            count_summary = await self._buffer_count_input(buffer)
            count_summaries = await self._gather_counts(count_summary)
            starts = tuple(
                accumulate((item.rows for item in count_summaries), initial=0)
            )
            sends, sources, destinations = self._count_plan(
                policy, tuple(pairwise(starts))
            )
            return await self._exchange(
                buffer,
                policy,
                sends,
                sources,
                destinations,
                starts[self.comm.rank],
            )
        except BaseException:
            buffer.clear()
            raise

    async def _buffer_count_input(self, buffer: ChunkStore) -> _CountSummary:
        """Buffer local chunks and return their total row count."""
        rows = 0
        while (msg := await self.ch_in.recv(self.context)) is not None:
            chunk = TableChunk.from_message(msg, br=self.context.br())
            rows += chunk.shape[0]
            buffer.insert(Message(0, chunk))
        return _CountSummary(rows)

    async def _buffer_range_input(
        self,
        buffer: ChunkStore,
        policy: RangeOverlapPolicy,
    ) -> tuple[_RangeSummary, tuple[RangeValues | None, ...]]:
        """Buffer local chunks and collect host range values."""
        row_offset = 0
        validation = RangeValidation.VALID
        summary_values = None
        host_values: list[RangeValues | None] = []
        while (msg := await self.ch_in.recv(self.context)) is not None:
            table_chunk = TableChunk.from_message(msg, br=self.context.br())
            nrows, _ = table_chunk.shape
            chunk = await policy.prepare_chunk(
                self.context,
                table_chunk,
                row_offset=row_offset,
                validate_index=False,
            )
            policy.observe(chunk)
            table_chunk = chunk.chunk
            values = chunk.overlap.host_values
            if nrows:
                status = policy.index_validation(
                    chunk.overlap.index_column, table_chunk.stream
                )
                if status is RangeValidation.VALID:
                    assert values is not None
                    if (
                        summary_values is not None
                        and values.first_owned < summary_values.last_owned
                    ):
                        status = RangeValidation.UNSORTED
                if validation is RangeValidation.VALID:
                    validation = status
                if status is RangeValidation.VALID:
                    assert values is not None
                    summary_values = RangeValues(
                        summary_values.first_owned
                        if summary_values
                        else values.first_owned,
                        values.last_owned,
                        summary_values.first_needed
                        if summary_values
                        else values.first_needed,
                        values.last_needed,
                    )
            host_values.append(values)
            buffer.insert(Message(0, table_chunk))
            row_offset += nrows
        return _RangeSummary(row_offset, validation, summary_values), tuple(host_values)

    async def _allgather_ints(
        self, payload: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        """All-gather fixed-width signed integer metadata in rank order."""
        fmt = f"<{len(payload)}q"
        gather = AllGather(self.context, self.comm, self.collective_id)
        try:
            gather.insert(
                0,
                PackedData.from_host_bytes(
                    struct.pack(fmt, *payload), self.context.br()
                ),
            )
        finally:
            gather.insert_finished()
        return tuple(
            struct.unpack(fmt, item.to_host_bytes())
            for item in await gather.extract_all(self.context, ordered=True)
        )

    async def _gather_counts(self, summary: _CountSummary) -> tuple[_CountSummary, ...]:
        """Gather the row count owned by every rank."""
        return tuple(
            _CountSummary(item[0])
            for item in await self._allgather_ints((summary.rows,))
        )

    async def _gather_ranges(self, summary: _RangeSummary) -> tuple[_RangeSummary, ...]:
        """Gather range summaries after checking signed 64-bit bounds."""
        values = summary.range_values
        validation = summary.validation
        if values is not None and any(
            not -(2**63) <= value < 2**63
            for value in (
                values.first_owned,
                values.last_owned,
                values.first_needed,
                values.last_needed,
            )
        ):
            validation = RangeValidation.OUT_OF_BOUNDS
        packed_values = (
            (
                values.first_owned,
                values.last_owned,
                values.first_needed,
                values.last_needed,
            )
            if values is not None and validation is RangeValidation.VALID
            else (0, 0, 0, 0)
        )
        return tuple(
            _RangeSummary(
                item[0],
                RangeValidation(item[1]),
                RangeValues(*item[2:])
                if item[0] and item[1] == RangeValidation.VALID
                else None,
            )
            for item in await self._allgather_ints(
                (summary.rows, int(validation), *packed_values)
            )
        )

    @staticmethod
    def _validate_ranges(
        policy: RangeOverlapPolicy,
        summaries: Sequence[_RangeSummary],
    ) -> None:
        """Raise the same range-validation error on every rank."""
        validations = {item.validation for item in summaries}
        if RangeValidation.OUT_OF_BOUNDS in validations:
            raise NotImplementedError(
                "Multi-rank rolling does not support index values outside the "
                "signed 64-bit range"
            )
        if RangeValidation.NULL in validations:
            policy.raise_validation(RangeValidation.NULL)
        previous = None
        for item in summaries:
            if item.validation is RangeValidation.UNSORTED:
                break
            if not item.rows:
                continue
            values = item.range_values
            if (
                values is None
                or values.first_owned > values.last_owned
                or (previous is not None and values.first_owned < previous)
            ):
                break
            previous = values.last_owned
        else:
            return
        policy.raise_validation(RangeValidation.UNSORTED)

    @staticmethod
    def _intersection(left: RowRange, right: RowRange) -> RowRange | None:
        """Return the intersection of two half-open global row spans."""
        start, stop = max(left[0], right[0]), min(left[1], right[1])
        return (start, stop) if start < stop else None

    def _count_plan(
        self,
        policy: RowCountOverlapPolicy,
        spans: Sequence[RowRange],
    ) -> tuple[list[GhostSend], tuple[int, ...], tuple[int, ...]]:
        """
        Plan count-based ghost slices in global row coordinates.

        ``spans`` contains the half-open row span owned by each rank. Sends are
        ``(destination, start, stop)`` tuples in the same coordinate system.
        """
        total = spans[-1][1] if spans else 0
        requests = tuple(
            tuple(
                span
                for span in (
                    (max(0, start - policy.preceding), start),
                    (stop, min(total, stop + policy.following)),
                )
                if start != stop and span[0] < span[1]
            )
            for start, stop in spans
        )
        rank = self.comm.rank
        sends: list[GhostSend] = []
        sources: list[int] = []
        destinations: set[int] = set()
        for other, span in enumerate(spans):
            if other == rank:
                continue
            if any(self._intersection(span, request) for request in requests[rank]):
                sources.append(other)
            for request in requests[other]:
                if (piece := self._intersection(spans[rank], request)) is not None:
                    sends.append((other, *piece))
                    destinations.add(other)
        return sends, tuple(sources), tuple(destinations)

    def _range_plan(
        self,
        local: _RangeSummary,
        summaries: Sequence[_RangeSummary],
    ) -> tuple[list[GhostSend], tuple[int, ...], tuple[int, ...]]:
        """
        Select source and destination ranks for range overlap.

        This rank-level plan is conservative. ``_range_pieces`` later applies
        the policy's open or closed endpoints to find exact chunk slices.
        """
        rank = self.comm.rank
        local_values = local.range_values
        sources = tuple(
            source
            for source, item in enumerate(summaries)
            if source != rank
            and local_values is not None
            and item.range_values is not None
            and item.range_values.last_owned >= local_values.first_needed
            and item.range_values.first_owned <= local_values.last_needed
        )
        destinations = tuple(
            destination
            for destination, item in enumerate(summaries)
            if destination != rank
            and local_values is not None
            and item.range_values is not None
            and local_values.last_owned >= item.range_values.first_needed
            and local_values.first_owned <= item.range_values.last_needed
        )
        return [], sources, destinations

    async def _exchange(
        self,
        buffer: ChunkStore,
        policy: RangeOverlapPolicy | RowCountOverlapPolicy,
        sends: Sequence[GhostSend],
        sources: Sequence[int],
        destinations: Sequence[int],
        global_row_offset: int,
        *,
        host_values: tuple[RangeValues | None, ...] = (),
        summaries: Sequence[_RangeSummary] = (),
    ) -> _PreparedInput:
        """
        Exchange ghost slices and return buffered owned input.

        Count-based sends use half-open global row coordinates. Range-based
        sends are resolved from rank summaries to chunk-local slices before
        insertion into the sparse all-to-all.
        """
        sources = tuple(sorted(sources))
        destinations = tuple(sorted(set(destinations)))
        if not (sources or destinations):
            return _PreparedInput(buffer, None, sources, host_values)
        exchange = SparseAlltoall(
            self.context,
            self.comm,
            self.collective_id,
            srcs=sources,
            dsts=destinations,
        )
        source, output = buffer, ChunkStore(self.context)
        start = 0
        try:
            for position, msg in enumerate(source):
                chunk = TableChunk.from_message(msg, br=self.context.br())
                size = chunk.shape[0]
                pieces: list[tuple[int, int, int]] = []
                range_values = None
                if isinstance(policy, RangeOverlapPolicy) and size:
                    range_values = host_values[position]
                    assert range_values is not None
                    range_requests = tuple(
                        (destination, destination_values)
                        for destination in destinations
                        if (destination_values := summaries[destination].range_values)
                        is not None
                        and range_values.last_owned >= destination_values.first_needed
                        and range_values.first_owned <= destination_values.last_needed
                    )
                else:
                    span = (
                        start + global_row_offset,
                        start + global_row_offset + size,
                    )
                    pieces = [
                        (destination, piece[0] - span[0], piece[1] - span[0])
                        for destination, request_start, request_stop in sends
                        if (
                            piece := self._intersection(
                                span, (request_start, request_stop)
                            )
                        )
                        is not None
                    ]
                    range_requests = ()
                if not range_requests and not pieces:
                    output.insert(Message(0, chunk))
                    start += size
                    continue
                available, extra = await make_table_chunks_available_or_wait(
                    self.context,
                    chunk,
                    reserve_extra=0,
                    net_memory_delta=0,
                )
                with opaque_memory_usage(extra):
                    if (
                        isinstance(policy, RangeOverlapPolicy)
                        and range_values is not None
                    ):
                        pieces = await self._range_pieces(
                            policy,
                            available,
                            size,
                            range_values,
                            range_requests,
                        )
                    for destination, begin, end in pieces:
                        await self._send(exchange, destination, available, begin, end)
                    output.insert(Message(0, available))
                start += size
        except BaseException:
            output.clear()
            raise
        finally:
            try:
                source.clear()
            except BaseException:
                output.clear()
                raise
            finally:
                try:
                    await exchange.insert_finished(self.context)
                except BaseException:
                    output.clear()
                    raise
        return _PreparedInput(output, exchange, sources, host_values)

    async def _range_pieces(
        self,
        policy: RangeOverlapPolicy,
        chunk: TableChunk,
        size: int,
        chunk_values: RangeValues,
        requests: Sequence[RangeRequest],
    ) -> list[tuple[int, int, int]]:
        """
        Resolve range requests to half-open row slices within one chunk.

        Each result is ``(destination, start, stop)``. Search selection follows
        the policy's open or closed window endpoints.
        """
        pieces = []
        index = chunk.table_view().columns()[policy.index_position]
        for destination, requested_values in requests:
            first_needed = requested_values.first_needed
            last_needed = requested_values.last_needed
            begin = (
                0
                if first_needed < chunk_values.first_owned
                or (policy.start_closed and first_needed == chunk_values.first_owned)
                else await policy.insertion_point(
                    self.context,
                    index,
                    first_needed,
                    policy.find_start,
                    chunk.stream,
                )
            )
            end = (
                size
                if last_needed > chunk_values.last_owned
                or (policy.end_closed and last_needed == chunk_values.last_owned)
                else await policy.insertion_point(
                    self.context,
                    index,
                    last_needed,
                    policy.find_end,
                    chunk.stream,
                )
            )
            if begin < end:
                pieces.append((destination, begin, end))
        return pieces

    async def _send(
        self,
        exchange: SparseAlltoall,
        destination: int,
        chunk: TableChunk,
        start: int,
        stop: int,
    ) -> None:
        """Pack and send one half-open chunk slice to a destination rank."""
        table = (
            plc.Table(chunk.table_view().columns())
            if start == 0 and stop == chunk.shape[0]
            else plc.copying.slice(
                chunk.table_view(), [start, stop], stream=chunk.stream
            )[0]
        )
        piece = TableChunk.from_pylibcudf_table(
            table,
            chunk.stream,
            exclusive_view=False,
            br=self.context.br(),
        )
        piece, extra = await make_table_chunks_available_or_wait(
            self.context,
            piece,
            reserve_extra=piece.into_packed_data_cost(),
            net_memory_delta=0,
        )
        exchange.insert(destination, piece.into_packed_data(extra))
