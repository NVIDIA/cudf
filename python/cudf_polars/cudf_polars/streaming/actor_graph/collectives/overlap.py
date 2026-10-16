# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-overlap planning and exchange for streaming operators."""

from __future__ import annotations

import operator
import struct
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Any

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

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import Rolling


type RowRange = tuple[int, int]
type RangeValues = tuple[Any, Any, Any, Any]
type GhostSend = tuple[int, int, int]


class RangeValidation(IntEnum):
    """Validation status for one rank's range index."""

    VALID = 0
    NULL = 1
    UNSORTED = 2
    OUT_OF_BOUNDS = 3


@dataclass
class BufferedChunk:
    """An available input chunk with its global row span."""

    chunk: TableChunk
    row_start: int
    num_rows: int

    @property
    def row_stop(self) -> int:
        """Return the first row after this chunk."""
        return self.row_start + self.num_rows


@dataclass
class StagedChunk:
    """A prepared chunk marked as locally owned or ghost data."""

    chunk: BufferedChunk
    owned: bool


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


class _OverlapPolicy[BufferedChunkT: BufferedChunk]:
    """Base policy for finding the rows needed by a rolling cursor."""

    def observe(self, chunk: BufferedChunkT) -> None:
        del chunk

    def close(self) -> None:
        pass

    def validate_cursor(self, cursor: BufferedChunkT) -> None:
        del cursor

    def evict_history(
        self,
        history: list[BufferedChunkT],
        cursor: BufferedChunkT,
    ) -> list[BufferedChunkT]:
        raise NotImplementedError

    def has_complete_future(
        self,
        latest: BufferedChunkT,
        current: BufferedChunkT,
    ) -> bool:
        raise NotImplementedError

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunkT,
    ) -> BufferedChunkT | None:
        """Return the cursor rows that future cursors may need."""
        del context
        return cursor

    async def overlap_bounds(
        self,
        context: Context,
        cursor: BufferedChunkT,
        chunks: Sequence[BufferedChunkT],
    ) -> tuple[int, int]:
        raise NotImplementedError

    async def prepare_chunk(
        self,
        context: Context,
        chunk: TableChunk,
        *,
        row_offset: int,
        host_values: RangeValues | None = None,
    ) -> BufferedChunkT:
        raise NotImplementedError


@dataclass
class RangeOverlapPolicy(_OverlapPolicy[RangeBufferedChunk]):
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
    async def _insertion_point(
        context: Context,
        index_column: plc.Column,
        needle: plc.Column,
        find: Callable[..., plc.Column],
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
                    stream=stream,
                    mr=context.br().device_mr,
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
        value: Any,
        find: Callable[..., plc.Column],
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

    def validate_cursor(self, cursor: RangeBufferedChunk) -> None:
        """Check that the index remains sorted across chunk boundaries."""
        if cursor.num_rows == 0:
            return
        assert cursor.overlap.first is not None
        assert cursor.overlap.last is not None
        if (
            self.previous_index_value is not None
            and cursor.overlap.first < self.previous_index_value
        ):
            self.raise_validation(RangeValidation.UNSORTED)
        self.previous_index_value = cursor.overlap.last

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
            index_column = chunk.table_view().columns()[self.index]
            if host_values is None and validate_index:
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
                if host_values is None:
                    host_values = (
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
                overlap = RangeOverlap(
                    index_column,
                    lower_bound,
                    upper_bound,
                    *host_values,
                )
        return RangeBufferedChunk(chunk, row_offset, nrows, overlap)

    def evict_history(
        self,
        history: list[RangeBufferedChunk],
        cursor: RangeBufferedChunk,
    ) -> list[RangeBufferedChunk]:
        """Drop history chunks that cannot contribute to the cursor chunk."""
        assert cursor.overlap.lower_bound is not None
        keep = operator.ge if self.start_closed else operator.gt
        return [
            chunk
            for chunk in history
            if chunk.overlap.last is not None
            and keep(chunk.overlap.last, cursor.overlap.lower_bound)
        ]

    def has_complete_future(
        self,
        latest: RangeBufferedChunk,
        current: RangeBufferedChunk,
    ) -> bool:
        """Return whether latest contains current's upper insertion point."""
        if latest.overlap.last is None:
            return False
        assert current.overlap.upper_bound is not None
        complete = operator.gt if self.end_closed else operator.ge
        return complete(latest.overlap.last, current.overlap.upper_bound)

    async def overlap_bounds(
        self,
        context: Context,
        cursor: RangeBufferedChunk,
        chunks: Sequence[RangeBufferedChunk],
    ) -> tuple[int, int]:
        """Return the rows needed to evaluate a range cursor."""
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
        return ghost_start, ghost_stop


@dataclass
class RowCountOverlapPolicy(_OverlapPolicy[BufferedChunk]):
    """Overlap policy for fixed-size rolling expressions."""

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
        ghost_start = max(0, cursor.row_start - self.preceding)
        return [chunk for chunk in history if chunk.row_stop > ghost_start]

    def has_complete_future(
        self,
        latest: BufferedChunk,
        current: BufferedChunk,
    ) -> bool:
        """Return whether latest contains enough leading rows."""
        return latest.row_stop >= current.row_stop + self.following

    async def history_chunk(
        self,
        context: Context,
        cursor: BufferedChunk,
    ) -> BufferedChunk | None:
        """Keep only emitted rows that a future fixed-size window can use."""
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
        ghost_start = max(0, cursor.row_start - self.preceding)
        ghost_stop = min(cursor.row_stop + self.following, chunks[-1].row_stop)
        return ghost_start, ghost_stop


type OverlapPolicy = _OverlapPolicy[Any]


@dataclass(frozen=True)
class _RangeSummary:
    rows: int
    validation: RangeValidation = RangeValidation.VALID
    lower: Any | None = None
    upper: Any | None = None
    first: Any | None = None
    last: Any | None = None


class OverlapInput:
    """Yield local input with the ghost rows required from other ranks."""

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        ch_in: Channel[TableChunk],
        policy: OverlapPolicy,
        *,
        duplicated: bool,
        collective_id: int,
    ) -> None:
        self.context = context
        self.comm = comm
        self.ch_in = ch_in
        self.policy = policy
        self.range_policy = policy if isinstance(policy, RangeOverlapPolicy) else None
        self.multirank = comm.nranks != 1 and not duplicated
        self.collective_id = collective_id
        self.buffer: ChunkStore | None = None
        self.exchange: SparseAlltoall | None = None
        self.left_sources: tuple[int, ...] = ()
        self.right_sources: tuple[int, ...] = ()
        self.host_values: list[RangeValues | None] = []

    async def __aenter__(self) -> OverlapInput:
        """Buffer and exchange input when it spans multiple ranks."""
        try:
            if self.multirank:
                await self._prepare()
        except BaseException:
            if self.buffer is not None:
                self.buffer.clear()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Release messages still owned by this input."""
        if self.buffer is not None:
            self.buffer.clear()
        self.exchange = None

    def __aiter__(self) -> AsyncIterator[StagedChunk]:
        """Yield prepared chunks in rolling evaluation order."""
        return self._chunks()

    async def _raw_owned_chunks(
        self,
    ) -> AsyncIterator[tuple[TableChunk, RangeValues | None]]:
        """Yield locally owned input chunks."""
        if not self.multirank:
            while (msg := await self.ch_in.recv(self.context)) is not None:
                yield TableChunk.from_message(msg, br=self.context.br()), None
        else:
            assert self.buffer is not None
            for position, msg in enumerate(self.buffer):
                yield (
                    TableChunk.from_message(msg, br=self.context.br()),
                    self.host_values[position]
                    if self.range_policy is not None
                    else None,
                )

    async def _raw_ghost_chunks(
        self, sources: Iterable[int]
    ) -> AsyncIterator[tuple[TableChunk, None]]:
        """Yield exchanged chunks from each source in rank order."""
        for source in sources:
            assert self.exchange is not None
            for packed in self.exchange.extract(source):
                yield TableChunk.from_packed_data(packed, self.context.br()), None

    async def _chunks(self) -> AsyncIterator[StagedChunk]:
        """Prepare owned and ghost chunks in logical row order."""
        row_offset = 0
        inputs = (
            (False, self._raw_ghost_chunks(self.left_sources)),
            (True, self._raw_owned_chunks()),
            (False, self._raw_ghost_chunks(self.right_sources)),
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

    async def _prepare(self) -> None:
        self.buffer = ChunkStore(self.context)
        summary = await self._buffer_input()
        gathered = await self._allgather(summary)
        starts = tuple(accumulate((item.rows for item in gathered), initial=0))
        if self.range_policy is not None:
            self._validate_ranges(gathered)
            sends, sources, destinations = self._range_plan(summary, gathered)
        else:
            sends, sources, destinations = self._count_plan(tuple(pairwise(starts)))
        await self._exchange(
            sends,
            sources,
            destinations,
            gathered,
            starts[self.comm.rank],
        )

    async def _buffer_input(self) -> _RangeSummary:
        assert self.buffer is not None
        range_policy = self.range_policy
        row_offset = 0
        validation = RangeValidation.VALID
        lower = upper = first = last = previous = None
        while (msg := await self.ch_in.recv(self.context)) is not None:
            table_chunk = TableChunk.from_message(msg, br=self.context.br())
            nrows, _ = table_chunk.shape
            if range_policy is not None:
                chunk = await range_policy.prepare_chunk(
                    self.context,
                    table_chunk,
                    row_offset=row_offset,
                    validate_index=False,
                )
                self.policy.observe(chunk)
                table_chunk = chunk.chunk
                overlap = chunk.overlap
                host_values = None
                if nrows:
                    status = range_policy.index_validation(
                        overlap.index_column, table_chunk.stream
                    )
                    if (
                        status is RangeValidation.VALID
                        and previous is not None
                        and overlap.first < previous
                    ):
                        status = RangeValidation.UNSORTED
                    if validation is RangeValidation.VALID:
                        validation = status
                    if status is RangeValidation.VALID:
                        if first is None:
                            first, lower = overlap.first, overlap.lower_bound
                        last, upper, previous = (
                            overlap.last,
                            overlap.upper_bound,
                            overlap.last,
                        )
                    host_values = (
                        overlap.first,
                        overlap.last,
                        overlap.lower_bound,
                        overlap.upper_bound,
                    )
                self.host_values.append(host_values)
            self.buffer.insert(Message(0, table_chunk))
            row_offset += nrows
        return _RangeSummary(row_offset, validation, lower, upper, first, last)

    async def _allgather(self, summary: _RangeSummary) -> tuple[_RangeSummary, ...]:
        values = (summary.lower, summary.upper, summary.first, summary.last)
        validation = summary.validation
        if any(value is not None and not -(2**63) <= value < 2**63 for value in values):
            validation = RangeValidation.OUT_OF_BOUNDS
        payload = (
            (
                summary.rows,
                int(validation),
                *(
                    value
                    if value is not None and validation != RangeValidation.OUT_OF_BOUNDS
                    else 0
                    for value in values
                ),
            )
            if self.range_policy is not None
            else (summary.rows,)
        )
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
        unpacked = (
            struct.unpack(fmt, item.to_host_bytes())
            for item in await gather.extract_all(self.context, ordered=True)
        )
        if self.range_policy is None:
            return tuple(_RangeSummary(item[0]) for item in unpacked)
        return tuple(
            _RangeSummary(
                item[0],
                RangeValidation(item[1]),
                *(item[i] if item[0] else None for i in range(2, 6)),
            )
            for item in unpacked
        )

    def _validate_ranges(self, summaries: Sequence[_RangeSummary]) -> None:
        assert self.range_policy is not None
        validations = {item.validation for item in summaries}
        if RangeValidation.OUT_OF_BOUNDS in validations:
            raise NotImplementedError(
                "Multi-rank rolling does not support index values outside the "
                "signed 64-bit range"
            )
        if RangeValidation.NULL in validations:
            self.range_policy.raise_validation(RangeValidation.NULL)
        previous = None
        for item in summaries:
            if item.validation is RangeValidation.UNSORTED:
                break
            if not item.rows:
                continue
            if (
                item.first is None
                or item.last is None
                or item.first > item.last
                or (previous is not None and item.first < previous)
            ):
                break
            previous = item.last
        else:
            return
        self.range_policy.raise_validation(RangeValidation.UNSORTED)

    @staticmethod
    def _intersection(left: RowRange, right: RowRange) -> RowRange | None:
        start, stop = max(left[0], right[0]), min(left[1], right[1])
        return (start, stop) if start < stop else None

    def _count_plan(
        self, spans: Sequence[RowRange]
    ) -> tuple[list[GhostSend], tuple[int, ...], tuple[int, ...]]:
        assert isinstance(self.policy, RowCountOverlapPolicy)
        total = spans[-1][1] if spans else 0
        requests = tuple(
            tuple(
                span
                for span in (
                    (max(0, start - self.policy.preceding), start),
                    (stop, min(total, stop + self.policy.following)),
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
        rank = self.comm.rank
        sources = tuple(
            source
            for source, item in enumerate(summaries)
            if source != rank
            and local.lower is not None
            and local.upper is not None
            and item.first is not None
            and item.last is not None
            and item.last >= local.lower
            and item.first <= local.upper
        )
        destinations = tuple(
            destination
            for destination, item in enumerate(summaries)
            if destination != rank
            and local.first is not None
            and local.last is not None
            and item.lower is not None
            and item.upper is not None
            and local.last >= item.lower
            and local.first <= item.upper
        )
        return [], sources, destinations

    async def _exchange(
        self,
        sends: Sequence[GhostSend],
        sources: Sequence[int],
        destinations: Sequence[int],
        summaries: Sequence[_RangeSummary],
        row_offset: int,
    ) -> None:
        assert self.buffer is not None
        self.left_sources = tuple(
            source for source in sources if source < self.comm.rank
        )
        self.right_sources = tuple(
            source for source in sources if source > self.comm.rank
        )
        destinations = tuple(sorted(set(destinations)))
        if not (self.left_sources or self.right_sources or destinations):
            return
        self.exchange = SparseAlltoall(
            self.context,
            self.comm,
            self.collective_id,
            srcs=self.left_sources + self.right_sources,
            dsts=destinations,
        )
        source, self.buffer = self.buffer, ChunkStore(self.context)
        start = 0
        try:
            for position, msg in enumerate(source):
                chunk = TableChunk.from_message(msg, br=self.context.br())
                size = chunk.shape[0]
                pieces: list[tuple[int, int, int]] = []
                if self.range_policy is not None and size:
                    values = self.host_values[position]
                    assert values is not None
                    first, last = values[:2]
                    candidates = tuple(
                        destination
                        for destination in destinations
                        if summaries[destination].lower is not None
                        and summaries[destination].upper is not None
                        and last >= summaries[destination].lower
                        and first <= summaries[destination].upper
                    )
                else:
                    span = (start + row_offset, start + row_offset + size)
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
                    candidates = ()
                if not candidates and not pieces:
                    self.buffer.insert(Message(0, chunk))
                    start += size
                    continue
                available, extra = await make_table_chunks_available_or_wait(
                    self.context,
                    chunk,
                    reserve_extra=0,
                    net_memory_delta=0,
                )
                with opaque_memory_usage(extra):
                    if candidates:
                        pieces = await self._range_pieces(
                            available,
                            size,
                            first,
                            last,
                            candidates,
                            summaries,
                        )
                    for destination, begin, end in pieces:
                        await self._send(destination, available, begin, end)
                    self.buffer.insert(Message(0, available))
                start += size
        finally:
            source.clear()
            await self.exchange.insert_finished(self.context)

    async def _range_pieces(
        self,
        chunk: TableChunk,
        size: int,
        first: Any,
        last: Any,
        destinations: Sequence[int],
        summaries: Sequence[_RangeSummary],
    ) -> list[tuple[int, int, int]]:
        policy = self.range_policy
        assert policy is not None
        pieces = []
        index = chunk.table_view().columns()[policy.index]
        for destination in destinations:
            lower = summaries[destination].lower
            upper = summaries[destination].upper
            assert lower is not None
            assert upper is not None
            begin = (
                0
                if lower < first or (policy.start_closed and lower == first)
                else await policy.insertion_point(
                    self.context,
                    index,
                    lower,
                    policy.find_start,
                    chunk.stream,
                )
            )
            end = (
                size
                if upper > last or (policy.end_closed and upper == last)
                else await policy.insertion_point(
                    self.context,
                    index,
                    upper,
                    policy.find_end,
                    chunk.stream,
                )
            )
            if begin < end:
                pieces.append((destination, begin, end))
        return pieces

    async def _send(
        self, destination: int, chunk: TableChunk, start: int, stop: int
    ) -> None:
        assert self.exchange is not None
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
        self.exchange.insert(destination, piece.into_packed_data(extra))
