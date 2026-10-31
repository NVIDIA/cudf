# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
External merge sort for one range partition of the streaming ``Sort``.

After the range shuffle, each rank owns a few partitions whose packed pieces
live in the shuffler (spillable by rapidsmpf, to disk when a spill directory
is configured). Instead of unpacking a whole partition into device memory and
sorting it in place, this module:

* **Phase A** unpacks the pieces in batches bounded by a device budget, merges
  each batch (every piece is already sorted), and copies it out of device
  memory as a *run* of pages; and
* **Phase B** streams a k-way merge of the runs (``plc.merge.merge`` on the
  current page of each run), emitting only rows that no unread page can
  precede, and refilling the run(s) that bound the emission. Runs beyond the
  fan-in are merged in passes.

The partition is never resident as a whole. All device allocations go through
``reserve_memory`` so they wait for the shuffler to spill rather than fail.
When a partition fits the batch budget, the previous in-memory behaviour is
used unchanged.

Run pages are packed ``TableChunk`` copies in rapidsmpf ``DISK`` memory, i.e.
files under the ``BufferResource``'s spill directory (``disk_spill_dir``) that
are deleted when the page is released. Without a spill directory the pages
fall back to ``HOST`` memory.

This mirrors spark-rapids' ``GpuSortExec`` and the standalone
``rapidsmpf-disk-sort`` prototype that validated the approach at 1 TB.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    Partitioning,
    RowOrdering,
    RowOrderingScope,
)
from cudf_streaming.partition_utils import (
    unpack_and_concat_cost as py_unpack_and_concat_cost,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory
from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.shuffle import ShuffleManager
from cudf_polars.streaming.actor_graph.collectives.sort_utils import (
    append_row_sequence_column,
    build_post_sort_ir,
    compute_sort_boundaries,
    insert_chunk_with_sort_splits,
    select_local_split_candidates,
    sort_to_order_keys,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChunkStore,
    _sample_chunks,
    allgather_reduce,
    chunk_to_frame,
    evaluate_chunk,
    send_metadata,
)
from cudf_polars.utils.config import get_total_device_memory
from cudf_polars.utils.cuda_stream import join_cuda_streams

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from cudf_streaming.channel_metadata import OrderKey
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.memory.packed_data import PackedData
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IRExecutionContext, Sort
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.typing import Schema
    from cudf_polars.utils.config import StreamingExecutor


# Runs merged per pass. With ~4-8 runs per partition (see ``batch_bytes_for``)
# a single pass is the norm; more runs simply add passes.
DEFAULT_MERGE_FANIN = 16

# Device budget for one sorted run, relative to the executor's chunk size and
# to device memory. Derived rather than configured: the partition count is
# already governed by ``target_partition_size``.
# One sorted run holds about one partition's worth of rows. Phase A keeps the
# unpacked batch and its sorted copy resident while the shuffler still owns
# most of the device budget for the partitions not yet extracted, so a larger
# multiple (4x, i.e. 32 GiB at an 8 GiB target) ran out of device memory at
# 3TB on 8 nodes; the standalone reference sort uses 8 GiB batches.
_BATCH_BYTES_TARGET_MULTIPLE = 1
_BATCH_BYTES_DEVICE_FRACTION = 0.10

# Lower bound on rows per run page, so a tiny budget (tests, or a very wide
# table) cannot degenerate into one page per row.
_MIN_PAGE_ROWS = 1024
# Merged output is emitted page by page (~batch/(2*fanin) bytes); the sink
# writes one file per emitted chunk, which produced ~48k ~8 MB files for a
# 3TB sort. Coalesce pages to at least this many bytes (or batch/4) first.
_MIN_COALESCE_BYTES = 256 << 20


class _PartitionSort(enum.Enum):
    """How :func:`sort_partition` sorted a partition."""

    IN_MEMORY = enum.auto()
    """The partition fit the batch budget and was sorted in device memory."""
    EXTERNAL = enum.auto()
    """Sorted runs were written out of device memory and merged."""


def batch_bytes_for(target_partition_size: int) -> int:
    """Device budget (bytes) for one sorted run of the external merge sort."""
    budget = _BATCH_BYTES_TARGET_MULTIPLE * target_partition_size
    total = get_total_device_memory()
    if total is not None:
        budget = min(budget, int(total * _BATCH_BYTES_DEVICE_FRACTION))
    return max(1, budget)


def run_memory_type(br: BufferResource) -> MemoryType:
    """
    Memory type that holds run pages.

    ``DISK`` when the buffer resource has a spill directory (rapidsmpf's
    ``disk_spill_dir``), so the pages share the shuffler's disk tier; otherwise
    ``HOST``.
    """
    return MemoryType.DISK if br.spill_directory is not None else MemoryType.HOST


def table_nbytes(table: plc.Table) -> int:
    """Device bytes held by ``table``."""
    return sum(column.device_buffer_size() for column in table.columns())


@dataclass
class RunPage:
    """One page of a sorted run, packed outside device memory."""

    chunk: TableChunk
    num_rows: int


@dataclass
class Run:
    """A sorted sequence of pages."""

    pages: list[RunPage] = field(default_factory=list)

    @property
    def num_rows(self) -> int:  # noqa: D102
        return sum(page.num_rows for page in self.pages)


class RunWriter:
    """Write sorted tables as pages of a run."""

    def __init__(self, context: Context, page_rows: int) -> None:
        self._context = context
        self._mem_type = run_memory_type(context.br())
        self._page_rows = max(1, page_rows)
        self._run = Run()

    async def write(
        self, table: plc.Table, stream: Stream, ir_context: IRExecutionContext
    ) -> None:
        """Append a sorted table (split into pages) to the run."""
        nrows = table.num_rows()
        if nrows == 0:
            return
        br = self._context.br()
        splits = list(range(self._page_rows, nrows, self._page_rows))
        pages = plc.copying.split(table, splits, stream=stream) if splits else [table]
        for page in pages:
            chunk = TableChunk.from_pylibcudf_table(
                page, stream, exclusive_view=False, br=br
            )
            nbytes = chunk.data_alloc_size(MemoryType.DEVICE)
            # Copying out of device memory packs the page into a temporary
            # device buffer first, which the copy itself does not reserve.
            pack_reservation = await reserve_memory(
                self._context, nbytes, net_memory_delta=0
            )
            reservation, _ = br.reserve(self._mem_type, nbytes, allow_overbooking=True)
            with opaque_memory_usage(pack_reservation):
                packed = await ir_context.to_thread(chunk.copy, reservation)
            self._run.pages.append(RunPage(packed, page.num_rows()))

    def finish(self) -> Run:
        """Return the completed run."""
        return self._run


class RunCursor:
    """Sequential reader over a run's pages, remembering the last key read."""

    def __init__(self, run: Run) -> None:
        self._pages: deque[RunPage] = deque(run.pages)
        # Key columns of the last row of the most recently loaded page. Every
        # unread row of this run compares >= tail, which is what bounds the
        # rows the merge may emit.
        self.tail: plc.Table | None = None

    def has_more(self) -> bool:
        """Whether unread pages remain."""
        return bool(self._pages)

    async def load_next(
        self,
        context: Context,
        ir_context: IRExecutionContext,
        key_indices: Sequence[int],
        stream: Stream,
    ) -> plc.Table:
        """
        Copy the next page into device memory (reserving it first).

        The page's off-device copy is released (and its file deleted) once loaded.
        """
        packed = self._pages.popleft().chunk
        nbytes = packed.make_available_cost()
        reservation = await reserve_memory(context, nbytes, net_memory_delta=nbytes)
        chunk = await ir_context.to_thread(packed.make_available, reservation)
        del packed
        join_cuda_streams(downstreams=[stream], upstreams=[chunk.stream])
        # The view keeps ``chunk`` alive.
        table = chunk.table_view()
        nrows = table.num_rows()
        keys = plc.Table([table.columns()[i] for i in key_indices])
        # Owning copy: a slice view would keep the whole page alive.
        self.tail = plc.copying.slice(keys, [nrows - 1, nrows], stream=stream)[0].copy(
            stream=stream
        )
        return table


def _emit_bound(
    merged_keys: plc.Table,
    pending: list[RunCursor],
    orders: list[plc.types.Order],
    null_orders: list[plc.types.NullOrder],
    stream: Stream,
) -> tuple[int, list[RunCursor]]:
    """
    Rows of ``merged`` that are final, and the cursors to refill.

    A row is final when its key is <= the smallest tail among runs with
    unread pages (nothing unread can precede it). The cursors whose tail
    equals that minimum are the ones that must be refilled next.
    """
    pending_tails = [cursor.tail for cursor in pending]
    assert all(tail is not None for tail in pending_tails)
    tails = plc.concatenate.concatenate(
        [tail for tail in pending_tails if tail is not None], stream=stream
    )
    order = plc.sorting.sorted_order(tails, orders, null_orders, stream=stream)
    min_tail = plc.copying.gather(
        tails,
        plc.copying.slice(order, [0, 1], stream=stream)[0],
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )
    cut_col = plc.search.upper_bound(
        merged_keys, min_tail, orders, null_orders, stream=stream
    )
    n_equal_col = plc.search.upper_bound(
        plc.copying.gather(
            tails, order, plc.copying.OutOfBoundsPolicy.DONT_CHECK, stream=stream
        ),
        min_tail,
        orders,
        null_orders,
        stream=stream,
    )
    # Everything above stays on the device, so a single sync covers all three
    # host copies. ``to_pylist`` copies on the default stream, hence the sync.
    # The columns have one INT32 (``size_type``) entry per pending run (at most
    # the merge fan-in), never one per row, so the copies are cheap.
    stream.synchronize()
    ranked = order.to_pylist()
    (cut,) = cut_col.to_pylist()
    (n_equal,) = n_equal_col.to_pylist()
    refill = [pending[i] for i in ranked[: max(1, n_equal)]]
    return cut, refill


async def merge_runs(
    context: Context,
    ir_context: IRExecutionContext,
    runs: Sequence[Run],
    order_keys: Sequence[OrderKey],
    emit: Callable[[plc.Table, Stream], Awaitable[None]],
    stream: Stream,
) -> None:
    """
    Stream a k-way merge of ``runs`` into ``emit``.

    At most one page per run plus the carried-over (not yet emittable) rows
    are resident at any time. ``stream`` must be the stream the runs' pages
    were written on: a loaded page is freed on that stream.
    """
    key_indices = [key.column_index for key in order_keys]
    orders = [key.order for key in order_keys]
    null_orders = [key.null_order for key in order_keys]
    mr = context.br().device_mr

    cursors = [RunCursor(run) for run in runs if run.pages]
    tables = [
        await cursor.load_next(context, ir_context, key_indices, stream)
        for cursor in cursors
    ]
    carry: plc.Table | None = None
    while tables or carry is not None:
        inputs = ([carry] if carry is not None else []) + tables
        if len(inputs) == 1:
            merged = inputs[0]
        else:
            reservation = await reserve_memory(
                context,
                sum(table_nbytes(table) for table in inputs),
                # The merged output replaces its inputs.
                net_memory_delta=0,
            )
            with opaque_memory_usage(reservation):
                merged = plc.merge.merge(
                    inputs, key_indices, orders, null_orders, stream=stream, mr=mr
                )
        del inputs, tables
        carry = None

        pending = [cursor for cursor in cursors if cursor.has_more()]
        if pending:
            merged_keys = plc.Table([merged.columns()[i] for i in key_indices])
            cut, refill = _emit_bound(merged_keys, pending, orders, null_orders, stream)
        else:
            cut, refill = merged.num_rows(), []

        nrows = merged.num_rows()
        if cut >= nrows:
            await emit(merged, stream)
        else:
            head, rest = plc.copying.split(merged, [cut], stream=stream)
            if cut > 0:
                await emit(head.copy(stream=stream), stream)
            carry = rest.copy(stream=stream)
        del merged

        tables = [
            await cursor.load_next(context, ir_context, key_indices, stream)
            for cursor in refill
        ]


async def _unpack_and_merge(
    context: Context,
    pieces: Sequence[PackedData],
    order_keys: Sequence[OrderKey],
    stream: Stream,
) -> plc.Table:
    """
    Unpack packed pieces and merge them into one sorted table.

    Every piece is a contiguous slice of a locally sorted input chunk (see
    :class:`_ChunkPreparer`), so a k-way merge suffices; no re-sort is needed.
    """
    br = context.br()
    # Device-resident pieces unpack as zero-copy views; spilled pieces are
    # copied back to device first.
    chunks, _ = await make_table_chunks_available_or_wait(
        context,
        [TableChunk.from_packed_data(piece, br) for piece in pieces],
        reserve_extra=0,
        # Representation change: packed input consumed as the views are produced.
        net_memory_delta=0,
    )
    join_cuda_streams(downstreams=[stream], upstreams=[c.stream for c in chunks])
    # Keep one chunk, even if empty, to carry the schema.
    chunks = [c for c in chunks if c.table_view().num_rows() > 0] or chunks[:1]
    if len(chunks) == 1:
        # The view keeps the chunk alive.
        return chunks[0].table_view()
    # The merged output replaces its inputs.
    merge_reservation = await reserve_memory(
        context,
        sum(c.data_alloc_size(MemoryType.DEVICE) for c in chunks),
        net_memory_delta=0,
    )
    with opaque_memory_usage(merge_reservation):
        merged = plc.merge.merge(
            [c.table_view() for c in chunks],
            [key.column_index for key in order_keys],
            [key.order for key in order_keys],
            [key.null_order for key in order_keys],
            stream=stream,
            mr=br.device_mr,
        )
    # The inputs are freed on their own streams once dropped.
    join_cuda_streams(downstreams=[c.stream for c in chunks], upstreams=[stream])
    return merged


async def sort_partition(
    context: Context,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    pieces: list[PackedData],
    order_keys: Sequence[OrderKey],
    output_schema: Schema,
    partition_id: int,
    *,
    batch_bytes: int,
    tracer: ActorTracer | None,
) -> _PartitionSort:
    """Sort one owned partition, given its packed ``pieces``, and send it on ``ch_out``."""
    ncols_out = len(output_schema)

    async def emit(table: plc.Table, stream: Stream) -> None:
        if table.num_rows() == 0:
            return
        if table.num_columns() > ncols_out:
            table = plc.Table(table.columns()[:ncols_out])
        chunk = TableChunk.from_pylibcudf_table(
            table, stream, exclusive_view=True, br=context.br()
        )
        await send_chunk(context, ch_out, chunk, partition_id, tracer=tracer)

    sizes = [py_unpack_and_concat_cost([piece]) for piece in pieces]
    total = sum(sizes)
    stream = ir_context.get_cuda_stream()

    if total <= batch_bytes:
        table = await _unpack_and_merge(context, pieces, order_keys, stream)
        await emit(table, stream)
        return _PartitionSort.IN_MEMORY

    # Run pages live outside device memory (on disk with a spill directory)
    # and are released with the runs, also when an exception unwinds.
    fanin = DEFAULT_MERGE_FANIN
    # Phase A: sorted runs.
    runs: list[Run] = []
    page_rows: int | None = None
    batch: list[PackedData] = []
    batch_nbytes = 0

    async def flush() -> None:
        nonlocal batch, batch_nbytes, page_rows
        if not batch:
            return
        table = await _unpack_and_merge(context, batch, order_keys, stream)
        if page_rows is None:
            # A page is 1/(2*fanin) of the budget so that `fanin` pages
            # plus the merge output fit within one batch budget.
            bytes_per_row = max(1, batch_nbytes // max(1, table.num_rows()))
            page_rows = max(
                _MIN_PAGE_ROWS, (batch_bytes // (2 * fanin)) // bytes_per_row
            )
        writer = RunWriter(context, page_rows)
        await writer.write(table, stream, ir_context)
        del table
        runs.append(writer.finish())
        batch, batch_nbytes = [], 0

    for piece, size in zip(pieces, sizes, strict=True):
        if batch and batch_nbytes + size > batch_bytes:
            await flush()
        batch.append(piece)
        batch_nbytes += size
    await flush()
    del pieces

    # Phase B: merge passes until the runs fit one fan-in, then stream out.
    while len(runs) > fanin:
        next_runs: list[Run] = []
        for start in range(0, len(runs), fanin):
            writer = RunWriter(context, page_rows or 1)

            async def to_run(
                table: plc.Table, s: Stream, w: RunWriter = writer
            ) -> None:
                await w.write(table, s, ir_context)

            await merge_runs(
                context,
                ir_context,
                runs[start : start + fanin],
                order_keys,
                to_run,
                stream,
            )
            next_runs.append(writer.finish())
        runs = next_runs
    # Final pass: coalesce merged pages into sink-sized chunks before emitting.
    coalesce_bytes = max(_MIN_COALESCE_BYTES, batch_bytes // 4)
    pending_tables: list[plc.Table] = []
    pending_bytes = 0

    async def flush_coalesced(s: Stream) -> None:
        nonlocal pending_tables, pending_bytes
        if not pending_tables:
            return
        if len(pending_tables) == 1:
            table = pending_tables[0]
        else:
            # The concatenated copy is transient: inputs are dropped right after.
            reservation = await reserve_memory(
                context, pending_bytes, net_memory_delta=0
            )
            with opaque_memory_usage(reservation):
                table = plc.concatenate.concatenate(pending_tables, stream=s)
        pending_tables, pending_bytes = [], 0
        await emit(table, s)

    async def emit_coalesced(table: plc.Table, s: Stream) -> None:
        nonlocal pending_bytes
        if table.num_rows() == 0:
            return
        pending_tables.append(table)
        pending_bytes += table_nbytes(table)
        if pending_bytes >= coalesce_bytes:
            await flush_coalesced(s)

    await merge_runs(context, ir_context, runs, order_keys, emit_coalesced, stream)
    await flush_coalesced(stream)
    return _PartitionSort.EXTERNAL


# Number of leading chunks sampled per rank to derive range-partition boundaries
# when dynamic planning is disabled (with dynamic planning,
# ``DynamicPlanningOptions.sample_chunk_count`` is used). Boundaries only need to
# be *consistent* across ranks for correctness; sample size affects balance.
_BOUNDARY_SAMPLE_CHUNKS = 8


async def _sample_chunks_for_boundaries(
    context: Context,
    comm: Communicator,
    ch_in: Channel[TableChunk],
    num_partitions: int,
    metadata_in: ChannelMetadata,
    executor: StreamingExecutor,
    collective_ids: list[int],
) -> tuple[ChunkStore, int]:
    """
    Sample an input prefix for range boundaries and the size estimate.

    With dynamic planning the same sample also derives ``num_partitions``.

    Unlike the previous ``ChunkStore``-based sort, the sample is the *only*
    input buffered before the shuffle starts; every later chunk is
    range-split and inserted into the (disk-spilling) Shuffler as it arrives.
    """
    target_partition_size = executor.target_partition_size
    if executor.dynamic_planning is not None:
        sample_chunk_count = executor.dynamic_planning.sample_chunk_count
    else:
        sample_chunk_count = _BOUNDARY_SAMPLE_CHUNKS

    sample = await _sample_chunks(
        context,
        ch_in,
        sample_chunk_count,
        # Cap the sample at one target partition per sampled chunk.
        sample_chunk_count * target_partition_size,
        metadata_in.local_count,
    )

    if executor.dynamic_planning is not None:
        size_estimate_id = collective_ids.pop()
        if comm.nranks > 1 and not metadata_in.duplicated:
            (global_size,) = await allgather_reduce(
                context, comm, size_estimate_id, sample.total_size
            )
        else:
            global_size = sample.total_size
        num_partitions = max(1, math.ceil(global_size / target_partition_size))

    return sample.local_sample.chunks, num_partitions


class _ChunkPreparer:
    """
    Locally sort input chunks and tag them for a stable sort.

    For stable sorts a sequence column is appended that keeps the post-shuffle
    sort stable across chunks and ranks.

    Chunks must be prepared in arrival order: the sequence column encodes
    ``(rank << 48) + running local row offset``.
    """

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        ir: Sort,
        ir_context: IRExecutionContext,
    ) -> None:
        self._context = context
        self._comm = comm
        self._ir = ir
        self._ir_context = ir_context
        self._local_row_offset = 0

    async def prepare(self, msg: Message) -> tuple[int, DataFrame, plc.Table]:
        """Return ``(seq_num, sorted frame, table with optional sequence column)``."""
        df = chunk_to_frame(
            # Make sure chunks are pre-sorted
            await evaluate_chunk(
                self._context,
                TableChunk.from_message(msg, br=self._context.br()),
                self._ir,
                ir_context=self._ir_context,
            ),
            self._ir,
        )
        if self._ir.stable:
            tbl = append_row_sequence_column(
                df, self._comm.rank, self._local_row_offset
            )
            self._local_row_offset += tbl.num_rows()
        else:
            tbl = df.table
        return msg.sequence_number, df, tbl

    async def prepare_sample(
        self, sampled_chunks: ChunkStore, by: list[str], num_partitions: int
    ) -> tuple[ChunkStore, list[TableChunk]]:
        """Prepare the sampled prefix (before any other chunk); return it and its split candidates."""
        br = self._context.br()
        prepared_sample = ChunkStore(self._context)
        local_candidates_list: list[TableChunk] = []
        for msg in sampled_chunks:
            seq_num, df, tbl = await self.prepare(msg)
            local_candidates_list.append(
                TableChunk.from_pylibcudf_table(
                    select_local_split_candidates(
                        df.select(by), by, num_partitions, seq_num
                    ).table,
                    df.stream,
                    exclusive_view=True,
                    br=br,
                )
            )
            prepared_sample.insert(
                Message(
                    seq_num,
                    TableChunk.from_pylibcudf_table(
                        tbl, df.stream, exclusive_view=True, br=br
                    ),
                )
            )
            del df
        return prepared_sample, local_candidates_list


async def _stream_chunks_into_shuffle(
    context: Context,
    comm: Communicator,
    ir: Sort,
    ir_context: IRExecutionContext,
    shuffle: ShuffleManager,
    prepared_sample: ChunkStore,
    preparer: _ChunkPreparer,
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    sort_boundaries_df: DataFrame,
    by: list[str],
) -> None:
    """
    Range-split and insert the sample, then every remaining chunk, into ``shuffle``.

    Chunks after the sample are prepared and inserted one at a time as they
    arrive.

    At most one input chunk (plus shuffler-owned packed buffers, which
    rapidsmpf spills — to disk when configured) is resident per rank.
    """
    by_indices = names_to_indices(tuple(by), ir.schema)

    skip_insert = metadata_in.duplicated and comm.rank != 0

    async def insert_prepared(seq_num: int, chunk: TableChunk) -> None:
        await insert_chunk_with_sort_splits(
            context,
            inserter,
            chunk,
            seq_num,
            ir,
            ir_context,
            sort_boundaries_df,
            by_indices,
        )

    async with shuffle.inserting() as inserter:
        # 1. The sampled prefix, already sorted and sequence-tagged.
        for msg in prepared_sample:
            if skip_insert:
                continue
            await insert_prepared(
                msg.sequence_number, TableChunk.from_message(msg, br=context.br())
            )
        # 2. Everything else, one chunk at a time, straight from the input.
        while (msg := await ch_in.recv(context)) is not None:
            if skip_insert:
                continue
            seq_num, df, tbl = await preparer.prepare(msg)
            chunk = TableChunk.from_pylibcudf_table(
                tbl, df.stream, exclusive_view=True, br=context.br()
            )
            del df
            await insert_prepared(seq_num, chunk)


async def global_sort_external(
    context: Context,
    comm: Communicator,
    ir: Sort,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    by: list[str],
    num_partitions: int,
    executor: StreamingExecutor,
    collective_ids: list[int],
    *,
    tracer: ActorTracer | None,
) -> None:
    """
    Out-of-core global sort (``sort_strategy="external"``).

    Range boundaries come from a sampled input prefix only (as in Spark's
    ``RangePartitioner``); the rest of the input is never buffered but
    streamed into the (disk-spilling) shuffler. Each owned partition is then
    sorted with the external merge sort, so neither the shard nor a partition
    is ever resident whole.

    The tracer decision ``"external_sort"`` is recorded only when runs were
    actually written: partitions that fit the run budget are sorted in memory
    even under this strategy.
    """
    sampled_chunks, num_partitions = await _sample_chunks_for_boundaries(
        context,
        comm,
        ch_in,
        num_partitions,
        metadata_in,
        executor,
        collective_ids,
    )

    preparer = _ChunkPreparer(context, comm, ir, ir_context)
    prepared_sample, local_candidates_list = await preparer.prepare_sample(
        sampled_chunks, by, num_partitions
    )

    need_allgather = comm.nranks > 1 and not metadata_in.duplicated
    sort_boundaries_df = await compute_sort_boundaries(
        context,
        comm,
        ir_context,
        local_candidates_list,
        ir,
        by,
        num_partitions,
        collective_ids.pop() if need_allgather else None,
    )

    await send_metadata(
        ch_out,
        context,
        ChannelMetadata(
            local_count=max(1, num_partitions // comm.nranks),
            # Only claim row ordering: the emitted chunks are not guaranteed
            # to line up with ``sort_boundaries_df``, so no boundaries promise.
            partitioning=Partitioning(
                row_orderings=(
                    RowOrdering(sort_to_order_keys(ir), RowOrderingScope.GLOBAL),
                ),
            ),
        ),
    )

    shuffle = ShuffleManager(
        context,
        comm,
        num_partitions,
        collective_ids.pop(),
        partition_assignment=PartitionAssignment.CONTIGUOUS,
    )

    await _stream_chunks_into_shuffle(
        context,
        comm,
        ir,
        ir_context,
        shuffle,
        prepared_sample,
        preparer,
        ch_in,
        metadata_in,
        sort_boundaries_df,
        by,
    )

    order_keys = sort_to_order_keys(build_post_sort_ir(ir))
    batch_bytes = batch_bytes_for(executor.target_partition_size)
    any_external = False
    for partition_id in shuffle.local_partitions():
        method = await sort_partition(
            context,
            ir_context,
            ch_out,
            shuffle.extract_pieces(partition_id),
            order_keys,
            ir.schema,
            partition_id,
            batch_bytes=batch_bytes,
            tracer=tracer,
        )
        any_external |= method is _PartitionSort.EXTERNAL
    if tracer is not None and any_external:
        tracer.decision = "external_sort"

    await ch_out.drain(context)
