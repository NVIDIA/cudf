# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid scan prefetch pipeline."""

from __future__ import annotations

import asyncio
import concurrent.futures
import ctypes
import sys
import threading
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import nvtx

import pylibcudf as plc
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

from cudf_polars.dsl.ir import _prepare_parquet_predicate
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.dsl.tracing import CUDF_POLARS_NVTX_DOMAIN, nvtx_annotate_cudf_polars
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.io import (
    PinnedBatch,
    PrefetchedByteRanges,
    SplitScan,
    StreamingScan,
    decide_pass_mode,
    hybrid_scan_eligible,
    split_row_group_indices,
)
from cudf_polars.utils.config import HybridScanPassMode
from cudf_polars.utils.cuda_stream import get_cuda_stream

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from kvikio.cufile import CuFile, IOFuture
    from kvikio.remote_file import RemoteFile

    import pylibcudf.expressions as plc_expr
    from rapidsmpf.memory.buffer import Buffer, BufferHostView
    from rapidsmpf.memory.memory_reservation import MemoryReservation
    from rapidsmpf.memory.pinned_memory_resource import PinnedMemoryResource
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext


# TODO: kvikio is adding a remote batch API that coalesces and splits ranges
# itself. Once that lands, coalesce_adjacent_ranges and issue_pread_calls
# below should likely be replaced by calls into it instead of maintaining our
# own coalescing logic.
class CoalescedRange(NamedTuple):
    """One run of consecutive, file-adjacent ranges, merged into a single span."""

    host_start: int
    """Start offset into the destination host buffer."""
    host_end: int
    """End offset into the destination host buffer."""
    file_offset: int
    """Start offset into the source file."""
    file_size: int
    """Number of bytes to read from the source file."""


@nvtx_annotate_cudf_polars(message="coalesce_adjacent_ranges")
def coalesce_adjacent_ranges(ranges: list[Any]) -> list[CoalescedRange]:
    """
    Merge runs of consecutive, file-adjacent ranges into single spans.

    Pure endpoint-adjustment logic, no I/O: computes where each merged span
    starts and ends in both the destination host buffer and the source
    file, for a caller to issue reads against.

    Parameters
    ----------
    ranges
        Byte ranges to coalesce, in the order the caller needs them back in
        (positional, matching a ``HybridScanReader`` byte-range call). Only
        *runs* of already-adjacent entries get merged, not the whole list
        resorted by file offset. Column chunk ranges come back roughly
        file-ordered already, so this still catches most of the coalescing
        benefit without touching the order the caller depends on.

    Returns
    -------
    One :class:`CoalescedRange` per coalesced run, in file order.
    """
    groups = []
    offset = 0
    i = 0
    n = len(ranges)
    while i < n:
        group_start = offset
        group_file_start = ranges[i].offset
        group_file_end = group_file_start + ranges[i].size
        offset += ranges[i].size
        j = i + 1
        while j < n and ranges[j].offset == group_file_end:
            group_file_end += ranges[j].size
            offset += ranges[j].size
            j += 1
        groups.append(
            CoalescedRange(
                group_start, offset, group_file_start, group_file_end - group_file_start
            )
        )
        i = j
    return groups


@nvtx_annotate_cudf_polars(message="issue_pread_calls")
def issue_pread_calls(
    handle: CuFile | RemoteFile, ranges: list[Any], host: memoryview
) -> list[IOFuture]:
    """
    Issue one ``pread`` call per run of consecutive, file-adjacent ranges.

    Parameters
    ----------
    handle
        Open kvikio handle to read from.
    ranges
        Byte ranges to read; coalesced via :func:`coalesce_adjacent_ranges`
        before issuing reads.
    host
        Pinned host buffer to read into, sized to fit every range in
        ``ranges`` back to back.

    Returns
    -------
    One ``pread`` future per coalesced run, in file order.
    """
    return [
        handle.pread(
            host[coalesced.host_start : coalesced.host_end],
            size=coalesced.file_size,
            file_offset=coalesced.file_offset,
        )
        for coalesced in coalesce_adjacent_ranges(ranges)
    ]


@nvtx_annotate_cudf_polars(message="issue_reads_into_pinned_buffer")
def issue_reads_into_pinned_buffer(
    buf: Buffer, handle: CuFile | RemoteFile, ranges: list[Any]
) -> tuple[BufferHostView, memoryview, list[IOFuture]]:
    """
    Take a pinned host view of a buffer and issue reads into it.

    A single unit of work so it can be offloaded to a thread together:
    kvikio's ``pread`` submission releases the GIL, but that only lets other
    *threads* make progress. It's still a synchronous call from the event
    loop's perspective (no ``await``), so a slow submission, e.g. contention
    on kvikio's reactor from many concurrent prefetch tasks submitting
    around the same time, would otherwise stall the event loop and every
    other task on it.

    Deliberately doesn't use ``with buf.host_view() as host:`` here: the
    view's exclusive write lock is meant to stay held until the writes
    it's guarding are actually done, but the ``pread`` futures issued
    below are still in flight when this function returns, they're only
    awaited much later (see ``copy_pinned_batch_to_device``). Exiting the
    context manager here would release that lock before the writes it's
    protecting have finished. The caller is responsible for exiting the
    returned view once the returned futures actually complete.

    Parameters
    ----------
    buf
        The pinned buffer to view and read into.
    handle
        Open kvikio handle to read from.
    ranges
        Byte ranges to read.

    Returns
    -------
    The still-open view, the pinned host view read into, and one ``pread``
    future per coalesced run issued against it, in file order.
    """
    view = buf.host_view()
    host = view.__enter__()
    try:
        futures = issue_pread_calls(handle, ranges, host)
    except BaseException:
        view.__exit__(*sys.exc_info())
        raise
    return view, host, futures


@nvtx_annotate_cudf_polars(message="make_buffer_and_issue_reads")
def make_buffer_and_issue_reads(
    br: Any,
    total: int,
    stream: Any,
    reservation: Any,
    handle: CuFile | RemoteFile,
    ranges: list[Any],
) -> tuple[Buffer, BufferHostView, memoryview, list[IOFuture]]:
    """
    Allocate a pinned buffer and issue its reads, in one call on one thread.

    Combines what were previously two separate ``to_prefetch_thread`` hops
    (``br.make_buffer`` then :func:`issue_reads_into_pinned_buffer`) into a
    single dispatch, so a split only has to round-trip through the event
    loop once here instead of twice.

    Parameters
    ----------
    br
        The buffer resource to allocate the pinned buffer from.
    total
        Size, in bytes, of the pinned buffer to allocate.
    stream
        CUDA stream to allocate on.
    reservation
        Memory reservation covering ``total`` bytes, already claimed on the
        event loop before this call was dispatched.
    handle
        Open kvikio handle to read from.
    ranges
        Byte ranges to read.

    Returns
    -------
    The allocated buffer, the still-open view backing ``host``, the pinned
    host view read into, and one ``pread`` future per coalesced run.
    """
    buf = br.make_buffer(total, stream, reservation)
    view, host, futures = issue_reads_into_pinned_buffer(buf, handle, ranges)
    return buf, view, host, futures


async def reserve_pinned_batch(
    context: Context,
    ir_context: IRExecutionContext,
    handle: CuFile | RemoteFile,
    ranges: list[Any],
) -> PinnedBatch | None:
    """
    Reserve pinned host memory and issue reads for one batch of byte ranges.

    Parameters
    ----------
    context
        The rapidsmpf context to reserve memory through.
    ir_context
        The execution context to offload the buffer allocation and reads to.
    handle
        Open kvikio handle to read from.
    ranges
        Byte ranges to reserve for and read.

    Returns
    -------
    The reserved, in-flight batch, or ``None`` when ``ranges`` is empty.
    """
    if not ranges:
        return None
    total = sum(r.size for r in ranges)
    br = context.br()
    # `nvtx.start_range`/`end_range` (not the push/pop `nvtx_annotate_cudf_polars`
    # context manager) since this span crosses `await` points; many prefetch
    # tasks interleave on the same event-loop thread, which would corrupt a
    # thread-local push/pop stack.
    batch_range = nvtx.start_range(
        message="reserve_pinned_batch", domain=CUDF_POLARS_NVTX_DOMAIN, payload=total
    )
    try:
        # TODO: a reservation here can queue behind other pinned memory
        # contention (e.g. shuffle spill) with no way to proactively free up
        # our own holdings.
        wait_range = nvtx.start_range(
            message="reserve_memory_wait", domain=CUDF_POLARS_NVTX_DOMAIN, payload=total
        )
        try:
            reservation = await reserve_memory(
                context,
                size=total,
                net_memory_delta=total,
                mem_type=MemoryType.PINNED_HOST,
            )
        finally:
            nvtx.end_range(wait_range)
        buf, view, host, futures = await ir_context.to_prefetch_thread(
            make_buffer_and_issue_reads,
            br,
            total,
            br.stream_pool.get_stream(),
            reservation,
            handle,
            ranges,
        )
    finally:
        nvtx.end_range(batch_range)
    return PinnedBatch(ranges=ranges, host=host, futures=futures, buf=buf, view=view)


@nvtx_annotate_cudf_polars(message="prepare_prefetch")
def prepare_prefetch(
    scan: SplitScan,
) -> (
    tuple[
        plc_expr.Expression, list[int], HybridScanPassMode, list[Any], list[Any] | None
    ]
    | None
):
    """
    Prune row groups for one scan task and compute its byte ranges.

    Parameters
    ----------
    scan
        The scan task to prune and compute byte ranges for.

    Returns
    -------
    ``None`` when the predicate can't be expressed as a parquet filter.
    Otherwise ``(plc_filter, row_group_indices, pass_mode, primary_ranges,
    payload_ranges)``. ``payload_ranges`` is ``None`` under ``SINGLE_PASS``
    (``primary_ranges`` already covers every column), and a second list
    under ``TWO_PASS`` (``primary_ranges`` is the filter columns' ranges).
    """
    cached_info = scan.cached_parquet_info
    assert cached_info is not None
    predicate = scan.base_scan.predicate
    assert predicate is not None
    stream = get_cuda_stream()

    plc_filter, residual = to_parquet_filter(
        _prepare_parquet_predicate(
            predicate.value, scan.paths, scan.schema, scan.base_scan.with_columns
        ),
        stream=stream,
    )
    if plc_filter is None or residual is not None:
        return None

    row_group_indices = split_row_group_indices(
        len(cached_info[0].file_metadata.row_group_num_rows),
        scan.total_splits,
        scan.split_index,
    )
    row_group_count_before_pruning = len(row_group_indices)

    options = cached_info[0].default_reader_options()
    if scan.base_scan.with_columns is not None:
        options.set_column_names(scan.base_scan.with_columns)
    options.set_filter(plc_filter)
    reader = cached_info[0].hybrid_scan_reader(options)

    parquet_options = scan.parquet_options
    if parquet_options._hybrid_scan_stats_pruning:
        row_group_indices = reader.filter_row_groups_with_stats(
            row_group_indices, options, stream=stream
        )
        if row_group_indices:
            bloom_ranges = reader.bloom_filters_byte_ranges(row_group_indices, options)
            if bloom_ranges:
                source_info = plc.io.SourceInfo(
                    [
                        plc.io.types.FilepathSource(
                            cached_info[0].path, cached_info[0].size
                        )
                    ]
                )
                bloom_chunks = plc.io.parquet_io_utils.fetch_byte_ranges_to_device(
                    source_info, bloom_ranges, stream=stream
                )
                row_group_indices = reader.filter_row_groups_with_bloom_filters(
                    bloom_chunks, row_group_indices, options, stream=stream
                )

    if not row_group_indices:
        return plc_filter, [], HybridScanPassMode.SINGLE_PASS, [], None

    pass_mode = decide_pass_mode(
        parquet_options.pass_mode, row_group_indices, row_group_count_before_pruning
    )
    if pass_mode is HybridScanPassMode.SINGLE_PASS:
        ranges = reader.all_column_chunks_byte_ranges(row_group_indices, options)
        return plc_filter, row_group_indices, pass_mode, ranges, None
    filter_ranges = reader.filter_column_chunks_byte_ranges(row_group_indices, options)
    payload_ranges = reader.payload_column_chunks_byte_ranges(
        row_group_indices, options
    )
    return plc_filter, row_group_indices, pass_mode, filter_ranges, payload_ranges


def submit_prefetch_plans_for_ir(
    root: IR,
    py_executor: concurrent.futures.Executor,
) -> dict[SplitScan, concurrent.futures.Future[Any]]:
    """
    Submit row-group pruning for every hybrid-scan-eligible split in a query.

    Pruning (:func:`prepare_prefetch`) only needs a split's already-attached
    ``cached_parquet_info`` -- no pinned memory, no actor graph. Called once,
    before the actor graph starts, right after cached parquet metadata is
    attached, so pruning for every eligible split across the whole query
    graph runs concurrently with actor graph construction instead of
    waiting for each scan node to start.

    Parameters
    ----------
    root
        The root of the IR graph to traverse.
    py_executor
        The thread pool to submit pruning work to.

    Returns
    -------
    One future per eligible split, keyed by the split itself.
    """
    plans: dict[SplitScan, concurrent.futures.Future[Any]] = {}
    for node in traversal([root]):
        if isinstance(node, StreamingScan) and node.base_scan.typ == "parquet":
            for scan in node.scans:
                if isinstance(scan, SplitScan) and _eligible_for_planning(scan):
                    plans[scan] = py_executor.submit(prepare_prefetch, scan)
    return plans


def _eligible_for_planning(scan: SplitScan) -> bool:
    if not hybrid_scan_eligible(
        scan.parquet_options,
        cached_parquet_info=scan.cached_parquet_info,
        row_index=scan.base_scan.row_index,
        include_file_paths=scan.base_scan.include_file_paths,
        predicate=scan.base_scan.predicate,
    ):
        return False
    assert scan.cached_parquet_info is not None
    total_row_groups = len(scan.cached_parquet_info[0].file_metadata.row_group_num_rows)
    # Matches `prefetch_eligible`'s guard: beyond this, the split isn't
    # row-group-aligned and won't actually be prefetched, so pruning it
    # here would be wasted work.
    return scan.total_splits <= total_row_groups


async def resolve_prefetch_plan(
    scan: SplitScan, ir_context: IRExecutionContext
) -> (
    tuple[
        plc_expr.Expression, list[int], HybridScanPassMode, list[Any], list[Any] | None
    ]
    | None
):
    """
    Return one split's pruning result, submitted ahead of time for ``scan``.

    Parameters
    ----------
    scan
        The split to prune.
    ir_context
        The execution context holding the future submitted by
        :func:`submit_prefetch_plans_for_ir` for ``scan``.

    Returns
    -------
    Same as :func:`prepare_prefetch`.
    """
    return await asyncio.wrap_future(ir_context.pending_prefetch_plans[scan])


class BatchedSplit(NamedTuple):
    """One split's placement within a shared fetch batch."""

    seq_num: int
    """This split's index into its scan's ``scans``, matching :class:`PlannedSplit`."""
    ranges: list[Any]
    """This split's own byte ranges, unchanged from planning."""
    host_offset: int
    """Start offset of this split's data within the shared batch buffer."""


class FetchBatch(NamedTuple):
    """One group of splits' byte ranges to fetch together, sharing one buffer."""

    splits: list[BatchedSplit]
    """Every split sharing this batch's buffer, in the order packed."""
    total_bytes: int
    """Combined size of every split's ranges in this batch."""


def pack_batches(
    entries: Iterable[tuple[int, list[Any]]], batch_bytes: int
) -> list[FetchBatch]:
    """
    Greedily group splits' byte ranges into byte-budget-bounded batches.

    Preserves the order ``entries`` is given in: a batch never reorders
    splits for coalescing efficiency, it only decides where one batch ends
    and the next begins. A single split's own ranges are never split
    across two batches -- if one split's ranges alone exceed
    ``batch_bytes``, it becomes its own, over-budget batch; the budget
    bounds how many splits get grouped together, not an absolute ceiling
    on any single batch.

    Parameters
    ----------
    entries
        ``(seq_num, ranges)`` pairs, one per split, in packing order.
        Splits with no ranges are skipped.
    batch_bytes
        Target maximum combined size of one batch, in bytes.

    Returns
    -------
    One :class:`FetchBatch` per group, in packing order.
    """
    batches: list[FetchBatch] = []
    current: list[BatchedSplit] = []
    current_total = 0
    for seq_num, ranges in entries:
        size = sum(r.size for r in ranges)
        if size == 0:
            continue
        if current and current_total + size > batch_bytes:
            batches.append(FetchBatch(current, current_total))
            current = []
            current_total = 0
        current.append(BatchedSplit(seq_num, ranges, current_total))
        current_total += size
    if current:
        batches.append(FetchBatch(current, current_total))
    return batches


class _OnceFuture:
    """
    Wraps an ``IOFuture`` so ``.get()`` is safe to call more than once.

    ``IOFuture.get()`` consumes the underlying ``std::future``'s state, so
    calling it a second time raises. Every split sharing a batch gets the
    same combined future list (see :func:`fetch_batch`), so more than one
    caller can end up calling ``.get()`` on the same future -- this makes
    only the first call do real work, caching the outcome (result or
    exception) for the rest.
    """

    __slots__ = ("_done", "_exc", "_future", "_lock")

    def __init__(self, future: IOFuture) -> None:
        self._future = future
        self._lock = threading.Lock()
        self._done = False
        self._exc: BaseException | None = None

    def get(self) -> None:
        """Wait for the wrapped future, raising its exception on every call if it failed."""
        with self._lock:
            if not self._done:
                try:
                    self._future.get()
                except BaseException as e:
                    self._exc = e
                finally:
                    self._done = True
        if self._exc is not None:
            raise self._exc


def _make_batch_buffer_and_issue_reads(
    br: Any,
    total: int,
    stream: Any,
    reservation: Any,
    batch: FetchBatch,
    scans: Mapping[int, SplitScan],
) -> tuple[Buffer, BufferHostView, memoryview, list[_OnceFuture]]:
    """
    Allocate one shared pinned buffer and issue every split's reads into it.

    Each split's own ranges are coalesced and read independently (via
    :func:`issue_pread_calls`), into that split's own slice of the shared
    buffer at ``split.host_offset``; ranges aren't coalesced *across*
    splits, even when two splits happen to be adjacent in the same file.
    Futures are wrapped in :class:`_OnceFuture` since every split sharing
    this batch gets the same combined future list (see :func:`fetch_batch`).

    Parameters
    ----------
    br
        The buffer resource to allocate the pinned buffer from.
    total
        Combined size, in bytes, of every split's ranges in ``batch``.
    stream
        CUDA stream to allocate on.
    reservation
        Memory reservation covering ``total`` bytes, already claimed on the
        event loop before this call was dispatched.
    batch
        The batch to fetch.
    scans
        Maps each split's ``seq_num`` (as used in ``batch``) to the
        ``SplitScan`` it belongs to, to look up its remote handle.

    Returns
    -------
    The allocated buffer, the still-open view backing ``host``, the pinned
    host view read into, and every split's ``pread`` futures combined.
    """
    buf = br.make_buffer(total, stream, reservation)
    view = buf.host_view()
    host = view.__enter__()
    try:
        futures: list[_OnceFuture] = []
        for split in batch.splits:
            handle = scans[split.seq_num].cached_parquet_info[0].remote_handle()  # type: ignore[index]
            size = sum(r.size for r in split.ranges)
            split_host = host[split.host_offset : split.host_offset + size]
            futures.extend(
                _OnceFuture(f)
                for f in issue_pread_calls(handle, split.ranges, split_host)
            )
    except BaseException:
        view.__exit__(*sys.exc_info())
        raise
    return buf, view, host, futures


async def fetch_batch(
    batch: FetchBatch,
    scans: Mapping[int, SplitScan],
    context: Context,
    ir_context: IRExecutionContext,
) -> dict[int, PinnedBatch]:
    """
    Reserve one shared pinned buffer for a batch and issue every split's reads.

    All of ``batch``'s splits share one reservation, one pinned buffer, and
    one exclusive write lock (:meth:`~rapidsmpf.memory.buffer.Buffer.host_view`).
    That lock can only be safely released once every write into the shared
    buffer has completed, not just the ones a particular split cares
    about -- unlocking early would let something else (e.g. a spill path)
    treat the buffer as safe to touch while another split's write is still
    in flight. So every returned ``PinnedBatch`` carries the *combined*
    futures for the whole batch, not just that split's own reads: whichever
    split's consumer runs ``copy_pinned_batch_to_device`` first ends up
    waiting for the whole batch's I/O, not just its own share of it.
    Releasing the lock more than once is safe (it's a no-op once already
    unlocked), so every split's consumer calling it independently is fine.

    Parameters
    ----------
    batch
        The batch to fetch, from :func:`pack_batches`.
    scans
        Maps each split's ``seq_num`` (as used in ``batch``) to the
        ``SplitScan`` it belongs to, to look up its remote handle.
    context
        The rapidsmpf context to reserve memory through.
    ir_context
        The execution context to offload the buffer allocation and reads to.

    Returns
    -------
    One :class:`PinnedBatch` per split in ``batch``, keyed by ``seq_num``,
    sharing one buffer and one combined future list.
    """
    total = batch.total_bytes
    br = context.br()
    batch_range = nvtx.start_range(
        message="reserve_pinned_batch", domain=CUDF_POLARS_NVTX_DOMAIN, payload=total
    )
    try:
        wait_range = nvtx.start_range(
            message="reserve_memory_wait", domain=CUDF_POLARS_NVTX_DOMAIN, payload=total
        )
        try:
            reservation = await reserve_memory(
                context,
                size=total,
                net_memory_delta=total,
                mem_type=MemoryType.PINNED_HOST,
            )
        finally:
            nvtx.end_range(wait_range)
        buf, view, host, futures = await ir_context.to_prefetch_thread(
            _make_batch_buffer_and_issue_reads,
            br,
            total,
            br.stream_pool.get_stream(),
            reservation,
            batch,
            scans,
        )
    finally:
        nvtx.end_range(batch_range)
    return {
        split.seq_num: PinnedBatch(
            ranges=split.ranges,
            host=host[
                split.host_offset : split.host_offset
                + sum(r.size for r in split.ranges)
            ],
            futures=futures,
            buf=buf,
            view=view,
        )
        for split in batch.splits
    }


async def run_batch_prefetch_pipeline(
    scans: Sequence[SplitScan],
    context: Context,
    ir_context: IRExecutionContext,
    *,
    batch_bytes: int,
    max_concurrent_batches: int,
) -> dict[int, asyncio.Future[PrefetchedByteRanges | None]]:
    """
    Plan, batch, and fetch every split in a scan under the batch pipeline.

    Every split's plan is resolved concurrently (already submitted ahead of
    time by :func:`submit_prefetch_plans_for_ir`), then split into two
    independent todo lists -- primary (``SINGLE_PASS``'s only pass, or
    ``TWO_PASS``'s filter columns) and payload (``TWO_PASS`` only) -- each
    packed into batches (:func:`pack_batches`) and fetched
    (:func:`fetch_batch`) on its own, never mixing a primary range and a
    payload range in the same batch. A ``TWO_PASS`` split waits on both its
    primary and payload batch before it's ready.

    Parameters
    ----------
    scans
        The splits to prefetch, in ``StreamingScan.scans`` order; a
        split's index into ``scans`` is its key in the returned mapping.
    context
        The rapidsmpf context to reserve memory through.
    ir_context
        The execution context to offload work to and read submitted plans
        from.
    batch_bytes
        Target maximum combined size of one batch, in bytes.
    max_concurrent_batches
        How many batches (primary and payload combined) can be
        concurrently reserving and fetching at once.

    Returns
    -------
    One task per split, keyed by its index into ``scans``, each resolving
    to the same result :func:`prepare_prefetch` would have for that split.
    """
    # Deferred: `actor_graph`'s package init eagerly imports `actor_graph.io`,
    # which imports from this module, so a module-level import here would be
    # circular.
    from cudf_polars.streaming.actor_graph.utils import gather_in_task_group

    prepared = await gather_in_task_group(
        *(resolve_prefetch_plan(scan, ir_context) for scan in scans)
    )

    results: dict[int, PrefetchedByteRanges | None] = {}
    primary_entries: list[tuple[int, list[Any]]] = []
    payload_entries: list[tuple[int, list[Any]]] = []
    for seq_num, one in enumerate(prepared):
        if one is None:
            results[seq_num] = None
            continue
        plc_filter, row_group_indices, _, primary_ranges, payload_ranges = one
        if not row_group_indices:
            results[seq_num] = PrefetchedByteRanges.empty(plc_filter)
            continue
        primary_entries.append((seq_num, primary_ranges))
        if payload_ranges is not None:
            payload_entries.append((seq_num, payload_ranges))

    primary_batches = pack_batches(primary_entries, batch_bytes)
    payload_batches = pack_batches(payload_entries, batch_bytes)

    scans_by_seq_num = dict(enumerate(scans))
    semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def _fetch(fetch_batch_: FetchBatch) -> dict[int, PinnedBatch]:
        async with semaphore:
            return await fetch_batch(
                fetch_batch_, scans_by_seq_num, context, ir_context
            )

    primary_task_by_seq_num: dict[int, asyncio.Task[dict[int, PinnedBatch]]] = {}
    for one_batch in primary_batches:
        task = asyncio.create_task(_fetch(one_batch))
        for split in one_batch.splits:
            primary_task_by_seq_num[split.seq_num] = task
    payload_task_by_seq_num: dict[int, asyncio.Task[dict[int, PinnedBatch]]] = {}
    for one_batch in payload_batches:
        task = asyncio.create_task(_fetch(one_batch))
        for split in one_batch.splits:
            payload_task_by_seq_num[split.seq_num] = task

    async def _resolve(
        task_by_seq_num: dict[int, asyncio.Task[dict[int, PinnedBatch]]],
        seq_num: int,
    ) -> PinnedBatch | None:
        task = task_by_seq_num.get(seq_num)
        return None if task is None else (await task)[seq_num]

    async def _finalize(seq_num: int) -> PrefetchedByteRanges | None:
        if seq_num in results:
            return results[seq_num]
        one = prepared[seq_num]
        assert one is not None
        plc_filter, row_group_indices, pass_mode, _, _ = one
        primary = await _resolve(primary_task_by_seq_num, seq_num)
        if pass_mode is HybridScanPassMode.SINGLE_PASS:
            return PrefetchedByteRanges(
                row_group_indices=row_group_indices,
                pass_mode=pass_mode,
                plc_filter=plc_filter,
                all_columns=primary,
            )
        payload = await _resolve(payload_task_by_seq_num, seq_num)
        return PrefetchedByteRanges(
            row_group_indices=row_group_indices,
            pass_mode=pass_mode,
            plc_filter=plc_filter,
            filter=primary,
            payload=payload,
        )

    return {
        seq_num: asyncio.create_task(_finalize(seq_num))
        for seq_num in range(len(scans))
    }


async def prefetch_scan_byte_ranges(
    scan: SplitScan,
    context: Context,
    ir_context: IRExecutionContext,
    *,
    wait_for: asyncio.Event | None,
    own_turn: asyncio.Event,
) -> PrefetchedByteRanges | None:
    """
    Prune row groups for one scan task and prefetch its byte ranges.

    Pruning and byte-range computation are offloaded to ``ir_context``'s
    main thread pool and run freely, out of order across tasks. Claiming
    pinned memory and issuing reads is offloaded to the dedicated prefetch
    thread pool instead, and waits for ``wait_for`` first (the previous
    task's own attempt, within the same producer), so a task due for
    consumption soon can't lose its reservation to one that isn't, then
    signals ``own_turn`` before returning, whether or not a reservation
    succeeded.

    Parameters
    ----------
    scan
        The scan task to prefetch.
    context
        The rapidsmpf context to reserve memory through.
    ir_context
        The execution context to offload pruning and byte-range computation to.
    wait_for
        Event to wait on before claiming pinned memory and issuing reads, or
        ``None`` for the first task in a producer's chain.
    own_turn
        Event this task sets once it's done claiming resources, whether or
        not a reservation succeeded, so the next task in the chain can proceed.

    Returns
    -------
    The prefetched byte ranges, or ``None`` when the predicate can't be
    expressed as a parquet filter, the caller falls back to
    ``SplitScan.do_evaluate`` in that case.
    """
    task_range = nvtx.start_range(
        message="prefetch_scan_byte_ranges", domain=CUDF_POLARS_NVTX_DOMAIN
    )
    try:
        prepared = await ir_context.to_thread(prepare_prefetch, scan)
        if wait_for is not None:
            wait_range = nvtx.start_range(
                message="prefetch_wait_for_turn", domain=CUDF_POLARS_NVTX_DOMAIN
            )
            try:
                await wait_for.wait()
            finally:
                nvtx.end_range(wait_range)
        try:
            if prepared is None:
                return None
            plc_filter, row_group_indices, pass_mode, primary_ranges, payload_ranges = (
                prepared
            )
            if not row_group_indices:
                return PrefetchedByteRanges.empty(plc_filter)

            assert scan.cached_parquet_info is not None
            handle = scan.cached_parquet_info[0].remote_handle()

            if pass_mode is HybridScanPassMode.SINGLE_PASS:
                all_columns = await reserve_pinned_batch(
                    context, ir_context, handle, primary_ranges
                )
                return PrefetchedByteRanges(
                    row_group_indices=row_group_indices,
                    pass_mode=pass_mode,
                    plc_filter=plc_filter,
                    all_columns=all_columns,
                )

            filter_batch = await reserve_pinned_batch(
                context, ir_context, handle, primary_ranges
            )
            payload_batch = await reserve_pinned_batch(
                context, ir_context, handle, payload_ranges or []
            )
            return PrefetchedByteRanges(
                row_group_indices=row_group_indices,
                pass_mode=pass_mode,
                plc_filter=plc_filter,
                filter=filter_batch,
                payload=payload_batch,
            )
        finally:
            own_turn.set()
    finally:
        nvtx.end_range(task_range)


class ByteBudget:
    """
    Async, byte-weighted admission gate.

    Bounds how many bytes' worth of work can be concurrently admitted,
    rather than how many callers -- a plain ``asyncio.Semaphore`` only
    counts callers, it doesn't weigh them. A single request larger than
    the whole budget is still admitted once nothing else is outstanding,
    rather than deadlocking; the budget bounds how much gets admitted
    together, not an absolute ceiling on any one request.
    """

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._outstanding = 0
        self._condition = asyncio.Condition()

    async def acquire(self, nbytes: int) -> None:
        """Wait until ``nbytes`` fits under the budget, then claim it."""
        async with self._condition:
            await self._condition.wait_for(
                lambda: (
                    self._outstanding == 0 or self._outstanding + nbytes <= self._budget
                )
            )
            self._outstanding += nbytes

    async def release(self, nbytes: int) -> None:
        """Return ``nbytes`` to the budget, from the loop this budget runs on."""
        async with self._condition:
            self._outstanding -= nbytes
            self._condition.notify_all()

    def release_threadsafe(self, nbytes: int, loop: asyncio.AbstractEventLoop) -> None:
        """Return ``nbytes`` to the budget, safe to call from any thread."""
        asyncio.run_coroutine_threadsafe(self.release(nbytes), loop).result()


async def prefetch_scan_byte_ranges_paced(
    scan: SplitScan,
    context: Context,
    ir_context: IRExecutionContext,
    budget: ByteBudget,
    loop: asyncio.AbstractEventLoop,
    *,
    wait_for: asyncio.Event | None,
    own_turn: asyncio.Event,
) -> PrefetchedByteRanges | None:
    """
    Prune, byte-pace, and prefetch one split -- no shared buffer.

    Reserves and reads this split's own byte ranges independently (own
    reservation, own buffer, own lock), the same as
    :func:`prefetch_scan_byte_ranges`, so no split ever waits on another's
    *I/O*. Concurrency is bounded by ``budget`` instead of a fixed
    producer/thread count: a split only starts reserving pinned memory
    once enough of the byte budget is free, and returns its share only
    once the buffer it was holding is actually released (see
    ``PrefetchedByteRanges.on_release``), not merely once its reads are
    issued.

    Claiming a share of ``budget`` still waits for ``wait_for`` first (a
    strict FIFO chain across every split in the scan): without an
    ordering guarantee here, a split due for consumption *later* could
    win a race for limited budget over one due *sooner*, and since a
    split's share is only returned once it's actually consumed, that
    starves the split the consumer is waiting on -- a deadlock, not just
    a slowdown, if the budget is small enough that only the "wrong"
    splits fit at once. Only the *acquire* step is ordered; once a split
    has its share, its own reservation and reads proceed independently of
    every other split, same as before.

    Parameters
    ----------
    scan
        The scan task to prefetch.
    context
        The rapidsmpf context to reserve memory through.
    ir_context
        The execution context to offload pruning and reservation to, and
        to read this split's plan from (already submitted ahead of time
        by :func:`submit_prefetch_plans_for_ir`).
    budget
        The byte budget admission for this split's reservation is gated on.
    loop
        Event loop ``budget`` runs on, for :meth:`ByteBudget.release_threadsafe`.
    wait_for
        Event to wait on before attempting to claim a share of ``budget``,
        or ``None`` for the first split.
    own_turn
        Event this task sets once it's done claiming its share of
        ``budget`` (whether or not claiming it succeeded), so the next
        split can attempt to claim its own.

    Returns
    -------
    The prefetched byte ranges, or ``None`` when the predicate can't be
    expressed as a parquet filter, the caller falls back to
    ``SplitScan.do_evaluate`` in that case.
    """
    prepared = await resolve_prefetch_plan(scan, ir_context)
    if prepared is None:
        if wait_for is not None:
            await wait_for.wait()
        own_turn.set()
        return None
    plc_filter, row_group_indices, pass_mode, primary_ranges, payload_ranges = prepared
    if not row_group_indices:
        if wait_for is not None:
            await wait_for.wait()
        own_turn.set()
        return PrefetchedByteRanges.empty(plc_filter)

    assert scan.cached_parquet_info is not None
    handle = scan.cached_parquet_info[0].remote_handle()

    total_bytes = sum(r.size for r in primary_ranges)
    if payload_ranges is not None:
        total_bytes += sum(r.size for r in payload_ranges)

    if wait_for is not None:
        await wait_for.wait()
    try:
        await budget.acquire(total_bytes)
    finally:
        own_turn.set()
    try:
        if pass_mode is HybridScanPassMode.SINGLE_PASS:
            all_columns = await reserve_pinned_batch(
                context, ir_context, handle, primary_ranges
            )
            prefetched = PrefetchedByteRanges(
                row_group_indices=row_group_indices,
                pass_mode=pass_mode,
                plc_filter=plc_filter,
                all_columns=all_columns,
            )
        else:
            filter_batch = await reserve_pinned_batch(
                context, ir_context, handle, primary_ranges
            )
            payload_batch = await reserve_pinned_batch(
                context, ir_context, handle, payload_ranges or []
            )
            prefetched = PrefetchedByteRanges(
                row_group_indices=row_group_indices,
                pass_mode=pass_mode,
                plc_filter=plc_filter,
                filter=filter_batch,
                payload=payload_batch,
            )
    except BaseException:
        await budget.release(total_bytes)
        raise
    prefetched.on_release = lambda: budget.release_threadsafe(total_bytes, loop)
    return prefetched


async def run_paced_prefetch_pipeline(
    scans: Sequence[SplitScan],
    context: Context,
    ir_context: IRExecutionContext,
    *,
    budget_bytes: int,
    num_prefetch_producers: int = 1,
) -> dict[int, asyncio.Future[PrefetchedByteRanges | None]]:
    """
    Prefetch every split in a scan independently, paced by a byte budget.

    Parameters
    ----------
    scans
        The splits to prefetch, in ``StreamingScan.scans`` order; a
        split's index into ``scans`` is its key in the returned mapping.
    context
        The rapidsmpf context to reserve memory through.
    ir_context
        The execution context to offload work to and read submitted plans
        from.
    budget_bytes
        Maximum combined size of every split's pinned reservation
        outstanding at once.
    num_prefetch_producers
        Number of independent FIFO chains splits are round-robin assigned
        to when claiming a share of ``budget_bytes``. 1 (the default)
        means a single strict global order across every split. Higher
        values let that many splits attempt to claim budget concurrently
        (one per chain), reducing head-of-line blocking between splits in
        different chains, at the cost of a weaker ordering guarantee: two
        splits in different chains can now claim out of relative order.

    Returns
    -------
    One task per split, keyed by its index into ``scans``, each resolving
    to the same result :func:`prepare_prefetch` would have for that split.
    """
    loop = asyncio.get_running_loop()
    budget = ByteBudget(budget_bytes)
    # FIFO chains: within a chain, split N can't attempt to claim its
    # share of the budget until split N's predecessor in that same chain
    # has finished attempting to claim its own. Splits are round-robin
    # assigned to `num_prefetch_producers` chains, so up to that many
    # splits (one per chain) can be attempting to claim a share at once.
    num_scans = len(scans)
    effective_num_prefetch_producers = (
        1
        if (num_scans == 1 or num_prefetch_producers == 1)
        else min(num_prefetch_producers, num_scans)
    )
    own_turns = [asyncio.Event() for _ in range(num_scans)]
    wait_for: list[asyncio.Event | None] = [None] * num_scans
    for producer_id in range(effective_num_prefetch_producers):
        predecessor: asyncio.Event | None = None
        for task_idx in range(producer_id, num_scans, effective_num_prefetch_producers):
            wait_for[task_idx] = predecessor
            predecessor = own_turns[task_idx]
    return {
        seq_num: asyncio.create_task(
            prefetch_scan_byte_ranges_paced(
                scan,
                context,
                ir_context,
                budget,
                loop,
                wait_for=wait_for[seq_num],
                own_turn=own_turns[seq_num],
            )
        )
        for seq_num, scan in enumerate(scans)
    }


class PinnedBuffer:
    """
    Pinned host buffer allocated directly from a ``PinnedMemoryResource``.

    A raw ``ctypes``-backed allocation, bypassing ``BufferResource.make_buffer``
    entirely: no ``BufferResource`` mutex, no ``host_view()`` lock, no
    spill/eviction integration. Only used by the ``QUEUE`` prefetch
    pipeline.

    The reservation is released and the allocation freed whenever this
    object is garbage-collected, rather than at a deterministic point.
    """

    __slots__ = ("array", "mr", "nbytes", "ptr", "reservation", "stream")

    def __init__(
        self,
        mr: PinnedMemoryResource,
        nbytes: int,
        stream: Stream,
        reservation: MemoryReservation,
    ) -> None:
        self.mr = mr
        self.nbytes = nbytes
        self.stream = stream
        self.reservation = reservation
        self.ptr = mr.allocate(nbytes, stream)
        self.array = memoryview((ctypes.c_uint8 * nbytes).from_address(self.ptr))

    def __del__(self) -> None:
        """Release the reservation and free the allocation."""
        # Guard against partial init.
        if hasattr(self, "reservation"):
            self.reservation.clear()
        if hasattr(self, "ptr"):
            self.mr.deallocate(self.ptr, self.nbytes, self.stream)


def _reserve_pinned_batch_sync(
    context: Context,
    loop: asyncio.AbstractEventLoop,
    handle: CuFile | RemoteFile,
    ranges: list[Any],
) -> PinnedBatch | None:
    """
    Reserve pinned host memory and issue reads for one batch, synchronously.

    The queue pipeline's equivalent of :func:`reserve_pinned_batch`:
    allocates directly from the pinned memory resource via
    :class:`PinnedBuffer` instead of going through ``BufferResource``, and
    issues one ``pread`` per range with no coalescing. Runs entirely on the
    calling (worker) thread, only reaching back into ``loop`` for the one
    genuinely-async step (``reserve_memory``), via a blocking bridge
    rather than an ``await``.

    Parameters
    ----------
    context
        The rapidsmpf context to reserve memory through.
    loop
        Event loop to run ``reserve_memory`` on.
    handle
        Open kvikio handle to read from.
    ranges
        Byte ranges to reserve for and read.

    Returns
    -------
    The reserved, in-flight batch, or ``None`` when ``ranges`` is empty.
    """
    if not ranges:
        return None
    total = sum(r.size for r in ranges)
    br = context.br()
    pinned_mr = br.pinned_mr
    assert pinned_mr is not None
    with nvtx_annotate_cudf_polars(message="reserve_pinned_batch", payload=total):
        with nvtx_annotate_cudf_polars(message="reserve_memory_wait", payload=total):
            reservation = asyncio.run_coroutine_threadsafe(
                reserve_memory(
                    context,
                    size=total,
                    net_memory_delta=total,
                    mem_type=MemoryType.PINNED_HOST,
                ),
                loop,
            ).result()
        with nvtx_annotate_cudf_polars(
            message="issue_reads_into_pinned_buffer", payload=total
        ):
            stream = br.stream_pool.get_stream()
            buf = PinnedBuffer(pinned_mr, total, stream, reservation)
            futures = []
            offset = 0
            for r in ranges:
                futures.append(
                    handle.pread(
                        buf.array[offset : offset + r.size],
                        size=r.size,
                        file_offset=r.offset,
                    )
                )
                offset += r.size
    return PinnedBatch(
        ranges=ranges, host=buf.array, futures=futures, buf=buf, view=None
    )


@nvtx_annotate_cudf_polars(message="prefetch_scan_byte_ranges")
def prefetch_scan_byte_ranges_sync(
    scan: SplitScan,
    context: Context,
    loop: asyncio.AbstractEventLoop,
) -> PrefetchedByteRanges | None:
    """
    Prune row groups for one scan task and prefetch its byte ranges, synchronously.

    The queue pipeline's per-split unit of work: everything
    :func:`prefetch_scan_byte_ranges` does, but as one plain function
    running entirely on a single worker thread with no ``asyncio.Task`` of
    its own, and no ordering chain. A queue worker calls this once per
    split, moving straight to the next split once this one's reads are
    issued.

    Parameters
    ----------
    scan
        The scan task to prefetch.
    context
        The rapidsmpf context to reserve memory through.
    loop
        Event loop to run pinned memory reservation's admission control on.

    Returns
    -------
    The prefetched byte ranges, or ``None`` when the predicate can't be
    expressed as a parquet filter, the caller falls back to
    ``SplitScan.do_evaluate`` in that case.
    """
    prepared = prepare_prefetch(scan)
    if prepared is None:
        return None
    plc_filter, row_group_indices, pass_mode, primary_ranges, payload_ranges = prepared
    if not row_group_indices:
        return PrefetchedByteRanges.empty(plc_filter)

    assert scan.cached_parquet_info is not None
    handle = scan.cached_parquet_info[0].remote_handle()

    if pass_mode is HybridScanPassMode.SINGLE_PASS:
        all_columns = _reserve_pinned_batch_sync(context, loop, handle, primary_ranges)
        return PrefetchedByteRanges(
            row_group_indices=row_group_indices,
            pass_mode=pass_mode,
            plc_filter=plc_filter,
            all_columns=all_columns,
        )

    filter_batch = _reserve_pinned_batch_sync(context, loop, handle, primary_ranges)
    payload_batch = _reserve_pinned_batch_sync(
        context, loop, handle, payload_ranges or []
    )
    return PrefetchedByteRanges(
        row_group_indices=row_group_indices,
        pass_mode=pass_mode,
        plc_filter=plc_filter,
        filter=filter_batch,
        payload=payload_batch,
    )


class HybridScanPrefetchExecutor:
    """
    A small, fixed-size pool over a FIFO queue.

    Submits every split's full prune+reserve+read sequence
    (:func:`prefetch_scan_byte_ranges_sync`) to a plain
    ``ThreadPoolExecutor`` up front. Each worker thread processes one split
    completely before pulling the next off the queue: no ordering chain, no
    per-split ``asyncio.Task`` on the actor graph's event loop.
    """

    def __init__(
        self,
        futures: list[concurrent.futures.Future[PrefetchedByteRanges | None]],
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        self.futures = futures
        self._executor = executor

    @classmethod
    def from_scans(
        cls,
        scans: Sequence[SplitScan],
        num_workers: int,
        context: Context,
    ) -> Self:
        """
        Submit prefetch tasks for all scans.

        Parameters
        ----------
        scans
            Tasks to prefetch.
        num_workers
            Number of background worker threads.
        context
            The rapidsmpf context to reserve memory through.

        Returns
        -------
        HybridScanPrefetchExecutor
        """
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="cudf-polars-prefetch-queue"
        )
        futures = [
            executor.submit(prefetch_scan_byte_ranges_sync, scan, context, loop)
            for scan in scans
        ]
        return cls(futures, executor)

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Shut down the worker pool, cancelling any not-yet-started work."""
        self._executor.shutdown(cancel_futures=True, wait=False)

    def result(self, task_idx: int) -> PrefetchedByteRanges | None:
        """Block until the task's prefetch result is ready and return it."""
        return self.futures[task_idx].result()
