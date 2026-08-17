# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid scan prefetch pipeline."""

from __future__ import annotations

import asyncio
import ctypes
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Self

import pylibcudf as plc
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

from cudf_polars.dsl.ir import _prepare_parquet_predicate
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.dsl.tracing import nvtx_annotate_cudf_polars
from cudf_polars.streaming.io import PrefetchedByteRanges, _fetch_byte_ranges

if TYPE_CHECKING:
    from concurrent.futures import Future

    from kvikio.cufile import CuFile, IOFuture
    from kvikio.remote_file import RemoteFile

    from rapidsmpf.memory.memory_reservation import MemoryReservation
    from rapidsmpf.memory.pinned_memory_resource import PinnedMemoryResource
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.streaming.io import SplitScan


class PinnedBuffer:
    """Pinned host buffer backed by a rapidsmpf PinnedMemoryResource pool."""

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

    def __del__(self) -> None:  # noqa: D105
        # Guard against partial init.
        if hasattr(self, "reservation"):
            self.reservation.clear()
        if hasattr(self, "ptr"):
            self.mr.deallocate(self.ptr, self.nbytes, self.stream)


def pread_ranges(
    handle: CuFile | RemoteFile,
    ranges: list[plc.io.text.ByteRangeInfo],
    pinned_mr: PinnedMemoryResource,
    stream: Stream,
    context: Context,
    loop: asyncio.AbstractEventLoop,
) -> tuple[memoryview | None, list[IOFuture], PinnedBuffer | None]:
    """Issue concurrent async reads for each range into a single pinned host buffer."""
    total = sum(r.size for r in ranges)
    if not total:
        return None, [], None
    # Blocks this worker thread, not the event loop. The loop stays free to
    # run other coroutines while we wait for the reservation.
    with nvtx_annotate_cudf_polars(message="reserve_pinned_memory", payload=total):
        reservation = asyncio.run_coroutine_threadsafe(
            reserve_memory(
                context,
                size=total,
                net_memory_delta=total,
                mem_type=MemoryType.PINNED_HOST,
            ),
            loop,
        ).result()
    buf = PinnedBuffer(pinned_mr, total, stream, reservation)
    futures = []
    offset = 0
    with nvtx_annotate_cudf_polars(message="read_ranges:submit", payload=total):
        for r in ranges:
            futures.append(
                handle.pread(
                    buf.array[offset : offset + r.size],
                    size=r.size,
                    file_offset=r.offset,
                )
            )
            offset += r.size
    return buf.array, futures, buf


def prefetch_scan_byte_ranges(
    scan: SplitScan,
    stream: Stream,
    pinned_mr: PinnedMemoryResource,
    context: Context,
    loop: asyncio.AbstractEventLoop,
) -> PrefetchedByteRanges | None:
    """
    Run stats and bloom pruning for one SplitScan and issue async reads.

    Parameters
    ----------
    scan
        The split scan task to prefetch.
    stream
        CUDA stream used for filter expression compilation.
    pinned_mr
        Pinned memory resource to allocate host buffers from.
    context
        rapidsmpf context used for pinned memory reservation.
    loop
        Event loop used to submit the reservation coroutine from a worker thread.

    Returns
    -------
    PrefetchedByteRanges | None
        ``None`` when the predicate cannot be expressed as a parquet filter,
        in which case the producer falls back to ``SplitScan.do_evaluate``.
        :meth:`PrefetchedByteRanges.empty` when all row groups are pruned away.
    """
    cached_info = scan.cached_parquet_info
    assert cached_info is not None

    row_group_num_rows = cached_info[0].file_metadata.row_group_num_rows
    total_row_groups = len(row_group_num_rows)

    rg_stride = total_row_groups // scan.total_splits
    skip_rgs = rg_stride * scan.split_index
    end_rg = (
        total_row_groups
        if scan.split_index == scan.total_splits - 1
        else skip_rgs + rg_stride
    )
    row_group_indices = list(range(skip_rgs, end_rg))

    predicate = scan.base_scan.predicate
    assert predicate is not None

    with nvtx_annotate_cudf_polars(message="to_parquet_filter"):
        plc_filter = to_parquet_filter(
            _prepare_parquet_predicate(
                predicate.value, scan.paths, scan.schema, scan.base_scan.with_columns
            ),
            stream=stream,
        )
    if plc_filter is None:
        return None

    with nvtx_annotate_cudf_polars(message="build_reader_options"):
        options = (
            plc.io.parquet.ParquetReaderOptions.builder(
                plc.io.SourceInfo(
                    [
                        plc.io.types.FilepathSource(
                            cached_info[0].path, cached_info[0].size
                        )
                    ]
                )
            )
            .decimal_width(plc.TypeId.DECIMAL128)
            .build()
        )
        if scan.base_scan.with_columns is not None:
            options.set_column_names(scan.base_scan.with_columns)
        options.set_filter(plc_filter)

    with nvtx_annotate_cudf_polars(message="hybrid_scan_reader"):
        reader = cached_info[0].hybrid_scan_reader(options)

    if scan.parquet_options._hybrid_scan_stats_pruning:
        with nvtx_annotate_cudf_polars(message="filter_row_groups_with_stats"):
            row_group_indices = reader.filter_row_groups_with_stats(
                row_group_indices, options, stream=stream
            )

    if row_group_indices:
        bloom_ranges, _ = reader.secondary_filters_byte_ranges(
            row_group_indices, options
        )
        if bloom_ranges:
            with nvtx_annotate_cudf_polars(
                message="filter_row_groups_with_bloom_filters"
            ):
                bloom_chunks = _fetch_byte_ranges(
                    plc.io.SourceInfo(
                        [
                            plc.io.types.FilepathSource(
                                cached_info[0].path, cached_info[0].size
                            )
                        ]
                    ),
                    bloom_ranges,
                    stream,
                )
                row_group_indices = reader.filter_row_groups_with_bloom_filters(
                    bloom_chunks, row_group_indices, options, stream=stream
                )

    if not row_group_indices:
        return PrefetchedByteRanges.empty()

    with nvtx_annotate_cudf_polars(message="byte_range_computation"):
        filter_ranges = reader.filter_column_chunks_byte_ranges(
            row_group_indices, options
        )
        payload_ranges = reader.payload_column_chunks_byte_ranges(
            row_group_indices, options
        )

    handle = cached_info[0].remote_handle()
    filter_bytes = sum(r.size for r in filter_ranges)
    payload_bytes = sum(r.size for r in payload_ranges)
    # TODO: coalesce nearby ranges before issuing pread calls.
    # https://github.com/rapidsai/cudf/pull/23317#discussion_r3668809937
    with nvtx_annotate_cudf_polars(
        message="pread_filter_and_payload",
        payload=(
            id(scan.base_scan),
            scan.split_index + 1,
            scan.total_splits,
            filter_bytes,
            payload_bytes,
        ),
    ):
        filter_host, filter_futures, filter_buf = pread_ranges(
            handle, filter_ranges, pinned_mr, stream, context, loop
        )
        payload_host, payload_futures, payload_buf = pread_ranges(
            handle, payload_ranges, pinned_mr, stream, context, loop
        )

    return PrefetchedByteRanges(
        row_group_indices=row_group_indices,
        filter_ranges=filter_ranges,
        payload_ranges=payload_ranges,
        filter_host=filter_host,
        payload_host=payload_host,
        filter_futures=filter_futures,
        payload_futures=payload_futures,
        filter_buf=filter_buf,
        payload_buf=payload_buf,
    )


# TODO: Replace with a cucascade::io::datasource that accepts fadvise() hints
# issued before evaluation, so pre-reading is driven by the datasource layer
# rather than a separate host-pinned executor.
class HybridScanPrefetchExecutor:
    """Prefetch executor for SplitScan tasks."""

    def __init__(
        self,
        futures: list[Future[PrefetchedByteRanges | None]],
        executor: ThreadPoolExecutor,
    ):
        self.futures = futures
        self._executor = executor

    @classmethod
    def from_scans(
        cls,
        scans: list[SplitScan],
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
            rapidsmpf context. Pinned memory must be enabled.

        Returns
        -------
        HybridScanPrefetchExecutor

        Raises
        ------
        ValueError
            If ``context.br().pinned_mr`` is ``None``.
        """
        pinned_mr = context.br().pinned_mr
        if pinned_mr is None:
            raise ValueError(
                "HybridScanPrefetchExecutor requires a PinnedMemoryResource; "
                "enable pinned memory via --pinned-memory."
            )
        loop = asyncio.get_running_loop()
        stream_pool = context.br().stream_pool
        # TODO: Consider reusing ir_context.py_executor instead of a dedicated pool.
        executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="hybrid-prefetch",
        )

        def _task(s: SplitScan) -> PrefetchedByteRanges | None:
            return prefetch_scan_byte_ranges(
                s, stream_pool.get_stream(), pinned_mr, context, loop
            )

        futures = [executor.submit(_task, scan) for scan in scans]
        return cls(futures, executor)

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Shut down the thread pool, cancelling pending futures."""
        self._executor.shutdown(cancel_futures=True, wait=False)

    def result(self, task_idx: int) -> PrefetchedByteRanges | None:
        """Block until the tasks' prefetch result is ready and return it."""
        return self.futures[task_idx].result()
