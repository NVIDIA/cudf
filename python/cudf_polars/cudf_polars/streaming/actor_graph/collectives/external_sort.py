# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
External merge sort for one range partition of the streaming ``Sort``.

After the range shuffle, each rank owns a few partitions whose packed pieces
live in the shuffler (spillable by rapidsmpf, to disk when a spill directory
is configured). Instead of unpacking a whole partition into device memory and
sorting it in place, this module:

* **Phase A** unpacks the pieces in batches bounded by a device budget, sorts
  each batch, and writes it as a *run* of uncompressed parquet pages; and
* **Phase B** streams a k-way merge of the runs (``plc.merge.merge`` on the
  current page of each run), emitting only rows that no unread page can
  precede, and refilling the run(s) that bound the emission. Runs beyond the
  fan-in are merged in passes.

The partition is never resident as a whole. All device allocations go through
``reserve_memory`` so they wait for the shuffler to spill rather than fail.
When a partition fits the batch budget, the previous in-memory behaviour is
used unchanged.

This mirrors spark-rapids' ``GpuSortExec`` and the standalone
``rapidsmpf-disk-sort`` prototype that validated the approach at 1 TB.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc
from cudf_streaming.partition_utils import (
    unpack_and_concat as py_unpack_and_concat,
    unpack_and_concat_cost as py_unpack_and_concat_cost,
)
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.memory_reserve_or_wait import reserve_memory

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.streaming.actor_graph.tracing import send_chunk

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from cudf_streaming.channel_metadata import OrderKey
    from rapidsmpf.config import Options
    from rapidsmpf.memory.packed_data import PackedData
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext, Sort
    from cudf_polars.streaming.actor_graph.collectives.shuffle import ShuffleManager
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
# table) cannot degenerate into one parquet file per row.
_MIN_PAGE_ROWS = 1024


def _device_total_bytes() -> int | None:
    try:
        import rmm.mr

        _, total = rmm.mr.available_device_memory()
    except Exception:  # pragma: no cover - depends on the RMM build
        return None
    return int(total)


def batch_bytes_for(executor: StreamingExecutor) -> int:
    """Device budget (bytes) for one sorted run of the external merge sort."""
    budget = _BATCH_BYTES_TARGET_MULTIPLE * executor.target_partition_size
    total = _device_total_bytes()
    if total is not None:
        budget = min(budget, int(total * _BATCH_BYTES_DEVICE_FRACTION))
    return max(1, budget)


def merge_fanin() -> int:
    """Runs merged per pass (indirection so tests can shrink it)."""
    return DEFAULT_MERGE_FANIN


def make_run_directory(options: Options, rank: int) -> Path:
    """
    Directory for this rank's run pages.

    Uses rapidsmpf's ``disk_spill_dir`` option (env ``RAPIDSMPF_DISK_SPILL_DIR``)
    when set, so the shuffler's disk spill and the sort's run pages share one
    location; otherwise a temporary directory.
    """
    base = options.get_strings().get("disk_spill_dir")
    name = f"cudf-polars-sort-{rank}-{uuid.uuid4().hex[:8]}"
    if base is None or base.strip() == "" or base.strip().lower() == "false":
        return Path(tempfile.mkdtemp(prefix=name + "-"))
    path = Path(base) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def table_nbytes(table: plc.Table) -> int:
    """Device bytes held by ``table``."""
    return sum(column.device_buffer_size() for column in table.columns())


def _to_host_ints(column: plc.Column, stream: Stream) -> list[int]:
    """Stream-aware device->host copy of a small INT32 column."""
    return (
        DataFrame.from_table(
            plc.Table([column]), ["v"], [DataType(pl.Int32())], stream=stream
        )
        .to_polars()["v"]
        .to_list()
    )


def _owning_copy(table: plc.Table, stream: Stream) -> plc.Table:
    """Deep-copy a (possibly view) table so it owns its buffers."""
    return plc.Table([column.copy(stream=stream) for column in table.columns()])


def _write_parquet_page(path: Path, table: plc.Table, stream: Stream) -> None:
    options = (
        plc.io.parquet.ParquetWriterOptions.builder(
            plc.io.types.SinkInfo([str(path)]), table
        )
        .compression(plc.io.types.CompressionType.NONE)
        .build()
    )
    plc.io.parquet.write_parquet(options, stream=stream)


def _read_parquet_page(path: Path, stream: Stream, mr: object) -> plc.Table:
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.types.SourceInfo([str(path)])
    ).build()
    table = plc.io.parquet.read_parquet(options, stream=stream, mr=mr).tbl
    path.unlink(missing_ok=True)
    return table


@dataclass
class RunPage:
    """One parquet file of a sorted run."""

    path: Path
    num_rows: int
    nbytes: int


@dataclass
class Run:
    """A sorted sequence of pages."""

    pages: list[RunPage] = field(default_factory=list)

    @property
    def num_rows(self) -> int:  # noqa: D102
        return sum(page.num_rows for page in self.pages)


class RunWriter:
    """Write sorted tables as pages of a run."""

    def __init__(self, directory: Path, page_rows: int) -> None:
        self._dir = directory
        self._page_rows = max(1, page_rows)
        self._run = Run()
        self._counter = 0
        directory.mkdir(parents=True, exist_ok=True)

    async def write(
        self, table: plc.Table, stream: Stream, ir_context: IRExecutionContext
    ) -> None:
        """Append a sorted table (split into pages) to the run."""
        nrows = table.num_rows()
        if nrows == 0:
            return
        nbytes = table_nbytes(table)
        splits = list(range(self._page_rows, nrows, self._page_rows))
        pages = plc.copying.split(table, splits, stream=stream) if splits else [table]
        for page in pages:
            path = self._dir / f"page-{self._counter:08d}.parquet"
            self._counter += 1
            page_rows = page.num_rows()
            await ir_context.to_thread(_write_parquet_page, path, page, stream)
            self._run.pages.append(
                RunPage(path, page_rows, max(1, nbytes * page_rows // nrows))
            )

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
        """Read the next page into device memory (reserving it first)."""
        page = self._pages.popleft()
        reservation = await reserve_memory(
            context, page.nbytes, net_memory_delta=page.nbytes
        )
        with opaque_memory_usage(reservation):
            table = await ir_context.to_thread(
                _read_parquet_page, page.path, stream, context.br().device_mr
            )
        nrows = table.num_rows()
        keys = plc.Table([table.columns()[i] for i in key_indices])
        last = plc.Column.from_scalar(
            plc.Scalar.from_py(
                nrows - 1, plc.DataType(plc.TypeId.INT32), stream=stream
            ),
            1,
            stream=stream,
        )
        self.tail = plc.copying.gather(
            keys, last, plc.copying.OutOfBoundsPolicy.DONT_CHECK, stream=stream
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
    tails = plc.concatenate.concatenate(
        [cursor.tail for cursor in pending], stream=stream
    )
    order = plc.sorting.sorted_order(tails, orders, null_orders, stream=stream)
    ranked = _to_host_ints(order, stream)
    min_tail = plc.copying.gather(
        tails,
        plc.copying.slice(order, [0, 1], stream=stream)[0],
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    )
    (cut,) = _to_host_ints(
        plc.search.upper_bound(
            merged_keys, min_tail, orders, null_orders, stream=stream
        ),
        stream,
    )
    (n_equal,) = _to_host_ints(
        plc.search.upper_bound(
            plc.copying.gather(
                tails, order, plc.copying.OutOfBoundsPolicy.DONT_CHECK, stream=stream
            ),
            min_tail,
            orders,
            null_orders,
            stream=stream,
        ),
        stream,
    )
    refill = [pending[i] for i in ranked[: max(1, n_equal)]]
    return cut, refill


async def merge_runs(
    context: Context,
    ir_context: IRExecutionContext,
    runs: Sequence[Run],
    order_keys: Sequence[OrderKey],
    emit: Callable[[plc.Table, Stream], Awaitable[None]],
) -> None:
    """
    Stream a k-way merge of ``runs`` into ``emit``.

    At most one page per run plus the carried-over (not yet emittable) rows
    are resident at any time.
    """
    key_indices = [key.column_index for key in order_keys]
    orders = [key.order for key in order_keys]
    null_orders = [key.null_order for key in order_keys]
    stream = ir_context.get_cuda_stream()
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
                await emit(_owning_copy(head, stream), stream)
            carry = _owning_copy(rest, stream)
        del merged

        tables = [
            await cursor.load_next(context, ir_context, key_indices, stream)
            for cursor in refill
        ]


async def _unpack_and_sort(
    context: Context,
    ir_context: IRExecutionContext,
    pieces: Sequence[PackedData],
    nbytes: int,
    post_sort_ir: Sort,
    stream: Stream,
) -> plc.Table:
    """Unpack packed pieces into one table and sort it (with the stable key)."""
    reservation = await reserve_memory(
        context,
        py_unpack_and_concat_cost(pieces),
        # Representation change: packed input consumed as the table is produced.
        net_memory_delta=0,
    )
    table = py_unpack_and_concat(
        partitions=list(pieces),
        stream=stream,
        br=context.br(),
        reservation=reservation,
    )
    if table.num_rows() == 0:
        return table
    # Sorting needs the output plus temporaries of about the input size.
    sort_reservation = await reserve_memory(context, 2 * nbytes, net_memory_delta=0)
    with opaque_memory_usage(sort_reservation):
        return post_sort_ir.do_evaluate(
            *post_sort_ir._non_child_args,
            DataFrame.from_table(
                table,
                list(post_sort_ir.schema.keys()),
                list(post_sort_ir.schema.values()),
                stream,
            ),
            context=ir_context,
        ).table


async def sort_partition(
    context: Context,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    shuffle: ShuffleManager,
    post_sort_ir: Sort,
    order_keys: Sequence[OrderKey],
    output_schema: Schema,
    partition_id: int,
    *,
    batch_bytes: int,
    run_root: Path,
    tracer: ActorTracer | None,
) -> str:
    """
    Sort one owned partition and send it on ``ch_out``.

    Returns ``"in_memory"`` when the partition fit the batch budget and was
    sorted like before, or ``"external"`` when runs were written and merged.
    """
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

    pieces = shuffle.extract_pieces(partition_id)
    sizes = [py_unpack_and_concat_cost([piece]) for piece in pieces]
    total = sum(sizes)
    stream = ir_context.get_cuda_stream()

    if total <= batch_bytes:
        table = await _unpack_and_sort(
            context, ir_context, pieces, total, post_sort_ir, stream
        )
        await emit(table, stream)
        return "in_memory"

    fanin = merge_fanin()
    part_dir = run_root / f"part-{partition_id:05d}"
    try:
        # Phase A: sorted runs.
        runs: list[Run] = []
        page_rows: int | None = None
        batch: list[PackedData] = []
        batch_nbytes = 0

        async def flush() -> None:
            nonlocal batch, batch_nbytes, page_rows
            if not batch:
                return
            table = await _unpack_and_sort(
                context, ir_context, batch, batch_nbytes, post_sort_ir, stream
            )
            if page_rows is None:
                # A page is 1/(2*fanin) of the budget so that `fanin` pages
                # plus the merge output fit within one batch budget.
                bytes_per_row = max(1, batch_nbytes // max(1, table.num_rows()))
                page_rows = max(
                    _MIN_PAGE_ROWS, (batch_bytes // (2 * fanin)) // bytes_per_row
                )
            writer = RunWriter(part_dir / f"run-{len(runs):05d}", page_rows)
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
        pass_no = 0
        while len(runs) > fanin:
            next_runs: list[Run] = []
            for start in range(0, len(runs), fanin):
                writer = RunWriter(
                    part_dir / f"pass-{pass_no:02d}-run-{len(next_runs):05d}",
                    page_rows or 1,
                )

                async def to_run(
                    table: plc.Table, s: Stream, w: RunWriter = writer
                ) -> None:
                    await w.write(table, s, ir_context)

                await merge_runs(
                    context, ir_context, runs[start : start + fanin], order_keys, to_run
                )
                next_runs.append(writer.finish())
            runs = next_runs
            pass_no += 1
        await merge_runs(context, ir_context, runs, order_keys, emit)
        return "external"
    finally:
        await ir_context.to_thread(shutil.rmtree, part_dir, ignore_errors=True)
