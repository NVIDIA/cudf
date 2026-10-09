# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the out-of-core Sort (``sort_strategy="external"``)."""

from __future__ import annotations

import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_streaming.channel_metadata import OrderKey
from rapidsmpf.config import Options

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph.collectives import external_sort as es
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils import sorting

if TYPE_CHECKING:
    import pylibcudf as plc

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

BASE = {
    "max_rows_per_partition": 2_100,
    "fallback_mode": "raise",
    "raise_on_fail": True,
    "sort_strategy": "external",
}


@pytest.fixture
def force_disk(monkeypatch):
    """
    Every partition exceeds the run budget and every merge exceeds the fan-in.

    The monkeypatch only reaches in-process engines (``spmd``, ``spmd-small``);
    on ``dask``/``ray`` the workers are separate processes, so those variants
    exercise the streaming insertion and the in-memory fast path instead.
    """
    monkeypatch.setattr(es, "batch_bytes_for", lambda executor: 1)
    monkeypatch.setattr(es, "merge_fanin", lambda: 2)


@pytest.fixture
def engine_external(streaming_engine_factory):
    return streaming_engine_factory(StreamingOptions(**BASE))


# --------------------------------------------------------------------------- #
# Direct unit tests of the run writer / cursor / k-way merge
# --------------------------------------------------------------------------- #

_I64 = DataType(pl.Int64())


def _run_async(coro):
    return asyncio.run(coro)


def _frame_table(frame: pl.DataFrame, stream) -> plc.Table:
    return DataFrame.from_polars(frame, stream).table


def _table_frame(table: plc.Table, names: list[str], stream) -> pl.DataFrame:
    return DataFrame.from_table(table, names, [_I64] * len(names), stream).to_polars()


def _ir_context(context) -> tuple[ThreadPoolExecutor, IRExecutionContext]:
    pool = ThreadPoolExecutor(max_workers=2)
    return pool, IRExecutionContext(
        pool, get_cuda_stream=context.br().stream_pool.get_stream
    )


async def _write_run(
    context, ir_context, directory: Path, frame: pl.DataFrame, page_rows: int
):
    stream = ir_context.get_cuda_stream()
    writer = es.RunWriter(directory, page_rows)
    await writer.write(_frame_table(frame, stream), stream, ir_context)
    return writer.finish()


def _sorted_frame(
    frame: pl.DataFrame, *, descending: bool, nulls_last: bool
) -> pl.DataFrame:
    return frame.sort(
        ["k", "v"], descending=descending, nulls_last=nulls_last, maintain_order=True
    )


def _order_keys(*, descending: bool, nulls_last: bool) -> list[OrderKey]:
    # Same polars -> libcudf mapping the Sort IR uses (null precedence is
    # expressed relative to the reversed order for descending columns).
    # Merge on both columns so the result is fully determined (no ties).
    orders, null_orders = sorting.sort_order(
        [descending, descending], nulls_last=[nulls_last, nulls_last], num_keys=2
    )
    return [OrderKey(i, orders[i], null_orders[i]) for i in range(2)]


def _random_frame(n: int, seed: int, *, with_nulls: bool) -> pl.DataFrame:
    rng = random.Random(seed)
    k = [rng.randint(0, 20) for _ in range(n)]  # many duplicate keys
    v = list(range(n))
    if with_nulls:
        k = [None if rng.random() < 0.1 else x for x in k]
    return pl.DataFrame({"k": k, "v": v}, schema={"k": pl.Int64, "v": pl.Int64})


@pytest.mark.spmd
def test_run_writer_and_cursor_roundtrip(spmd_engine, tmp_path) -> None:
    context = spmd_engine.context
    frame = _random_frame(100, 1, with_nulls=False).sort("k", "v")

    async def _run():
        pool, ir_context = _ir_context(context)
        with pool:
            run = await _write_run(context, ir_context, tmp_path / "run", frame, 7)
            assert len(run.pages) == 15  # ceil(100 / 7)
            assert run.num_rows == 100
            assert all(page.path.exists() for page in run.pages)
            cursor = es.RunCursor(run)
            stream = ir_context.get_cuda_stream()
            parts = []
            while cursor.has_more():
                table = await cursor.load_next(context, ir_context, [0, 1], stream)
                parts.append(_table_frame(table, ["k", "v"], stream))
                # tail = key columns of the last row of the page just loaded
                tail = _table_frame(cursor.tail, ["k", "v"], stream)
                assert tail.row(0) == parts[-1].row(-1)
            assert pl.concat(parts).equals(frame)
            # Pages are deleted as they are consumed.
            assert not any(page.path.exists() for page in run.pages)

    _run_async(_run())


@pytest.mark.spmd
@pytest.mark.parametrize("n_runs", [1, 2, 5])
@pytest.mark.parametrize("page_rows", [3, 50])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("with_nulls", [False, True])
def test_merge_runs_matches_sorted_concat(
    spmd_engine, tmp_path, n_runs, page_rows, descending, with_nulls
) -> None:
    context = spmd_engine.context
    nulls_last = not descending  # exercise both null placements
    frames = [
        _sorted_frame(
            _random_frame(60 + 13 * i, i, with_nulls=with_nulls),
            descending=descending,
            nulls_last=nulls_last,
        )
        for i in range(n_runs)
    ]
    expected = _sorted_frame(
        pl.concat(frames), descending=descending, nulls_last=nulls_last
    )

    async def _run():
        pool, ir_context = _ir_context(context)
        with pool:
            runs = [
                await _write_run(
                    context, ir_context, tmp_path / f"run{i}", f, page_rows
                )
                for i, f in enumerate(frames)
            ]
            emitted: list[pl.DataFrame] = []

            async def emit(table, stream):
                emitted.append(_table_frame(table, ["k", "v"], stream))

            await es.merge_runs(
                context,
                ir_context,
                runs,
                _order_keys(descending=descending, nulls_last=nulls_last),
                emit,
            )
            return emitted

    emitted = _run_async(_run())
    assert emitted, "merge emitted nothing"
    result = pl.concat(emitted)
    assert result.equals(expected)
    # Every emitted chunk is itself sorted and chunks do not overlap.
    for chunk in emitted:
        assert chunk.equals(
            _sorted_frame(chunk, descending=descending, nulls_last=nulls_last)
        )


@pytest.mark.spmd
def test_merge_runs_refills_every_run_sharing_the_minimum_tail(
    spmd_engine, tmp_path
) -> None:
    """Two runs whose first pages end on the same key must both be refilled."""
    context = spmd_engine.context
    # page_rows=3: run A pages end on keys 5, 9; run B pages end on 5, 12; run C is all larger.
    a = pl.DataFrame({"k": [1, 3, 5, 6, 8, 9], "v": [0, 1, 2, 3, 4, 5]})
    b = pl.DataFrame({"k": [2, 4, 5, 10, 11, 12], "v": [6, 7, 8, 9, 10, 11]})
    c = pl.DataFrame({"k": [20, 21, 22], "v": [12, 13, 14]})
    expected = pl.concat([a, b, c]).sort("k", "v")

    async def _run():
        pool, ir_context = _ir_context(context)
        with pool:
            runs = [
                await _write_run(context, ir_context, tmp_path / n, f, 3)
                for n, f in (("a", a), ("b", b), ("c", c))
            ]
            emitted = []

            async def emit(table, stream):
                emitted.append(_table_frame(table, ["k", "v"], stream))

            await es.merge_runs(
                context,
                ir_context,
                runs,
                _order_keys(descending=False, nulls_last=True),
                emit,
            )
            return emitted

    emitted = _run_async(_run())
    assert pl.concat(emitted).equals(expected)


@pytest.mark.spmd
def test_merge_runs_empty_inputs(spmd_engine, tmp_path) -> None:
    context = spmd_engine.context

    async def _run():
        pool, ir_context = _ir_context(context)
        with pool:
            calls = []

            async def emit(table, stream):
                calls.append(table.num_rows())

            await es.merge_runs(
                context,
                ir_context,
                [],
                _order_keys(descending=False, nulls_last=True),
                emit,
            )
            await es.merge_runs(
                context,
                ir_context,
                [es.Run(), es.Run()],
                _order_keys(descending=False, nulls_last=True),
                emit,
            )
            return calls

    assert _run_async(_run()) == []


def test_make_run_directory(tmp_path) -> None:
    spill = tmp_path / "spill"
    path = es.make_run_directory(Options({"disk_spill_dir": str(spill)}), rank=3)
    assert path.parent == spill
    assert path.name.startswith("cudf-polars-sort-3-")
    assert path.is_dir()
    for options in (Options({}), Options({"disk_spill_dir": "false"})):
        fallback = es.make_run_directory(options, rank=0)
        assert fallback.is_dir()
        assert spill not in fallback.parents
        fallback.rmdir()
    # An explicit run_dir wins over disk_spill_dir.
    runs = tmp_path / "runs"
    explicit = es.make_run_directory(
        Options({"disk_spill_dir": str(spill)}), rank=1, run_dir=str(runs)
    )
    assert explicit.parent == runs
    assert explicit.is_dir()


# --------------------------------------------------------------------------- #
# Query-level tests through the engines
# --------------------------------------------------------------------------- #


def _dtype_cases():
    n = 2_000  # enough for several partitions, pages and merge passes per engine
    rng = random.Random(0)
    ints = [rng.randint(-1_000_000, 1_000_000) for _ in range(n)]
    small = [rng.randint(-100, 100) for _ in range(n)]
    floats = [rng.uniform(-1e6, 1e6) for _ in range(n)]
    strs = [f"s{rng.randint(0, 999):04d}" for _ in range(n)]
    bools = [rng.random() < 0.5 for _ in range(n)]
    days = [rng.randint(0, 20_000) for _ in range(n)]
    yield pytest.param(pl.Series("c", small, dtype=pl.Int8), True, id="int8")
    yield pytest.param(pl.Series("c", small, dtype=pl.Int16), True, id="int16")
    yield pytest.param(pl.Series("c", ints, dtype=pl.Int32), True, id="int32")
    yield pytest.param(pl.Series("c", ints, dtype=pl.Int64), True, id="int64")
    yield pytest.param(
        pl.Series("c", [abs(x) for x in ints], dtype=pl.UInt32), True, id="uint32"
    )
    yield pytest.param(pl.Series("c", floats, dtype=pl.Float32), True, id="float32")
    yield pytest.param(pl.Series("c", floats, dtype=pl.Float64), True, id="float64")
    yield pytest.param(pl.Series("c", bools, dtype=pl.Boolean), True, id="bool")
    yield pytest.param(pl.Series("c", strs, dtype=pl.String), True, id="string")
    yield pytest.param(
        pl.Series("c", days, dtype=pl.Int32).cast(pl.Date), True, id="date"
    )
    yield pytest.param(
        pl.Series("c", [d * 86_400_000 for d in days], dtype=pl.Int64).cast(
            pl.Datetime("ms")
        ),
        True,
        id="datetime_ms",
    )
    yield pytest.param(
        pl.Series("c", [d * 86_400_000_000 for d in days], dtype=pl.Int64).cast(
            pl.Datetime("us")
        ),
        True,
        id="datetime_us",
    )
    yield pytest.param(
        pl.Series("c", [d * 1_000 for d in days], dtype=pl.Int64).cast(
            pl.Duration("ms")
        ),
        True,
        id="duration_ms",
    )
    yield pytest.param(
        pl.Series("c", [x / 1000 for x in ints]).cast(pl.Decimal(18, 3)),
        True,
        id="decimal_18_3",
    )
    yield pytest.param(
        pl.Series("c", [[x, x + 1] for x in small], dtype=pl.List(pl.Int64)),
        False,
        id="list_int64_payload",
    )


@pytest.mark.parametrize("column,sortable", list(_dtype_cases()))
def test_external_sort_dtypes(column, sortable, engine_external, force_disk) -> None:
    """Every dtype must survive the run-page round trip and merge, as key and payload."""
    n = len(column)
    rng = random.Random(1)
    keys = [rng.randint(0, 200) for _ in range(n)]  # duplicates -> ties across pages
    df = pl.LazyFrame({"k": keys, "c": column, "idx": list(range(n))})
    # As payload (sort by k, stable so the payload order is determined).
    q = df.sort("k", "idx")
    assert_gpu_result_equal(q, engine=engine_external)
    if sortable:
        # As the sort key, both directions, nulls-last/first.
        q = df.sort("c", "idx", descending=True, nulls_last=True)
        assert_gpu_result_equal(q, engine=engine_external)
        q = df.sort("c", "idx")
        assert_gpu_result_equal(q, engine=engine_external)


@pytest.mark.parametrize("stable", [False, True])
def test_external_sort_many_pages(
    stable, engine_external, force_disk, monkeypatch
) -> None:
    """Hundreds of page refills per merge: tiny pages, tiny budget, fan-in 2."""
    monkeypatch.setattr(es, "_MIN_PAGE_ROWS", 32)
    n = 12_000
    rng = random.Random(7)
    df = pl.LazyFrame(
        {
            "a": [rng.randint(0, 30) for _ in range(n)],  # heavy duplication
            "b": [rng.uniform(0, 1) for _ in range(n)],
            "idx": list(range(n)),
        }
    )
    q = (
        df.sort(["a", "b"], maintain_order=stable)
        if stable
        else df.sort(["a", "b", "idx"])
    )
    assert_gpu_result_equal(q, engine=engine_external)


def _leftover_run_dirs(spill: Path) -> list[Path]:
    return sorted(p for p in spill.glob("cudf-polars-sort-*"))


def _require_in_process(engine) -> None:
    # Monkeypatches only reach the test process; dask/ray run the sort in
    # separate worker processes, so tests relying on them are meaningless there.
    if type(engine).__name__ != "SPMDEngine":
        pytest.skip("requires an in-process (SPMD) engine for monkeypatching")


def test_external_sort_uses_and_cleans_disk_spill_dir(
    streaming_engine_factory, force_disk, tmp_path
) -> None:
    spill = tmp_path / "spill"
    spill.mkdir()
    engine = streaming_engine_factory(
        StreamingOptions(**BASE, disk_spill_dir=str(spill))
    )
    _require_in_process(engine)
    seen: list[Path] = []
    original = es.RunWriter.__init__

    def spy(self, directory, page_rows):
        seen.append(Path(directory))
        original(self, directory, page_rows)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(es.RunWriter, "__init__", spy)
        df = pl.LazyFrame({"x": list(range(10_000, 0, -1)), "y": list(range(10_000))})
        assert_gpu_result_equal(df.sort("x"), engine=engine)
    assert seen, "no runs were written"
    assert all(spill in d.parents for d in seen), seen
    assert _leftover_run_dirs(spill) == []


def test_external_sort_cleans_run_dir_on_failure(
    streaming_engine_factory, force_disk, tmp_path, monkeypatch
) -> None:
    spill = tmp_path / "spill"
    spill.mkdir()
    engine = streaming_engine_factory(
        StreamingOptions(**BASE, disk_spill_dir=str(spill))
    )
    _require_in_process(engine)

    async def boom(*args, **kwargs):
        raise RuntimeError("injected merge failure")

    monkeypatch.setattr(es, "merge_runs", boom)
    df = pl.LazyFrame({"x": list(range(10_000, 0, -1))})
    with pytest.raises(BaseException) as excinfo:
        df.sort("x").collect(engine=engine)
    assert "injected merge failure" in _flatten_exception(excinfo.value)
    assert _leftover_run_dirs(spill) == []


def _flatten_exception(error: BaseException) -> str:
    """Messages of ``error`` and, recursively, of any ExceptionGroup members."""
    parts = [str(error)]
    for sub in getattr(error, "exceptions", ()):
        parts.append(_flatten_exception(sub))
    return " | ".join(parts)


@pytest.mark.spmd
def test_external_sort_matches_in_memory_strategy(
    spmd_engine_factory, force_disk
) -> None:
    """
    A/B of the two strategies on the same engine (multi-rank under rrun).

    Comparing against CPU polars is not meaningful with more than one rank (an
    in-memory LazyFrame is present on every rank), so compare the strategies
    against each other instead: they must agree exactly.
    """
    from polars.testing import assert_frame_equal

    external = spmd_engine_factory(StreamingOptions(**BASE))
    in_memory = spmd_engine_factory(
        StreamingOptions(**{**BASE, "sort_strategy": "in-memory"})
    )
    n = 20_000
    rng = random.Random(3)
    df = pl.LazyFrame(
        {
            "k": [rng.randint(0, 500) for _ in range(n)],
            "s": [f"v{rng.randint(0, 99):03d}" for _ in range(n)],
            "idx": list(range(n)),
        }
    )
    for q in (
        df.sort("k", "s", "idx"),
        df.sort("k", descending=True, maintain_order=True),
        df.sort("s", "idx", nulls_last=True),
    ):
        assert_frame_equal(q.collect(engine=external), q.collect(engine=in_memory))
