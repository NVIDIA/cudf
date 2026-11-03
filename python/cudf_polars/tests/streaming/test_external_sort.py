# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the out-of-core Sort (``sort_strategy="external"``)."""

from __future__ import annotations

import gc
import random
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

from rapidsmpf.memory.buffer import MemoryType

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph.collectives import external_sort as es
from cudf_polars.testing.asserts import assert_gpu_result_equal

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.engine_params(["spmd", "dask", "ray"])

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

BASE: dict[str, Any] = {
    "max_rows_per_partition": 2_100,
    "fallback_mode": "raise",
    "raise_on_fail": True,
    "sort_strategy": "external",
}


@pytest.fixture
def force_disk(monkeypatch):
    """
    Every partition exceeds the run budget and every merge exceeds the fan-in.

    The monkeypatch only reaches in-process engines (``spmd``); on
    ``dask``/``ray`` the workers are separate processes, so those variants
    exercise the streaming insertion and the in-memory fast path instead.
    """
    monkeypatch.setattr(es, "batch_bytes_for", lambda target_partition_size: 1)
    monkeypatch.setattr(es, "DEFAULT_MERGE_FANIN", 2)


@pytest.fixture
def engine_external(streaming_engine_factory):
    return streaming_engine_factory(StreamingOptions(**BASE))


def _spill_files(spill: Path) -> list[Path]:
    return sorted(p for p in spill.rglob("*") if p.is_file())


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
    seen: list[MemoryType] = []
    original = es.RunWriter.__init__

    def spy(self, context, page_rows):
        original(self, context, page_rows)
        seen.append(self._mem_type)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(es.RunWriter, "__init__", spy)
        df = pl.LazyFrame({"x": list(range(10_000, 0, -1)), "y": list(range(10_000))})
        assert_gpu_result_equal(df.sort("x"), engine=engine)
    assert seen, "no runs were written"
    assert set(seen) == {MemoryType.DISK}
    assert _spill_files(spill) == []


def test_external_sort_releases_run_pages_on_failure(
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
    # Run pages are released with the frames that reference them.
    del excinfo
    gc.collect()
    assert _spill_files(spill) == []


def _flatten_exception(error: BaseException) -> str:
    """Messages of ``error`` and, recursively, of any ExceptionGroup members."""
    parts = [str(error)]
    parts.extend(_flatten_exception(sub) for sub in getattr(error, "exceptions", ()))
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
