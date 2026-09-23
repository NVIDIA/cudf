# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for statistics gathering on streaming engines: rapidsmpf and kvikio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import kvikio
import pytest

import polars as pl

from rapidsmpf.statistics import Statistics

from cudf_polars.engine.options import StreamingOptions

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cudf_polars.engine.core import StreamingEngine

# Runs the spmd variant even under rrun with nranks > 1. The ray/dask
# variants skip themselves in that environment.
pytestmark = [
    pytest.mark.spmd,
]


@pytest.fixture
def engine(
    streaming_engine_factory: Callable[..., StreamingEngine],
) -> StreamingEngine:
    """Yield each supported streaming engine with both kinds of statistics on."""
    return streaming_engine_factory(
        StreamingOptions(
            statistics=True, kvikio_statistics=True, max_rows_per_partition=10
        ),
    )


def test_statistics(engine: StreamingEngine) -> None:
    """gather_statistics / global_statistics / clear round-trip."""
    # gather_statistics returns one enabled Statistics per rank.
    stats = engine.gather_statistics()
    assert len(stats) == engine.nranks
    for s in stats:
        assert isinstance(s, Statistics)
        assert s.enabled

    # global_statistics returns a single merged, enabled Statistics.
    merged = engine.global_statistics()
    assert isinstance(merged, Statistics)
    assert merged.enabled

    # gather_statistics(clear=True) captures and then empties each rank.
    engine.gather_statistics(clear=True)
    stats = engine.gather_statistics()
    assert len(stats) == engine.nranks
    for s in stats:
        assert s.enabled
        # The allgather of the statistics can return locally on a single
        # rank and clear the stats before the allgather event loop is
        # removed. So we might see an event loop stat, but no other.
        assert s.list_stat_names() == [] or s.list_stat_names() == ["event-loop-total"]


@pytest.fixture
def scan_query(tmp_path: Path) -> pl.LazyFrame:
    """A parquet scan, so that the engine actually performs I/O."""
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": range(1000), "b": range(1000)}).write_parquet(path)
    return pl.scan_parquet(path)


def test_io_summary(engine: StreamingEngine, scan_query: pl.LazyFrame) -> None:
    """gather_io_summary reports what each rank read."""
    scan_query.collect(engine=engine)

    # Statistics are enabled on this fixture, so every rank is counting.
    gathered = engine.gather_io_summary()
    assert sorted(gathered) == list(range(engine.nranks))
    summaries = list(gathered.values())
    # Only some ranks may have been given a file to read, but not none of them.
    assert sum(s.num_ops for s in summaries) > 0
    assert sum(s.bytes_read for s in summaries) > 0

    # kvikio owns Summary and its invariants, so the only thing to check here is
    # that each rank handed back one intact.
    assert all(isinstance(s, kvikio.Summary) for s in summaries)


def test_io_summary_clear_starts_a_new_span(
    engine: StreamingEngine, scan_query: pl.LazyFrame
) -> None:
    """clear=True returns the totals so far and restarts the span."""
    scan_query.collect(engine=engine)

    before = engine.gather_io_summary(clear=True).values()
    assert sum(s.num_ops for s in before) > 0

    # Nothing between the clear and this gather, so the new span is empty.
    after = engine.gather_io_summary()
    assert len(after) == engine.nranks
    assert sum(s.num_ops for s in after.values()) == 0
    assert sum(s.bytes_transferred for s in after.values()) == 0


def test_io_summary_is_independent_of_rapidsmpf_statistics(
    streaming_engine_factory: Callable[..., StreamingEngine],
    scan_query: pl.LazyFrame,
) -> None:
    """``kvikio_statistics`` gates I/O counting on its own, and survives a reset."""
    # RapidsMPF statistics on, kvikio off. One does not imply the other.
    engine = streaming_engine_factory(
        StreamingOptions(
            statistics=True, kvikio_statistics=False, max_rows_per_partition=10
        )
    )
    scan_query.collect(engine=engine)
    assert engine.gather_statistics()[0].enabled
    # An absent rank is "this rank was not counting", which is not the same as
    # a zeroed summary meaning "this rank did no I/O".
    assert engine.gather_io_summary() == {}

    # The factory resets the one shared engine, so this exercises the monitor
    # being created on an engine that was running without one.
    reset = streaming_engine_factory(
        StreamingOptions(
            statistics=False, kvikio_statistics=True, max_rows_per_partition=10
        )
    )
    assert reset is engine
    scan_query.collect(engine=engine)
    summaries = engine.gather_io_summary()
    assert sorted(summaries) == list(range(engine.nranks))
    assert sum(s.num_ops for s in summaries.values()) > 0
