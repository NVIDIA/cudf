# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``gather_statistics`` / ``global_statistics`` on streaming engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rapidsmpf.statistics import Statistics

from cudf_polars.engine.options import StreamingOptions

if TYPE_CHECKING:
    from collections.abc import Callable

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
    """Yield each supported streaming engine with statistics enabled."""
    return streaming_engine_factory(
        StreamingOptions(statistics=True, max_rows_per_partition=10),
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
