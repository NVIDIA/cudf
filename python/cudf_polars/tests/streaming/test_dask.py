# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Dask execution mode."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

import polars as pl

from rapidsmpf.bootstrap import is_running_with_rrun

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import DaskContext

# try/except instead of pytest.importorskip so mypy can use distributed for type checking
try:
    import distributed
except ImportError:
    pytest.skip("distributed not installed", allow_module_level=True)

from cudf_polars.engine.dask import DaskEngine

if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = [
    pytest.mark.skipif(
        is_running_with_rrun(),
        reason="DaskEngine must not be created from within an rrun cluster",
    ),
]


@pytest.fixture(scope="module")
def dask_client() -> Iterator[distributed.Client]:
    # Use for DaskEngine constructor tests to avoid re-creating the cluster
    # Otherwise, use session-scoped DaskEngine in conftest
    with (
        distributed.LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            silence_logs=logging.WARNING,
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        yield client


# ---------------------------------------------------------------------------
# GPU tests — reuse the session-scoped Dask cluster from conftest
# ---------------------------------------------------------------------------


def test_from_options(dask_client: distributed.Client) -> None:
    """DaskEngine.from_options with default StreamingOptions creates a valid engine."""
    opts = StreamingOptions(fallback_mode="silent")
    with DaskEngine.from_options(opts, dask_client=dask_client) as engine:
        assert engine.nranks >= 1


def test_yields_engine(dask_engine: DaskEngine) -> None:
    """DaskEngine is a GPUEngine with at least one rank."""
    assert isinstance(dask_engine, pl.GPUEngine)
    assert dask_engine.nranks >= 1


def test_executor_options_forwarded(dask_engine: DaskEngine) -> None:
    """Reserved executor_options keys are injected into the engine config."""
    opts = dask_engine.config["executor_options"]
    assert opts["cluster"] == "dask"
    assert isinstance(opts["dask_context"], DaskContext)


def test_gather_cluster_info(dask_engine: DaskEngine) -> None:
    """gather_cluster_info returns one ClusterInfo per rank with expected fields."""
    infos = dask_engine.gather_cluster_info()
    assert len(infos) == dask_engine.nranks
    for info in infos:
        assert isinstance(info.pid, int)
        assert isinstance(info.hostname, str)
    # Each worker runs in its own process.
    assert len({info.pid for info in infos}) == dask_engine.nranks


def test_worker_host_memory_limit(dask_engine: DaskEngine) -> None:
    """Memory limit is respected."""
    scheduler_info = dask_engine._dask_ctx.client.scheduler_info(n_workers=-1)
    worker = next(iter(scheduler_info["workers"].values()))
    assert worker["memory_limit"] == distributed.system.MEMORY_LIMIT


def test_from_options_creates_engine(dask_client: distributed.Client) -> None:
    """DaskEngine.from_options produces a working engine and runs a query."""
    opts = StreamingOptions(max_rows_per_partition=10, fallback_mode="silent")
    with DaskEngine.from_options(opts, dask_client=dask_client) as eng:
        assert isinstance(eng, pl.GPUEngine)
        assert eng.nranks >= 1
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        assert_gpu_result_equal(lf, engine=eng, check_row_order=False)


def test_run(dask_engine: DaskEngine) -> None:
    result = dask_engine._run(os.getpid)
    assert len(set(result)) == dask_engine.nranks


@pytest.fixture(scope="module")
def reset_engine(dask_client: distributed.Client) -> Iterator[DaskEngine]:
    """Module-scoped engine for reset tests — independent of ``dask_engine``.

    These tests exercise :meth:`DaskEngine._reset` (which mutates the
    engine in-place). A dedicated fixture keeps those mutations from
    leaking into the conftest-shared ``dask_engine``.

    Note: Do not use this fixture if you call _reset with a new dask_client.
    """
    with DaskEngine(
        executor_options={"max_rows_per_partition": 10},
        dask_client=dask_client,
    ) as e:
        yield e


def test_reset_keeps_workers_alive(reset_engine: DaskEngine) -> None:
    """``_reset`` must not respawn dask workers."""
    workers_before = sorted(
        reset_engine._dask_ctx.client.scheduler_info(n_workers=-1)["workers"]
    )
    pids_before = sorted(reset_engine._run(os.getpid))

    reset_engine._reset(executor_options={"max_rows_per_partition": 7})

    workers_after = sorted(
        reset_engine._dask_ctx.client.scheduler_info(n_workers=-1)["workers"]
    )
    pids_after = sorted(reset_engine._run(os.getpid))

    # Same worker addresses …
    assert workers_before == workers_after
    # … and the workers are running in the same OS processes.
    assert pids_before == pids_after


def test_reset_updates_executor_options(reset_engine: DaskEngine) -> None:
    """``_reset`` updates the polars-layer config to the new options."""
    reset_engine._reset(executor_options={"max_rows_per_partition": 42})

    opts = reset_engine.config["executor_options"]
    assert opts["max_rows_per_partition"] == 42
    # Reserved keys are still injected by ``_reset``.
    assert opts["cluster"] == "dask"
    assert isinstance(opts["dask_context"], DaskContext)


def test_reset_collects_after_options_change(reset_engine: DaskEngine) -> None:
    """The engine still drives a real query after ``_reset``."""
    reset_engine._reset(executor_options={"max_rows_per_partition": 3})
    assert_gpu_result_equal(
        pl.LazyFrame({"a": [1, 2, 3, 4, 5]}),
        engine=reset_engine,
        check_row_order=False,
    )


def test_reset_after_shutdown_raises(dask_client: distributed.Client) -> None:
    """``shutdown`` is idempotent; ``_reset`` after shutdown raises every time."""
    engine = DaskEngine(
        dask_client=dask_client,
        executor_options={"max_rows_per_partition": 10},
    )
    engine.shutdown()
    engine.shutdown()  # idempotent
    with pytest.raises(RuntimeError, match="shut-down"):
        engine._reset()
    with pytest.raises(RuntimeError, match="shut-down"):
        engine._reset()  # still raises on a second attempt
    engine.shutdown()  # still safe after a failed _reset


def test_reset_rejects_construction_time_executor_options(
    reset_engine: DaskEngine,
) -> None:
    """``_reset`` rejects ``executor_options`` keys read at worker setup."""
    with pytest.raises(ValueError, match="num_py_executors"):
        reset_engine._reset(executor_options={"num_py_executors": 4})


def test_reset_rejects_construction_time_engine_options(
    reset_engine: DaskEngine,
) -> None:
    """``_reset`` rejects ``engine_options`` keys read at worker setup."""
    from cudf_polars.engine.hardware_binding import (
        HardwareBindingPolicy,
    )

    with pytest.raises(ValueError, match="hardware_binding"):
        reset_engine._reset(
            engine_options={
                "hardware_binding": HardwareBindingPolicy(enabled=False),
            },
        )
    with pytest.raises(ValueError, match="memory_resource_config"):
        reset_engine._reset(engine_options={"memory_resource_config": None})
