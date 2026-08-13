# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ray execution mode."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

import polars as pl

from rapidsmpf.bootstrap import is_running_with_rrun

from cudf_polars.engine.hardware_binding import HardwareBindingPolicy
from cudf_polars.utils.config import RayContext

ray = pytest.importorskip("ray")
from cudf_polars.engine.ray import RayEngine  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = [
    pytest.mark.skipif(
        is_running_with_rrun(),
        reason="RayEngine must not be created from within an rrun cluster",
    ),
]


# ---------------------------------------------------------------------------
# Context-manager smoke tests (no GPU required)
# ---------------------------------------------------------------------------


def test_reserved_executor_keys() -> None:
    """executor_options rejects reserved keys."""
    for key in ("cluster", "spmd_context", "ray_context"):
        with pytest.raises(TypeError, match="reserved"):
            RayEngine(executor_options={key: "anything"})


def test_reserved_engine_options_keys() -> None:
    """engine_options rejects reserved keys."""
    for key in ("memory_resource", "executor"):
        with pytest.raises(TypeError, match="reserved"):
            RayEngine(engine_options={key: "anything"})


def test_raises_inside_rrun() -> None:
    """RayEngine must not be created from within an rrun cluster."""
    with (
        patch(
            "rapidsmpf.bootstrap.is_running_with_rrun",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="rrun"),
    ):
        RayEngine()


def test_num_ranks_requires_allow_gpu_sharing(ray_num_ranks: int) -> None:
    """num_ranks requires engine_options['allow_gpu_sharing']=True."""
    with pytest.raises(ValueError, match="allow_gpu_sharing"):
        RayEngine(num_ranks=ray_num_ranks)
    with pytest.raises(ValueError, match="allow_gpu_sharing"):
        RayEngine(num_ranks=ray_num_ranks, engine_options={"allow_gpu_sharing": False})


def test_num_ranks_must_be_positive() -> None:
    """num_ranks must be at least 1."""
    with pytest.raises(ValueError, match="num_ranks"):
        RayEngine(num_ranks=0, engine_options={"allow_gpu_sharing": True})


# ---------------------------------------------------------------------------
# GPU tests — reuse the session-scoped Ray cluster from conftest
# ---------------------------------------------------------------------------


def test_yields_engine(ray_engine: RayEngine) -> None:
    """RayEngine is a GPUEngine with at least one rank."""
    assert isinstance(ray_engine, pl.GPUEngine)
    assert ray_engine.nranks >= 1


def test_executor_options_forwarded(ray_engine: RayEngine) -> None:
    """Reserved executor_options keys are injected into the engine config."""
    opts = ray_engine.config["executor_options"]
    assert opts["cluster"] == "ray"
    assert isinstance(opts["ray_context"], RayContext)
    assert ray_engine.rank_actors == opts["ray_context"].rank_actors
    assert len(ray_engine.rank_actors) == ray_engine.nranks


def test_gather_cluster_info(ray_engine: RayEngine) -> None:
    """gather_cluster_info returns one ClusterInfo per rank with expected fields."""
    infos = ray_engine.gather_cluster_info()
    assert len(infos) == ray_engine.nranks
    for info in infos:
        assert isinstance(info.hostname, str)
        assert isinstance(info.pid, int)
    # Each actor runs in its own process.
    assert len({info.pid for info in infos}) == ray_engine.nranks


def test_run(ray_engine: RayEngine) -> None:
    result = ray_engine._run(os.getpid)
    assert len(set(result)) == ray_engine.nranks


def test_num_ranks_oversubscribes(ray_engine: RayEngine, ray_num_ranks: int) -> None:
    """num_ranks creates the requested number of actors sharing GPU 0."""
    assert ray_engine.nranks == ray_num_ranks
    assert len(ray_engine.rank_actors) == ray_num_ranks
    result = pl.LazyFrame({"a": [1, 2, 3, 4]}).collect(engine=ray_engine)
    assert sorted(result["a"].to_list()) == [1, 2, 3, 4]


@pytest.fixture(scope="module")
def reset_engine(
    ray_num_ranks: int,
    ray_init_options: dict[str, Any],
) -> Iterator[RayEngine]:
    """Module-scoped engine for reset tests — independent of ``ray_engine``.

    These tests exercise :meth:`RayEngine._reset` (which mutates the
    engine in-place). A dedicated fixture keeps those mutations from
    leaking into the conftest-shared ``ray_engine``.
    """
    with RayEngine(
        executor_options={"max_rows_per_partition": 10},
        engine_options={"allow_gpu_sharing": True},
        num_ranks=ray_num_ranks,
        ray_init_options=ray_init_options,
    ) as e:
        yield e


def test_reset_keeps_actors_alive(reset_engine: RayEngine) -> None:
    """``_reset`` must not respawn rank actor processes."""
    actors_before = list(reset_engine.rank_actors)
    pids_before = reset_engine._run(os.getpid)

    reset_engine._reset(
        executor_options={"max_rows_per_partition": 7},
        engine_options={"allow_gpu_sharing": True},
    )

    actors_after = list(reset_engine.rank_actors)
    pids_after = reset_engine._run(os.getpid)

    # The Python ``ActorHandle`` objects are the same instances …
    assert all(a is b for a, b in zip(actors_before, actors_after, strict=True))
    # … and the actors are running in the same OS processes.
    assert pids_before == pids_after


def test_reset_updates_executor_options(reset_engine: RayEngine) -> None:
    """``_reset`` updates the polars-layer config to the new options."""
    reset_engine._reset(
        executor_options={"max_rows_per_partition": 42},
        engine_options={"allow_gpu_sharing": True},
    )

    opts = reset_engine.config["executor_options"]
    assert opts["max_rows_per_partition"] == 42
    # Reserved keys are still injected by ``_reset``.
    assert opts["cluster"] == "ray"
    assert isinstance(opts["ray_context"], RayContext)
    assert opts["ray_context"].rank_actors == reset_engine.rank_actors


def test_reset_collects_after_options_change(reset_engine: RayEngine) -> None:
    """The engine still drives a real query after ``_reset``."""
    reset_engine._reset(
        executor_options={"max_rows_per_partition": 3},
        engine_options={"allow_gpu_sharing": True},
    )
    result = pl.LazyFrame({"a": [1, 2, 3, 4, 5]}).collect(engine=reset_engine)
    assert sorted(result["a"].to_list()) == [1, 2, 3, 4, 5]


def test_reset_after_shutdown_raises(
    ray_num_ranks: int,
    ray_init_options: dict[str, Any],
) -> None:
    """``shutdown`` is idempotent; ``_reset`` after shutdown raises every time."""
    engine = RayEngine(
        executor_options={"max_rows_per_partition": 10},
        engine_options={"allow_gpu_sharing": True},
        num_ranks=ray_num_ranks,
        ray_init_options=ray_init_options,
    )
    engine.shutdown()
    engine.shutdown()  # idempotent
    with pytest.raises(RuntimeError, match="shut-down"):
        engine._reset()
    with pytest.raises(RuntimeError, match="shut-down"):
        engine._reset()  # still raises on a second attempt
    engine.shutdown()  # still safe after a failed _reset


def test_reset_rejects_construction_time_executor_options(
    reset_engine: RayEngine,
) -> None:
    """``_reset`` rejects ``executor_options`` keys read at actor construction."""
    with pytest.raises(ValueError, match="num_py_executors"):
        reset_engine._reset(
            executor_options={"num_py_executors": 4},
            engine_options={"allow_gpu_sharing": True},
        )


def test_reset_rejects_construction_time_engine_options(
    reset_engine: RayEngine,
) -> None:
    """``_reset`` rejects ``engine_options`` keys read at actor construction."""
    with pytest.raises(ValueError, match="hardware_binding"):
        reset_engine._reset(
            engine_options={
                "allow_gpu_sharing": True,
                "hardware_binding": HardwareBindingPolicy(enabled=False),
            },
        )
    with pytest.raises(ValueError, match="memory_resource_config"):
        reset_engine._reset(
            engine_options={
                "allow_gpu_sharing": True,
                "memory_resource_config": None,
            },
        )


def test_shutdown_skips_when_ray_not_initialized(
    ray_num_ranks: int,
    ray_init_options: dict[str, Any],
) -> None:
    """``shutdown`` short-circuits if ``ray.is_initialized()`` is ``False``."""
    engine = RayEngine(
        executor_options={"max_rows_per_partition": 10},
        engine_options={"allow_gpu_sharing": True},
        num_ranks=ray_num_ranks,
        ray_init_options=ray_init_options,
    )
    try:
        with patch("ray.is_initialized", return_value=False):
            engine.shutdown()  # must not raise, must not auto-init
        # The exit stack is closed; rank_actors is cleared.
        assert engine._rank_actors is None
    finally:
        # Defensive: if the guard somehow didn't fire, make sure the
        # actors are released so the next test's fixture isn't blocked.
        if engine._rank_actors is not None:
            engine.shutdown()
