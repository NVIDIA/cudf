# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private pooled-engine support used by the cudf-polars test fixtures."""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING, Any

from cudf_polars.testing.engine_utils import configure_streaming_engine

if TYPE_CHECKING:
    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.engine.options import StreamingOptions


class EnginePool:
    """Keep ordinary Ray and Dask test engines alive within one pytest worker."""

    def __init__(
        self,
        *,
        ray_num_ranks: int,
        ray_init_options: dict[str, Any],
    ) -> None:
        self._ray_num_ranks = ray_num_ranks
        self._ray_init_options = ray_init_options
        self._engines: dict[tuple[Any, ...], StreamingEngine] = {}
        self._construct_count = {"dask": 0, "ray": 0}
        self._reuse_count = {"dask": 0, "ray": 0}
        self._discard_count = {"dask": 0, "ray": 0}
        self._construct_seconds = {"dask": 0.0, "ray": 0.0}
        self._reset_seconds = {"dask": 0.0, "ray": 0.0}
        self._health_seconds = {"dask": 0.0, "ray": 0.0}
        self._discard_reasons: list[str] = []

    def acquire(self, engine_name: str) -> StreamingEngine:
        """Return the idle standard engine for ``engine_name``, creating it once."""
        key = self._key(engine_name)
        try:
            engine = self._engines[key]
        except KeyError:
            start = time.perf_counter()
            engine = self._construct(engine_name)
            self._construct_seconds[engine_name] += time.perf_counter() - start
            self._engines[key] = engine
            self._construct_count[engine_name] += 1
        else:
            self._reuse_count[engine_name] += 1
        return engine

    def release(
        self,
        engine_name: str,
        engine: StreamingEngine,
        options: StreamingOptions,
        *,
        test_failed: bool,
        nodeid: str,
    ) -> None:
        """Reset and health-check a successful test's engine before reuse."""
        if test_failed:
            self._discard_reasons.append(f"{engine_name} {nodeid}: test failed")
            self._discard(engine_name, engine)
            return
        try:
            start = time.perf_counter()
            configure_streaming_engine(engine, options)
            self._reset_seconds[engine_name] += time.perf_counter() - start
            start = time.perf_counter()
            self._health_check(engine_name, engine)
            self._health_seconds[engine_name] += time.perf_counter() - start
        except Exception as error:
            self._discard_reasons.append(
                f"{engine_name} {nodeid}: {type(error).__name__}: {error}"
            )
            self._discard(engine_name, engine)

    def close(self) -> None:
        """Shut down all engines still owned by this worker-local pool."""
        for key, engine in tuple(self._engines.items()):
            self._discard(key[0], engine, count_discard=False)

    def timing_lines(self) -> list[str]:
        """Return lifecycle timings for pytest's terminal-summary hook."""
        return [
            (
                f"  {engine_name}: constructed={self._construct_count[engine_name]}, "
                f"reused={self._reuse_count[engine_name]}, "
                f"discarded={self._discard_count[engine_name]}, "
                f"construct={self._construct_seconds[engine_name]:.2f}s, "
                f"reset={self._reset_seconds[engine_name]:.2f}s, "
                f"health={self._health_seconds[engine_name]:.2f}s"
            )
            for engine_name in ("dask", "ray")
        ] + [f"  discard: {reason}" for reason in self._discard_reasons]

    def _construct(self, engine_name: str) -> StreamingEngine:
        if engine_name == "dask":
            from cudf_polars.engine.dask import DaskEngine

            return DaskEngine(engine_options={"allow_gpu_sharing": True})
        if engine_name == "ray":
            from cudf_polars.engine.ray import RayEngine

            return RayEngine(
                num_ranks=self._ray_num_ranks,
                engine_options={"allow_gpu_sharing": True},
                ray_init_options=self._ray_init_options,
            )
        raise ValueError(f"Unknown pooled engine: {engine_name!r}")

    def _key(self, engine_name: str) -> tuple[Any, ...]:
        """Return the immutable construction configuration for a pooled engine."""
        if engine_name == "dask":
            return ("dask", ("allow_gpu_sharing", True), "owned-local-cluster")
        if engine_name == "ray":
            return (
                "ray",
                self._ray_num_ranks,
                ("allow_gpu_sharing", True),
                _freeze(self._ray_init_options),
            )
        raise ValueError(f"Unknown pooled engine: {engine_name!r}")

    def _health_check(self, engine_name: str, engine: StreamingEngine) -> None:
        """Verify that the reset retained one responsive resource per rank."""
        if engine_name == "dask":
            from cudf_polars.engine.dask import DaskEngine

            assert isinstance(engine, DaskEngine)
            if engine._dask_context is None:
                raise RuntimeError("Dask engine was shut down")
            workers = engine._dask_context.client.scheduler_info(n_workers=-1)[
                "workers"
            ]
            if len(workers) != engine.nranks:
                raise RuntimeError("dask engine lost a worker")
        elif engine_name == "ray":
            from cudf_polars.engine.ray import RayEngine

            assert isinstance(engine, RayEngine)
            if engine._rank_actors is None:
                raise RuntimeError("Ray engine was shut down")
            if len(engine._run(os.getpid)) != engine.nranks:
                raise RuntimeError("ray engine lost a worker")
        else:  # pragma: no cover - guarded by acquire
            raise ValueError(f"Unknown pooled engine: {engine_name!r}")

    def _discard(
        self, engine_name: str, engine: StreamingEngine, *, count_discard: bool = True
    ) -> None:
        key = self._key(engine_name)
        if self._engines.get(key) is engine:
            del self._engines[key]
        if count_discard:
            self._discard_count[engine_name] += 1
        with contextlib.suppress(Exception):
            engine.shutdown()


def _freeze(value: Any) -> Any:
    """Make standard engine-construction settings usable as a pool key."""
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
