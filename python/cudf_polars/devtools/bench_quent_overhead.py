#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure cudf-polars reservation telemetry overhead on a streaming query."""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from typing import TYPE_CHECKING

import polars as pl

from rapidsmpf import bootstrap
from rapidsmpf.communicator.single import new_communicator
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.progress_thread import ProgressThread

from cudf_polars.engine.spmd import SPMDEngine
from cudf_polars.quent import QuentConfig

if TYPE_CHECKING:
    from cudf_polars.quent._runtime import ReservationEventMetrics


def _engine(quent_context: QuentConfig | None) -> SPMDEngine:
    progress = ProgressThread()
    comm = (
        bootstrap.create_ucxx_comm(
            progress_thread=progress, type=bootstrap.BackendType.AUTO
        )
        if bootstrap.is_running_with_rrun()
        else new_communicator(Options(get_environment_variables()), progress)
    )
    return SPMDEngine(
        executor_options={
            "fallback_mode": "silent",
            "quent_context": quent_context,
        },
        comm=comm,
    )


def _measure(
    query: pl.LazyFrame, repeats: int, *, quent_enabled: bool
) -> tuple[list[float], ReservationEventMetrics | None]:
    with tempfile.TemporaryDirectory(prefix="quent-overhead-") as output_root:
        context = QuentConfig(output_root=output_root) if quent_enabled else None
        engine = _engine(context)
        metrics_runtime = None
        try:
            query.collect(engine=engine)  # warm up
            if quent_enabled:
                metrics_runtime = engine._quent_worker_runtime
                assert metrics_runtime is not None
                metrics_runtime.enable_reservation_event_metrics()
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                query.collect(engine=engine)
                samples.append(time.perf_counter() - start)
        finally:
            engine.shutdown()
        metrics = (
            metrics_runtime.reservation_event_metrics()
            if metrics_runtime is not None
            else None
        )
        return samples, metrics


def main() -> None:
    """Run the benchmark and print median end-to-end overhead."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    query = (
        pl.LazyFrame({"key": pl.arange(0, args.rows, eager=True) % 1024})
        .group_by("key")
        .agg(pl.len())
    )
    baseline, _ = _measure(query, args.repeats, quent_enabled=False)
    instrumented, metrics = _measure(query, args.repeats, quent_enabled=True)
    assert metrics is not None
    baseline_median = statistics.median(baseline)
    instrumented_median = statistics.median(instrumented)
    print(f"baseline median: {baseline_median:.6f}s")
    print(f"Quent median:    {instrumented_median:.6f}s")
    print(f"overhead:        {(instrumented_median / baseline_median - 1) * 100:.2f}%")
    print(f"events:          {metrics.event_count}")
    print(f"peak batch:      {metrics.peak_buffered_events} events")
    print(f"peak batch size: {metrics.peak_buffered_bytes} Python bytes")
    print(f"drain conversion:{metrics.drain_conversion_seconds * 1_000:.3f}ms")


if __name__ == "__main__":
    main()
