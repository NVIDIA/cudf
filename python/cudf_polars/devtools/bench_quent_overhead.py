#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure cudf-polars reservation telemetry overhead on a streaming query."""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time

import polars as pl

from rapidsmpf import bootstrap
from rapidsmpf.communicator.single import new_communicator
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.progress_thread import ProgressThread

from cudf_polars.engine.spmd import SPMDEngine
from cudf_polars.quent import QuentConfig


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


def _measure(query: pl.LazyFrame, repeats: int, *, quent_enabled: bool) -> list[float]:
    with tempfile.TemporaryDirectory(prefix="quent-overhead-") as output_root:
        context = QuentConfig(output_root=output_root) if quent_enabled else None
        engine = _engine(context)
        try:
            query.collect(engine=engine)  # warm up
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                query.collect(engine=engine)
                samples.append(time.perf_counter() - start)
            return samples
        finally:
            engine.shutdown()


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
    baseline = _measure(query, args.repeats, quent_enabled=False)
    instrumented = _measure(query, args.repeats, quent_enabled=True)
    baseline_median = statistics.median(baseline)
    instrumented_median = statistics.median(instrumented)
    print(f"baseline median: {baseline_median:.6f}s")
    print(f"Quent median:    {instrumented_median:.6f}s")
    print(f"overhead:        {(instrumented_median / baseline_median - 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
