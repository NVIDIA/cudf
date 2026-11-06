#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Measure reservation-event overhead for many small reservations.

Runs the full telemetry path against a real Quent Collector, without a query:

    BufferResource (C++ recording)
      -> drain_reservation_events()      (Cython bridge)
      -> QuentWorkerRuntime ingest       (cudf-polars Python / Rust ingestor)
      -> Quent Collector                 (flushed on close)

Each ``reserve`` + ``clear`` pair records four events: request submitted,
request granted, reservation created, and reservation released. The script
reports nanoseconds per event for each stage, so it works unchanged on either
the per-event-dict implementation or the fixed-record batch implementation.

Example::

    python devtools/bench_reservation_events.py --lifecycles 100000
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import rmm.mr
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource

from cudf_polars.quent import QuentConfig
from cudf_polars.quent._runtime import QuentControllerRuntime, QuentWorkerRuntime

if TYPE_CHECKING:
    from collections.abc import Mapping

EVENTS_PER_LIFECYCLE = 4
ATTRIBUTES: dict[str, Mapping[str, object] | None] = {
    "none": None,
    "three": {"rapidsmpf.site": "bench", "operator": 7, "active": True},
}


@dataclasses.dataclass
class Result:
    """Wall-clock seconds spent in each stage, and the event count."""

    events: int = 0
    produce_disabled: float = 0.0
    produce_enabled: float = 0.0
    drain: float = 0.0
    # Ingest is split into three stages when the ingestor exposes the Python
    # parse/validate/forward steps; otherwise only `ingest` (the whole call) is set.
    ingest_parse: float = 0.0
    ingest_validate: float = 0.0
    ingest_forward: float = 0.0
    ingest: float = 0.0
    close: float = 0.0
    collector_lines: int = 0


def _new_buffer_resource() -> BufferResource:
    return BufferResource(
        rmm.mr.CudaMemoryResource(),
        memory_limits={MemoryType.DEVICE: 1 << 30},
        periodic_spill_check=None,
    )


def _produce(
    br: BufferResource, lifecycles: int, attributes: Mapping[str, object] | None
) -> float:
    start = time.perf_counter()
    for _ in range(lifecycles):
        reservation, _ = br.reserve(
            MemoryType.DEVICE, 64, allow_overbooking=False, attributes=attributes
        )
        reservation.clear()
    return time.perf_counter() - start


def _ingest_split(worker: QuentWorkerRuntime, events: object, result: Result) -> None:
    """
    Ingest one batch, timing each stage of the Python ingestor separately.

    This mirrors ``ReservationEventIngestor.ingest`` step by step:

    1. parse: ``_parse_event`` turns each event dict into a checked ``_Event``
       (type and range checks on every field, attribute-map checks).
    2. validate: ``_validate`` checks sequence continuity, ID and lifecycle
       rules and enum values against a copied lifecycle.
    3. forward: ``_forward`` + ``_apply`` translate attributes, derive UUIDs
       and call the generated Quent bindings.

    Ingestors without those private steps (e.g. a native one) are timed as a
    single ``ingest`` call.
    """
    ingestor = worker._reservation_ingestor
    parse_event = getattr(
        sys.modules.get(type(ingestor).__module__), "_parse_event", None
    )
    if ingestor is None or parse_event is None or not hasattr(ingestor, "_validate"):
        start = time.perf_counter()
        worker._ingest_reservation_events(events)
        result.ingest += time.perf_counter() - start
        return

    start = time.perf_counter()
    parsed = [parse_event(event, index) for index, event in enumerate(events)]
    result.ingest_parse += time.perf_counter() - start

    start = time.perf_counter()
    next_lifecycle = ingestor._lifecycle.copy()
    for event in parsed:
        ingestor._validate(event, next_lifecycle)
    result.ingest_validate += time.perf_counter() - start

    start = time.perf_counter()
    current = ingestor._lifecycle.copy()
    for event in parsed:
        ingestor._forward(event, current)
        ingestor._apply(event, current)
    ingestor._lifecycle = next_lifecycle
    result.ingest_forward += time.perf_counter() - start


def _count_collector_lines(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as f:
                total += sum(1 for _ in f)
    return total


def run_once(
    lifecycles: int, drain_every: int, attributes: Mapping[str, object] | None
) -> Result:
    """Run the full pipeline once and return per-stage timings."""
    result = Result()
    br = _new_buffer_resource()
    # Baseline: the same producer loop with recording off.
    _produce(br, min(lifecycles, 1_000), attributes)  # warm up
    result.produce_disabled = _produce(br, lifecycles, attributes)

    with tempfile.TemporaryDirectory(prefix="reservation-events-") as output_root:
        config = QuentConfig(output_root=output_root)
        controller = QuentControllerRuntime.create(config, backend="bench")
        worker = QuentWorkerRuntime.create(
            config,
            controller.collector_address,
            worker_id=uuid.uuid4(),
            rank=0,
            nranks=1,
            instance_name="bench",
            buffer_resource=br,
        )
        try:
            remaining = lifecycles
            while remaining > 0:
                chunk = min(remaining, drain_every)
                remaining -= chunk
                result.produce_enabled += _produce(br, chunk, attributes)

                start = time.perf_counter()
                events = br.drain_reservation_events()
                result.drain += time.perf_counter() - start
                result.events += len(events)

                # Equivalent to the runtime's ``drain_reservation_events``, with
                # drain and each ingest stage timed separately.
                _ingest_split(worker, events, result)
        finally:
            start = time.perf_counter()
            worker.close()
            controller.close()
            result.close = time.perf_counter() - start
        result.collector_lines = _count_collector_lines(Path(output_root))
    return result


def _ns_per_event(seconds: float, events: int) -> float:
    return seconds / events * 1e9


def main() -> None:
    """Run the benchmark and print per-stage costs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--lifecycles",
        type=int,
        default=100_000,
        help="reserve+clear pairs per run (4 events each)",
    )
    parser.add_argument(
        "--drain-every",
        type=int,
        default=10_000,
        help="lifecycles recorded between drains",
    )
    parser.add_argument("--attributes", choices=sorted(ATTRIBUTES), default="none")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    attributes = ATTRIBUTES[args.attributes]
    run_once(1_000, 1_000, attributes)  # warm up imports, collector, JIT-ish paths
    results = [
        run_once(args.lifecycles, args.drain_every, attributes)
        for _ in range(args.repeats)
    ]

    events = results[0].events
    stages = {
        "produce, recording off": [r.produce_disabled for r in results],
        "produce, recording on": [r.produce_enabled for r in results],
        "drain (Cython bridge)": [r.drain for r in results],
        "ingest: parse": [r.ingest_parse for r in results],
        "ingest: validate": [r.ingest_validate for r in results],
        "ingest: forward to Quent": [r.ingest_forward for r in results],
        "ingest: unsplit (whole call)": [r.ingest for r in results],
        "close + collector flush": [r.close for r in results],
    }
    # The recording-off loop runs `lifecycles` iterations, so it is per-lifecycle.
    print(
        f"{args.lifecycles} lifecycles = {events} events, "
        f"drain every {args.drain_every} lifecycles, "
        f"attributes={args.attributes}, median of {args.repeats}"
    )
    for name, samples in stages.items():
        median = statistics.median(samples)
        if median == 0.0 and name.startswith("ingest:"):
            continue  # stage not applicable to this ingestor
        per_event = (
            median / args.lifecycles * 1e9
            if name == "produce, recording off"
            else _ns_per_event(median, events)
        )
        unit = "ns/lifecycle" if name == "produce, recording off" else "ns/event"
        print(f"  {name:32s} {median * 1e3:10.1f} ms  {per_event:10.0f} {unit}")
    overhead = statistics.median(
        (r.produce_enabled - r.produce_disabled)
        + r.drain
        + r.ingest
        + r.ingest_parse
        + r.ingest_validate
        + r.ingest_forward
        + r.close
        for r in results
    )
    print(
        f"  {'telemetry total (added cost)':32s} {overhead * 1e3:10.1f} ms  "
        f"{_ns_per_event(overhead, events):10.0f} ns/event"
    )
    lines = results[-1].collector_lines
    print(f"  collector output lines (last run): {lines}")


if __name__ == "__main__":
    main()
