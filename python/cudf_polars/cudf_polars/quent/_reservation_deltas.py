# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate rapidsmpf reservation deltas into aggregate Quent occupancy."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, cast

from rapidsmpf.memory.buffer import MemoryType

if TYPE_CHECKING:
    from collections.abc import Mapping

    import cudf_polars_quent as quent_bindings

    from cudf_polars.quent._context import ReservationContextRegistry

_MEMORY_TYPES = {
    int(MemoryType.DEVICE): "device",
    int(MemoryType.HOST): "host",
    int(MemoryType.PINNED_HOST): "pinned_host",
    int(MemoryType.DISK): "disk",
}
_U64_MAX = 2**64 - 1


class _ReservationDelta(Protocol):
    timestamp_ns: object
    reservation_context: object
    memory_type: object
    byte_delta: object


class _ReservationBatch(Protocol):
    events: Iterable[object]
    allocation_failure_count: object


class ReservationDeltaIngestor:
    """Maintain checked aggregate balances and emit occupancy transitions."""

    def __init__(
        self,
        context: quent_bindings.Context,
        worker_id: uuid.UUID,
        reservable_memory_ids: Mapping[str, uuid.UUID],
        registry: ReservationContextRegistry,
    ) -> None:
        self._observer = context.reservation_occupancy_observer()
        self._worker_id = worker_id
        self._memory_ids = {
            value: reservable_memory_ids[name]
            for value, name in _MEMORY_TYPES.items()
            if name in reservable_memory_ids
        }
        self._registry = registry
        self._balances: dict[tuple[int, int], int] = {}
        self._handles: dict[
            tuple[int, int], quent_bindings.ReservationOccupancyDynamicFsmHandle
        ] = {}
        self._diagnostics: list[str] = []
        self._closed = False

    def ingest(self, batch: object) -> None:
        """Ingest a fixed-shape batch; malformed telemetry remains non-fatal."""
        if self._closed:
            self._diagnostics.append("discarded reservation deltas after shutdown")
            return
        events, allocation_failures = _batch_parts(batch)
        if allocation_failures:
            self._diagnostics.append(
                f"{allocation_failures} reservation telemetry allocation failure(s)"
            )
        for index, value in enumerate(events):
            try:
                timestamp, token, memory_type, delta = _event_parts(value)
                self._ingest_one(timestamp, token, memory_type, delta)
            except Exception as error:
                self._diagnostics.append(
                    f"discarded reservation delta {index}: {error}"
                )

    def _ingest_one(
        self, timestamp: int, token: int, memory_type: int, delta: int
    ) -> None:
        attribution = self._registry.resolve(token)
        if attribution is None:
            self._diagnostics.append(f"unknown reservation context token {token}")
            return
        memory_id = self._memory_ids.get(memory_type)
        if memory_id is None:
            self._diagnostics.append(f"unknown reservation memory type {memory_type}")
            return
        key = (token, memory_type)
        previous = self._balances.get(key, 0)
        total = previous + delta
        if total < 0:
            self._diagnostics.append(
                f"reservation occupancy underflow for token {token}, "
                f"memory type {memory_type}: {previous} + {delta}"
            )
            return
        if total > _U64_MAX:
            self._diagnostics.append(
                f"reservation occupancy overflow for token {token}, "
                f"memory type {memory_type}: {previous} + {delta}"
            )
            return

        handle = self._handles.get(key)
        if handle is None:
            actor, site = attribution
            identifier = uuid.uuid5(
                self._worker_id, f"reservation-occupancy:{token}:{memory_type}"
            )
            handle = self._observer.handle(identifier).into_dynamic()
            handle.created_at(
                timestamp,
                worker=self._worker_id,
                actor=actor,
                site=site,
            )
            self._handles[key] = handle
        handle.held_at(
            timestamp,
            memory={"target": memory_id, "data": {"bytes": total}},
        )
        self._balances[key] = total

    def diagnostics(self) -> list[str]:
        """Return accumulated non-fatal telemetry diagnostics."""
        return self._diagnostics.copy()

    def close(self) -> None:
        """End all aggregate occupancy traces at worker-runtime shutdown."""
        if self._closed:
            return
        self._closed = True
        timestamp = time.time_ns()
        for key, handle in list(self._handles.items()):
            try:
                handle.released_at(timestamp)
            except Exception as error:
                self._diagnostics.append(
                    f"failed to release reservation occupancy {key}: {error}"
                )
        self._handles.clear()


def _batch_parts(batch: object) -> tuple[Iterable[object], int]:
    if hasattr(batch, "events"):
        typed_batch = cast("_ReservationBatch", batch)
        raw_events: object = typed_batch.events
        failures = typed_batch.allocation_failure_count
    elif (
        isinstance(batch, tuple)
        and len(batch) == 2
        and not isinstance(batch[0], int)
        and isinstance(batch[1], int)
    ):
        raw_events, failures = batch
    else:
        raw_events, failures = batch, 0
    if not isinstance(failures, int) or isinstance(failures, bool) or failures < 0:
        raise ValueError("allocation failure count must be a non-negative integer")
    if not isinstance(raw_events, Iterable):
        raise TypeError("reservation events must be iterable")
    return iter(raw_events), failures


def _event_parts(value: object) -> tuple[int, int, int, int]:
    if isinstance(value, tuple) and len(value) == 4:
        fields = value
    elif all(
        hasattr(value, name)
        for name in ("timestamp_ns", "reservation_context", "memory_type", "byte_delta")
    ):
        event = cast("_ReservationDelta", value)
        fields = (
            event.timestamp_ns,
            event.reservation_context,
            event.memory_type,
            event.byte_delta,
        )
    else:
        raise TypeError("event must be a four-item tuple or reservation delta object")
    timestamp = _integer(fields[0], "timestamp_ns", signed=True)
    token = _integer(fields[1], "reservation_context", signed=False)
    memory_type = _integer(fields[2], "memory_type", signed=False)
    delta = _integer(fields[3], "byte_delta", signed=True)
    return timestamp, token, memory_type, delta


def _integer(value: object, name: str, *, signed: bool) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    minimum = -(2**63) if signed else 0
    maximum = 2**63 - 1 if signed else _U64_MAX
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its 64-bit range")
    return value


__all__ = ["ReservationDeltaIngestor"]
