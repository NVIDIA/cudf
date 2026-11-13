# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate rapidsmpf reservation deltas into aggregate Quent occupancy."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from rapidsmpf.memory.buffer import MemoryType

if TYPE_CHECKING:
    from collections.abc import Mapping

    import cudf_polars_quent as quent_bindings

    from rapidsmpf.memory.buffer_resource import ReservationEventBatch
    from rapidsmpf.memory.reservation_events import ReservationContext

    from cudf_polars.quent._context import ReservationContextRegistry

_MEMORY_TYPES = {int(member): member.name.lower() for member in MemoryType}
_U64_MAX = 2**64 - 1


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
        self._balances: dict[tuple[ReservationContext, int], int] = {}
        self._handles: dict[
            tuple[ReservationContext, int],
            quent_bindings.ReservationOccupancyDynamicFsmHandle,
        ] = {}
        self._diagnostics: list[str] = []
        self._latest_timestamp = 0
        self._closed = False

    def ingest(self, batch: ReservationEventBatch) -> None:
        """Ingest a batch; malformed telemetry remains non-fatal."""
        if self._closed:
            self._diagnostics.append("discarded reservation deltas after shutdown")
            return
        if batch.dropped_event_count:
            self._diagnostics.append(
                f"{batch.dropped_event_count} dropped reservation telemetry event(s)"
            )
        for index, event in enumerate(batch.events):
            try:
                self._ingest_one(
                    event.timestamp_ns,
                    event.reservation_context,
                    event.memory_type,
                    event.byte_delta,
                )
            except Exception as error:
                self._diagnostics.append(
                    f"discarded reservation delta {index}: {error}"
                )

    def _ingest_one(
        self,
        timestamp: int,
        token: ReservationContext,
        memory_type: int,
        delta: int,
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
        self._latest_timestamp = max(self._latest_timestamp, timestamp)

    def diagnostics(self) -> list[str]:
        """Return accumulated non-fatal telemetry diagnostics."""
        return self._diagnostics.copy()

    def close(self) -> None:
        """End all aggregate occupancy traces at worker-runtime shutdown."""
        if self._closed:
            return
        self._closed = True
        import cudf_polars_quent as _quent

        # Quent rejects decreasing timestamps, and ingested timestamps come from a
        # separately anchored rapidsmpf clock.
        timestamp = max(_quent.timestamp(), self._latest_timestamp)
        for key, handle in list(self._handles.items()):
            try:
                handle.released_at(timestamp)
            except Exception as error:
                self._diagnostics.append(
                    f"failed to release reservation occupancy {key}: {error}"
                )
        self._handles.clear()


__all__ = ["ReservationDeltaIngestor"]
