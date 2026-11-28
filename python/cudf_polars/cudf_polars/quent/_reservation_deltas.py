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
        self._completed_tokens: set[ReservationContext] = set()
        self._generation_namespace = uuid.uuid4()
        self._next_generation = 0
        self._dropped_event_count = 0
        self._diagnostics: dict[str, tuple[int, str]] = {}

    def _diagnose(self, category: str, message: str) -> None:
        count, _ = self._diagnostics.get(category, (0, message))
        self._diagnostics[category] = (count + 1, message)

    def ingest(self, batch: ReservationEventBatch) -> None:
        """Ingest a batch; malformed telemetry remains non-fatal."""
        self._dropped_event_count += batch.dropped_event_count
        for index, event in enumerate(batch.events):
            try:
                self._ingest_one(
                    event.timestamp_ns,
                    event.reservation_context,
                    event.memory_type,
                    event.byte_delta,
                )
            except Exception as error:
                self._diagnose(
                    "discarded-delta", f"discarded reservation delta {index}: {error}"
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
            self._diagnose(
                "unknown-context", f"unknown reservation context token {token}"
            )
            return
        memory_id = self._memory_ids.get(memory_type)
        if memory_id is None:
            self._diagnose(
                "unknown-memory-type",
                f"unknown reservation memory type {memory_type}",
            )
            return
        key = (token, memory_type)
        previous = self._balances.get(key, 0)
        total = previous + delta
        if total < 0:
            # Preserve the last valid balance; applying a malformed delta would
            # make every subsequent aggregate transition unreliable.
            self._diagnose(
                "underflow",
                f"reservation occupancy underflow for token {token}, "
                f"memory type {memory_type}: {previous} + {delta}",
            )
            return
        if total > _U64_MAX:
            # As with underflow, keep the last valid balance rather than resync
            # to a value that cannot be represented by the telemetry schema.
            self._diagnose(
                "overflow",
                f"reservation occupancy overflow for token {token}, "
                f"memory type {memory_type}: {previous} + {delta}",
            )
            return

        handle = self._handles.get(key)
        if handle is None:
            actor, site = attribution
            generation = self._next_generation
            self._next_generation += 1
            identifier = uuid.uuid5(
                self._generation_namespace,
                f"reservation-occupancy:{generation}:{memory_type}",
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
        if total == 0 and token in self._completed_tokens:
            self._release(key, timestamp)

    def diagnostics(self) -> list[str]:
        """Return accumulated non-fatal telemetry diagnostics."""
        diagnostics = [
            message if count == 1 else f"{message} ({count} occurrences)"
            for count, message in self._diagnostics.values()
        ]
        if self._dropped_event_count:
            diagnostics.insert(
                0,
                f"{self._dropped_event_count} dropped reservation telemetry event(s)",
            )
        return diagnostics

    @property
    def dropped_event_count(self) -> int:
        """Return the cumulative number of dropped reservation events."""
        return self._dropped_event_count

    def complete_actor(
        self, actor_id: uuid.UUID, tokens: set[ReservationContext]
    ) -> None:
        """Release zero-balance occupancy after an Actor has completed."""
        import cudf_polars_quent

        timestamp = cudf_polars_quent.timestamp()
        del actor_id  # Tokens are already scoped to the completed Actor.
        self._completed_tokens.update(tokens)
        for key in [
            key
            for key, balance in self._balances.items()
            if key[0] in tokens and balance == 0
        ]:
            self._release(key, timestamp)
        for token in tokens:
            self._forget_if_finished(token)

    def _release(self, key: tuple[ReservationContext, int], timestamp: int) -> None:
        token, _ = key
        handle = self._handles.pop(key)
        self._balances.pop(key, None)
        try:
            handle.released_at(timestamp)
        except Exception as error:
            self._diagnose(
                "release-failure",
                f"failed to release reservation occupancy {key}: {error}",
            )
        self._forget_if_finished(token)

    def _forget_if_finished(self, token: ReservationContext) -> None:
        if token in self._completed_tokens and not any(
            key_token == token for key_token, _ in self._balances
        ):
            self._completed_tokens.remove(token)
            self._registry.forget(token)

    def close(self) -> None:
        """End all aggregate occupancy traces at worker-runtime shutdown."""
        import cudf_polars_quent

        timestamp = cudf_polars_quent.timestamp()
        for key, handle in list(self._handles.items()):
            try:
                handle.released_at(timestamp)
            except Exception as error:
                self._diagnose(
                    "release-failure",
                    f"failed to release reservation occupancy {key}: {error}",
                )
        self._handles.clear()
        self._balances.clear()
        for token in self._completed_tokens:
            self._registry.forget(token)
        self._completed_tokens.clear()


__all__ = ["ReservationDeltaIngestor"]
