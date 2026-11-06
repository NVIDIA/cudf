# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import ReservationEvent, ReservationEventBatch
from rapidsmpf.memory.reservation_events import (
    ReservationContext,
    ReservationContextCreator,
)

from cudf_polars.quent._context import ReservationContextRegistry
from cudf_polars.quent._reservation_deltas import ReservationDeltaIngestor


def _batch(
    *deltas: tuple[int, ReservationContext, int], dropped_event_count: int = 0
) -> ReservationEventBatch:
    return ReservationEventBatch(
        events=[
            ReservationEvent(
                timestamp_ns=timestamp_ns,
                reservation_context=reservation_context,
                memory_type=MemoryType.DEVICE,
                byte_delta=byte_delta,
            )
            for timestamp_ns, reservation_context, byte_delta in deltas
        ],
        dropped_event_count=dropped_event_count,
    )


def _ingestor() -> tuple[
    ReservationDeltaIngestor, ReservationContextRegistry, MagicMock
]:
    context = MagicMock()
    initial = context.reservation_occupancy_observer.return_value.handle.return_value
    initial.into_dynamic.return_value = MagicMock()
    registry = ReservationContextRegistry()
    ingestor = ReservationDeltaIngestor(
        context,
        uuid.UUID(int=1),
        {"device": uuid.UUID(int=2)},
        registry,
    )
    return ingestor, registry, initial.into_dynamic.return_value


def test_ingests_checked_aggregate_balances_including_zero() -> None:
    ingestor, registry, handle = _ingestor()
    actor = uuid.uuid4()
    token = registry.token(actor, "join.evaluate")

    ingestor.ingest(_batch((100, token, 64), (101, token, -16), (102, token, -48)))

    handle.created_at.assert_called_once_with(
        100, worker=uuid.UUID(int=1), actor=actor, site="join.evaluate"
    )
    assert [
        call.kwargs["memory"]["data"]["bytes"] for call in handle.held_at.call_args_list
    ] == [64, 48, 0]
    ingestor.close()
    handle.released_at.assert_called_once()


def test_release_does_not_precede_latest_ingested_timestamp() -> None:
    ingestor, registry, handle = _ingestor()
    token = registry.token(uuid.uuid4(), "join.evaluate")
    future = 2**62
    ingestor.ingest(_batch((future, token, 1)))

    ingestor.close()

    handle.released_at.assert_called_once_with(future)


def test_unattributed_context_is_valid() -> None:
    ingestor, _, handle = _ingestor()
    ingestor.ingest(_batch((100, ReservationContext(), 1)))
    handle.created_at.assert_called_once_with(
        100, worker=uuid.UUID(int=1), actor=None, site=None
    )
    assert ingestor.diagnostics() == []


def test_diagnostics_are_nonfatal_and_preserve_checked_balance() -> None:
    ingestor, registry, handle = _ingestor()
    token = registry.token(uuid.uuid4(), "join.evaluate")
    unknown = ReservationContextCreator().create()
    batch = _batch(
        (100, unknown, 1),
        (101, token, -1),
        (102, token, 2**63 - 1),
        (103, token, 2**63 - 1),
        (104, token, 2**63 - 1),
        dropped_event_count=3,
    )

    ingestor.ingest(batch)

    diagnostics = "; ".join(ingestor.diagnostics())
    assert "3 dropped reservation telemetry event" in diagnostics
    assert "unknown reservation context token" in diagnostics
    assert "underflow" in diagnostics
    assert "overflow" in diagnostics
    assert handle.held_at.call_count == 2
