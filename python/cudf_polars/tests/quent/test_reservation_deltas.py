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
from cudf_polars.quent._reservation_sites import ReservationSite


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
    ReservationDeltaIngestor, ReservationContextRegistry, MagicMock, MagicMock
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
    return ingestor, registry, initial.into_dynamic.return_value, context


def test_ingests_checked_aggregate_balances_including_zero() -> None:
    ingestor, registry, handle, _ = _ingestor()
    actor = uuid.uuid4()
    token = registry.token(actor, ReservationSite.EVALUATE_MAIN)

    ingestor.ingest(_batch((100, token, 64), (101, token, -16), (102, token, -48)))

    handle.created_at.assert_called_once_with(
        100, worker=uuid.UUID(int=1), actor=actor, site="evaluate.main"
    )
    assert [
        call.kwargs["memory"]["data"]["bytes"] for call in handle.held_at.call_args_list
    ] == [64, 48, 0]
    ingestor.close()
    handle.released_at.assert_called_once()


def test_unattributed_context_is_valid() -> None:
    ingestor, _, handle, _ = _ingestor()
    ingestor.ingest(_batch((100, ReservationContext(), 1)))
    handle.created_at.assert_called_once_with(
        100, worker=uuid.UUID(int=1), actor=None, site=None
    )
    assert ingestor.diagnostics() == []


def test_diagnostics_are_nonfatal_and_preserve_checked_balance() -> None:
    ingestor, registry, handle, _ = _ingestor()
    token = registry.token(uuid.uuid4(), ReservationSite.EVALUATE_MAIN)
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


def test_completed_actor_releases_at_zero_and_prunes_registry() -> None:
    ingestor, registry, handle, context = _ingestor()
    actor = uuid.uuid4()
    token = registry.token(actor, ReservationSite.EVALUATE_MAIN)
    ingestor.ingest(_batch((100, token, 8), (101, token, -8)))

    ingestor.complete_actor(actor, registry.prune_actor(actor))

    handle.released_at.assert_called_once()
    assert registry.resolve(token) is None

    recreated = registry.token(actor, ReservationSite.EVALUATE_MAIN)
    assert recreated != token
    ingestor.ingest(_batch((102, recreated, 1)))
    identifiers = [
        call.args[0]
        for call in context.reservation_occupancy_observer.return_value.handle.call_args_list
    ]
    assert len(identifiers) == 2
    assert identifiers[0] != identifiers[1]


def test_completed_actor_releases_when_delayed_delta_reaches_zero() -> None:
    ingestor, registry, handle, _ = _ingestor()
    actor = uuid.uuid4()
    token = registry.token(actor, ReservationSite.EVALUATE_MAIN)
    ingestor.ingest(_batch((100, token, 8)))

    ingestor.complete_actor(actor, registry.prune_actor(actor))

    handle.released_at.assert_not_called()
    assert registry.resolve(token) == (actor, "evaluate.main")

    ingestor.ingest(_batch((101, token, -8)))

    handle.released_at.assert_called_once_with(101)
    assert registry.resolve(token) is None


def test_runtime_recreation_does_not_reuse_occupancy_identifier() -> None:
    identifiers = []
    for _ in range(2):
        ingestor, registry, _, context = _ingestor()
        token = registry.token(uuid.uuid4(), ReservationSite.EVALUATE_MAIN)
        ingestor.ingest(_batch((100, token, 1)))
        identifiers.append(
            context.reservation_occupancy_observer.return_value.handle.call_args.args[0]
        )

    assert identifiers[0] != identifiers[1]


def test_diagnostics_are_bounded_by_category() -> None:
    ingestor, _, _, _ = _ingestor()
    creator = ReservationContextCreator()
    ingestor.ingest(_batch(*((index, creator.create(), 1) for index in range(100))))

    diagnostics = ingestor.diagnostics()
    assert len(diagnostics) == 1
    assert "100 occurrences" in diagnostics[0]
