# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
import uuid
from unittest.mock import MagicMock

from rapidsmpf.memory.buffer import MemoryType

from cudf_polars.quent._context import ReservationContextRegistry
from cudf_polars.quent._reservation_deltas import ReservationDeltaIngestor


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

    ingestor.ingest(
        [
            (100, token, MemoryType.DEVICE, 64),
            (101, token, MemoryType.DEVICE, -16),
            (102, token, MemoryType.DEVICE, -48),
        ]
    )

    handle.created_at.assert_called_once_with(
        100, worker=uuid.UUID(int=1), actor=actor, site="join.evaluate"
    )
    assert [
        call.kwargs["memory"]["data"]["bytes"] for call in handle.held_at.call_args_list
    ] == [64, 48, 0]
    ingestor.close()
    handle.released_at.assert_called_once()


def test_unattributed_context_zero_is_valid() -> None:
    ingestor, _, handle = _ingestor()
    ingestor.ingest([(100, 0, MemoryType.DEVICE, 1)])
    handle.created_at.assert_called_once_with(
        100, worker=uuid.UUID(int=1), actor=None, site=None
    )
    assert ingestor.diagnostics() == []


def test_diagnostics_are_nonfatal_and_preserve_checked_balance() -> None:
    ingestor, registry, handle = _ingestor()
    token = registry.token(uuid.uuid4(), "join.evaluate")
    batch = types.SimpleNamespace(
        events=[
            (100, 999, MemoryType.DEVICE, 1),
            (101, token, MemoryType.DEVICE, -1),
            (102, token, MemoryType.DEVICE, 2**63 - 1),
            (103, token, MemoryType.DEVICE, 2**63 - 1),
            (104, token, MemoryType.DEVICE, 2**63 - 1),
            ("bad", token, MemoryType.DEVICE, 1),
        ],
        allocation_failure_count=3,
    )

    ingestor.ingest(batch)

    diagnostics = "; ".join(ingestor.diagnostics())
    assert "3 reservation telemetry allocation failure" in diagnostics
    assert "unknown reservation context token 999" in diagnostics
    assert "underflow" in diagnostics
    assert "overflow" in diagnostics
    assert "timestamp_ns must be an integer" in diagnostics
    assert handle.held_at.call_count == 2
