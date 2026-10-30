# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import (
    ReservationReleaseReason,
    ReservationRequestPolicy,
    ReservationResizeReason,
)

from cudf_polars.quent._reservation_events import ReservationEventIngestor


@pytest.fixture
def bindings() -> tuple[ReservationEventIngestor, MagicMock, MagicMock, MagicMock]:
    context = MagicMock()
    request_observer = context.memory_reservation_request_observer.return_value
    reservation_observer = context.memory_reservation_observer.return_value
    memory_observer = context.reservable_memory_observer.return_value
    _record_dynamic_handles(request_observer)
    _record_dynamic_handles(reservation_observer)
    ingestor = ReservationEventIngestor(
        context,
        uuid.UUID(int=1),
        {"device": uuid.UUID(int=2)},
    )
    return ingestor, request_observer, reservation_observer, memory_observer


def _initial_handle() -> MagicMock:
    handle = MagicMock()
    handle.into_dynamic.return_value = MagicMock()
    return handle


def _record_dynamic_handles(observer: MagicMock) -> None:
    observer.dynamic_handles = []

    def create(_: uuid.UUID) -> MagicMock:
        initial = _initial_handle()
        observer.dynamic_handles.append(initial.into_dynamic.return_value)
        return initial

    observer.handle.side_effect = create


def _submitted(sequence: int = 0, identifier: int = 1) -> dict[str, object]:
    return {
        "sequence": sequence,
        "steady_time_ns": 1_000 + sequence,
        "kind": "request_submitted",
        "id": identifier,
        "size": 64,
        "memory_types": [MemoryType.DEVICE],
        "policy": ReservationRequestPolicy.WAIT,
        "net_memory_delta": None,
        "attributes": {},
    }


def _terminal(
    sequence: int,
    identifier: int = 1,
    kind: str = "request_failed",
) -> dict[str, object]:
    event: dict[str, object] = {
        "sequence": sequence,
        "steady_time_ns": 1_000 + sequence,
        "kind": kind,
        "id": identifier,
    }
    if kind == "request_granted":
        event.update(
            reservation_id=None,
            memory_type=MemoryType.DEVICE,
            size=0,
            overbooked_bytes=0,
            timed_out=False,
        )
    elif kind == "request_empty":
        event["timed_out"] = True
    elif kind == "request_failed":
        event["error"] = "failure"
    return event


def _created(
    sequence: int,
    identifier: int,
    *,
    request_id: int = 0,
    parent_id: int = 0,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "steady_time_ns": 1_000 + sequence,
        "kind": "reservation_created",
        "id": identifier,
        "memory_type": MemoryType.DEVICE,
        "size": 64,
        "attributes": {},
        "request_id": request_id or None,
        "parent_id": parent_id or None,
    }


def _released(sequence: int, identifier: int) -> dict[str, object]:
    return {
        "sequence": sequence,
        "steady_time_ns": 1_000 + sequence,
        "kind": "reservation_released",
        "id": identifier,
        "reason": ReservationReleaseReason.CLEARED,
    }


def test_requires_strict_sequence_continuity_across_batches(
    bindings: tuple[ReservationEventIngestor, MagicMock, MagicMock, MagicMock],
) -> None:
    ingestor = bindings[0]
    ingestor.ingest([_submitted()])

    for invalid in (0, 2):
        with pytest.raises(ValueError, match=f"expected 1, got {invalid}"):
            ingestor.ingest(
                [
                    {
                        "sequence": invalid,
                        "steady_time_ns": 2_000,
                        "kind": "request_waiting",
                        "id": 1,
                    }
                ]
            )


@pytest.mark.parametrize(
    "kind",
    [
        "request_empty",
        "request_failed",
        "request_cancelled",
    ],
)
def test_forwards_terminal_request_outcomes_at_source_timestamps(
    bindings: tuple[ReservationEventIngestor, MagicMock, MagicMock, MagicMock],
    kind: str,
) -> None:
    ingestor, request_observer, _, _ = bindings
    ingestor.ingest([_submitted()])
    handle = request_observer.dynamic_handles[0]

    ingestor.ingest([_terminal(1, kind=kind)])

    method = {
        "request_empty": handle.empty_at,
        "request_failed": handle.failed_at,
        "request_cancelled": handle.cancelled_at,
    }[kind]
    assert method.call_args.args[0] == 1_001
    ingestor.close()


def test_forwards_grant_split_resize_release_and_limit_change(
    bindings: tuple[ReservationEventIngestor, MagicMock, MagicMock, MagicMock],
) -> None:
    ingestor, request_observer, reservation_observer, memory_observer = bindings
    ingestor.ingest(
        [
            _submitted(),
            _created(1, 10, request_id=1),
            {
                **_terminal(2, kind="request_granted"),
                "reservation_id": 10,
                "size": 64,
                "overbooked_bytes": 4,
            },
            _released(3, 10),
            _created(4, 11, parent_id=10),
            {
                "sequence": 5,
                "steady_time_ns": 1_005,
                "kind": "reservation_resized",
                "id": 11,
                "size": 32,
                "reason": ReservationResizeReason.SPLIT_OFF,
            },
            _released(6, 11),
            {
                "sequence": 7,
                "steady_time_ns": 1_007,
                "kind": "memory_limit_changed",
                "memory_type": MemoryType.DEVICE,
                "configured_limit": -1,
            },
        ]
    )

    request = request_observer.dynamic_handles[0]
    request.granted_at.assert_called_once()
    assert reservation_observer.handle.call_count == 2
    first_reservation, second_reservation = reservation_observer.dynamic_handles
    assert first_reservation.held_at.call_args.args[0] == 1_001
    assert second_reservation.held_at.call_args_list[0].kwargs["change"] == (
        "split_child"
    )
    assert second_reservation.held_at.call_args_list[1].kwargs["change"] == (
        "split_off"
    )
    second = reservation_observer.handle.call_args_list[1]
    assert second.args[0] != reservation_observer.handle.call_args_list[0].args[0]
    memory_observer.handle.return_value.resized_at.assert_called_once_with(
        1_007,
        limits={"bytes": 0},
        configured_limit=-1,
        sequence=7,
    )
    ingestor.close()


def test_attribute_translation_and_worker_scoped_ids() -> None:
    context = MagicMock()
    request_observer = context.memory_reservation_request_observer.return_value
    _record_dynamic_handles(request_observer)
    actor_id = uuid.uuid4()
    worker_a = uuid.UUID(int=1)
    worker_b = uuid.UUID(int=2)
    ingestor_a = ReservationEventIngestor(
        context, worker_a, {"device": uuid.UUID(int=3)}
    )
    ingestor_b = ReservationEventIngestor(
        context, worker_b, {"device": uuid.UUID(int=4)}
    )
    event = _submitted()
    event["attributes"] = {
        "rapidsmpf.site": "test.site",
        "quent.actor_id": str(actor_id),
        "custom": 7,
    }

    ingestor_a.ingest([event])
    ingestor_b.ingest([event])

    first_id, second_id = [
        item.args[0] for item in request_observer.handle.call_args_list
    ]
    assert first_id != second_id
    for handle in request_observer.dynamic_handles:
        handle.submitted_at.assert_called_once()
        assert handle.submitted_at.call_args.kwargs["actor"] == actor_id
        assert handle.submitted_at.call_args.kwargs["site"] == "test.site"
        assert handle.submitted_at.call_args.kwargs["attributes"] == {"custom": 7}
    assert ingestor_a.diagnostics() == []


def test_malformed_batch_is_atomic(
    bindings: tuple[ReservationEventIngestor, MagicMock, MagicMock, MagicMock],
) -> None:
    ingestor, request_observer, _, _ = bindings
    with pytest.raises(ValueError, match="waiting for unknown request 99"):
        ingestor.ingest(
            [
                _submitted(),
                {
                    "sequence": 1,
                    "steady_time_ns": 1_001,
                    "kind": "request_waiting",
                    "id": 99,
                },
            ]
        )
    request_observer.handle.assert_not_called()
    ingestor.ingest([_submitted()])


def test_failed_batch_can_be_skipped(
    bindings: tuple[ReservationEventIngestor, MagicMock, MagicMock, MagicMock],
) -> None:
    ingestor = bindings[0]
    bad_batch = [
        _submitted(),
        {
            "sequence": 1,
            "steady_time_ns": 1_001,
            "kind": "request_waiting",
            "id": 99,
        },
    ]
    with pytest.raises(ValueError, match="waiting for unknown request 99") as caught:
        ingestor.ingest(bad_batch)

    ingestor.skip_failed_batch(bad_batch, caught.value)
    ingestor.ingest([_submitted(sequence=2, identifier=2), _terminal(3, identifier=2)])
    assert "discarded reservation event batch" in ingestor.diagnostics()[0]
    ingestor.close()


def test_recording_sessions_namespace_reused_ids(
    bindings: tuple[ReservationEventIngestor, MagicMock, MagicMock, MagicMock],
) -> None:
    ingestor, request_observer, _, _ = bindings
    ingestor.ingest([_submitted(), _terminal(1)])
    first_id = request_observer.handle.call_args.args[0]

    ingestor.begin_recording_session()
    ingestor.ingest([_submitted(), _terminal(1)])
    second_id = request_observer.handle.call_args.args[0]

    assert first_id != second_id
    ingestor.close()


def test_lifecycle_history_is_bounded(
    bindings: tuple[ReservationEventIngestor, MagicMock, MagicMock, MagicMock],
) -> None:
    ingestor = bindings[0]
    for identifier in range(1, 101):
        sequence = (identifier - 1) * 2
        ingestor.ingest(
            [
                _submitted(sequence, identifier),
                _terminal(sequence + 1, identifier),
            ]
        )

    assert ingestor._lifecycle.requests == set()
    assert ingestor._lifecycle.reservations == {}
    assert ingestor._lifecycle.max_request_id == 100
    ingestor.close()


def test_close_rejects_incomplete_and_ingest_after_close(
    bindings: tuple[ReservationEventIngestor, MagicMock, MagicMock, MagicMock],
) -> None:
    ingestor = bindings[0]
    ingestor.ingest([_submitted()])
    with pytest.raises(RuntimeError, match="1 live reservation request"):
        ingestor.close()
    ingestor.ingest([_terminal(1)])
    ingestor.close()
    with pytest.raises(ValueError, match="ingestor is closed"):
        ingestor.ingest([])


def test_reports_invalid_reserved_attributes(
    bindings: tuple[ReservationEventIngestor, MagicMock, MagicMock, MagicMock],
) -> None:
    ingestor = bindings[0]
    event = _submitted()
    event["attributes"] = {
        "rapidsmpf.site": 1,
        "quent.actor_id": "not-a-uuid",
    }
    ingestor.ingest([event])
    assert ingestor.diagnostics() == [
        "rapidsmpf.site is not a string",
        "invalid actor UUID: not-a-uuid",
    ]
