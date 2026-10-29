# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate buffered rapidsmpf reservation events into generated Quent calls."""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import (
    ReservationReleaseReason,
    ReservationRequestPolicy,
    ReservationResizeReason,
)

if TYPE_CHECKING:
    import cudf_polars_quent as quent_bindings

_MEMORY_TYPES = {
    MemoryType.DEVICE: "device",
    MemoryType.HOST: "host",
    MemoryType.PINNED_HOST: "pinned_host",
    MemoryType.DISK: "disk",
}
_MEMORY_TYPE_VALUES = {name: value for value, name in _MEMORY_TYPES.items()}
_POLICIES = {
    ReservationRequestPolicy.IMMEDIATE: "immediate",
    ReservationRequestPolicy.OVERBOOK: "overbook",
    ReservationRequestPolicy.SPILL: "spill",
    ReservationRequestPolicy.WAIT: "wait",
    ReservationRequestPolicy.WAIT_OR_OVERBOOK: "wait_or_overbook",
    ReservationRequestPolicy.WAIT_OR_FAIL: "wait_or_fail",
}
_RESIZE_REASONS = {
    ReservationResizeReason.CONSUMED: "consumed",
    ReservationResizeReason.SPLIT_OFF: "split_off",
}
_RELEASE_REASONS = {
    ReservationReleaseReason.CLEARED: "cleared",
    ReservationReleaseReason.RELEASED_CONSUMED: "consumed",
    ReservationReleaseReason.RELEASED_SPLIT: "split",
}
_REQUEST_TERMINAL_KINDS = {
    "request_granted",
    "request_empty",
    "request_failed",
    "request_cancelled",
}


@dataclasses.dataclass(frozen=True)
class _Event:
    sequence: int
    timestamp_ns: int
    kind: str
    data: dict[str, Any]


@dataclasses.dataclass
class _Lifecycle:
    next_sequence: int = 0
    requests: set[int] = dataclasses.field(default_factory=set)
    max_request_id: int = 0
    reservations: dict[int, int] = dataclasses.field(default_factory=dict)
    max_reservation_id: int = 0

    def copy(self) -> _Lifecycle:
        """Return an independent lifecycle snapshot."""
        return _Lifecycle(
            next_sequence=self.next_sequence,
            requests=self.requests.copy(),
            max_request_id=self.max_request_id,
            reservations=self.reservations.copy(),
            max_reservation_id=self.max_reservation_id,
        )


class ReservationEventIngestor:
    """Validate and forward buffered reservation events in sequence order."""

    def __init__(
        self,
        context: quent_bindings.Context,
        worker_id: uuid.UUID | str,
        reservable_memory_ids: Mapping[str, uuid.UUID | str],
    ) -> None:
        self._worker_id = _uuid(worker_id, "worker_id")
        self._memory_ids: dict[int, uuid.UUID] = {}
        for name, identifier in reservable_memory_ids.items():
            if name not in _MEMORY_TYPE_VALUES:
                raise ValueError(f"invalid reservable memory type {name!r}")
            memory_type = _MEMORY_TYPE_VALUES[name]
            if memory_type in self._memory_ids:
                raise ValueError(f"duplicate reservable memory type {name!r}")
            self._memory_ids[memory_type] = _uuid(
                identifier, f"reservable_memory_ids[{name!r}]"
            )

        self._request_observer = context.memory_reservation_request_observer()
        self._reservation_observer = context.memory_reservation_observer()
        self._memory_observer = context.reservable_memory_observer()
        self._requests: dict[
            int, quent_bindings.MemoryReservationRequestDynamicFsmHandle
        ] = {}
        self._reservations: dict[
            int, quent_bindings.MemoryReservationDynamicFsmHandle
        ] = {}
        self._lifecycle = _Lifecycle()
        self._session_nonce = 0
        self._diagnostics: list[str] = []
        self._closed = False

    def ingest(self, events: Iterable[Mapping[str, object]]) -> None:
        """Validate one complete batch, then forward it to generated handles."""
        if self._closed:
            raise ValueError("reservation event ingestor is closed")

        parsed = [_parse_event(event, index) for index, event in enumerate(events)]
        next_lifecycle = self._lifecycle.copy()
        for event in parsed:
            self._validate(event, next_lifecycle)

        current = self._lifecycle.copy()
        for event in parsed:
            self._forward(event, current)
            self._apply(event, current)
        self._lifecycle = next_lifecycle

    def begin_recording_session(self) -> None:
        """Reset producer-local IDs and namespace the next recording session."""
        if self._lifecycle.requests or self._lifecycle.reservations:
            self._diagnostics.append(
                "recording session ended with "
                f"{len(self._lifecycle.requests)} live request(s) and "
                f"{len(self._lifecycle.reservations)} live reservation(s)"
            )
        self._session_nonce += 1
        self._lifecycle = _Lifecycle()
        self._requests.clear()
        self._reservations.clear()

    def skip_failed_batch(
        self, events: Iterable[Mapping[str, object]], error: BaseException
    ) -> None:
        """Advance beyond a destructively drained batch that could not be ingested."""
        sequences = [
            value
            for event in events
            if isinstance(event, Mapping)
            and isinstance((value := event.get("sequence")), int)
            and not isinstance(value, bool)
        ]
        if sequences:
            self._lifecycle.next_sequence = max(
                self._lifecycle.next_sequence, max(sequences) + 1
            )
        self._diagnostics.append(f"discarded reservation event batch: {error}")

    def diagnostics(self) -> list[str]:
        """Return non-fatal attribute conversion diagnostics."""
        return self._diagnostics.copy()

    def close(self) -> None:
        """Reject incomplete lifecycles and prevent further ingestion."""
        if self._closed:
            return
        errors = []
        if self._lifecycle.requests:
            errors.append(
                f"{len(self._lifecycle.requests)} live reservation request(s)"
            )
        if self._lifecycle.reservations:
            errors.append(f"{len(self._lifecycle.reservations)} live reservation(s)")
        if errors:
            raise RuntimeError("; ".join(errors))
        self._closed = True

    def _validate(self, event: _Event, lifecycle: _Lifecycle) -> None:
        if event.sequence != lifecycle.next_sequence:
            raise ValueError(
                "reservation event sequence discontinuity: "
                f"expected {lifecycle.next_sequence}, got {event.sequence}"
            )
        if lifecycle.next_sequence == 2**64 - 1:
            raise ValueError("reservation event sequence overflow")
        lifecycle.next_sequence += 1

        data = event.data
        if event.kind == "request_submitted":
            identifier = data["id"]
            if identifier <= lifecycle.max_request_id:
                raise ValueError(f"duplicate request id {identifier}")
            for memory_type in data["memory_types"]:
                _memory_type_name(memory_type)
            _enum_name(_POLICIES, data["policy"], "reservation policy")
        elif event.kind == "request_waiting":
            identifier = data["id"]
            if identifier not in lifecycle.requests:
                raise ValueError(f"waiting for unknown request {identifier}")
        elif event.kind in _REQUEST_TERMINAL_KINDS:
            identifier = data["id"]
            if identifier not in lifecycle.requests:
                raise ValueError(f"resolution for unknown request {identifier}")
            if event.kind == "request_granted":
                _memory_type_name(data["memory_type"])
                reservation_id = data["reservation_id"]
                if reservation_id is not None and not (
                    0 < reservation_id <= lifecycle.max_reservation_id
                ):
                    raise ValueError(
                        f"request {identifier} granted unknown reservation "
                        f"{reservation_id}"
                    )
        elif event.kind == "reservation_created":
            identifier = data["id"]
            if data["size"] == 0:
                raise ValueError(
                    f"reservation {identifier} was created with zero bytes"
                )
            memory_type = data["memory_type"]
            if memory_type not in self._memory_ids:
                raise ValueError(
                    "reservation uses undeclared "
                    f"{_memory_type_description(memory_type)}"
                )
            if identifier <= lifecycle.max_reservation_id:
                raise ValueError(f"duplicate reservation id {identifier}")
            request_id = data["request_id"]
            if request_id and request_id not in lifecycle.requests:
                raise ValueError(
                    f"reservation {identifier} references unknown request {request_id}"
                )
            parent_id = data["parent_id"]
            if parent_id and parent_id > lifecycle.max_reservation_id:
                raise ValueError(
                    f"reservation {identifier} references unknown parent {parent_id}"
                )
        elif event.kind == "reservation_resized":
            identifier = data["id"]
            if data["size"] == 0:
                raise ValueError(f"reservation {identifier} resized to zero bytes")
            _enum_name(_RESIZE_REASONS, data["reason"], "reservation resize reason")
            if identifier not in lifecycle.reservations:
                raise ValueError(f"resize for unknown reservation {identifier}")
        elif event.kind == "reservation_released":
            identifier = data["id"]
            _enum_name(_RELEASE_REASONS, data["reason"], "reservation release reason")
            if identifier not in lifecycle.reservations:
                raise ValueError(f"release for unknown reservation {identifier}")
        else:
            memory_type = data["memory_type"]
            if memory_type not in self._memory_ids:
                raise ValueError(
                    "memory limit changed for undeclared "
                    f"{_memory_type_description(memory_type)}"
                )
        self._apply(event, lifecycle)

    @staticmethod
    def _apply(event: _Event, lifecycle: _Lifecycle) -> None:
        data = event.data
        if event.kind == "request_submitted":
            identifier = data["id"]
            lifecycle.max_request_id = identifier
            lifecycle.requests.add(identifier)
        elif event.kind in _REQUEST_TERMINAL_KINDS:
            lifecycle.requests.remove(data["id"])
        elif event.kind == "reservation_created":
            identifier = data["id"]
            lifecycle.max_reservation_id = identifier
            lifecycle.reservations[identifier] = data["memory_type"]
        elif event.kind == "reservation_released":
            del lifecycle.reservations[data["id"]]

    def _forward(self, event: _Event, lifecycle: _Lifecycle) -> None:
        data = event.data
        timestamp = event.timestamp_ns
        sequence = event.sequence
        if event.kind == "request_submitted":
            site, actor, attributes = self._translate_attributes(data["attributes"])
            request_handle = self._request_observer.handle(
                _stable_id(self._worker_id, "request", self._session_nonce, data["id"])
            ).into_dynamic()
            request_handle.submitted_at(
                timestamp,
                worker=self._worker_id,
                actor=actor,
                site=site,
                size=data["size"],
                memory_types=[
                    _memory_type_name(value) for value in data["memory_types"]
                ],
                policy=_enum_name(_POLICIES, data["policy"], "reservation policy"),
                net_memory_delta=data["net_memory_delta"],
                attributes=attributes,
                sequence=sequence,
            )
            self._requests[data["id"]] = request_handle
        elif event.kind == "request_waiting":
            self._requests[data["id"]].waiting_at(timestamp, sequence=sequence)
        elif event.kind in _REQUEST_TERMINAL_KINDS:
            request_handle = self._requests[data["id"]]
            if event.kind == "request_granted":
                request_handle.granted_at(
                    timestamp,
                    reservation=(
                        self._reservation_id(data["reservation_id"])
                        if data["reservation_id"] is not None
                        else None
                    ),
                    memory_type=_memory_type_name(data["memory_type"]),
                    size=data["size"],
                    overbooked_bytes=data["overbooked_bytes"],
                    timed_out=data["timed_out"],
                    sequence=sequence,
                )
            elif event.kind == "request_empty":
                request_handle.empty_at(
                    timestamp, timed_out=data["timed_out"], sequence=sequence
                )
            elif event.kind == "request_failed":
                request_handle.failed_at(
                    timestamp, error=data["error"], sequence=sequence
                )
            else:
                request_handle.cancelled_at(timestamp, sequence=sequence)
            del self._requests[data["id"]]
        elif event.kind == "reservation_created":
            site, actor, attributes = self._translate_attributes(data["attributes"])
            identifier = data["id"]
            reservation_handle = self._reservation_observer.handle(
                self._reservation_id(identifier)
            ).into_dynamic()
            reservation_handle.created_at(
                timestamp,
                worker=self._worker_id,
                actor=actor,
                site=site,
                request=(
                    self._request_id(data["request_id"]) if data["request_id"] else None
                ),
                parent=(
                    self._reservation_id(data["parent_id"])
                    if data["parent_id"]
                    else None
                ),
                attributes=attributes,
                sequence=sequence,
            )
            reservation_handle.held_at(
                timestamp,
                memory=self._memory_usage(data["memory_type"], data["size"]),
                change="granted" if data["parent_id"] == 0 else "split_child",
                sequence=sequence,
            )
            self._reservations[identifier] = reservation_handle
        elif event.kind == "reservation_resized":
            identifier = data["id"]
            memory_type = lifecycle.reservations[identifier]
            self._reservations[identifier].held_at(
                timestamp,
                memory=self._memory_usage(memory_type, data["size"]),
                change=_enum_name(
                    _RESIZE_REASONS,
                    data["reason"],
                    "reservation resize reason",
                ),
                sequence=sequence,
            )
        elif event.kind == "reservation_released":
            identifier = data["id"]
            self._reservations[identifier].released_at(
                timestamp,
                reason=_enum_name(
                    _RELEASE_REASONS,
                    data["reason"],
                    "reservation release reason",
                ),
                sequence=sequence,
            )
            del self._reservations[identifier]
        else:
            memory_type = data["memory_type"]
            self._memory_observer.handle(self._memory_ids[memory_type]).resized_at(
                timestamp,
                limits={"bytes": max(data["configured_limit"], 0)},
                configured_limit=data["configured_limit"],
                sequence=sequence,
            )

    def _request_id(self, identifier: int) -> uuid.UUID:
        return _stable_id(self._worker_id, "request", self._session_nonce, identifier)

    def _reservation_id(self, identifier: int) -> uuid.UUID:
        return _stable_id(
            self._worker_id, "reservation", self._session_nonce, identifier
        )

    def _memory_usage(
        self, memory_type: int, size: int
    ) -> quent_bindings.ReservableMemoryUsageRefDict:
        return {
            "target": self._memory_ids[memory_type],
            "data": {"bytes": size},
        }

    def _translate_attributes(
        self, values: Mapping[str, bool | float | int | str]
    ) -> tuple[str, uuid.UUID | None, dict[str, bool | float | int | str]]:
        site: str = "unknown"
        actor: uuid.UUID | None = None
        dynamic: dict[str, bool | float | int | str] = {}
        for key, value in values.items():
            if key == "rapidsmpf.site":
                if isinstance(value, str):
                    site = value
                else:
                    self._diagnostics.append("rapidsmpf.site is not a string")
            elif key == "quent.actor_id":
                if isinstance(value, str):
                    try:
                        actor = uuid.UUID(value)
                    except ValueError:
                        self._diagnostics.append(f"invalid actor UUID: {value}")
                        dynamic["invalid_actor_id"] = value
                else:
                    self._diagnostics.append("quent.actor_id is not a string")
                    dynamic["invalid_actor_id"] = value
            else:
                dynamic[key] = value
        return site, actor, dynamic


def _parse_event(event: Mapping[str, object], index: int) -> _Event:
    if not isinstance(event, Mapping):
        raise ValueError(f"event {index} is not a mapping")  # noqa: TRY004
    try:
        sequence = _integer(event["sequence"], "sequence", 64, signed=False)
        timestamp = _integer(
            event["steady_time_ns"], "steady_time_ns", 64, signed=False
        )
        kind = _string(event["kind"], "kind")
        data = _parse_data(kind, event)
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, KeyError):
            error = ValueError(f"missing field {error.args[0]!r}")
        raise ValueError(f"event {index}: {error}") from error
    return _Event(sequence, timestamp, kind, data)


def _parse_data(kind: str, event: Mapping[str, object]) -> dict[str, Any]:
    def u64(name: str) -> int:
        return _integer(event[name], name, 64, signed=False)

    def u32(name: str) -> int:
        return _integer(event[name], name, 32, signed=False)

    if kind == "request_submitted":
        memory_types = event["memory_types"]
        if isinstance(memory_types, (str, bytes)) or not isinstance(
            memory_types, Iterable
        ):
            raise TypeError("memory_types must be iterable")
        return {
            "id": u64("id"),
            "size": u64("size"),
            "memory_types": [
                _integer(value, "memory_types item", 32, signed=False)
                for value in memory_types
            ],
            "policy": u32("policy"),
            "net_memory_delta": _optional_i64(
                event["net_memory_delta"], "net_memory_delta"
            ),
            "attributes": _attributes(event["attributes"]),
        }
    if kind == "request_waiting":
        return {"id": u64("id")}
    if kind == "request_granted":
        timed_out = event["timed_out"]
        if not isinstance(timed_out, bool):
            raise TypeError("timed_out must be a bool")
        return {
            "id": u64("id"),
            "reservation_id": _optional_u64(event["reservation_id"], "reservation_id"),
            "memory_type": u32("memory_type"),
            "size": u64("size"),
            "overbooked_bytes": u64("overbooked_bytes"),
            "timed_out": timed_out,
        }
    if kind == "request_empty":
        timed_out = event["timed_out"]
        if not isinstance(timed_out, bool):
            raise TypeError("timed_out must be a bool")
        return {"id": u64("id"), "timed_out": timed_out}
    if kind == "request_failed":
        return {"id": u64("id"), "error": _string(event["error"], "error")}
    if kind == "request_cancelled":
        return {"id": u64("id")}
    if kind == "reservation_created":
        return {
            "id": u64("id"),
            "memory_type": u32("memory_type"),
            "size": u64("size"),
            "attributes": _attributes(event["attributes"]),
            "request_id": _optional_u64(event["request_id"], "request_id"),
            "parent_id": _optional_u64(event["parent_id"], "parent_id"),
        }
    if kind == "reservation_resized":
        return {"id": u64("id"), "size": u64("size"), "reason": u32("reason")}
    if kind == "reservation_released":
        return {"id": u64("id"), "reason": u32("reason")}
    if kind == "memory_limit_changed":
        return {
            "memory_type": u32("memory_type"),
            "configured_limit": _integer(
                event["configured_limit"], "configured_limit", 64, signed=True
            ),
        }
    raise ValueError(f"unknown event kind {kind!r}")


def _attributes(value: object) -> dict[str, bool | float | int | str]:
    if not isinstance(value, Mapping):
        raise TypeError("attributes is not a mapping")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("attribute keys must be strings")
        if isinstance(item, (bool, str, float)):
            result[key] = item
        elif isinstance(item, int):
            result[key] = _integer(item, f"attribute {key!r}", 64, signed=True)
        else:
            raise TypeError(f"attribute {key!r} must be str, int, bool, or float")
    return result


def _integer(value: object, field: str, bits: int, *, signed: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    minimum = -(2 ** (bits - 1)) if signed else 0
    maximum = 2 ** (bits - int(signed)) - 1
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} does not fit in {'i' if signed else 'u'}{bits}")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _optional_i64(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field, 64, signed=True)


def _optional_u64(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field, 64, signed=False)


def _uuid(value: uuid.UUID | str, field: str) -> uuid.UUID:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"{field}: {error}") from error


def _stable_id(
    worker_id: uuid.UUID, namespace: str, session_nonce: int, identifier: int
) -> uuid.UUID:
    return uuid.uuid5(worker_id, f"{namespace}:{session_nonce}:{identifier}")


def _memory_type_name(value: int) -> str:
    return _enum_name(_MEMORY_TYPES, value, "memory type")


def _memory_type_description(value: int) -> str:
    return _MEMORY_TYPES.get(value, f"memory type {value}")


def _enum_name(values: Mapping[int, str], value: int, label: str) -> str:
    try:
        return values[value]
    except KeyError:
        raise ValueError(f"invalid {label} {value}") from None


__all__ = ["ReservationEventIngestor"]
