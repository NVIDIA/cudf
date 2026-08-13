# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

# TODO: https://github.com/rapidsai/cudf/issues/22949 use the bindings from quent

from __future__ import annotations

import dataclasses
import enum
import sys
import time
import uuid
from typing import Any, TypeAlias

from cudf_polars import __version__

QUENT_SCOPE = "QUENT"


class EventName(enum.Enum):
    """Quent event names."""

    ENGINE = "Engine"
    WORKER = "Worker"
    QUERY_GROUP = "QueryGroup"
    QUERY = "Query"
    PLAN = "Plan"
    OPERATOR = "Operator"
    PORT = "Port"
    TASK = "Task"


if sys.version_info >= (3, 14):  # pragma: no cover; requires Python 3.14+
    new_quent_id = uuid.uuid7
else:  # pragma: no cover; requires Python 3.13 or earlier
    new_quent_id = uuid.uuid4


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
    """Quent event envelope: id + timestamp + data payload."""

    id: uuid.UUID
    timestamp: int
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp,
            "data": self.data,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Implementation:
    """Engine implementation metadata."""

    name: str = "cudf-polars"
    version: str = __version__
    custom_attributes: list[Attribute] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "name": self.name,
            "version": self.version,
            "custom_attributes": [attr.serialize() for attr in self.custom_attributes],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Operator:
    """
    A Quent Operator.

    Parameters
    ----------
    parent_operators: list[Operator]
        The operators that are the parents of this operator.
        Note that these are *not* related to the children from cudf-polars' IR.
        Instead, this expresses some kind of lowering relationship (i.e. this node
        was lowered from the given operators).

    Examples
    --------
    {"id":"019dd571-1062-77c2-9803-62c7c1941381","timestamp":1777402450018384340,"data":{"Operator":{"Declaration":{"plan_id":"019dd571-1062-77c2-9803-642b6c301d29","parent_operator_ids":[],"instance_name":"Scan-NodeIndex(0)","type_name":"Scan","custom_attributes":[]}}}}
    """

    id: uuid.UUID
    plan: Plan
    parent_operators: list[Operator]
    instance_name: str
    type_name: str
    custom_attributes: list[Attribute] = dataclasses.field(default_factory=list)

    @property
    def plan_id(self) -> uuid.UUID:
        """Compatibility accessor for the operator's plan UUID."""
        return self.plan.id

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "id": str(self.id),
            "plan_id": str(self.plan.id),
            "parent_operator_ids": [
                str(operator.id) for operator in self.parent_operators
            ],
            "instance_name": self.instance_name,
            "type_name": self.type_name,
            "custom_attributes": [attr.serialize() for attr in self.custom_attributes],
        }

    def declare(self, timestamp: int | None = None) -> Event:
        """Declare a Quent Operator."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.OPERATOR.value: {"Declaration": self.to_dict()}},
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Engine:
    """A Quent Engine."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    implementation: Implementation = dataclasses.field(default_factory=Implementation)

    def _init(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent engine init event.

        Examples
        --------
        {"id":"019dd571-105a-7c53-a15b-713cbdd7666b","timestamp":1777402450018164995,"data":{"Engine":{"Init":{"implementation":{"name":"Simulator","version":"0.0.0-PoC","custom_attributes":[]},"instance_name":"holodeck-9dfbdcf7"}}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.ENGINE.value: {
                    "Init": {
                        "implementation": self.implementation.to_dict(),
                        "instance_name": f"cudf-polars-{str(self.id)[:8]}",
                    }
                }
            },
        )

    def _exit(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent engine exit event.

        Examples
        --------
        {"id":"019dd571-105a-7c53-a15b-713cbdd7666b","timestamp":1777402451406253343,"data":{"Engine":{"Exit":null}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.ENGINE.value: {"Exit": None}},
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Worker:
    """A Quent Worker."""

    id: uuid.UUID
    engine: Engine
    instance_name: str

    @property
    def engine_id(self) -> uuid.UUID:
        """Compatibility accessor for the worker's parent engine UUID."""
        return self.engine.id

    def _init(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent worker init event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-6179ddb14b3d","timestamp":1777402450018191773,"data":{"Worker":{"Init":{"parent_engine_id":"019dd571-105a-7c53-a15b-713cbdd7666b","instance_name":"drone-0"}}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.WORKER.value: {
                    "Init": {
                        "parent_engine_id": str(self.engine_id),
                        "instance_name": self.instance_name,
                    }
                }
            },
        )

    def _exit(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent worker exit event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-618b9db790c2","timestamp":1777402451406250693,"data":{"Worker":{"Exit":null}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.WORKER.value: {"Exit": None}},
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """
    A Quent Plan.

    Examples
    --------
    {"id":"019dd571-1062-77c2-9803-642b6c301d29","timestamp":1777402450018374018,"data":{"Plan":{"Declaration":{"parent":{"query_id":"019dd571-1062-77c2-9803-62bd37658144","plan_id":null},"instance_name":"logical","edges":[{"source":"019dd571-1062-77c2-9803-62e0c8278c73","target":"019dd571-1062-77c2-9803-62f9fc6f8dd2"},{"source":"019dd571-1062-77c2-9803-6321681d89a7","target":"019dd571-1062-77c2-9803-6337bca9d715"},{"source":"019dd571-1062-77c2-9803-63542123336f","target":"019dd571-1062-77c2-9803-636aad7178d1"},{"source":"019dd571-1062-77c2-9803-637d87343372","target":"019dd571-1062-77c2-9803-6389c3b5868a"},{"source":"019dd571-1062-77c2-9803-63a1b8236d47","target":"019dd571-1062-77c2-9803-63bebe4c1970"},{"source":"019dd571-1062-77c2-9803-63d888c0ba46","target":"019dd571-1062-77c2-9803-63e1b00c3ae0"},{"source":"019dd571-1062-77c2-9803-6400931a8db7","target":"019dd571-1062-77c2-9803-6412f5addd70"}],"worker_id":null}}}}
    """

    id: uuid.UUID
    query: Query | None
    parent_plan: Plan | None
    instance_name: str  # TODO: Literal? logical / physical
    edges: list[Edge]
    worker: Worker | None

    @property
    def query_id(self) -> uuid.UUID | None:
        """Compatibility accessor for the parent query UUID."""
        return self.query.id if self.query is not None else None

    @property
    def parent_plan_id(self) -> uuid.UUID | None:
        """Compatibility accessor for the parent plan UUID."""
        return self.parent_plan.id if self.parent_plan is not None else None

    @property
    def worker_id(self) -> uuid.UUID | None:
        """Compatibility accessor for the worker UUID."""
        return self.worker.id if self.worker is not None else None

    def declare(self, timestamp: int | None = None) -> Event:
        """Declare a Quent Plan."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.PLAN.value: {
                    "Declaration": {
                        "parent": {
                            "query_id": str(self.query.id)
                            if self.query is not None
                            else None,
                            "plan_id": str(self.parent_plan.id)
                            if self.parent_plan is not None
                            else None,
                        },
                        "instance_name": self.instance_name,
                        "edges": [
                            {
                                "source": str(edge.source.id),
                                "target": str(edge.target.id),
                            }
                            for edge in self.edges
                        ],
                        "worker_id": str(self.worker.id)
                        if self.worker is not None
                        else None,
                    }
                }
            },
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Edge:
    """Plan edge connecting a source port to a target port."""

    source: Port
    target: Port


@dataclasses.dataclass(frozen=True, slots=True)
class Port:
    """Plan port."""

    id: uuid.UUID
    operator: Operator
    instance_name: str

    def declare(self, timestamp: int | None = None) -> Event:
        """
        Declare a Quent Port.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-62e0c8278c73","timestamp":1777402450018384708,"data":{"Port":{"Declaration":{"operator_id":"019dd571-1062-77c2-9803-62c7c1941381","instance_name":"out"}}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.PORT.value: {
                    "Declaration": {
                        "operator_id": str(self.operator.id),
                        "instance_name": self.instance_name,
                    }
                }
            },
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Query:
    """A Quent Query with lifecycle state transitions."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    instance_name: str | None = None

    def _init(self, query_group: QueryGroup, timestamp: int | None = None) -> Event:
        """
        Build a Quent Query Init event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-62bd37658144","timestamp":1777402450018294782,"data":{"Query":{"seq":0,"state":{"Init":{"instance_name":"Q0","query_group_id":"019dd571-1062-77c2-9803-62a66b6e0c5f"}}}}}
        """
        name = self.instance_name or self.id.hex[:8]
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.QUERY.value: {
                    "seq": 0,
                    "state": {
                        "Init": {
                            "instance_name": name,
                            "query_group_id": str(query_group.id),
                        }
                    },
                }
            },
        )

    def _planning(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent Query Planning event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-62bd37658144","timestamp":1777402450018327459,"data":{"Query":{"seq":1,"state":{"Planning":{}}}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.QUERY.value: {"seq": 1, "state": {"Planning": {}}}},
        )

    def _executing(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent Query Executing event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-62bd37658144","timestamp":1777402450018327459,"data":{"Query":{"seq":2,"state":{"Executing":{}}}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.QUERY.value: {"seq": 2, "state": {"Executing": {}}}},
        )

    def _exit(self, timestamp: int | None = None) -> Event:
        """
        Build a Quent Query Exit event.

        Examples
        --------
        {"id":"019dd571-1062-77c2-9803-62bd37658144","timestamp":1777402450365535083,"data":{"Query":{"seq":3,"state":"Exit"}}}
        """
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.QUERY.value: {"seq": 3, "state": "Exit"}},
        )


@dataclasses.dataclass(frozen=True, slots=True)
class QueryGroup:
    """Build a Quent Query Group."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    instance_name: str | None = None

    def _declare(self, engine: Engine, timestamp: int | None = None) -> Event:
        """Declare a Quent QueryGroup."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.QUERY_GROUP.value: {
                    "Declaration": {
                        "instance_name": self.instance_name,
                        "engine_id": str(engine.id),
                    }
                }
            },
        )


ScalarValue = int | float | str | bool
HomogeneousListValue = list[int] | list[float] | list[str] | list[bool]
Value: TypeAlias = ScalarValue | HomogeneousListValue | dict[str, "Value"]


@dataclasses.dataclass(frozen=True, slots=True)
class Attribute:
    """A Quent custom attribute."""

    name: str
    value: Value | None

    def serialize(self) -> dict[str, Any]:
        return {"key": self.name, "value": _serialize_value(self.value)}

    @classmethod
    def deserialize(cls, payload: dict[str, Any]) -> Attribute:
        return cls(name=payload["key"], value=_deserialize_value(payload["value"]))


def _serialize_value(value: Value | None) -> dict[str, Any] | None:
    match value:
        case None:
            return None
        case bool():
            # Bool is not a native Quent Value variant.
            return {"U8": int(value)}
        case int():
            if value >= 0:
                if value <= 2**8 - 1:
                    return {"U8": value}
                if value <= 2**16 - 1:
                    return {"U16": value}
                if value <= 2**32 - 1:
                    return {"U32": value}
                if value <= 2**64 - 1:
                    return {"U64": value}
            else:
                if -(2**7) <= value <= 2**7 - 1:
                    return {"I8": value}
                if -(2**15) <= value <= 2**15 - 1:
                    return {"I16": value}
                if -(2**31) <= value <= 2**31 - 1:
                    return {"I32": value}
                if -(2**63) <= value <= 2**63 - 1:
                    return {"I64": value}
            raise ValueError(
                f"Integer value {value} does not fit any Quent integer type."
            )
        case float():
            return {"F64": value}
        case str():
            return {"String": value}
        case list() | dict():
            raise NotImplementedError("List and dict attributes are not supported yet.")
        case _:  # pragma: no cover; should be exhaustive
            raise TypeError(f"Unsupported Quent custom attribute type: {type(value)}")


def _deserialize_value(value: dict[str, Any] | None) -> Value | None:
    if value is None:
        return None
    n = len(value)
    if n != 1:
        raise ValueError(
            f"Expected Quent attribute value envelope with exactly one variant, got '{n}' instead."
        )

    variant, deserialized = next(iter(value.items()))
    if variant in {"U8", "U16", "U32", "U64", "I8", "I16", "I32", "I64"}:
        return int(deserialized)
    if variant == "F64":
        return float(deserialized)
    if variant == "String":
        return str(deserialized)
    raise ValueError(f"Unsupported Quent custom attribute variant: '{variant}'")
