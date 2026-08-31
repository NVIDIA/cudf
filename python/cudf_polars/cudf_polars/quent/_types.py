# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quent telemetry tracing."""

# TODO: https://github.com/NVIDIA/cudf/issues/22949 use the bindings from quent

from __future__ import annotations

import dataclasses
import enum
import itertools
import sys
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias

from cudf_polars import __version__

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent._context import QuentIRExecutionContext

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
    MEMORY = "Memory"
    CHANNEL = "Channel"
    THREAD_POOL = "ThreadPool"
    PROCESSOR = "Processor"
    NETWORK = "Network"


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
class StatisticsAttribute:
    """Typed key/value pair for Quent statistics custom attributes."""

    key: str
    value_type: Literal["U64", "F64", "String"]
    value: int | float | str

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": {self.value_type: self.value}}


@dataclasses.dataclass(frozen=True, slots=True)
class Statistics:
    """Operator statistics payload."""

    input_bytes: int
    output_bytes: int
    output_rows: int
    custom_attributes: list[StatisticsAttribute] = dataclasses.field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to Quent's custom attributes format."""
        base_attributes: list[StatisticsAttribute] = [
            StatisticsAttribute(
                key="input_bytes", value_type="U64", value=self.input_bytes
            ),
            StatisticsAttribute(
                key="output_bytes", value_type="U64", value=self.output_bytes
            ),
            StatisticsAttribute(
                key="output_rows", value_type="U64", value=self.output_rows
            ),
        ]
        return {
            "custom_attributes": [
                *(attribute.to_dict() for attribute in base_attributes),
                *(attribute.to_dict() for attribute in self.custom_attributes),
            ]
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
            "instance_name": f"{self.type_name}-{self.id.hex[:8]}",
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

    def statistics(self, statistics: Statistics, timestamp: int | None = None) -> Event:
        """Emit post-execution operator statistics."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.OPERATOR.value: {"Statistics": statistics.to_dict()}},
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
StructValue: TypeAlias = dict[str, "Value | None"]
HomogeneousListValue = (
    list[int] | list[float] | list[str] | list[bool] | list[StructValue]
)
Value: TypeAlias = ScalarValue | HomogeneousListValue | StructValue

_INT_VARIANTS: tuple[tuple[str, int, int], ...] = (
    ("U8", 0, 2**8 - 1),
    ("U16", 0, 2**16 - 1),
    ("U32", 0, 2**32 - 1),
    ("U64", 0, 2**64 - 1),
    ("I8", -(2**7), 2**7 - 1),
    ("I16", -(2**15), 2**15 - 1),
    ("I32", -(2**31), 2**31 - 1),
    ("I64", -(2**63), 2**63 - 1),
)
_INT_VARIANT_NAMES = frozenset(name for name, _, _ in _INT_VARIANTS)


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


def _integer_variant(value: int) -> str:
    """Return the narrowest Quent integer variant that can hold ``value``."""
    for variant, lo, hi in _INT_VARIANTS:
        if lo <= value <= hi:
            return variant
    raise ValueError(f"Integer value {value} does not fit any Quent integer type.")


def _common_integer_variant(values: list[int]) -> str:
    """Return the narrowest Quent integer variant that can hold all ``values``."""
    for variant, lo, hi in _INT_VARIANTS:
        if all(lo <= value <= hi for value in values):
            return variant
    raise ValueError("Integer list values do not fit any Quent integer type.")


def _serialize_struct(value: StructValue) -> list[dict[str, Any]]:
    """Serialize a dict as a Quent ``Struct`` (list of attributes)."""
    return [
        {"key": key, "value": _serialize_value(item)} for key, item in value.items()
    ]


def _serialize_list(values: list[Any]) -> dict[str, Any]:
    """
    Serialize a homogeneous list as a Quent ``List`` payload.

    Empty lists default to ``String`` because the element type cannot be
    inferred. Nested lists are not supported by Quent
    (see https://github.com/rapidsai/quent/issues/79).
    """
    if not values:
        return {"String": []}
    # bool is a subclass of int; check it before int.
    if all(isinstance(item, bool) for item in values):
        return {"U8": [int(item) for item in values]}
    if all(isinstance(item, int) for item in values):
        return {_common_integer_variant(values): values}
    if all(isinstance(item, float) for item in values):
        return {"F64": values}
    if all(isinstance(item, str) for item in values):
        return {"String": values}
    if all(isinstance(item, dict) for item in values):
        return {"Struct": [_serialize_struct(item) for item in values]}
    if any(isinstance(item, list) for item in values):
        raise NotImplementedError("Nested list attributes are not supported by Quent.")
    raise TypeError(
        "Quent list attributes must be homogeneous "
        f"(int, float, str, bool, or dict); got {[type(v).__name__ for v in values]}"
    )


def _serialize_value(value: Value | None) -> dict[str, Any] | None:
    match value:
        case None:
            return None
        case bool():
            # Bool is not a native Quent Value variant.
            return {"U8": int(value)}
        case int():
            return {_integer_variant(value): value}
        case float():
            return {"F64": value}
        case str():
            return {"String": value}
        case list():
            return {"List": _serialize_list(value)}
        case dict():
            return {"Struct": _serialize_struct(value)}
        case _:  # pragma: no cover; should be exhaustive
            raise TypeError(f"Unsupported Quent custom attribute type: {type(value)}")


def _deserialize_struct(payload: list[dict[str, Any]]) -> StructValue:
    return {item["key"]: _deserialize_value(item["value"]) for item in payload}


def _deserialize_list(payload: dict[str, Any]) -> HomogeneousListValue:
    n = len(payload)
    if n != 1:
        raise ValueError(
            f"Expected Quent List envelope with exactly one variant, got '{n}' instead."
        )
    variant, deserialized = next(iter(payload.items()))
    if variant in _INT_VARIANT_NAMES:
        return [int(item) for item in deserialized]
    if variant == "F64":
        return [float(item) for item in deserialized]
    if variant == "String":
        return [str(item) for item in deserialized]
    if variant == "Struct":
        return [_deserialize_struct(item) for item in deserialized]
    raise ValueError(f"Unsupported Quent List variant: '{variant}'")


def _deserialize_value(value: dict[str, Any] | list[Any] | None) -> Value | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(
            "Expected Quent attribute value envelope as a single-variant object, "
            f"got {type(value).__name__}."
        )
    n = len(value)
    if n != 1:
        raise ValueError(
            f"Expected Quent attribute value envelope with exactly one variant, got '{n}' instead."
        )

    variant, deserialized = next(iter(value.items()))
    if variant in _INT_VARIANT_NAMES:
        return int(deserialized)
    if variant == "F64":
        return float(deserialized)
    if variant == "String":
        return str(deserialized)
    if variant == "Struct":
        return _deserialize_struct(deserialized)
    if variant == "List":
        return _deserialize_list(deserialized)
    raise ValueError(f"Unsupported Quent custom attribute variant: '{variant}'")


# Resource capacity helpers
#
# Quent distinguishes unit resources (Processor/thread), occupancy capacities
# (Memory), and rate capacities (Channel). See quent/docs/modeling/resource.md.


def occupancy_usage_capacity_bytes(capacity_bytes: int) -> dict[str, int]:
    """Usage capacity for a Memory resource (occupancy over the usage span)."""
    return {"capacity_bytes": capacity_bytes}


def rate_usage_capacity_bytes(capacity_bytes: int) -> dict[str, int]:
    """
    Usage capacity for a Channel resource (total bytes over the usage span).

    Rate capacity values represent the total quantity transferred during the
    span, not bytes per second.
    """
    return {"capacity_bytes": capacity_bytes}


# Resource types


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Memory:
    """A Quent Memory resource."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    instance_name: str
    resource_type_name: str
    parent_group_id: uuid.UUID

    def initializing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Memory Initializing event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.MEMORY.value: {
                    "seq": 0,
                    "state": {
                        "MemoryInitializing": {
                            "instance_name": self.instance_name,
                            "parent_group_id": str(self.parent_group_id),
                            "resource_type_name": self.resource_type_name,
                        }
                    },
                }
            },
        )

    def operating(
        self, capacity_bytes: int | None = None, timestamp: int | None = None
    ) -> Event:
        """Build a Quent Memory Operating event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.MEMORY.value: {
                    "seq": 1,
                    "state": {"MemoryOperating": {"capacity_bytes": capacity_bytes}},
                }
            },
        )

    def finalizing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Memory Finalizing event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.MEMORY.value: {"seq": 2, "state": {"MemoryFinalizing": None}}
            },
        )

    def exit(self, timestamp: int | None = None) -> Event:
        """Build a Quent Memory Exit event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.MEMORY.value: {"seq": 3, "state": "Exit"}},
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Channel:
    """
    A Quent Channel resource.

    A Channel is a unidirectional data-transfer resource between two entities.
    Examples include disk-to-device I/O channels and inter-rank network links.
    """

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    instance_name: str
    resource_type_name: str
    parent_group_id: uuid.UUID
    source: Memory
    target: Memory

    def initializing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Channel Initializing event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.CHANNEL.value: {
                    "seq": 0,
                    "state": {
                        "ChannelInitializing": {
                            "instance_name": self.instance_name,
                            "parent_group_id": str(self.parent_group_id),
                            "resource_type_name": self.resource_type_name,
                            "source_id": str(self.source.id),
                            "target_id": str(self.target.id),
                        }
                    },
                }
            },
        )

    def operating(
        self, capacity_bytes: int | None = None, timestamp: int | None = None
    ) -> Event:
        """Build a Quent Channel Operating event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.CHANNEL.value: {
                    "seq": 1,
                    "state": {"ChannelOperating": {"capacity_bytes": capacity_bytes}},
                }
            },
        )

    def finalizing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Channel Finalizing event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.CHANNEL.value: {
                    "seq": 2,
                    "state": {"ChannelFinalizing": None},
                }
            },
        )

    def exit(self, timestamp: int | None = None) -> Event:
        """Build a Quent Channel Exit event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.CHANNEL.value: {"seq": 3, "state": "Exit"}},
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Network:
    """A Quent Network resource group."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    engine_id: uuid.UUID

    def declare(self, timestamp: int | None = None) -> Event:
        """Build a Network declaration event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.NETWORK.value: {
                    "Declaration": {
                        "instance_name": "Network",
                        "parent_group_id": str(self.engine_id),
                    }
                }
            },
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ThreadPool:
    """A Quent ThreadPool resource group."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    worker_id: uuid.UUID

    def declare(self, timestamp: int | None = None) -> Event:
        """Build a ThreadPool declaration event."""
        instance_name = f"Thread Pool {self.id.hex[:8]}"
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.THREAD_POOL.value: {
                    "Declaration": {
                        "instance_name": instance_name,
                        "parent_group_id": str(self.worker_id),
                    }
                },
            },
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Processor:
    """A Quent Processor resource representing a CPU thread."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    pool_id: uuid.UUID

    def initializing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Processor Initializing event."""
        instance_name = f"Thread {self.id.hex[:8]}"
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.PROCESSOR.value: {
                    "seq": 0,
                    "state": {
                        "ProcessorInitializing": {
                            "instance_name": instance_name,
                            "parent_group_id": str(self.pool_id),
                            "resource_type_name": "processor",
                        }
                    },
                }
            },
        )

    def operating(self, timestamp: int | None = None) -> Event:
        """Build a Quent Processor Operating event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.PROCESSOR.value: {
                    "seq": 1,
                    "state": {"ProcessorOperating": None},
                }
            },
        )

    def finalizing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Processor Finalizing event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.PROCESSOR.value: {
                    "seq": 2,
                    "state": {"ProcessorFinalizing": None},
                }
            },
        )

    def exit(self, timestamp: int | None = None) -> Event:
        """Build a Quent Processor Exit event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.PROCESSOR.value: {"seq": 3, "state": "Exit"}},
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Task:
    """A Quent Task representing a unit of work on an operator."""

    id: uuid.UUID = dataclasses.field(default_factory=new_quent_id)
    operator_id: uuid.UUID
    instance_name: str | None = None
    _seq: Iterator[int] = dataclasses.field(
        default_factory=itertools.count, compare=False, repr=False
    )

    @classmethod
    def from_ir(
        cls, ir_type: type[IR], quent_ir_execution_context: QuentIRExecutionContext
    ) -> Self:
        """
        Build an operator-scoped Quent Task from an IR execution context.

        Parameters
        ----------
        ir_type
            The IR type of the operator.
        quent_ir_execution_context
            The Quent IR execution context, which is used to get the operator ID.

        Returns
        -------
        Task | None
            The operator-scoped Quent Task, or ``None`` if the IR execution context
            is not bound to a Quent operator.
        """
        token = uuid.uuid4()
        return cls(
            instance_name=(
                f"{ir_type.__name__}-{quent_ir_execution_context.quent_operator.id.hex[:8]}-"
                f"{token.hex[:8]}"
            ),
            operator_id=quent_ir_execution_context.quent_operator.id,
        )

    def queueing(self, timestamp: int | None = None) -> Event:
        """Build a Quent Task Queueing event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.TASK.value: {
                    "seq": next(self._seq),
                    "state": {
                        "Queueing": {
                            "instance_name": self.instance_name or self.id.hex[:8],
                            "operator_id": str(self.operator_id),
                        }
                    },
                }
            },
        )

    def allocating(
        self,
        resource_id: uuid.UUID,
        timestamp: int | None = None,
    ) -> Event:
        """Build a Quent Task Allocating event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.TASK.value: {
                    "seq": next(self._seq),
                    "state": {
                        "Allocating": {
                            "use_thread": {
                                "resource_id": str(resource_id),
                                "capacity": None,
                            }
                        }
                    },
                }
            },
        )

    def loading(
        self,
        use_thread: Processor | None = None,
        use_channel: Channel | None = None,
        channel_capacity_bytes: int = 0,
        use_memory: Memory | None = None,
        memory_capacity_bytes: int = 0,
        timestamp: int | None = None,
    ) -> Event:
        """Build a Quent Task Loading event."""
        loading_data: dict[str, dict[str, Any]] = {}
        if use_thread is not None:
            loading_data["use_thread"] = {
                "resource_id": str(use_thread.id),
                "capacity": None,
            }
        if use_channel is not None:
            loading_data["use_fs_to_mem"] = {
                "resource_id": str(use_channel.id),
                "capacity": rate_usage_capacity_bytes(channel_capacity_bytes),
            }
        if use_memory is not None:
            loading_data["use_memory"] = {
                "resource_id": str(use_memory.id),
                "capacity": occupancy_usage_capacity_bytes(memory_capacity_bytes),
            }
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.TASK.value: {
                    "seq": next(self._seq),
                    "state": {"Loading": loading_data},
                }
            },
        )

    def computing(
        self,
        use_thread: Processor | None = None,
        use_memory: Memory | None = None,
        input_bytes: int = 0,
        memory_capacity_bytes: int = 0,
        timestamp: int | None = None,
    ) -> Event:
        """Build a Quent Task Computing event."""
        computing_data: dict[str, Any] = {}
        computing_data["instance_name"] = ""
        computing_data["input_bytes"] = input_bytes
        if use_thread is not None:
            computing_data["use_thread"] = {
                "resource_id": str(use_thread.id),
                "capacity": None,
            }
        if use_memory is not None:
            computing_data["use_memory"] = {
                "resource_id": str(use_memory.id),
                "capacity": occupancy_usage_capacity_bytes(memory_capacity_bytes),
            }
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.TASK.value: {
                    "seq": next(self._seq),
                    "state": {"Computing": computing_data},
                }
            },
        )

    def sending(
        self,
        use_thread: Processor | None = None,
        use_link: Channel | None = None,
        link_capacity_bytes: int = 0,
        timestamp: int | None = None,
    ) -> Event:
        """Build a Quent Task Sending event."""
        sending_data: dict[str, dict[str, Any]] = {}
        if use_thread is not None:
            sending_data["use_thread"] = {
                "resource_id": str(use_thread.id),
                "capacity": None,
            }
        if use_link is not None:
            sending_data["use_link"] = {
                "resource_id": str(use_link.id),
                "capacity": rate_usage_capacity_bytes(link_capacity_bytes),
            }
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={
                EventName.TASK.value: {
                    "seq": next(self._seq),
                    "state": {"Sending": sending_data},
                }
            },
        )

    def exit(self, timestamp: int | None = None) -> Event:
        """Build a Quent Task Exit event."""
        return Event(
            id=self.id,
            timestamp=timestamp if timestamp is not None else time.time_ns(),
            data={EventName.TASK.value: {"seq": next(self._seq), "state": "Exit"}},
        )
