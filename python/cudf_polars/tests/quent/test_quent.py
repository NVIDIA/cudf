# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for schema-generated Quent telemetry."""

from __future__ import annotations

import dataclasses
import json
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

quent_bindings = pytest.importorskip("cudf_polars_quent")

from cudf_polars.quent._context import (  # noqa: E402
    ProcessorRegistry,
    QuentContext,
    WorkerResources,
)
from cudf_polars.quent._plan import (  # noqa: E402
    _emit_operator_details,
    emit_plan,
)
from cudf_polars.quent._runtime import QuentCollector, QuentSession  # noqa: E402


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    return tmp_path / "quent"


@pytest.fixture
def collector(output_root: Path) -> Iterator[QuentCollector]:
    collector = QuentCollector(output_root)
    yield collector
    collector.close()


def _events(root: Path) -> list[dict[str, Any]]:
    events = []
    for path in root.glob("*/*/*.ndjson"):
        entity_name = path.parent.name
        for line in path.read_text().splitlines():
            event = json.loads(line)
            event["data"] = {entity_name: event["data"]}
            events.append(event)
    return sorted(events, key=lambda event: event["timestamp"])


def _finish(
    session: QuentSession, collector: QuentCollector, root: Path
) -> list[dict[str, Any]]:
    session.close()
    collector.close()
    return _events(root)


def test_context_lifecycle_uses_generated_handles(
    quent_context: QuentContext,
    collector: QuentCollector,
    output_root: Path,
) -> None:
    session = QuentSession(collector.address)
    query_id = uuid.uuid4()

    quent_context._emit_engine_init_events(session, backend="test")
    quent_context._emit_query_group_events(session)
    quent_context._emit_query_events(session, query_id)
    quent_context._emit_query_completed_event(session, query_id)
    quent_context._emit_engine_exit_events(session)

    events = _finish(session, collector, output_root)
    assert [next(iter(event["data"])) for event in events] == [
        "Engine",
        "QueryGroup",
        "Query",
        "Query",
        "Query",
        "Query",
        "Engine",
    ]
    query_events = [event for event in events if "Query" in event["data"]]
    assert [next(iter(event["data"]["Query"])) for event in query_events] == [
        "Initialized",
        "Planning",
        "Executing",
        "Completed",
    ]
    assert str(quent_context.engine_id) == events[0]["id"]
    assert str(query_id) == query_events[0]["id"]


def test_query_group_is_declared_once_across_derived_configs(
    quent_context: QuentContext,
    collector: QuentCollector,
    output_root: Path,
) -> None:
    session = QuentSession(collector.address)
    quent_context._emit_query_group_events(session)
    for iteration in range(2):
        dataclasses.replace(
            quent_context, query_name=f"Iteration {iteration}"
        )._emit_query_group_events(session)

    assert [
        next(iter(event["data"])) for event in _finish(session, collector, output_root)
    ] == ["QueryGroup"]


def test_fsm_start_handle_is_consumed_after_transition() -> None:
    context = quent_bindings.Context()
    handle = context.engine_observer().handle(uuid.uuid4())
    implementation = {
        "name": "cudf-polars",
        "version": "test",
        "backend": "spmd",
        "custom_attributes": {"backend": "spmd"},
    }
    handle.init(instance_name="engine", implementation=implementation)
    with pytest.raises(quent_bindings.HandleConsumedError):
        handle.init(instance_name="engine", implementation=implementation)
    context.close()


def test_context_serialization_preserves_configuration(
    quent_context: QuentContext,
) -> None:
    assert QuentContext._deserialize(quent_context._serialize()) == quent_context


def test_plan_entities_are_deterministic(
    quent_context: QuentContext,
    monkeypatch: pytest.MonkeyPatch,
    collector: QuentCollector,
    output_root: Path,
) -> None:
    nodes = {
        "0": SimpleNamespace(type="DataFrameScan", children=[], properties={}),
        "1": SimpleNamespace(
            type="Sort",
            children=["0"],
            properties={"by": ["x"], "order": ["ASCENDING"]},
        ),
    }
    monkeypatch.setattr(
        "cudf_polars.quent._plan.SerializablePlan.from_ir",
        lambda *args, **kwargs: SimpleNamespace(nodes=nodes),
    )
    plan_id = uuid.uuid4()
    query_id = uuid.uuid4()
    worker_id = uuid.uuid4()

    first_session = QuentSession(collector.address)
    first = emit_plan(
        first_session,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        query_id,
        plan_id,
        worker_id,
    )
    first_session.close()

    second_session = QuentSession(collector.address)
    second = emit_plan(
        second_session,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        query_id,
        plan_id,
        worker_id,
        emit=False,
    )
    second_session.close()
    collector.close()
    events = _events(output_root)

    assert first == second
    assert events[0]["data"]["Plan"]["Declared"]["edges"][0] == {
        "source": {
            "target": str(uuid.uuid5(first["0"], "port:out")),
            "data": None,
        },
        "target": {
            "target": str(uuid.uuid5(first["1"], "port:in")),
            "data": None,
        },
    }
    assert any(
        event["data"].get("Operator", {}).get("SortDetails", {}).get("values")
        == {"by": ["x"], "order": ["ASCENDING"]}
        for event in events
    )


def test_processor_registry_declares_each_thread_once(
    collector: QuentCollector,
    output_root: Path,
) -> None:
    session = QuentSession(collector.address)
    registry = ProcessorRegistry()
    pool_id = uuid.uuid4()

    first = registry.get_or_declare_processor(session, 10, pool_id)
    assert registry.get_or_declare_processor(session, 10, pool_id) == first
    assert registry.get_or_declare_processor(session, 11, pool_id) != first

    events = _finish(session, collector, output_root)
    assert len(events) == 2
    assert all("Processor" in event["data"] for event in events)


def test_inter_rank_channel_targets_remote_memory(
    monkeypatch: pytest.MonkeyPatch,
    collector: QuentCollector,
    output_root: Path,
) -> None:
    monkeypatch.setattr(
        "cudf_polars.quent._context.get_total_device_memory", lambda: 1024
    )
    engine_id = uuid.uuid4()
    rank0 = WorkerResources.build("rank-0", engine_id, uuid.uuid4(), 0, 2)
    rank1 = WorkerResources.build("rank-1", engine_id, uuid.uuid4(), 1, 2)
    session = QuentSession(collector.address)
    rank0.declare(session)

    channel = next(
        event["data"]["DataChannel"]["Declared"]
        for event in _finish(session, collector, output_root)
        if event["data"].get("DataChannel", {}).get("Declared", {}).get("channel_type")
        == "inter-rank"
    )
    assert channel["source"]["target"] == str(rank0.device_memory_id)
    assert channel["target"]["target"] == str(rank1.device_memory_id)


def test_operator_details_preserve_static_types() -> None:
    class Operator:
        values: object = None

        def sort_details(self, *, values: object) -> None:
            self.values = values

    operator = Operator()
    _emit_operator_details(
        operator,  # type: ignore[arg-type]
        "Sort",
        {"by": ["x"], "order": ["ASCENDING"]},
    )
    assert operator.values == {"by": ["x"], "order": ["ASCENDING"]}
