# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for process-local Quent lifecycle management."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("cudf_polars_quent")

from cudf_polars.quent._context import QuentContext
from cudf_polars.quent._runtime import QuentRuntime, start_collector

if TYPE_CHECKING:
    from pathlib import Path


def _events(root: Path) -> list[dict[str, Any]]:
    events = []
    for path in root.glob("*/*/*.ndjson"):
        entity_name = path.parent.name
        for line in path.read_text().splitlines():
            event = json.loads(line)
            event["data"] = {entity_name: event["data"]}
            events.append(event)
    return sorted(events, key=lambda event: event["timestamp"])


def test_runtime_combines_controller_and_worker_lifecycles(tmp_path: Path) -> None:
    context = QuentContext(output_root=str(tmp_path))
    collector = start_collector(context.run_root)
    worker_id = uuid.uuid4()
    query_id = uuid.uuid4()
    runtime = QuentRuntime.create(
        context,
        collector.address,
        backend="test",
        collector=collector,
        worker_id=worker_id,
        rank=0,
        nranks=1,
        instance_name="rank-0",
    )

    local_context = runtime.local_context(query_id)
    assert local_context.worker_id == worker_id
    assert local_context.session is runtime.session
    with runtime.query(query_id):
        pass
    runtime.close()

    events = _events(context.run_root)
    assert [
        next(iter(event["data"]))
        for event in events
        if next(iter(event["data"])) in {"Engine", "Worker", "Query"}
    ] == [
        "Engine",
        "Worker",
        "Query",
        "Query",
        "Query",
        "Query",
        "Worker",
        "Engine",
    ]


def test_runtime_query_failure_and_idempotent_close(tmp_path: Path) -> None:
    context = QuentContext(output_root=str(tmp_path))
    collector = start_collector(context.run_root)
    query_id = uuid.uuid4()
    runtime = QuentRuntime.create(
        context,
        collector.address,
        backend="test",
        collector=collector,
    )

    with pytest.raises(RuntimeError, match="failed"), runtime.query(query_id):
        raise RuntimeError("failed")
    runtime.close()
    runtime.close()

    query_events = [
        event for event in _events(context.run_root) if "Query" in event["data"]
    ]
    assert "Failed" in query_events[-1]["data"]["Query"]


def test_runtime_worker_only(tmp_path: Path) -> None:
    context = QuentContext(output_root=str(tmp_path))
    collector = start_collector(context.run_root)
    runtime = QuentRuntime.create(
        context,
        collector.address,
        collector=collector,
        worker_id=uuid.uuid4(),
        rank=0,
        nranks=1,
        instance_name="rank-0",
    )

    runtime.local_context(uuid.uuid4())
    runtime.close()

    events = _events(context.run_root)
    assert not any("Engine" in event["data"] for event in events)
    assert len([event for event in events if "Worker" in event["data"]]) == 2
