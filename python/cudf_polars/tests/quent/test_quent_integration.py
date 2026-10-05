# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for custom-schema Quent telemetry."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

pytest.importorskip("cudf_polars_quent")

import cudf_polars.quent
from cudf_polars.dsl.tracing import LOG_TRACES
from cudf_polars.engine.spmd import SPMDEngine

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.quent import QuentContext


@pytest.fixture(params=["ray", "dask", "spmd"])
def engine_with_quent_context(
    request: pytest.FixtureRequest,
    quent_context: QuentContext,
    ray_num_ranks: int,
    ray_init_options: dict[str, Any],
) -> Iterator[StreamingEngine]:
    backend = request.param
    engine: StreamingEngine
    if backend == "ray":
        pytest.importorskip("ray")
        from cudf_polars.engine.ray import RayEngine

        engine = RayEngine(
            executor_options={"quent_context": quent_context},
            engine_options={"allow_gpu_sharing": True},
            ray_init_options=ray_init_options,
            num_ranks=ray_num_ranks,
        )
    elif backend == "dask":
        pytest.importorskip("distributed")
        from cudf_polars.engine.dask import DaskEngine

        engine = DaskEngine(executor_options={"quent_context": quent_context})
    else:
        from rapidsmpf import bootstrap
        from rapidsmpf.communicator.single import new_communicator
        from rapidsmpf.config import Options, get_environment_variables
        from rapidsmpf.progress_thread import ProgressThread

        comm = (
            bootstrap.create_ucxx_comm(
                progress_thread=ProgressThread(), type=bootstrap.BackendType.AUTO
            )
            if bootstrap.is_running_with_rrun()
            else new_communicator(
                Options(get_environment_variables()), ProgressThread()
            )
        )
        engine = SPMDEngine(
            executor_options={"quent_context": quent_context}, comm=comm
        )
    try:
        yield engine
    finally:
        engine.shutdown()


def _stored_events(root: Path) -> list[dict[str, Any]]:
    events = []
    for path in root.glob("*/*/*.ndjson"):
        entity_name = path.parent.name
        for line in path.read_text().splitlines():
            event = json.loads(line)
            event["data"] = {entity_name: event["data"]}
            events.append(event)
    return sorted(events, key=lambda event: event["timestamp"])


def _of_type(events: list[dict[str, Any]], entity: str) -> list[dict[str, Any]]:
    return [event for event in events if entity in event["data"]]


def test_custom_schema_events(
    engine_with_quent_context: StreamingEngine, quent_context: QuentContext
) -> None:
    query = pl.LazyFrame({"x": [1, 2]}).filter(pl.col("x") > 1)
    with engine_with_quent_context:
        query.collect(engine=engine_with_quent_context)

    assert engine_with_quent_context._quent_output_root is not None
    events = _stored_events(engine_with_quent_context._quent_output_root)
    engine_events = _of_type(events, "Engine")
    assert len(engine_events) == 2
    assert engine_events[0]["id"] == str(quent_context.engine_id)
    assert "Init" in engine_events[0]["data"]["Engine"]
    assert "Exit" in engine_events[1]["data"]["Engine"]

    worker_events = _of_type(events, "Worker")
    initialized = [
        event for event in worker_events if "Init" in event["data"]["Worker"]
    ]
    exited = [event for event in worker_events if "Exit" in event["data"]["Worker"]]
    assert {event["id"] for event in initialized} == {event["id"] for event in exited}

    assert len(_of_type(events, "QueryGroup")) == 1
    query_events = _of_type(events, "Query")
    assert len(query_events) == 4
    assert query_events[0]["id"] != str(quent_context.query_group_id)
    assert _of_type(events, "Plan")
    assert _of_type(events, "Operator")
    assert _of_type(events, "Actor")
    assert _of_type(events, "DeviceMemory")
    assert _of_type(events, "Storage")
    if LOG_TRACES:
        assert _of_type(events, "Evaluate")


def test_multiple_collects_get_distinct_queries_and_plans(
    engine_with_quent_context: StreamingEngine,
) -> None:
    query = pl.LazyFrame({"x": [1, 2, 3]}).filter(pl.col("x") > 1)
    with engine_with_quent_context:
        query.collect(engine=engine_with_quent_context)
        query.collect(engine=engine_with_quent_context)

    assert engine_with_quent_context._quent_output_root is not None
    events = _stored_events(engine_with_quent_context._quent_output_root)
    initialized_queries = [
        event
        for event in _of_type(events, "Query")
        if isinstance(event["data"]["Query"], dict)
        and "Initialized" in event["data"]["Query"]
    ]
    assert len({event["id"] for event in initialized_queries}) == 2

    logical_plans = [
        event
        for event in _of_type(events, "Plan")
        if event["data"]["Plan"]["Declared"]["instance_name"] == "logical"
    ]
    assert len({event["id"] for event in logical_plans}) == 2

    memory_ids = [
        event["id"]
        for entity in ("DeviceMemory", "Storage")
        for event in _of_type(events, entity)
    ]
    assert len(memory_ids) == len(set(memory_ids))


@pytest.mark.spmd
def test_quent_context_user_provided(spmd_engine: SPMDEngine, tmp_path: Path) -> None:
    quent_context = cudf_polars.quent.QuentContext(
        engine_id=uuid.uuid4(),
        implementation_name="test_implementation",
        implementation_version="0.0.0",
        query_group_name="test_query_group",
        query_name="test_query",
        output_root=str(tmp_path / "quent"),
    )

    with SPMDEngine(
        comm=spmd_engine.comm, executor_options={"quent_context": quent_context}
    ) as engine:
        assert engine.config["executor_options"]["quent_context"] == quent_context
        with pytest.raises(ValueError, match="quent_context cannot be changed"):
            engine._reset(
                executor_options={"quent_context": cudf_polars.quent.QuentContext()}
            )


@pytest.mark.spmd
def test_quent_context_default(spmd_engine: SPMDEngine) -> None:
    with SPMDEngine(comm=spmd_engine.comm) as engine:
        assert engine.config["executor_options"].get("quent_context") is None
