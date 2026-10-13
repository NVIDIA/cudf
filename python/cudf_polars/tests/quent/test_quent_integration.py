# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for custom-schema Quent telemetry."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

from cudf_polars.utils.config import ConfigOptions

pytest.importorskip("cudf_polars_quent")

import cudf_polars.quent
from cudf_polars.dsl.tracing import LOG_TRACES

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from pathlib import Path

    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.quent import QuentConfig


@pytest.fixture(params=["ray", "dask", "spmd"])
def engine_with_quent_context(
    request: pytest.FixtureRequest,
    quent_context: QuentConfig,
    ray_num_ranks: int,
    ray_init_options: dict[str, Any],
) -> Iterator[StreamingEngine]:
    backend = request.param
    executor_options = {
        "quent_context": quent_context,
        "fallback_mode": "silent",
    }
    engine: StreamingEngine
    if backend == "ray":
        pytest.importorskip("ray")
        from cudf_polars.engine.ray import RayEngine

        engine = RayEngine(
            executor_options=executor_options,
            engine_options={"allow_gpu_sharing": True},
            ray_init_options=ray_init_options,
            num_ranks=ray_num_ranks,
        )
    elif backend == "dask":
        pytest.importorskip("distributed")
        from cudf_polars.engine.dask import DaskEngine

        engine = DaskEngine(executor_options=executor_options)
    else:
        from rapidsmpf import bootstrap
        from rapidsmpf.communicator.single import new_communicator
        from rapidsmpf.config import Options, get_environment_variables
        from rapidsmpf.progress_thread import ProgressThread

        from cudf_polars.engine.spmd import SPMDEngine

        comm = (
            bootstrap.create_ucxx_comm(
                progress_thread=ProgressThread(), type=bootstrap.BackendType.AUTO
            )
            if bootstrap.is_running_with_rrun()
            else new_communicator(
                Options(get_environment_variables()), ProgressThread()
            )
        )
        engine = SPMDEngine(executor_options=executor_options, comm=comm)
    try:
        yield engine
    finally:
        engine.shutdown()


def _stored_events(root: Path) -> list[dict[str, Any]]:
    events = []
    for path in root.glob("*/*/*.ndjson"):
        entity_name = path.parent.name
        for seq, line in enumerate(path.read_text().splitlines()):
            event = json.loads(line)
            event["data"] = {entity_name: event["data"]}
            event["sequence"] = seq
            events.append(event)
    return sorted(events, key=lambda event: (event["timestamp"], event["sequence"]))


def _of_type(events: list[dict[str, Any]], entity: str) -> list[dict[str, Any]]:
    return [event for event in events if entity in event["data"]]


def _disable_logging(level: int) -> None:
    """Set the process-wide logging threshold in a Dask worker."""
    logging.disable(level)


@contextlib.contextmanager
def suppress_worker_exceptions(
    engine: StreamingEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Suppress backend logging for a deliberately failed remote query."""
    dask_client = None
    try:
        from cudf_polars.engine.ray import RayEngine
    except ImportError:
        pass
    else:
        if isinstance(engine, RayEngine):
            # Ray otherwise reports intentionally unhandled remote errors.
            monkeypatch.setenv("RAY_IGNORE_UNHANDLED_ERRORS", "1")

    try:
        from cudf_polars.engine.dask import DaskEngine
    except ImportError:
        pass
    else:
        if isinstance(engine, DaskEngine):
            dask_context = engine._dask_context
            assert dask_context is not None
            dask_client = dask_context.client

    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    if dask_client is not None:
        # Dask emits task-failure logs in its worker processes.
        dask_client.run(_disable_logging, logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_logging_disable)
        if dask_client is not None:
            dask_client.run(_disable_logging, logging.NOTSET)


@pytest.mark.filterwarnings("ignore:Rolling.*:UserWarning")
def test_quent_lifecycle(
    engine_with_quent_context: StreamingEngine,
    quent_context: QuentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = pl.LazyFrame({"x": [1, 2, 3]}).filter(pl.col("x") > 1)
    with engine_with_quent_context:
        assert (
            engine_with_quent_context.config["executor_options"]["quent_context"]
            == quent_context
        )
        current_context = dataclasses.replace(
            quent_context,
            query=dataclasses.replace(
                quent_context.query,
                query_group_id=uuid.uuid4(),
                query_group_name="Updated Query Group",
                query_name="Iteration 1",
            ),
        )
        engine_with_quent_context.config["executor_options"]["quent_context"] = (
            current_context
        )
        query.collect(engine=engine_with_quent_context)
        current_context = dataclasses.replace(
            current_context,
            query=dataclasses.replace(
                current_context.query,
                query_name="Iteration 2",
            ),
        )
        engine_with_quent_context.config["executor_options"]["quent_context"] = (
            current_context
        )
        query.collect(engine=engine_with_quent_context)
        current_context = dataclasses.replace(
            current_context,
            query=dataclasses.replace(
                current_context.query,
                query_name="Failed iteration",
            ),
        )
        engine_with_quent_context.config["executor_options"]["quent_context"] = (
            current_context
        )
        failed_query = (
            pl.LazyFrame({"orderby": [1, 2, 4, 2], "value": [1, 2, 3, 4]})
            .rolling("orderby", period="2i")
            .agg(pl.sum("value"))
        )
        with (
            suppress_worker_exceptions(engine_with_quent_context, monkeypatch),
            pytest.raises(Exception),  # noqa: B017 - backend-specific wrapper
        ):
            failed_query.collect(engine=engine_with_quent_context)
        with pytest.raises(ValueError, match="quent_context cannot be changed"):
            engine_with_quent_context._reset(
                executor_options={"quent_context": cudf_polars.quent.QuentConfig()}
            )

        # StreamingExecutor
        config_options = ConfigOptions.from_polars_engine(engine_with_quent_context)
        hash_a = hash(config_options)
        hash_b = hash(config_options)
        assert hash_a == hash_b

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

    query_group_events = _of_type(events, "QueryGroup")
    assert len(query_group_events) == 1
    assert query_group_events[0]["id"] == str(current_context.query.query_group_id)
    assert (
        query_group_events[0]["data"]["QueryGroup"]["Declared"]["instance_name"]
        == "Updated Query Group"
    )
    query_events = _of_type(events, "Query")
    assert len(query_events) == 12
    assert _of_type(events, "Plan")
    assert _of_type(events, "Operator")
    assert _of_type(events, "Actor")
    assert _of_type(events, "DeviceMemory")
    assert _of_type(events, "Storage")
    if LOG_TRACES:
        assert _of_type(events, "Evaluate")

    initialized_queries = [
        event
        for event in query_events
        if isinstance(event["data"]["Query"], dict)
        and "Initialized" in event["data"]["Query"]
    ]
    assert len({event["id"] for event in initialized_queries}) == 3
    assert [
        event["data"]["Query"]["Initialized"]["instance_name"]
        for event in initialized_queries
    ] == ["Iteration 1", "Iteration 2", "Failed iteration"]
    events_by_query = {
        initialized["id"]: [
            next(iter(event["data"]["Query"]))
            for event in query_events
            if event["id"] == initialized["id"]
        ]
        for initialized in initialized_queries
    }
    assert list(events_by_query.values()) == [
        ["Initialized", "Planning", "Executing", "Completed"],
        ["Initialized", "Planning", "Executing", "Completed"],
        ["Initialized", "Planning", "Executing", "Failed"],
    ]

    logical_plans = [
        event
        for event in _of_type(events, "Plan")
        if event["data"]["Plan"]["Declared"]["instance_name"] == "logical"
    ]
    assert len({event["id"] for event in logical_plans}) == 3

    memory_ids = [
        event["id"]
        for entity in ("DeviceMemory", "Storage")
        for event in _of_type(events, entity)
    ]
    assert len(memory_ids) == len(set(memory_ids))
