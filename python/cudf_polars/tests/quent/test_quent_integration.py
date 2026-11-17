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
from unittest.mock import MagicMock

import pytest

import polars as pl

from cudf_polars.utils.config import ConfigOptions

pytest.importorskip("cudf_polars_quent")

import cudf_polars.quent._runtime

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from pathlib import Path

    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.engine.options import StreamingOptions
    from cudf_polars.quent import QuentConfig


@pytest.fixture
def engine_with_quent_context(
    _unconfigured_engine: tuple[StreamingEngine, StreamingOptions],
    quent_context: QuentConfig,
    streaming_engine_factory: Callable[[StreamingOptions], StreamingEngine],
) -> Iterator[tuple[StreamingEngine, StreamingOptions]]:
    """
    Yield a streaming engine configured with Quent tracing.
    """
    # Uses ``_unconfigured_engine`` directly (rather than a public engine
    # fixture) so that quent setup and teardown can be expressed as a single
    # ``_reset`` call each way.
    engine, base_options = _unconfigured_engine
    engine._reset(
        rapidsmpf_options=base_options.to_rapidsmpf_options(),
        executor_options={
            **base_options.to_executor_options(),
            "quent_context": quent_context,
            "fallback_mode": "silent",
        },
        engine_options=base_options.to_engine_options(),
    )
    try:
        yield engine, base_options
    finally:
        engine._reset(
            rapidsmpf_options=base_options.to_rapidsmpf_options(),
            executor_options={
                **base_options.to_executor_options(),
                "quent_context": None,
                "fallback_mode": "silent",
            },
            engine_options=base_options.to_engine_options(),
        )


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


def test_quent_worker_cleanup_continues_after_handle_failure() -> None:
    session = MagicMock()
    worker_handle = MagicMock()
    worker_handle.exit.side_effect = RuntimeError("worker exit failed")
    runtime = cudf_polars.quent._runtime.QuentWorkerRuntime(
        config=MagicMock(),
        session=session,
        worker_resources=MagicMock(),
        _worker_handle=worker_handle,
    )

    with pytest.raises(ExceptionGroup, match="Quent worker shutdown failed"):
        runtime.close()

    session.close.assert_called_once_with()


def test_quent_controller_cleanup_continues_after_handle_failure() -> None:
    session = MagicMock()
    engine_handle = MagicMock()
    engine_handle.exit.side_effect = RuntimeError("engine exit failed")
    collector = MagicMock()
    runtime = cudf_polars.quent._runtime.QuentControllerRuntime(
        config=MagicMock(),
        session=session,
        collector=collector,
        _engine_handle=engine_handle,
    )

    with pytest.raises(ExceptionGroup, match="Quent controller shutdown failed"):
        runtime.close()

    session.close.assert_called_once_with()
    collector.close.assert_called_once_with(timeout=10.0)


def test_quent_lifecycle(
    engine_with_quent_context: tuple[StreamingEngine, StreamingOptions],
    quent_context: QuentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = pl.LazyFrame({"x": [1, 2, 3]}).filter(pl.col("x") > 1)
    engine, base_options = engine_with_quent_context
    assert engine.config["executor_options"]["quent_context"] == quent_context
    current_context = dataclasses.replace(
        quent_context,
        query=dataclasses.replace(
            quent_context.query,
            query_group_id=uuid.uuid4(),
            query_group_name="Updated Query Group",
            query_name="Iteration 1",
        ),
    )
    engine.config["executor_options"]["quent_context"] = current_context
    query.collect(engine=engine)
    current_context = dataclasses.replace(
        current_context,
        query=dataclasses.replace(
            current_context.query,
            query_name="Iteration 2",
        ),
    )
    engine.config["executor_options"]["quent_context"] = current_context
    query.collect(engine=engine)
    current_context = dataclasses.replace(
        current_context,
        query=dataclasses.replace(
            current_context.query,
            query_name="Failed iteration",
        ),
    )
    engine.config["executor_options"]["quent_context"] = current_context
    failed_query = (
        pl.LazyFrame({"orderby": [1, 2, 4, 2], "value": [1, 2, 3, 4]})
        .rolling("orderby", period="2i")
        .agg(pl.sum("value"))
    )
    with (
        suppress_worker_exceptions(engine, monkeypatch),
        pytest.raises(Exception),  # noqa: B017 - backend-specific wrapper
    ):
        failed_query.collect(engine=engine)
    # StreamingExecutor
    config_options = ConfigOptions.from_polars_engine(engine)
    hash_a = hash(config_options)
    hash_b = hash(config_options)
    assert hash_a == hash_b

    # Capture the output root, then close Quent to flush events to disk.
    # The fixture teardown calls _reset(..., quent_context=None) again, which is a no-op.
    quent_output_root = engine._quent_output_root
    assert quent_output_root is not None
    engine._reset(
        rapidsmpf_options=base_options.to_rapidsmpf_options(),
        executor_options={
            **base_options.to_executor_options(),
            "quent_context": None,
            "fallback_mode": "silent",
        },
        engine_options=base_options.to_engine_options(),
    )

    events = _stored_events(quent_output_root)
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
    operator_declarations = [
        event["data"]["Operator"]["Declared"]
        for event in _of_type(events, "Operator")
        if "Declared" in event["data"]["Operator"]
    ]
    assert operator_declarations
    assert all(
        "input_schemas" in declaration["schemas"]
        and "columns" in declaration["schemas"]["output_schema"]
        for declaration in operator_declarations
    )
    actor_events = _of_type(events, "Actor")
    assert actor_events
    actor_states: dict[str, list[str]] = {}
    for event in actor_events:
        actor_states.setdefault(event["id"], []).append(
            next(iter(event["data"]["Actor"]))
        )
    assert all(
        states[:2] == ["Started", "Running"]
        and states[-1] in {"Completed", "Failed"}
        and len(states) == 3
        for states in actor_states.values()
    )
    terminal_actor_events = [
        next(iter(event["data"]["Actor"].values()))
        for event in actor_events
        if not set(event["data"]["Actor"]).isdisjoint({"Completed", "Failed"})
    ]
    assert all(
        {
            "input_bytes",
            "output_bytes",
            "output_rows",
            "chunk_count",
            "duplicated",
            "decision",
        }.issubset(terminal["values"])
        for terminal in terminal_actor_events
    )
    assert _of_type(events, "DeviceMemory")
    assert _of_type(events, "Storage")
    evaluate_events = _of_type(events, "Evaluate")
    assert evaluate_events
    evaluate_states: dict[str, list[str]] = {}
    for event in evaluate_events:
        evaluate_states.setdefault(event["id"], []).append(
            next(iter(event["data"]["Evaluate"]))
        )
    assert all(
        states == ["Queued", "Running", states[-1]]
        and states[-1] in {"Completed", "Failed"}
        for states in evaluate_states.values()
    )
    running_evaluations = [
        event["data"]["Evaluate"]["Running"]
        for event in evaluate_events
        if "Running" in event["data"]["Evaluate"]
    ]
    assert all(
        {
            "io",
            "input_bytes",
            "input",
            "processor",
            "channel",
        }.issubset(running)
        for running in running_evaluations
    )
    assert all(
        all(set(dataframe) == {"shape"} for dataframe in running["input"]["dataframes"])
        for running in running_evaluations
    )
    chunk_evaluations = [
        running
        for running in running_evaluations
        if running["input"]["content_sizes"] is not None
    ]
    assert chunk_evaluations
    assert all(
        running["input"]["sequence_number"] is not None
        and running["input"]["spillable"] is not None
        for running in chunk_evaluations
    )
    completed_evaluations = [
        event["data"]["Evaluate"]["Completed"]
        for event in evaluate_events
        if "Completed" in event["data"]["Evaluate"]
    ]
    assert completed_evaluations
    assert all(
        set(completed["output_dataframe"]) == {"shape"}
        for completed in completed_evaluations
    )
    queued_io_evaluations = [
        event["data"]["Evaluate"]["Queued"]
        for event in evaluate_events
        if "Queued" in event["data"]["Evaluate"]
        and event["data"]["Evaluate"]["Queued"]["task"] is not None
    ]
    assert queued_io_evaluations
    assert all(
        {"node_id", "node_type"}.issubset(queued["task"])
        for queued in queued_io_evaluations
    )
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
