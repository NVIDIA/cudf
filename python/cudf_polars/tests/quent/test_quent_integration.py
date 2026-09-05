# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Quent telemetry tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

from cudf_polars.dsl.tracing import LOG_TRACES

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.quent._context import QuentContext

# Quent tracing requires structlog to emit events. Skip the whole module when
# it is unavailable so the engine fixture below is never even constructed.
pytest.importorskip("structlog")


@pytest.fixture(params=["ray", "dask", "spmd"])
def engine_with_quent_context(
    request: pytest.FixtureRequest,
    quent_context: QuentContext,
    ray_num_ranks: int,
    ray_init_options: dict[str, Any],
) -> Iterator[StreamingEngine]:
    """
    A streaming engine configured with a quent context from the 'quent_context'
    fixture.

    The engine owns a rapidsmpf ``Context`` that must be shut down on the thread
    that created it. This fixture guarantees that teardown even if the test body
    skips or raises before shutting the engine down itself; otherwise the
    ``Context`` would be finalized by the garbage collector on an arbitrary
    thread and abort the interpreter.
    """
    backend = request.param
    engine: StreamingEngine
    if backend == "ray":
        pytest.importorskip("ray")
        import cudf_polars.engine.ray

        # Always specify num_ranks: the default path sizes the engine from the
        # GPUs Ray reports, which fails if this test shares an xdist worker
        # with a test that already brought up the shared num_gpus=0 cluster.
        engine = cudf_polars.engine.ray.RayEngine(
            executor_options={"quent_context": quent_context},
            engine_options={"allow_gpu_sharing": True},
            ray_init_options=ray_init_options,
            num_ranks=ray_num_ranks,
        )
    elif backend == "dask":
        pytest.importorskip("distributed")
        import cudf_polars.engine.dask

        engine = cudf_polars.engine.dask.DaskEngine(
            executor_options={"quent_context": quent_context}
        )
    elif backend == "spmd":
        from rapidsmpf import bootstrap
        from rapidsmpf.communicator.single import (
            new_communicator as single_communicator,
        )
        from rapidsmpf.config import Options, get_environment_variables
        from rapidsmpf.progress_thread import ProgressThread

        import cudf_polars.engine.spmd

        if bootstrap.is_running_with_rrun():
            comm = bootstrap.create_ucxx_comm(
                progress_thread=ProgressThread(),
                type=bootstrap.BackendType.AUTO,
            )
        else:
            comm = single_communicator(
                Options(get_environment_variables()), ProgressThread()
            )

        engine = cudf_polars.engine.spmd.SPMDEngine(
            executor_options={"quent_context": quent_context},
            comm=comm,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")

    try:
        yield engine
    finally:
        # Idempotent: a no-op if the test body already shut the engine down.
        engine.shutdown()


def test_quent_events(
    engine_with_quent_context: StreamingEngine, quent_context: QuentContext
) -> None:
    # We need to create the engine, to ensure the lifecycle events are emitted properly.
    q = pl.LazyFrame({"x": [1, 2]}).filter(pl.col("x") > 1)

    with engine_with_quent_context:
        q.collect(engine=engine_with_quent_context)

    check_quent_events(engine_with_quent_context, quent_context)


def check_quent_events(engine: StreamingEngine, quent_context: QuentContext) -> None:
    quent_events = engine._quent_events
    engine_events = [x for x in quent_events if "Engine" in x["data"]]
    assert len(engine_events) == 2
    engine_init, engine_exit = engine_events
    assert engine_init["id"] == str(quent_context.engine.id)
    assert engine_exit["id"] == str(quent_context.engine.id)
    assert engine_exit["data"]["Engine"]["Exit"] is None
    assert (
        engine_init["data"]["Engine"]["Init"]["implementation"]["name"] == "cudf-polars"
    )

    worker_events = [x for x in quent_events if "Worker" in x["data"]]
    worker_init_events = sorted(
        [x for x in worker_events if "Init" in x["data"]["Worker"]],
        key=lambda x: x["id"],
    )
    worker_exit_events = sorted(
        [x for x in worker_events if "Exit" in x["data"]["Worker"]],
        key=lambda x: x["id"],
    )
    assert len(worker_init_events) == len(worker_exit_events)
    for worker_init, worker_exit in zip(
        worker_init_events, worker_exit_events, strict=True
    ):
        assert worker_init["id"] == worker_exit["id"]
        assert (
            worker_init["data"]["Worker"]["Init"]["parent_engine_id"]
            == engine_init["id"]
        )
        assert worker_exit["data"]["Worker"]["Exit"] is None

    query_group_events = [x for x in quent_events if "QueryGroup" in x["data"]]
    assert len(query_group_events) == 1
    query_group_declaration = query_group_events[0]
    assert query_group_declaration["id"] == str(quent_context.query_group.id)
    assert (
        query_group_declaration["data"]["QueryGroup"]["Declaration"]["engine_id"]
        == engine_init["id"]
    )

    query_events = [x for x in quent_events if "Query" in x["data"]]
    assert len(query_events) == 4

    query_init, query_planning, query_executing, query_exit = query_events
    # Each ``.collect()`` derives a fresh per-collect query id, so the emitted
    # id must be unique to this collect rather than the engine-scoped template
    # ``quent_context.query`` id.
    query_id = query_init["id"]
    assert query_id != str(quent_context.query.id)
    assert (
        query_init["data"]["Query"]["state"]["Init"]["query_group_id"]
        == query_group_declaration["id"]
    )
    assert query_init["data"]["Query"]["seq"] == 0
    assert query_planning["id"] == query_id
    assert query_planning["data"]["Query"]["seq"] == 1
    assert query_executing["id"] == query_id
    assert query_executing["data"]["Query"]["seq"] == 2
    assert query_exit["id"] == query_id
    assert query_exit["data"]["Query"]["seq"] == 3

    memory_events = [x for x in quent_events if "Memory" in x["data"]]
    task_events = [x for x in quent_events if "Task" in x["data"]]
    assert len(memory_events) > 0

    if LOG_TRACES:
        assert len(task_events) > 0

    # A single collect exercises the full processor lifecycle, so fold that
    # check in here rather than paying for a dedicated engine startup.
    check_processor_lifecycle(quent_events)


def test_quent_events_multiple_collects(
    engine_with_quent_context: StreamingEngine, quent_context: QuentContext
) -> None:
    # Everything that depends on running more than one collect against the same
    # engine is folded into this single test to avoid paying for extra engine
    # startups. Running the *same* query twice is the strongest scenario: query
    # ids are derived per-collect and ``get_stable_plan_id`` is a deterministic
    # function of the IR structure, so an un-namespaced plan id would collide
    # across the two identical collects.
    q = pl.LazyFrame({"x": [1, 2, 3]}).filter(pl.col("x") > 1)
    with engine_with_quent_context:
        q.collect(engine=engine_with_quent_context)
        q.collect(engine=engine_with_quent_context)

    quent_events = engine_with_quent_context._quent_events

    # The processor lifecycle stays balanced across multiple collects.
    check_processor_lifecycle(quent_events)

    # Device memory is engine/worker-scoped: it is initialized and finalized
    # exactly once per worker, matching the number of engine-scoped ThreadPool
    # declarations. Critically, running two collects must NOT re-declare it
    # (the per-query bug would produce a fresh device memory per collect, i.e.
    # twice as many inits as thread pools).
    memory_events = [x for x in quent_events if "Memory" in x["data"]]
    device_init_events = [
        x
        for x in memory_events
        if isinstance(x["data"]["Memory"]["state"], dict)
        and "MemoryInitializing" in x["data"]["Memory"]["state"]
        and "device memory"
        in x["data"]["Memory"]["state"]["MemoryInitializing"]["instance_name"]
    ]
    device_exit_events = [
        x
        for x in memory_events
        if x["data"]["Memory"]["state"] == "Exit"
        and x["id"] in {e["id"] for e in device_init_events}
    ]
    thread_pool_decls = [
        x
        for x in quent_events
        if "ThreadPool" in x["data"] and "Declaration" in x["data"]["ThreadPool"]
    ]
    assert len(thread_pool_decls) >= 1
    assert len(device_init_events) == len(thread_pool_decls)
    # Every device memory id is initialized once and exited once.
    init_ids = [x["id"] for x in device_init_events]
    assert len(set(init_ids)) == len(init_ids)
    assert {x["id"] for x in device_exit_events} == set(init_ids)

    # Each collect reuses the engine-scoped QuentContext but must emit a
    # distinct query id.
    query_init_ids = [
        x["id"]
        for x in quent_events
        if "Query" in x["data"] and "Init" in x["data"]["Query"].get("state", {})
    ]
    assert len(query_init_ids) == 2
    assert len(set(query_init_ids)) == 2
    assert str(quent_context.query.id) not in query_init_ids

    # Without namespacing by the per-collect query id, both identical collects
    # would emit the same logical plan id under different parent queries.
    logical_plan_decls = [
        x
        for x in quent_events
        if "Plan" in x["data"]
        and "Declaration" in x["data"]["Plan"]
        and x["data"]["Plan"]["Declaration"]["instance_name"] == "logical"
    ]
    assert len(logical_plan_decls) == 2
    plan_ids = [x["id"] for x in logical_plan_decls]
    assert len(set(plan_ids)) == 2
    # Each logical plan must hang off the distinct per-collect query id.
    parent_query_ids = [
        x["data"]["Plan"]["Declaration"]["parent"]["query_id"]
        for x in logical_plan_decls
    ]
    assert len(set(parent_query_ids)) == 2


def check_processor_lifecycle(quent_events: list[dict]) -> None:
    thread_pool_ids = {
        x["id"]
        for x in quent_events
        if "ThreadPool" in x["data"] and "Declaration" in x["data"]["ThreadPool"]
    }
    assert len(thread_pool_ids) >= 1

    processor_events = [x for x in quent_events if "Processor" in x["data"]]
    init_events = [
        x
        for x in processor_events
        if "ProcessorInitializing" in x["data"]["Processor"]["state"]
    ]
    finalizing_events = [
        x
        for x in processor_events
        if x["data"]["Processor"]["state"] == {"ProcessorFinalizing": None}
    ]
    exit_events = [
        x for x in processor_events if x["data"]["Processor"]["state"] == "Exit"
    ]

    assert len(init_events) == len(finalizing_events) == len(exit_events)

    if LOG_TRACES:
        assert len(init_events) > 0

        init_by_id = {x["id"]: x for x in init_events}
        finalizing_by_id = {x["id"]: x for x in finalizing_events}
        exit_by_id = {x["id"]: x for x in exit_events}
        assert init_by_id.keys() == finalizing_by_id.keys() == exit_by_id.keys()

        for init_event in init_by_id.values():
            parent_group_id = init_event["data"]["Processor"]["state"][
                "ProcessorInitializing"
            ]["parent_group_id"]
            assert parent_group_id in thread_pool_ids
