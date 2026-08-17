# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Quent telemetry tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.quent import QuentContext

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
    assert query_init["id"] == str(quent_context.query.id)
    assert (
        query_init["data"]["Query"]["state"]["Init"]["query_group_id"]
        == query_group_declaration["id"]
    )
    assert query_init["data"]["Query"]["seq"] == 0
    assert query_planning["id"] == str(quent_context.query.id)
    assert query_planning["data"]["Query"]["seq"] == 1
    assert query_executing["id"] == str(quent_context.query.id)
    assert query_executing["data"]["Query"]["seq"] == 2
    assert query_exit["id"] == str(quent_context.query.id)
    assert query_exit["data"]["Query"]["seq"] == 3
