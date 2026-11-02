# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Quent context state.

In general, integration tests are preferred for Quent testing. But these unit
test cover areas that are relatively difficult to trigger through normal
execution.
"""

from __future__ import annotations

import dataclasses
import uuid
from unittest.mock import MagicMock, call

import pytest

import cudf_polars.quent._runtime
from cudf_polars.engine.spmd import synchronize_quent_reset_contexts
from cudf_polars.quent._context import (
    ActorBoundQuentIRExecutionState,
    QuentConfig,
    QuentQueryConfig,
    UnboundQuentIRExecutionState,
    WorkerResources,
)
from cudf_polars.quent._runtime import QuentWorkerRuntime


def test_context_serialization_roundtrip(tmp_path) -> None:
    context = QuentConfig(
        engine_id=uuid.uuid4(),
        output_root=str(tmp_path),
        query=QuentQueryConfig(
            query_group_id=uuid.uuid4(),
            query_group_name="group",
            query_name="query",
        ),
    )

    assert QuentConfig._deserialize(context._serialize()) == context


@pytest.mark.parametrize(
    "existing_enabled,new_enabled",
    [
        pytest.param(False, True, id="enable"),
        pytest.param(True, True, id="replace"),
        pytest.param(True, False, id="disable"),
    ],
)
def test_synchronize_quent_reset_contexts(
    quent_context: QuentConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing_enabled: bool,
    new_enabled: bool,
) -> None:
    existing_context = quent_context if existing_enabled else None
    new_context = (
        dataclasses.replace(quent_context, engine_id=uuid.uuid4())
        if new_enabled
        else None
    )

    def fake_all_gather(*args: object) -> list[bytes]:
        data = args[-1]
        assert isinstance(data, bytes)
        return [data, b""]

    monkeypatch.setattr(
        "cudf_polars.engine.spmd.all_gather_host_data",
        fake_all_gather,
    )

    synchronized = synchronize_quent_reset_contexts(
        comm=MagicMock(nranks=2, rank=0),
        context=MagicMock(),
        existing_context=existing_context,
        new_context=new_context,
    )

    assert synchronized == (existing_context, new_context)


def test_ir_execution_state_bind_actor() -> None:
    actor_id = uuid.uuid4()
    state = UnboundQuentIRExecutionState(
        query_worker_state=MagicMock(),
        operator_id=uuid.uuid4(),
    )

    actor_state = state.bind_actor(actor_id)

    assert isinstance(actor_state, ActorBoundQuentIRExecutionState)
    assert actor_state.actor_id == actor_id
    assert actor_state.query_worker_state is state.query_worker_state
    assert actor_state.operator_id == state.operator_id


def test_worker_resources_declares_inter_rank_channel() -> None:
    engine_id = uuid.uuid4()
    worker_id = uuid.uuid4()
    resources = WorkerResources.build(
        instance_suffix="rank-0",
        engine_id=engine_id,
        worker_id=worker_id,
        rank=0,
        nranks=2,
    )

    session = MagicMock()
    resources.declare(session)

    data_channel_observer = session.binding_context.data_channel_observer.return_value
    link_channel_id = resources.link_channel_ids[1]
    assert data_channel_observer.handle.call_args_list == [
        call(resources.disk_to_device_channel_id),
        call(link_channel_id),
    ]
    data_channel_observer.handle.return_value.declared.assert_has_calls(
        [
            call(
                instance_name="rank-0 disk -> device",
                channel_type="disk-to-device",
                worker=worker_id,
                source=resources.filesystem_id,
                target=resources.device_memory_id,
            ),
            call(
                instance_name="rank-0 -> rank-1",
                channel_type="inter-rank",
                worker=worker_id,
                source=resources.device_memory_id,
                target=uuid.uuid5(
                    uuid.uuid5(engine_id, "worker:1"),
                    "device-memory",
                ),
            ),
        ]
    )


def test_emit_evaluate_failure() -> None:
    evaluate_id = uuid.uuid4()
    running_evaluation = MagicMock()
    runtime = QuentWorkerRuntime(
        config=QuentConfig(),
        session=MagicMock(),
        worker_resources=MagicMock(),
        _worker_handle=MagicMock(),
    )
    runtime.session._evaluations = {
        evaluate_id: running_evaluation,
    }
    error = RuntimeError("evaluation failed")

    runtime.emit_evaluate_end(
        evaluate_id,
        result=None,
        error=error,
    )

    running_evaluation.failed.assert_called_once_with(error=str(error))
    assert evaluate_id not in runtime.session._evaluations


def test_quent_controller_create_closes_collector_on_failure(
    quent_context: QuentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = MagicMock(address="127.0.0.1:1234")
    monkeypatch.setattr(
        cudf_polars.quent._runtime,
        "start_collector",
        MagicMock(return_value=collector),
    )
    monkeypatch.setattr(
        cudf_polars.quent._runtime,
        "QuentSession",
        MagicMock(side_effect=RuntimeError("session creation failed")),
    )

    with pytest.raises(RuntimeError, match="session creation failed"):
        cudf_polars.quent._runtime.QuentControllerRuntime.create(
            quent_context,
            backend="test",
        )

    collector.close.assert_called_once_with(timeout=10.0)
