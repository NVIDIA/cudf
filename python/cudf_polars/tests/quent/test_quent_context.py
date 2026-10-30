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
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call

import pytest

from rapidsmpf.memory.buffer import MemoryType

import cudf_polars.quent._runtime as quent_runtime
from cudf_polars.engine.spmd import synchronize_quent_reset_contexts
from cudf_polars.quent._context import (
    ActorBoundQuentIRExecutionState,
    QuentConfig,
    QuentQueryConfig,
    UnboundQuentIRExecutionState,
    WorkerResources,
)
from cudf_polars.quent._runtime import QuentWorkerRuntime
from cudf_polars.streaming.actor_graph.utils import reservation_attributes

if TYPE_CHECKING:
    from collections.abc import Callable


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


def test_reservation_attributes() -> None:
    actor_id = uuid.uuid4()
    state = MagicMock(actor_id=actor_id)

    assert reservation_attributes(None, "join.evaluate") is None
    assert reservation_attributes(state, "join.evaluate") == {
        "rapidsmpf.site": "join.evaluate",
        "quent.actor_id": str(actor_id),
    }
    with pytest.raises(ValueError, match="lowercase dotted"):
        reservation_attributes(state, "Join")
    with pytest.raises(TypeError, match="UUID"):
        reservation_attributes(MagicMock(actor_id="not-a-uuid"), "join.evaluate")


def test_worker_resources_declares_inter_rank_channel() -> None:
    engine_id = uuid.uuid4()
    worker_id = uuid.uuid4()
    buffer_resource = MagicMock()
    buffer_resource.has_memory_resource.side_effect = lambda memory_type: (
        memory_type in {MemoryType.DEVICE, MemoryType.HOST}
    )
    buffer_resource.memory_limit.side_effect = {
        MemoryType.DEVICE: 100,
        MemoryType.HOST: -1,
    }.__getitem__
    buffer_resource.spill_directory = None
    resources = WorkerResources.build(
        instance_suffix="rank-0",
        engine_id=engine_id,
        worker_id=worker_id,
        rank=0,
        nranks=2,
        buffer_resource=buffer_resource,
    )

    session = MagicMock()
    resources.declare(session)

    reservable_memory_observer = (
        session.binding_context.reservable_memory_observer.return_value
    )
    assert reservable_memory_observer.handle.call_args_list == [
        call(resources.reservable_memory_ids["device"]),
        call(resources.reservable_memory_ids["host"]),
    ]
    reservable_memory_observer.handle.return_value.declared.assert_has_calls(
        [
            call(
                instance_name="rank-0 device reservable memory",
                worker=worker_id,
                memory_type="device",
                limits={"bytes": 100},
                configured_limit=100,
            ),
            call(
                instance_name="rank-0 host reservable memory",
                worker=worker_id,
                memory_type="host",
                limits={"bytes": 0},
                configured_limit=-1,
            ),
        ]
    )
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


def test_worker_runtime_finalizes_events_before_worker_exit() -> None:
    events: list[str] = []
    final_batch = [{"sequence": 1}]
    buffer_resource = MagicMock()

    def stop_recording() -> list[dict[str, int]]:
        events.append("stop")
        return final_batch

    buffer_resource.stop_reservation_event_recording.side_effect = stop_recording
    ingestor = MagicMock()
    ingestor.ingest.side_effect = lambda batch: events.append(
        f"ingest:{batch is final_batch}"
    )

    def diagnostics() -> list[str]:
        events.append("diagnostics")
        return []

    ingestor.diagnostics.side_effect = diagnostics
    ingestor.close.side_effect = lambda: events.append("ingestor-close")
    worker_handle = MagicMock()
    worker_handle.exit.side_effect = lambda: events.append("worker-exit")
    session = MagicMock()
    session.close.side_effect = lambda: events.append("session-close")
    runtime = QuentWorkerRuntime(
        config=QuentConfig(),
        session=session,
        worker_resources=MagicMock(),
        _worker_handle=worker_handle,
        _buffer_resource=buffer_resource,
        _reservation_ingestor=ingestor,
    )

    runtime.close()

    assert events == [
        "stop",
        "ingest:True",
        "diagnostics",
        "ingestor-close",
        "worker-exit",
        "session-close",
    ]


def test_worker_runtime_create_starts_recording_after_declarations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    session = MagicMock()
    monkeypatch.setattr(quent_runtime, "QuentSession", lambda _: session)
    resources = MagicMock()
    resources.reservable_memory_ids = {"device": uuid.uuid4()}
    resources.declare.side_effect = lambda _: events.append("declare")
    monkeypatch.setattr(WorkerResources, "build", staticmethod(lambda **_: resources))
    ingestor = MagicMock()

    def create_ingestor(*_: object) -> MagicMock:
        events.append("ingestor")
        return ingestor

    monkeypatch.setattr(quent_runtime, "ReservationEventIngestor", create_ingestor)
    buffer_resource = MagicMock()
    buffer_resource.start_reservation_event_recording.side_effect = lambda: (
        events.append("start")
    )

    runtime = QuentWorkerRuntime.create(
        QuentConfig(),
        "collector:1234",
        worker_id=uuid.uuid4(),
        rank=0,
        nranks=1,
        instance_name="rank-0",
        buffer_resource=buffer_resource,
    )

    assert events == ["declare", "ingestor", "start"]
    assert runtime._reservation_ingestor is ingestor


def test_worker_runtime_drains_events() -> None:
    batch = [{"sequence": 0}]
    buffer_resource = MagicMock()
    buffer_resource.drain_reservation_events.return_value = batch
    ingestor = MagicMock()
    runtime = QuentWorkerRuntime(
        config=QuentConfig(),
        session=MagicMock(),
        worker_resources=MagicMock(),
        _worker_handle=MagicMock(),
        _buffer_resource=buffer_resource,
        _reservation_ingestor=ingestor,
    )

    runtime.enable_reservation_event_metrics()
    runtime.drain_reservation_events()

    ingestor.ingest.assert_called_once_with(batch)
    metrics = runtime.reservation_event_metrics()
    assert metrics.event_count == 1
    assert metrics.peak_buffered_events == 1
    assert metrics.peak_buffered_bytes > 0
    assert metrics.drain_conversion_seconds >= 0


def test_worker_runtime_retargets_and_seeds_memory_limits() -> None:
    events: list[str] = []

    def record(name: str, result: object = None) -> Callable[..., object]:
        def side_effect(*_: object) -> object:
            events.append(name)
            return result

        return side_effect

    old_resource = MagicMock()
    old_resource.stop_reservation_event_recording.side_effect = record(
        "stop", [{"sequence": 0}]
    )
    new_resource = MagicMock()
    new_resource.start_reservation_event_recording.side_effect = record("start")
    new_resource.memory_limit.side_effect = record("limit", 42)
    new_resource.set_memory_limit.side_effect = record("set-limit")
    ingestor = MagicMock()
    ingestor.ingest.side_effect = record("ingest")
    ingestor.begin_recording_session.side_effect = record("session")
    resources = MagicMock()
    resources.reservable_memory_ids = {"device": uuid.uuid4()}
    runtime = QuentWorkerRuntime(
        config=QuentConfig(),
        session=MagicMock(),
        worker_resources=resources,
        _worker_handle=MagicMock(),
        _buffer_resource=old_resource,
        _reservation_ingestor=ingestor,
    )

    runtime.retarget_buffer_resource(new_resource)

    assert events == ["stop", "ingest", "session", "start", "limit", "set-limit"]
    new_resource.set_memory_limit.assert_called_once_with(MemoryType.DEVICE, 42)


def test_worker_runtime_close_logs_event_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    buffer_resource = MagicMock()
    buffer_resource.stop_reservation_event_recording.side_effect = RuntimeError(
        "stop failed"
    )
    ingestor = MagicMock()
    ingestor.diagnostics.return_value = ["bad attribute"]
    ingestor.close.side_effect = RuntimeError("live reservation")
    worker_handle = MagicMock()
    session = MagicMock()
    runtime = QuentWorkerRuntime(
        config=QuentConfig(),
        session=session,
        worker_resources=MagicMock(),
        _worker_handle=worker_handle,
        _buffer_resource=buffer_resource,
        _reservation_ingestor=ingestor,
    )

    runtime.close()

    assert "Failed to stop reservation event recording" in caplog.text
    assert "Reservation event diagnostics: bad attribute" in caplog.text
    assert "Reservation event lifecycle was incomplete" in caplog.text
    ingestor.close.assert_called_once_with()
    worker_handle.exit.assert_called_once_with()
    session.close.assert_called_once_with()


def test_emit_evaluate_failure() -> None:
    evaluate_id = uuid.uuid4()
    running_evaluation = MagicMock()
    runtime = QuentWorkerRuntime(
        config=QuentConfig(),
        session=MagicMock(),
        worker_resources=MagicMock(),
        _worker_handle=MagicMock(),
        _buffer_resource=None,
        _reservation_ingestor=None,
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
        quent_runtime,
        "start_collector",
        MagicMock(return_value=collector),
    )
    monkeypatch.setattr(
        quent_runtime,
        "QuentSession",
        MagicMock(side_effect=RuntimeError("session creation failed")),
    )

    with pytest.raises(RuntimeError, match="session creation failed"):
        quent_runtime.QuentControllerRuntime.create(
            quent_context,
            backend="test",
        )

    collector.close.assert_called_once_with(timeout=10.0)
