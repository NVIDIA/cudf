# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Quent context state.

In general, integration tests are preferred for Quent testing. But these unit
test cover areas that are relatively difficult to trigger through normal
execution.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, call

from cudf_polars.quent._context import (
    QuentConfig,
    QuentQueryConfig,
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
