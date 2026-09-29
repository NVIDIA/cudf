# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudf-polars orchestration for schema-generated Quent handles."""

from __future__ import annotations

import dataclasses
import json
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from cudf_polars import __version__
from cudf_polars.quent._plan import build_parent_operators_map, emit_plan
from cudf_polars.quent._runtime import QuentSession
from cudf_polars.utils.config import (
    get_total_device_memory,
    resolve_quent_output_root,
)

if TYPE_CHECKING:
    from typing import Self

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor

__all__ = [
    "LocalQuentContext",
    "ProcessorRegistry",
    "QuentContext",
    "QuentIRExecutionContext",
    "QuentSession",
    "WorkerResources",
]


class ProcessorRegistry:
    """Map Python executor threads to generated Processor handles."""

    def __init__(self) -> None:
        self._processors: dict[int, uuid.UUID] = {}
        self._lock = threading.Lock()

    def get_or_declare_processor(
        self, session: QuentSession, thread_ident: int, pool_id: uuid.UUID
    ) -> uuid.UUID:
        """Get or declare the Processor associated with a host thread."""
        with self._lock:
            if thread_ident in self._processors:
                return self._processors[thread_ident]
            import cudf_polars_quent as _quent

            processor_id = _quent.now_v7()
            self._processors[thread_ident] = processor_id

        session.context.processor_observer().handle(processor_id).declared(
            instance_name=f"Thread {processor_id.hex[:8]}",
            thread_pool=pool_id,
        )
        return processor_id


@dataclasses.dataclass(frozen=True, kw_only=True)
class QuentContext:
    """
    Serializable Quent configuration shared by all ranks.

    Notes
    -----
    ``output_root`` is the controller-local staging directory used by the
    Collector. Workers send events over the network and do not access this path.
    """

    engine_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    query_group_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    query_group_name: str | None = None
    query_name: str | None = None
    implementation_name: str = "cudf-polars"
    implementation_version: str = __version__
    output_root: str = dataclasses.field(default_factory=resolve_quent_output_root)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_root", resolve_quent_output_root(self.output_root)
        )

    @property
    def run_root(self) -> Path:
        """Return this engine's controller-local Collector output directory."""
        return Path(self.output_root) / str(self.engine_id)

    def _serialize(self) -> bytes:
        payload = {
            **dataclasses.asdict(self),
            "engine_id": int(self.engine_id),
            "query_group_id": int(self.query_group_id),
        }
        return json.dumps(payload).encode()

    @classmethod
    def _deserialize(cls, data: bytes) -> Self:
        payload = json.loads(data)
        return cls(
            engine_id=uuid.UUID(int=int(payload["engine_id"])),
            query_group_id=uuid.UUID(int=int(payload["query_group_id"])),
            query_group_name=payload["query_group_name"],
            query_name=payload["query_name"],
            implementation_name=payload["implementation_name"],
            implementation_version=payload["implementation_version"],
            output_root=payload["output_root"],
        )

    def _emit_engine_init_events(self, session: QuentSession, *, backend: str) -> None:
        session._engines[self.engine_id] = (
            session.context.engine_observer()
            .handle(self.engine_id)
            .init(
                instance_name=f"cudf-polars-{str(self.engine_id)[:8]}",
                implementation={
                    "name": self.implementation_name,
                    "version": self.implementation_version,
                    "backend": backend,
                    "custom_attributes": {"backend": backend},
                },
            )
        )

    def _emit_engine_exit_events(self, session: QuentSession) -> None:
        session._engines.pop(self.engine_id).exit()

    def _emit_query_group_events(self, session: QuentSession) -> None:
        if not session.declare_once("QueryGroup", self.query_group_id):
            return
        session.context.query_group_observer().handle(self.query_group_id).declared(
            instance_name=self.query_group_name,
            engine=self.engine_id,
        )

    def _emit_query_events(self, session: QuentSession, query_id: uuid.UUID) -> None:
        initialized = (
            session.context.query_observer()
            .handle(query_id)
            .initialized(
                instance_name=self.query_name or query_id.hex[:8],
                query_group=self.query_group_id,
            )
        )
        session._queries[query_id] = initialized.planning().executing()

    def _emit_query_completed_event(
        self, session: QuentSession, query_id: uuid.UUID
    ) -> None:
        session._queries.pop(query_id).completed()

    def _emit_query_failed_event(
        self, session: QuentSession, query_id: uuid.UUID, error: BaseException
    ) -> None:
        session._queries.pop(query_id).failed(error=str(error))

    def _emit_physical_plan_events(
        self,
        session: QuentSession,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        plan_id: uuid.UUID,
        query_id: uuid.UUID,
        worker_id: uuid.UUID,
        *,
        parent_plan_id: uuid.UUID,
        node_map: dict[str, list[str]],
        logical_op_by_id: dict[str, uuid.UUID],
    ) -> dict[str, uuid.UUID]:
        parent_operators = build_parent_operators_map(node_map, logical_op_by_id)
        return emit_plan(
            session,
            ir,
            config_options,
            query_id=query_id,
            plan_id=plan_id,
            worker_id=worker_id,
            instance_name="physical",
            parent_plan_id=parent_plan_id,
            parent_operators_by_node_id=parent_operators,
        )

    def _emit_evaluate_begin_events(
        self,
        ir_type: type[IR],
        evaluate_id: uuid.UUID,
        instance_name: str,
        execution_context: QuentIRExecutionContext,
        input_frames_bytes: int,
    ) -> None:
        processor_id = execution_context.get_or_declare_processor(threading.get_ident())
        assert execution_context.actor_id is not None, (
            "Evaluate events must be emitted from an Actor scope"
        )
        queued = (
            execution_context.session.context.evaluate_observer()
            .handle(evaluate_id)
            .queued(instance_name=instance_name, actor=execution_context.actor_id)
        )
        execution_context.session._evaluations[evaluate_id] = queued.running(
            io=ir_type.is_io_node,
            input_bytes=input_frames_bytes,
            processor={"target": processor_id, "data": {}},
            channel={
                "target": execution_context.worker_resources.disk_to_device_channel_id,
                "data": {"bytes": input_frames_bytes},
            }
            if ir_type.is_io_node
            else None,
        )

    def _emit_evaluate_end_event(
        self,
        evaluate_id: uuid.UUID,
        execution_context: QuentIRExecutionContext,
        result: DataFrame | None,
        error: BaseException | None,
    ) -> None:
        if error is not None:
            execution_context.session._evaluations.pop(evaluate_id).failed(
                error=str(error)
            )
        else:
            assert result is not None
            execution_context.session._evaluations.pop(evaluate_id).completed(
                output_bytes=result._size_bytes,
            )


@dataclasses.dataclass(kw_only=True)
class WorkerResources:
    """Per-worker resource identities and generated declarations."""

    engine_id: uuid.UUID
    worker_id: uuid.UUID
    rank: int
    instance_suffix: str
    thread_pool_id: uuid.UUID
    processor_registry: ProcessorRegistry
    device_memory_id: uuid.UUID
    device_memory_bytes: int
    filesystem_id: uuid.UUID
    disk_to_device_channel_id: uuid.UUID
    link_channel_ids: dict[int, uuid.UUID]

    @classmethod
    def build(
        cls,
        instance_suffix: str,
        engine_id: uuid.UUID,
        worker_id: uuid.UUID,
        rank: int,
        nranks: int,
    ) -> Self:
        namespace = uuid.uuid5(engine_id, f"worker:{rank}")

        return cls(
            engine_id=engine_id,
            worker_id=worker_id,
            rank=rank,
            instance_suffix=instance_suffix,
            thread_pool_id=uuid.uuid5(namespace, "thread-pool"),
            processor_registry=ProcessorRegistry(),
            device_memory_id=uuid.uuid5(namespace, "device-memory"),
            device_memory_bytes=get_total_device_memory() or 0,
            filesystem_id=uuid.uuid5(namespace, "filesystem"),
            disk_to_device_channel_id=uuid.uuid5(namespace, "disk-to-device"),
            link_channel_ids={
                target_rank: uuid.uuid5(namespace, f"channel:{target_rank}")
                for target_rank in range(nranks)
                if target_rank != rank
            },
        )

    def declare(self, session: QuentSession) -> None:
        context = session.context
        context.device_memory_observer().handle(self.device_memory_id).declared(
            instance_name=f"{self.instance_suffix} device memory",
            worker=self.worker_id,
            limits={"bytes": self.device_memory_bytes},
        )
        context.storage_observer().handle(self.filesystem_id).declared(
            instance_name=f"{self.instance_suffix} filesystem",
            worker=self.worker_id,
        )
        context.thread_pool_observer().handle(self.thread_pool_id).declared(
            instance_name=f"Thread Pool {self.thread_pool_id.hex[:8]}",
            worker=self.worker_id,
        )
        context.data_channel_observer().handle(self.disk_to_device_channel_id).declared(
            instance_name=f"{self.instance_suffix} disk -> device",
            channel_type="disk-to-device",
            worker=self.worker_id,
            source=self.filesystem_id,
            target=self.device_memory_id,
        )
        for target_rank, channel_id in self.link_channel_ids.items():
            context.data_channel_observer().handle(channel_id).declared(
                instance_name=f"rank-{self.rank} -> rank-{target_rank}",
                channel_type="inter-rank",
                worker=self.worker_id,
                source=self.device_memory_id,
                target=uuid.uuid5(
                    uuid.uuid5(self.engine_id, f"worker:{target_rank}"),
                    "device-memory",
                ),
            )


@dataclasses.dataclass(kw_only=True)
class LocalQuentContext:
    """Rank-local generated Quent state."""

    context: QuentContext
    query_id: uuid.UUID
    worker_id: uuid.UUID
    session: QuentSession
    worker_resources: WorkerResources

    def get_or_declare_processor(self, thread_ident: int) -> uuid.UUID:
        return self.worker_resources.processor_registry.get_or_declare_processor(
            self.session, thread_ident, self.worker_resources.thread_pool_id
        )


@dataclasses.dataclass(kw_only=True)
class QuentIRExecutionContext(LocalQuentContext):
    """Rank-local state bound to an Operator and, while running, an Actor."""

    operator_id: uuid.UUID
    actor_id: uuid.UUID | None = None

    @classmethod
    def from_execution_context(
        cls, execution_context: LocalQuentContext, operator_id: uuid.UUID
    ) -> Self:
        return cls(
            operator_id=operator_id,
            context=execution_context.context,
            query_id=execution_context.query_id,
            worker_id=execution_context.worker_id,
            session=execution_context.session,
            worker_resources=execution_context.worker_resources,
        )
