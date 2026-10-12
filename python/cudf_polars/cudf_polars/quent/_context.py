# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and execution state for Quent tracing."""

from __future__ import annotations

import dataclasses
import json
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from cudf_polars import __version__
from cudf_polars.utils.config import (
    get_total_device_memory,
    resolve_quent_output_root,
)

if TYPE_CHECKING:
    from typing import Self

    from cudf_polars.quent._runtime import QuentSession, QuentWorkerRuntime

__all__ = [
    "QuentConfig",
    "QuentIRExecutionState",
    "QuentQueryConfig",
    "QuentQueryWorkerState",
    "WorkerResources",
]


class _ProcessorRegistry:
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

        session.binding_context.processor_observer().handle(processor_id).declared(
            instance_name=f"Thread {processor_id.hex[:8]}",
            thread_pool=pool_id,
        )
        return processor_id


@dataclasses.dataclass(frozen=True, kw_only=True)
class QuentQueryConfig:
    """User-facing metadata that may vary from one query to the next."""

    query_group_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    query_group_name: str | None = None
    query_name: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class QuentConfig:
    """Serializable engine configuration shared by all ranks."""

    engine_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    implementation_name: str = "cudf-polars"
    implementation_version: str = __version__
    output_root: str = dataclasses.field(default_factory=resolve_quent_output_root)
    query: QuentQueryConfig = dataclasses.field(default_factory=QuentQueryConfig)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_root", resolve_quent_output_root(self.output_root)
        )

    @property
    def run_root(self) -> Path:
        """Return this engine's controller-local Collector output directory."""
        return Path(self.output_root) / str(self.engine_id)

    def _serialize(self) -> bytes:  # TODO: coverage
        payload = {
            **dataclasses.asdict(self),
            "engine_id": int(self.engine_id),
            "query": {
                **dataclasses.asdict(self.query),
                "query_group_id": int(self.query.query_group_id),
            },
        }
        return json.dumps(payload).encode()

    @classmethod
    def _deserialize(cls, data: bytes) -> Self:  # TODO: coverage
        payload = json.loads(data)
        return cls(
            engine_id=uuid.UUID(int=int(payload["engine_id"])),
            implementation_name=payload["implementation_name"],
            implementation_version=payload["implementation_version"],
            output_root=payload["output_root"],
            query=QuentQueryConfig(
                query_group_id=uuid.UUID(int=int(payload["query"]["query_group_id"])),
                query_group_name=payload["query"]["query_group_name"],
                query_name=payload["query"]["query_name"],
            ),
        )


@dataclasses.dataclass(kw_only=True)
class WorkerResources:
    """Per-worker resource identities and generated declarations."""

    engine_id: uuid.UUID
    worker_id: uuid.UUID
    rank: int
    instance_suffix: str
    thread_pool_id: uuid.UUID
    processor_registry: _ProcessorRegistry
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
            processor_registry=_ProcessorRegistry(),
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
        context = session.binding_context
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
            context.data_channel_observer().handle(
                channel_id
            ).declared(  # TODO: coverage
                instance_name=f"rank-{self.rank} -> rank-{target_rank}",
                channel_type="inter-rank",
                worker=self.worker_id,
                source=self.device_memory_id,
                target=uuid.uuid5(
                    uuid.uuid5(self.engine_id, f"worker:{target_rank}"),
                    "device-memory",
                ),
            )


@dataclasses.dataclass(frozen=True, kw_only=True)
class QuentQueryWorkerState:
    """State for one query executing on one worker."""

    runtime: QuentWorkerRuntime
    query_id: uuid.UUID

    def get_or_declare_processor(self, thread_ident: int) -> uuid.UUID:
        resources = self.runtime.worker_resources
        return resources.processor_registry.get_or_declare_processor(
            self.runtime.session, thread_ident, resources.thread_pool_id
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class QuentIRExecutionState:
    """Query-worker state bound to an Operator and, while running, an Actor."""

    query_worker_state: QuentQueryWorkerState
    operator_id: uuid.UUID
    actor_id: uuid.UUID | None = None

    @classmethod
    def from_query_worker_state(
        cls, query_worker_state: QuentQueryWorkerState, operator_id: uuid.UUID
    ) -> Self:
        return cls(
            query_worker_state=query_worker_state,
            operator_id=operator_id,
        )
