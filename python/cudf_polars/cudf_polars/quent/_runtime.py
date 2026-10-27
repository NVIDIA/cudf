# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime state for schema-generated cudf-polars Quent bindings."""

from __future__ import annotations

import contextlib
import dataclasses
import ipaddress
import socket
import threading
from typing import TYPE_CHECKING

from cudf_polars.utils.cleanup import run_cleanup_steps

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Iterator, Sequence
    from os import PathLike

    import cudf_polars_quent as quent_bindings

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent._context import (
        ActorBoundQuentIRExecutionState,
        QuentConfig,
        QuentQueryConfig,
        QuentQueryWorkerState,
        WorkerResources,
    )

try:
    import cudf_polars_quent as _quent
except ImportError:  # pragma: no cover - depends on optional extension
    _quent = None  # type: ignore[assignment]


def _dataframe_statistics(frame: DataFrame) -> quent_bindings.DataFrameStatisticsDict:
    """Return typed shape statistics for one dataframe."""
    return {"shape": frame.table.shape()}


def _local_ipv4_address() -> str:
    """Return a non-loopback local address when one is routable."""
    try:
        address = socket.gethostbyname(socket.gethostname())
    except OSError:  # pragma: no cover
        address = "127.0.0.1"
    if not ipaddress.ip_address(address).is_loopback:
        return address  # pragma: no cover
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # UDP connect only consults the route table; it sends no packets.
            sock.connect(("192.0.2.1", 9))
            return str(sock.getsockname()[0])
    except OSError:  # pragma: no cover
        return address


class QuentSession:
    """Own one collector-backed generated context and its active FSM handles."""

    def __init__(self, collector_address: str) -> None:
        if _quent is None:  # pragma: no cover
            raise ImportError(
                "Quent tracing requires the cudf-polars Quent extension. "
                "Build python/cudf_polars/quent/bridge with maturin."
            )
        self._declarations_lock = threading.Lock()
        self._declared: set[tuple[str, uuid.UUID]] = set()
        self._queries: dict[uuid.UUID, quent_bindings.QueryExecutingHandle] = {}
        self._evaluations: dict[uuid.UUID, quent_bindings.EvaluateRunningHandle] = {}
        self._actors: dict[uuid.UUID, quent_bindings.ActorRunningHandle] = {}
        self._binding_context = _quent.Context(
            _quent.ExporterOptions.collector(collector_address)
        )
        self._closed = False

    @property
    def binding_context(self) -> quent_bindings.Context:
        """Return the generated instrumentation context."""
        return self._binding_context

    def declare_once(self, entity_name: str, identifier: uuid.UUID) -> bool:
        """Claim one declaration for an entity in this session."""
        key = (entity_name, identifier)
        with self._declarations_lock:
            if key in self._declared:
                return False
            self._declared.add(key)
            return True

    def close(self) -> None:
        """Close active handles and flush generated events to the collector."""
        if self._closed:
            return
        self._closed = True
        self._queries.clear()
        self._evaluations.clear()
        self._actors.clear()
        self._binding_context.close()


@dataclasses.dataclass
class QuentControllerRuntime:
    """
    Own controller-side Engine, Query, session, and Collector state.

    This is used in the "controller" process (e.g. Dask / Ray Client, or the SPMD engine's rank 0).
    Compare with :class:`QuentWorkerRuntime`, which is used in the "worker" process.
    """

    config: QuentConfig
    session: QuentSession
    collector: quent_bindings.Collector | None = None
    _engine_handle: quent_bindings.EngineInitHandle | None = None

    @classmethod
    def create(
        cls,
        config: QuentConfig,
        *,
        backend: str,
    ) -> QuentControllerRuntime:
        """Start the Collector and create its controller runtime."""
        with contextlib.ExitStack() as cleanup:
            collector = start_collector(config.run_root)
            cleanup.callback(lambda: collector.close(timeout=10.0))
            collector_address = collector.address
            if collector_address is None:  # pragma: no cover - Collector invariant
                raise RuntimeError("Quent Collector did not advertise an address")
            session = QuentSession(collector_address)
            cleanup.callback(session.close)
            engine_handle = (
                session.binding_context.engine_observer()
                .handle(config.engine_id)
                .init(
                    instance_name=f"cudf-polars-{str(config.engine_id)[:8]}",
                    implementation={
                        "name": config.implementation_name,
                        "version": config.implementation_version,
                        "backend": backend,
                        "custom_attributes": {"backend": backend},
                    },
                )
            )
            runtime = cls(
                config=config,
                session=session,
                collector=collector,
                _engine_handle=engine_handle,
            )
            cleanup.pop_all()
            return runtime

    @property
    def collector_address(self) -> str:
        """Return the live Collector's advertised address."""
        if self.collector is None:  # pragma: no cover - caller lifecycle invariant
            raise RuntimeError("Quent controller runtime is closed")
        address = self.collector.address
        if address is None:  # pragma: no cover - Collector invariant
            raise RuntimeError("Quent Collector did not advertise an address")
        return address

    @contextlib.contextmanager
    def query(
        self,
        query_id: uuid.UUID,
        *,
        query_config: QuentQueryConfig,
        emit: bool = True,
    ) -> Iterator[None]:
        """Emit one controller-side Query lifecycle."""
        if not emit:  # pragma: no cover
            yield
            return
        if self._engine_handle is None:  # pragma: no cover
            raise RuntimeError("Quent controller runtime is not initialized")
        if self.session.declare_once("QueryGroup", query_config.query_group_id):
            self.session.binding_context.query_group_observer().handle(
                query_config.query_group_id
            ).declared(
                instance_name=query_config.query_group_name,
                engine=self.config.engine_id,
            )
        initialized = (
            self.session.binding_context.query_observer()
            .handle(query_id)
            .initialized(
                instance_name=query_config.query_name or query_id.hex[:8],
                query_group=query_config.query_group_id,
            )
        )
        self.session._queries[query_id] = initialized.planning().executing()
        try:
            yield
        except BaseException as error:
            self.session._queries.pop(query_id).failed(error=str(error))
            raise
        else:
            self.session._queries.pop(query_id).completed()

    def close(self) -> None:
        """Close the Engine, session, and Collector in dependency order."""
        engine_handle = self._engine_handle
        collector = self.collector
        self._engine_handle = None
        self.collector = None
        steps: list[Callable[[], object]] = []
        if engine_handle is not None:
            steps.append(engine_handle.exit)
        steps.append(self.session.close)
        if collector is not None:
            steps.append(lambda: collector.close(timeout=10.0))
        run_cleanup_steps("Quent controller shutdown failed", *steps)


@dataclasses.dataclass
class QuentWorkerRuntime:
    """Own worker-side Worker, resource, plan, Actor, and Evaluate state."""

    config: QuentConfig
    session: QuentSession
    worker_resources: WorkerResources
    _worker_handle: quent_bindings.WorkerInitHandle | None

    @classmethod
    def create(
        cls,
        config: QuentConfig,
        collector_address: str,
        *,
        worker_id: uuid.UUID,
        rank: int,
        nranks: int,
        instance_name: str,
    ) -> QuentWorkerRuntime:
        """Create a worker runtime and declare its resources."""
        from cudf_polars.quent._context import WorkerResources

        session = QuentSession(collector_address)
        worker_handle: quent_bindings.WorkerInitHandle | None = None
        try:
            worker_handle = (
                session.binding_context.worker_observer()
                .handle(worker_id)
                .init(instance_name=instance_name, engine=config.engine_id)
            )
            resources = WorkerResources.build(
                instance_suffix=instance_name,
                engine_id=config.engine_id,
                worker_id=worker_id,
                rank=rank,
                nranks=nranks,
            )
            resources.declare(session)
        except BaseException:  # pragma: no cover
            if worker_handle is not None:
                with contextlib.suppress(Exception):
                    worker_handle.exit()
            with contextlib.suppress(Exception):
                session.close()
            raise
        return cls(
            config=config,
            session=session,
            worker_resources=resources,
            _worker_handle=worker_handle,
        )

    def query_worker_state(self, query_id: uuid.UUID) -> QuentQueryWorkerState:
        """Build state for one query executing on this worker."""
        from cudf_polars.quent._context import QuentQueryWorkerState

        return QuentQueryWorkerState(runtime=self, query_id=query_id)

    def emit_physical_plan(
        self,
        state: QuentQueryWorkerState,
        ir: IR,
        plan_id: uuid.UUID,
        *,
        parent_plan_id: uuid.UUID,
        node_map: dict[str, list[str]],
        logical_op_by_id: dict[str, uuid.UUID],
    ) -> dict[str, uuid.UUID]:
        """Emit a physical plan and return stable-node to operator IDs."""
        from cudf_polars.quent._plan import build_parent_operators_map, emit_plan

        parent_operators = build_parent_operators_map(node_map, logical_op_by_id)
        return emit_plan(
            self.session,
            ir,
            query_id=state.query_id,
            plan_id=plan_id,
            worker_id=self.worker_resources.worker_id,
            instance_name="physical",
            parent_plan_id=parent_plan_id,
            parent_operators_by_node_id=parent_operators,
        )

    def emit_evaluate_begin(
        self,
        ir_type: type[IR],
        evaluate_id: uuid.UUID,
        instance_name: str,
        state: ActorBoundQuentIRExecutionState,
        frames: Sequence[DataFrame],
    ) -> None:
        """Emit Evaluate queued/running events."""
        input_frames_bytes = sum(frame._size_bytes for frame in frames)
        processor_id = state.query_worker_state.get_or_declare_processor(
            threading.get_ident()
        )
        channel_bytes = (
            state.io_bytes
            if state.io_bytes is not None
            else input_frames_bytes
            if ir_type.is_io_node
            else None
        )
        input_attributes: quent_bindings.EvaluateInputDict = {
            "dataframes": [_dataframe_statistics(frame) for frame in frames],
            "sequence_number": state.sequence_number,
            "content_sizes": state.content_sizes,
            "spillable": state.spillable,
        }
        processor: quent_bindings.ProcessorUsageRefDict = {
            "target": processor_id,
            "data": {},
        }
        channel: quent_bindings.DataChannelUsageRefDict | None = (
            {
                "target": self.worker_resources.disk_to_device_channel_id,
                "data": {"bytes": channel_bytes},
            }
            if channel_bytes is not None
            else None
        )
        task: quent_bindings.EvaluateTaskDict | None = (
            {
                "node_id": state.scan_task_node_id,
                "node_type": state.scan_task_node_type,
            }
            if state.scan_task_node_id is not None
            and state.scan_task_node_type is not None
            else None
        )
        queued = (
            self.session.binding_context.evaluate_observer()
            .handle(evaluate_id)
            .queued(instance_name=instance_name, actor=state.actor_id, task=task)
        )
        self.session._evaluations[evaluate_id] = queued.running(
            io=ir_type.is_io_node,
            input_bytes=input_frames_bytes,
            input=input_attributes,
            processor=processor,
            channel=channel,
        )

    def emit_evaluate_end(
        self,
        evaluate_id: uuid.UUID,
        result: DataFrame | None,
        error: BaseException | None,
    ) -> None:
        """Emit an Evaluate terminal event."""
        if error is not None:
            self.session._evaluations.pop(evaluate_id).failed(error=str(error))
        else:
            assert result is not None
            self.session._evaluations.pop(evaluate_id).completed(
                output_bytes=result._size_bytes,
                output_dataframe=_dataframe_statistics(result),
            )

    def close(self) -> None:
        """Close the Worker and its process-local Collector client."""
        worker_handle = self._worker_handle
        self._worker_handle = None
        steps: list[Callable[[], object]] = []
        if worker_handle is not None:
            steps.append(worker_handle.exit)
        steps.append(self.session.close)
        run_cleanup_steps("Quent worker shutdown failed", *steps)


def start_collector(
    output_root: str | PathLike[str], *, advertised_host: str | None = None
) -> quent_bindings.Collector:
    """Start a collector that writes events to an NDJSON tree."""
    if _quent is None:  # pragma: no cover
        raise ImportError(
            "Quent tracing requires the cudf-polars Quent extension. "
            "Build python/cudf_polars/quent/bridge with maturin."
        )
    if advertised_host is None:
        advertised_host = _local_ipv4_address()
    bind_host = socket.gethostbyname(advertised_host)
    return _quent.start_collector(
        _quent.ExporterOptions.ndjson(output_root),
        bind_address=f"{bind_host}:0",
        advertised_host=advertised_host,
    )


__all__ = [
    "QuentControllerRuntime",
    "QuentSession",
    "QuentWorkerRuntime",
    "start_collector",
]
