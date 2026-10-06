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

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterator
    from os import PathLike

    import cudf_polars_quent as quent_bindings

    from cudf_polars.quent._context import (
        LocalQuentContext,
        QuentContext,
        WorkerResources,
    )

try:
    import cudf_polars_quent as _quent
except ImportError:  # pragma: no cover - depends on optional extension
    _quent = None  # type: ignore[assignment]


def _local_ipv4_address() -> str:
    """Return a non-loopback local address when one is routable."""
    try:
        address = socket.gethostbyname(socket.gethostname())
    except OSError:
        address = "127.0.0.1"
    if not ipaddress.ip_address(address).is_loopback:
        return address
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # UDP connect only consults the route table; it sends no packets.
            sock.connect(("192.0.2.1", 9))
            return str(sock.getsockname()[0])
    except OSError:
        return address


class QuentSession:
    """Own one collector-backed generated context and its active FSM handles."""

    def __init__(self, collector_address: str) -> None:
        if _quent is None:
            raise ImportError(
                "Quent tracing requires the cudf-polars Quent extension. "
                "Build python/cudf_polars/quent/bridge with maturin."
            )
        self._declarations_lock = threading.Lock()
        self._declared: set[tuple[str, uuid.UUID]] = set()
        self._engines: dict[uuid.UUID, quent_bindings.EngineInitHandle] = {}
        self._workers: dict[uuid.UUID, quent_bindings.WorkerInitHandle] = {}
        self._queries: dict[uuid.UUID, quent_bindings.QueryExecutingHandle] = {}
        self._evaluations: dict[uuid.UUID, quent_bindings.EvaluateRunningHandle] = {}
        self._actors: dict[uuid.UUID, quent_bindings.ActorRunningHandle] = {}
        self._context = _quent.Context(
            _quent.ExporterOptions.collector(collector_address)
        )
        self._closed = False

    @property
    def context(self) -> quent_bindings.Context:
        """Return the generated instrumentation context."""
        return self._context

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
        self._engines.clear()
        self._workers.clear()
        self._queries.clear()
        self._evaluations.clear()
        self._actors.clear()
        self._context.close()
        self._closed = True


@dataclasses.dataclass
class QuentRuntime:
    """
    Own the Quent resources used by one process.

    A runtime may represent a controller, a worker, or both. Controllers emit
    Engine and Query lifecycle events. Workers emit Worker, Plan, Operator,
    Actor, and Evaluate events.
    """

    context: QuentContext
    session: QuentSession
    collector: quent_bindings.Collector | None = None
    worker_id: uuid.UUID | None = None
    worker_resources: WorkerResources | None = None
    _engine_active: bool = False
    _worker_active: bool = False
    _session_closed: bool = False
    _collector_closed: bool = False

    @classmethod
    def create(
        cls,
        context: QuentContext,
        collector_address: str,
        *,
        backend: str | None = None,
        collector: quent_bindings.Collector | None = None,
        worker_id: uuid.UUID | None = None,
        rank: int | None = None,
        nranks: int | None = None,
        instance_name: str | None = None,
    ) -> QuentRuntime:
        """Create and register the requested process-local Quent roles."""
        runtime = cls(
            context=context,
            session=QuentSession(collector_address),
            collector=collector,
        )
        if backend is not None:
            runtime.start_engine(backend)
        if worker_id is not None:
            if rank is None or nranks is None or instance_name is None:
                raise ValueError(
                    "rank, nranks, and instance_name are required for a worker runtime"
                )
            runtime.start_worker(
                worker_id=worker_id,
                rank=rank,
                nranks=nranks,
                instance_name=instance_name,
            )
        return runtime

    def start_engine(self, backend: str) -> None:
        """Register the engine role for this process."""
        if self._engine_active:
            return
        self.context._emit_engine_init_events(self.session, backend=backend)
        self._engine_active = True

    def start_worker(
        self,
        *,
        worker_id: uuid.UUID,
        rank: int,
        nranks: int,
        instance_name: str,
    ) -> None:
        """Register a worker and declare its engine-scoped resources."""
        if self._worker_active:
            return
        from cudf_polars.quent._context import WorkerResources

        self.session._workers[worker_id] = (
            self.session.context.worker_observer()
            .handle(worker_id)
            .init(instance_name=instance_name, engine=self.context.engine_id)
        )
        resources = WorkerResources.build(
            instance_suffix=instance_name,
            engine_id=self.context.engine_id,
            worker_id=worker_id,
            rank=rank,
            nranks=nranks,
        )
        resources.declare(self.session)
        self.worker_id = worker_id
        self.worker_resources = resources
        self._worker_active = True

    def local_context(
        self, query_id: uuid.UUID, *, context: QuentContext | None = None
    ) -> LocalQuentContext:
        """Build the per-query context for rank-local execution."""
        from cudf_polars.quent._context import LocalQuentContext

        if (
            not self._worker_active
            or self.worker_id is None
            or self.worker_resources is None
        ):
            raise RuntimeError("Quent worker runtime is not initialized")
        context = context or self.context
        return LocalQuentContext(
            context=context,
            query_id=query_id,
            worker_id=self.worker_id,
            session=self.session,
            worker_resources=self.worker_resources,
        )

    @contextlib.contextmanager
    def query(
        self,
        query_id: uuid.UUID,
        *,
        context: QuentContext,
        emit: bool = True,
    ) -> Iterator[None]:
        """Emit one controller-side Query lifecycle using the current context."""
        if not emit:
            yield
            return
        if not self._engine_active:
            raise RuntimeError("Quent controller runtime is not initialized")
        context._emit_query_group_events(self.session)
        context._emit_query_events(self.session, query_id)
        try:
            yield
        except BaseException as error:
            context._emit_query_failed_event(self.session, query_id, error)
            raise
        else:
            context._emit_query_completed_event(self.session, query_id)

    def close_worker(self) -> None:
        """Emit Worker exit, if this runtime owns an active worker."""
        if not self._worker_active:
            return
        assert self.worker_id is not None
        self.session._workers.pop(self.worker_id).exit()
        self._worker_active = False

    def close_engine(self) -> None:
        """Emit Engine exit, if this runtime owns an active engine."""
        if not self._engine_active:
            return
        self.context._emit_engine_exit_events(self.session)
        self._engine_active = False

    def close_session(self) -> None:
        """Close the process-local collector client."""
        if self._session_closed:
            return
        self.close_worker()
        self.close_engine()
        self.session.close()
        self._session_closed = True

    def close_collector(self) -> None:
        """Close the controller-local Collector, if present."""
        if self.collector is None or self._collector_closed:
            return
        self.collector.close()
        self._collector_closed = True

    def close(self) -> None:
        """Close worker, engine, session, and Collector state in order."""
        self.close_session()
        self.close_collector()


def start_collector(
    output_root: str | PathLike[str], *, advertised_host: str | None = None
) -> quent_bindings.Collector:
    """Start a collector that writes events to an NDJSON tree."""
    if _quent is None:
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


__all__ = ["QuentRuntime", "QuentSession", "start_collector"]
