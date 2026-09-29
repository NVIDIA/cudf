# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime state for schema-generated cudf-polars Quent bindings."""

from __future__ import annotations

import ipaddress
import socket
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from os import PathLike

    import cudf_polars_quent as quent_bindings

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


class QuentCollector:
    """Collect events from local or remote sessions into one NDJSON tree."""

    def __init__(
        self,
        output_root: str | PathLike[str],
        *,
        advertised_host: str | None = None,
    ) -> None:
        if _quent is None:
            raise ImportError(
                "Quent tracing requires the cudf-polars Quent extension. "
                "Build python/cudf_polars/quent/bridge with maturin."
            )
        if advertised_host is None:
            advertised_host = _local_ipv4_address()
        bind_host = socket.gethostbyname(advertised_host)
        self._collector = _quent.start_collector(
            _quent.ExporterOptions.ndjson(output_root),
            bind_address=f"{bind_host}:0",
            advertised_host=advertised_host,
        )
        self._closed = False

    @property
    def address(self) -> str:
        """Return the address clients use to connect to this collector."""
        return self._collector.address

    def close(self) -> None:
        """Stop accepting clients and flush collected events."""
        if not self._closed:
            self._collector.close()
            self._closed = True


__all__ = ["QuentCollector", "QuentSession"]
