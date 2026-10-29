# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common utilities for collective operations."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Literal

from rapidsmpf.shuffler import Shuffler

from cudf_polars.dsl.ir import (
    CallbackSink,
    Distinct,
    GroupBy,
    Rolling,
    Sort,
    Union,
)
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.filter_hint import PushdownFilterHint
from cudf_polars.streaming.hint_sorted import HintSorted
from cudf_polars.streaming.io import StreamingScan, StreamingSink
from cudf_polars.streaming.join import Join
from cudf_polars.streaming.over import Over
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.rolling import FixedSizeRolling
from cudf_polars.streaming.shuffle import Shuffle

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from cudf_polars.dsl.ir import IR
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


class _CollectiveIDPool:
    """Pool of vacant collective IDs."""

    def __init__(self) -> None:
        # Since 3.7, dict.popitem() is guaranteed LIFO.
        self.vacancy: dict[int, None] = dict.fromkeys(
            range(Shuffler.max_concurrent_shuffles)
        )
        self.lock = threading.Lock()

    def acquire_unsafe(self) -> int:
        # Not thread safe, must be called with self.lock held
        try:
            # All ranks must choose the same collective IDs during lowering.
            collective_id, _ = self.vacancy.popitem()
        except KeyError:
            raise ValueError(
                f"Cannot shuffle more than {Shuffler.max_concurrent_shuffles} "
                "times in a single query."
            ) from None
        return collective_id

    def release_unsafe(self, collective_id: int) -> None:
        """Release a collective ID back to the vacancy set."""
        # Not thread safe, must be called with self.lock held
        if collective_id in self.vacancy:
            raise ValueError("Restoring ID that exists")
        self.vacancy[collective_id] = None


# Collective IDs are only meaningful relative to a communicator, and all ranks
# must agree on the IDs chosen. A process-global pool breaks that agreement as
# soon as the same process takes part in queries on different communicators,
# so the pool is scoped to the current context (see ``collective_id_scope``).
_collective_id_pool: ContextVar[_CollectiveIDPool] = ContextVar(
    "cudf_polars_collective_id_pool"
)


def _current_pool() -> _CollectiveIDPool:
    """Return the collective ID pool of the current context, creating it if unset."""
    try:
        return _collective_id_pool.get()
    except LookupError:
        pool = _CollectiveIDPool()
        _collective_id_pool.set(pool)
        return pool


@contextmanager
def collective_id_scope() -> Iterator[None]:
    """
    Use a fresh collective ID pool for the duration of the block.

    Every query should run in its own scope so that the collective IDs it
    reserves depend only on the query itself, and not on collectives that
    other queries in the same process (possibly on other communicators)
    currently hold.
    """
    token = _collective_id_pool.set(_CollectiveIDPool())
    try:
        yield
    finally:
        _collective_id_pool.reset(token)


class ReserveOpIDs:
    """
    Context manager to reserve collective IDs for pipeline execution.

    Parameters
    ----------
    ir : IR
        The root IR node of the pipeline.
    config_options : ConfigOptions, optional
        Configuration options (needed for dynamic planning).

    Notes
    -----
    This context manager:
    1. Identifies all IR nodes that may require collective operations
    2. Reserves collective IDs from the current context's pool
       (see :func:`collective_id_scope`)
    3. Creates a mapping from IR nodes to their reserved IDs
    4. Releases all IDs back to the pool on __exit__

    Each IR node may require multiple collective operation IDs
    (e.g., for metadata gathering, shuffling multiple sides of a join).
    """

    def __init__(
        self, ir: IR, config_options: ConfigOptions[StreamingExecutor] | None = None
    ):
        self.config_options = config_options

        # Check if dynamic planning is enabled
        self.dynamic_planning_enabled = (
            config_options is not None
            and config_options.executor.dynamic_planning is not None
        )

        # Find all collective IR nodes.
        collective_types: tuple[type, ...] = (
            CallbackSink,
            Shuffle,
            Join,
            HintSorted,
            Repartition,
            StreamingSink,
            StreamingScan,
            Sort,
            Rolling,
            FixedSizeRolling,
            Union,
        )
        if self.dynamic_planning_enabled:
            collective_types = (
                CallbackSink,
                Shuffle,
                Join,
                HintSorted,
                Repartition,
                StreamingSink,
                StreamingScan,
                Sort,
                Rolling,
                FixedSizeRolling,
                Union,
                GroupBy,
                Distinct,
                Over,
                PushdownFilterHint,
            )

        self.collective_nodes: list[IR] = [
            node for node in traversal([ir]) if isinstance(node, collective_types)
        ]
        self.collective_id_map: dict[IR, list[int]] = {}
        self._pool: _CollectiveIDPool | None = None

    def __enter__(self) -> dict[IR, list[int]]:
        """
        Reserve collective IDs and return the mapping.

        Returns
        -------
        collective_id_map : dict[IR, list[int]]
            Mapping from IR nodes to their reserved collective IDs.
            Each IR node gets a list of IDs to support multiple
            collective operations per node.
        """
        # Reserve IDs and map nodes to a list of IDs
        self._pool = pool = _current_pool()
        with pool.lock:
            for node in self.collective_nodes:
                if (
                    isinstance(node, (GroupBy, Distinct))
                    and self.dynamic_planning_enabled
                ):
                    # GroupBy/Distinct need 2 IDs: one for size allgather, one for shuffle
                    self.collective_id_map[node] = [
                        pool.acquire_unsafe(),
                        pool.acquire_unsafe(),
                    ]
                elif isinstance(node, Join) and self.dynamic_planning_enabled:
                    # Join needs 3 IDs: allgather, left shuffle, right shuffle.
                    self.collective_id_map[node] = [
                        pool.acquire_unsafe(),
                        pool.acquire_unsafe(),
                        pool.acquire_unsafe(),
                    ]
                elif isinstance(node, Sort):
                    if self.dynamic_planning_enabled:
                        # 3 IDs: size-estimate allgather, boundary allgather, shuffle
                        self.collective_id_map[node] = [
                            pool.acquire_unsafe(),
                            pool.acquire_unsafe(),
                            pool.acquire_unsafe(),
                        ]
                    else:
                        # 2 IDs: boundary allgather, shuffle
                        self.collective_id_map[node] = [
                            pool.acquire_unsafe(),
                            pool.acquire_unsafe(),
                        ]
                elif isinstance(node, Union):
                    # Union needs one ID per child. Only an order-preserving
                    # multi-rank union uses them: ID 0 for the chunk-count
                    # AllGather and then child 0's exchange (the AllGather
                    # completes before the exchange starts, so they can
                    # share), and ID i for child i's exchange.
                    self.collective_id_map[node] = [
                        pool.acquire_unsafe() for _ in node.children
                    ]
                elif isinstance(node, Over) and not node.is_scalar:
                    # Non-scalar Over needs 2 IDs: one for the size AllGather +
                    # forward shuffle (the AllGather completes before the forward
                    # shuffle starts, so they can share), and a separate ID for
                    # the return shuffle (which overlaps with the forward shuffle
                    # during extract+insert).
                    self.collective_id_map[node] = [
                        pool.acquire_unsafe(),
                        pool.acquire_unsafe(),
                    ]
                else:
                    self.collective_id_map[node] = [pool.acquire_unsafe()]

        return self.collective_id_map

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Release all reserved collective IDs back to the vacancy pool."""
        pool = self._pool
        assert pool is not None
        with pool.lock:
            for collective_ids in self.collective_id_map.values():
                for collective_id in collective_ids:
                    pool.release_unsafe(collective_id)
        return False


@contextmanager
def reserve_op_id() -> Iterator[int]:
    """
    Reserve a single collective operation ID.

    This function and the ID it yields must only be used **outside** of a
    ``run_actor_graph`` call. It is intended for SPMD mode, where operations
    such as gathering results across ranks are performed directly rather than
    through the actor graph. The contained block _must_ wait for completion of the collective.

    Yields
    ------
    collective_id : int
        A vacant collective ID reserved from the current context's pool
        (see :func:`collective_id_scope`).
    """
    pool = _current_pool()
    with pool.lock:
        collective_id = pool.acquire_unsafe()
    try:
        yield collective_id
    finally:
        with pool.lock:
            pool.release_unsafe(collective_id)
