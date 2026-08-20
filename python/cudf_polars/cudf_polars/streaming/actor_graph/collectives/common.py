# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common utilities for collective operations."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from rapidsmpf.shuffler import Shuffler

from cudf_polars.dsl.ir import Distinct, GroupBy, Sort
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.io import StreamingSink
from cudf_polars.streaming.join import Join
from cudf_polars.streaming.over import Over
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.shuffle import Shuffle

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from cudf_polars.dsl.ir import IR
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


# Set of available collective IDs
_collective_id_vacancy: dict[int, None] = dict.fromkeys(
    range(Shuffler.max_concurrent_shuffles)
)
_collective_id_vacancy_lock: threading.Lock = threading.Lock()


def _get_new_collective_id_unsafe() -> int:
    # Not thread safe, must be called with _collective_id_vacancy_lock held
    try:
        # All ranks must choose the same collective IDs during lowering.
        # Since 3.7, dict.popitem() is guaranteed LIFO.
        collective_id, _ = _collective_id_vacancy.popitem()
    except KeyError:
        raise ValueError(
            f"Cannot shuffle more than {Shuffler.max_concurrent_shuffles} "
            "times in a single query."
        ) from None
    return collective_id


def _release_collective_id_unsafe(collective_id: int) -> None:
    """Release a collective ID back to the vacancy set."""
    # Not thread safe, must be called with _collective_id_vacancy_lock held
    if collective_id in _collective_id_vacancy:
        raise ValueError("Restoring ID that exists")
    _collective_id_vacancy[collective_id] = None


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
    2. Reserves collective IDs from the vacancy pool
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
            Shuffle,
            Join,
            Repartition,
            StreamingSink,
            Sort,
        )
        if self.dynamic_planning_enabled:
            collective_types = (
                Shuffle,
                Join,
                Repartition,
                StreamingSink,
                Sort,
                GroupBy,
                Distinct,
                Over,
            )

        self.collective_nodes: list[IR] = [
            node for node in traversal([ir]) if isinstance(node, collective_types)
        ]
        self.collective_id_map: dict[IR, list[int]] = {}

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
        with _collective_id_vacancy_lock:
            for node in self.collective_nodes:
                if (
                    isinstance(node, (GroupBy, Distinct))
                    and self.dynamic_planning_enabled
                ):
                    # GroupBy/Distinct need 2 IDs: one for size allgather, one for shuffle
                    self.collective_id_map[node] = [
                        _get_new_collective_id_unsafe(),
                        _get_new_collective_id_unsafe(),
                    ]
                elif isinstance(node, Join) and self.dynamic_planning_enabled:
                    # Join needs 4 IDs: allgather, left shuffle, right shuffle,
                    # and bloom filter.
                    self.collective_id_map[node] = [
                        _get_new_collective_id_unsafe(),
                        _get_new_collective_id_unsafe(),
                        _get_new_collective_id_unsafe(),
                        _get_new_collective_id_unsafe(),
                    ]
                elif isinstance(node, Sort):
                    if self.dynamic_planning_enabled:
                        # 3 IDs: size-estimate allgather, boundary allgather, shuffle
                        self.collective_id_map[node] = [
                            _get_new_collective_id_unsafe(),
                            _get_new_collective_id_unsafe(),
                            _get_new_collective_id_unsafe(),
                        ]
                    else:
                        # 2 IDs: boundary allgather, shuffle
                        self.collective_id_map[node] = [
                            _get_new_collective_id_unsafe(),
                            _get_new_collective_id_unsafe(),
                        ]
                elif isinstance(node, Over) and not node.is_scalar:
                    # Non-scalar Over needs 2 IDs: one for the size AllGather +
                    # forward shuffle (the AllGather completes before the forward
                    # shuffle starts, so they can share), and a separate ID for
                    # the return shuffle (which overlaps with the forward shuffle
                    # during extract+insert).
                    self.collective_id_map[node] = [
                        _get_new_collective_id_unsafe(),
                        _get_new_collective_id_unsafe(),
                    ]
                else:
                    self.collective_id_map[node] = [_get_new_collective_id_unsafe()]

        return self.collective_id_map

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Release all reserved collective IDs back to the vacancy pool."""
        with _collective_id_vacancy_lock:
            for collective_ids in self.collective_id_map.values():
                for collective_id in collective_ids:
                    _release_collective_id_unsafe(collective_id)
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
        A vacant collective ID reserved from the global vacancy pool.
    """
    with _collective_id_vacancy_lock:
        collective_id = _get_new_collective_id_unsafe()
    try:
        yield collective_id
    finally:
        with _collective_id_vacancy_lock:
            _release_collective_id_unsafe(collective_id)
