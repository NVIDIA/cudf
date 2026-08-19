# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatching for the RapidsMPF streaming runtime."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, TypedDict

from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.utils import ChannelManager
    from cudf_polars.streaming.base import (
        PartitionInfo,
        StatsCollector,
    )
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


class FanoutInfo(NamedTuple):
    """A named tuple representing fanout information."""

    num_consumers: int
    """The number of consumers."""
    unbounded: bool
    """Whether the node needs unbounded fanout."""


class GenState(TypedDict):
    """
    State used for generating a streaming sub-network.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator the generation is collective over
    config_options
        GPUEngine configuration options.
    partition_info
        Partition information.
    fanout_nodes
        Dictionary mapping IR nodes to fanout information.
    ir_context
        The execution context for the IR node.
    max_concurrent_io_tasks
        The maximum number of concurrent IO tasks to use for a single IO node.
    stats
        Statistics collector.
    collective_id_map
        The mapping of IR nodes to lists of collective IDs.
    """

    context: Context
    comm: Communicator
    config_options: ConfigOptions[StreamingExecutor]
    partition_info: MutableMapping[IR, PartitionInfo]
    fanout_nodes: dict[IR, FanoutInfo]
    ir_context: IRExecutionContext
    max_concurrent_io_tasks: int
    stats: StatsCollector
    collective_id_map: dict[IR, list[int]]


SubNetGenerator: TypeAlias = GenericTransformer[
    "IR", "tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]", GenState
]
"""Protocol for Generating a streaming sub-network."""


@singledispatch
def generate_ir_sub_network(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """
    Generate a sub-network for the RapidsMPF streaming runtime.

    Parameters
    ----------
    ir
        IR node to generate tasks for.
    rec
        Recursive SubNetGenerator callable.

    Returns
    -------
    nodes
        Dictionary mapping each IR node to its list of streaming-network node(s).
    channels
        Dictionary mapping between each IR node and its
        corresponding output ChannelManager object.
    """
    raise AssertionError(f"Unhandled type {type(ir)}")  # pragma: no cover
