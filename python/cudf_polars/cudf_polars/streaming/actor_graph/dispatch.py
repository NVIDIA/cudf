# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatching for the RapidsMPF streaming runtime."""

from __future__ import annotations

import dataclasses
from functools import singledispatch
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, TypedDict

from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context

    import cudf_polars.quent._context
    import cudf_polars.quent._types
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.utils import ChannelManager
    from cudf_polars.streaming.base import (
        PartitionInfo,
        StatsCollector,
    )
    from cudf_polars.utils.config import (
        ConfigOptions,
        MaxConcurrentIOTasks,
        StreamingExecutor,
    )


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
        The local and remote IO task limits to use for scan nodes.
    stats
        Statistics collector.
    collective_id_map
        The mapping of IR nodes to lists of collective IDs.
    quent_operator_map
        Mapping from IR nodes to physical-plan Quent operators.
    quent_execution_context
        Rank-local Quent execution context.
    """

    context: Context
    comm: Communicator
    config_options: ConfigOptions[StreamingExecutor]
    partition_info: MutableMapping[IR, PartitionInfo]
    fanout_nodes: dict[IR, FanoutInfo]
    ir_context: IRExecutionContext
    max_concurrent_io_tasks: MaxConcurrentIOTasks
    stats: StatsCollector
    collective_id_map: dict[IR, list[int]]
    quent_operator_map: dict[IR, cudf_polars.quent._types.Operator] | None
    quent_execution_context: cudf_polars.quent._context.LocalQuentContext | None


def ir_context_for_node(rec: SubNetGenerator, ir: IR) -> IRExecutionContext:
    """
    Return ``ir_context`` with the physical Quent operator bound when tracing.

    Parameters
    ----------
    rec
        The recursive SubNetGenerator callable.
    ir
        The IR node to return the execution context for.

    Returns
    -------
    ir_context
        A clone of rec.state["ir_context"] with ``quent_ir_execution_context``
        bound to the physical Quent operator for the given IR node.
    """
    import cudf_polars.quent._context

    ir_context = rec.state["ir_context"]
    quent_operator_map = rec.state["quent_operator_map"]
    quent_execution_context = rec.state["quent_execution_context"]
    if quent_operator_map is not None and quent_execution_context is not None:
        quent_operator = quent_operator_map[ir]
        return dataclasses.replace(
            ir_context,
            quent_ir_execution_context=cudf_polars.quent._context.QuentIRExecutionContext.from_execution_context(
                execution_context=quent_execution_context,
                quent_operator=quent_operator,
            ),
        )
    return ir_context


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
