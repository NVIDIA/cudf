# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core RapidsMPF streaming-engine API."""

from __future__ import annotations

import dataclasses
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.leaf_actor import pull_from_channel

import cudf_polars.dsl.tracing
from cudf_polars.dsl.ir import (
    Join,
    Union,
)
from cudf_polars.dsl.traversal import CachingVisitor, traversal
from cudf_polars.streaming.actor_graph.dispatch import FanoutInfo
from cudf_polars.streaming.actor_graph.nodes import (
    generate_ir_sub_network_wrapper,
    metadata_drain_node,
)
from cudf_polars.streaming.over import Over
from cudf_polars.utils.config import SPMDContext

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    import polars as pl

    from cudf_streaming.channel_metadata import ChannelMetadata
    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.leaf_actor import DeferredMessages

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import (
        GenState,
        SubNetGenerator,
    )
    from cudf_polars.streaming.base import PartitionInfo, StatsCollector
    from cudf_polars.streaming.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


def evaluate_logical_plan(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Evaluate a logical plan with the RapidsMPF streaming runtime.

    Parameters
    ----------
    ir
        The IR node.
    config_options
        The configuration options.
    collect_metadata
        Whether to collect runtime metadata.

    Returns
    -------
    The output DataFrame and metadata collector.
    """
    # For default_singleton, inject the process-wide DefaultSingletonEngine instance
    # into config_options before treating it as a regular SPMDEngine.
    if config_options.executor.cluster == "default_singleton":
        from cudf_polars.engine.default_singleton_engine import (
            DefaultSingletonEngine,
        )

        engine = DefaultSingletonEngine.get_or_create()
        if config_options.executor.quent_context is not None:
            engine_id = config_options.executor.quent_context.engine.id
        else:
            engine_id = uuid.uuid4()
        config_options = dataclasses.replace(
            config_options,
            executor=dataclasses.replace(
                config_options.executor,
                spmd_context=SPMDContext(
                    comm=engine.comm,
                    context=engine.context,
                    py_executor=engine.py_executor,
                    engine_id=engine_id,
                    worker_id=engine._quent_worker.id,
                    quent_logger=engine._quent_logger,
                ),
            ),
        )

    query_id = uuid.uuid4()
    with cudf_polars.dsl.tracing.bound_contextvars(
        cudf_polars_query_id=str(query_id),
    ):
        match config_options.executor.cluster:
            case "spmd" | "default_singleton":
                from cudf_polars.engine.spmd import (
                    evaluate_pipeline_spmd_mode,
                )

                _gpu_result, metadata_collector = evaluate_pipeline_spmd_mode(
                    ir,
                    config_options,
                    collect_metadata=collect_metadata,
                    query_id=query_id,
                )
                result = _gpu_result.to_polars()
            case "ray":
                from cudf_polars.engine.ray import (
                    evaluate_pipeline_ray_mode,
                )

                result, metadata_collector = evaluate_pipeline_ray_mode(
                    ir,
                    config_options,
                    collect_metadata=collect_metadata,
                    query_id=query_id,
                )
            case "dask":
                from cudf_polars.engine.dask import (
                    evaluate_pipeline_dask_mode,
                )

                result, metadata_collector = evaluate_pipeline_dask_mode(
                    ir,
                    config_options,
                    collect_metadata=collect_metadata,
                    query_id=query_id,
                )
            case other:
                raise ValueError(f"Unknown cluster mode: {other}")

    return result, metadata_collector


def determine_fanout_nodes(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    ir_dep_count: defaultdict[IR, int],
) -> dict[IR, FanoutInfo]:
    """
    Determine which IR nodes need fanout and what type.

    Parameters
    ----------
    ir
        The root IR node.
    partition_info
        Partition information for each IR node.
    ir_dep_count
        The number of IR dependencies for each IR node.

    Returns
    -------
    Dictionary mapping IR nodes to FanoutInfo tuples where:
    - num_consumers: number of consumers
    - unbounded: whether the node needs unbounded fanout
    Only includes nodes that need fanout (i.e., have multiple consumers).
    """
    # Determine which nodes need unbounded fanout
    unbounded: set[IR] = set()

    def _mark_children_unbounded(node: IR) -> None:
        for child in node.children:
            unbounded.add(child)

    # Traverse the graph and identify nodes that need unbounded fanout
    for node in traversal([ir]):
        if node in unbounded:
            _mark_children_unbounded(node)
        elif isinstance(node, (Union, Join, Over)):
            # Union processes children sequentially; Join may broadcast one
            # side; Over buffers (or samples-then-replays) its input before
            # producing output. In every case the input source needs
            # unbounded fanout so other consumers don't block it.
            _mark_children_unbounded(node)
        elif len(node.children) > 1:
            # Check if this node is doing any broadcasting.
            # When we move to dynamic partitioning, we will need a
            # new way to indicate that a node is broadcasting 1+ children.
            counts = [partition_info[c].count for c in node.children]
            has_broadcast = any(c == 1 for c in counts) and not all(
                c == 1 for c in counts
            )
            if has_broadcast:
                # Broadcasting operation - children need unbounded fanout
                _mark_children_unbounded(node)

    # Build result dictionary: only include nodes with multiple consumers
    fanout_nodes: dict[IR, FanoutInfo] = {}
    for node, count in ir_dep_count.items():
        if count > 1:
            fanout_nodes[node] = FanoutInfo(
                num_consumers=count,
                unbounded=node in unbounded,
            )

    return fanout_nodes


def generate_network(
    context: Context,
    comm: Communicator,
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    stats: StatsCollector,
    *,
    ir_context: IRExecutionContext,
    collective_id_map: dict[IR, list[int]],
    metadata_collector: list[ChannelMetadata] | None,
) -> tuple[list[Any], DeferredMessages]:
    """
    Translate the IR graph to a RapidsMPF streaming network.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator the network generation is collective over.
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        The configuration options.
    stats
        Statistics collector.
    ir_context
        The execution context for the IR node.
    collective_id_map
        The mapping of IR nodes to lists of collective IDs.
    metadata_collector
        The list to collect the final metadata.
        This list will be mutated when the network is executed.
        If None, metadata will not be collected.

    Returns
    -------
    The network nodes and output hook.
    """
    # Count the number of IR dependencies
    ir_dep_count: defaultdict[IR, int] = defaultdict(int)
    for node in traversal([ir]):
        for child in node.children:
            ir_dep_count[child] += 1

    # Determine which nodes need fanout
    fanout_nodes = determine_fanout_nodes(ir, partition_info, ir_dep_count)

    # Generate the network
    state: GenState = {
        "context": context,
        "comm": comm,
        "config_options": config_options,
        "partition_info": partition_info,
        "fanout_nodes": fanout_nodes,
        "ir_context": ir_context,
        "max_concurrent_io_tasks": config_options.executor.max_concurrent_io_tasks,
        "stats": stats,
        "collective_id_map": collective_id_map,
    }
    mapper: SubNetGenerator = CachingVisitor(
        generate_ir_sub_network_wrapper, state=state
    )
    nodes_dict, channels = mapper(ir)
    ch_out = channels[ir].reserve_output_slot()

    # Add node to drain metadata before pull_from_channel
    # (since pull_from_channel doesn't handle metadata messages)
    ch_final_data: Channel[TableChunk] = context.create_channel()
    drain_node = metadata_drain_node(
        context,
        comm,
        ir,
        ir_context,
        ch_out,
        ch_final_data,
        metadata_collector,
    )

    # Add final node to pull from the output data channel
    output_node, output = pull_from_channel(context, ch_in=ch_final_data)

    # Flatten the nodes dictionary into a list for run_actor_network
    nodes: list[Any] = [node for node_list in nodes_dict.values() for node in node_list]
    nodes.extend([drain_node, output_node])

    # Return network and output hook
    return nodes, output
