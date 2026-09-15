# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone execution of optional pushdown-filter hints."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from cudf_streaming import CardinalityEstimator
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.dispatch import generate_ir_sub_network
from cudf_polars.streaming.actor_graph.join import add_prefilter
from cudf_polars.streaming.actor_graph.prefilter import (
    PrefilterExecution,
    choose_prefilter_method,
)
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    ChunkSampler,
    gather_in_task_group,
    process_children,
    recv_metadata,
    replay_buffered_channel,
    sample_inputs,
    shutdown_on_error,
)
from cudf_polars.streaming.filter_hint import PushdownFilterHint

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.utils import TableSizeStats
    from cudf_polars.utils.config import StreamingExecutor


@define_actor()
async def pushdown_filter_actor(
    context: Context,
    comm: Communicator,
    ir: PushdownFilterHint,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_target: Channel[TableChunk],
    ch_domain: Channel[TableChunk],
    executor: StreamingExecutor,
    collective_id: int,
) -> None:
    """Choose and optionally execute one standalone pushdown-filter hint."""
    collected_samples: Sequence[TableSizeStats] = []
    async with shutdown_on_error(
        context,
        ch_out,
        ch_target,
        ch_domain,
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        try:
            target_metadata, domain_metadata = await gather_in_task_group(
                recv_metadata(ch_target, context),
                recv_metadata(ch_domain, context),
            )
            dynamic_planning = executor.dynamic_planning
            if dynamic_planning is None:
                raise ValueError("Standalone prefilters require dynamic planning")
            collected_samples = await sample_inputs(
                context,
                comm,
                (
                    ChunkSampler(
                        context=context,
                        ch_in=ch_target,
                        max_chunks=dynamic_planning.sample_chunk_count,
                        max_bytes=executor.target_partition_size,
                        ch_in_chunk_count=target_metadata.local_count,
                    ),
                    ChunkSampler(
                        context=context,
                        ch_in=ch_domain,
                        max_chunks=dynamic_planning.sample_chunk_count,
                        max_bytes=executor.target_partition_size,
                        ch_in_chunk_count=domain_metadata.local_count,
                        cardinality_estimator=CardinalityEstimator(
                            context, comm, tag=collective_id
                        ),
                        cardinality_columns=names_to_indices(
                            ir.domain_on, ir.children[1].schema
                        ),
                    ),
                ),
                collective_id,
            )
            if len(collected_samples) != 2:
                raise ValueError("Standalone prefilters require two input samples")
            target_sample, domain_sample = collected_samples
            config = executor.join_filter_pushdown
            if config is None:
                raise ValueError("Standalone prefilter has no runtime configuration")
            decision = choose_prefilter_method(
                ir.domain_on,
                target_sample,
                domain_sample,
                broadcast_limit=executor.broadcast_limit,
                bloom_filter_max_size=config.bloom_filter_max_size,
            )
            trace = asdict(decision)
            trace["placement"] = "standalone"
            trace["target_on"] = [key.name for key in ir.target_on]
            trace["domain_on"] = [key.name for key in ir.domain_on]
            trace_stats = trace if tracer is not None else None
            if tracer is not None:
                tracer.decision = decision.method
                tracer.set_extra("prefilter", trace)

            if decision.method == "skip":
                domain_sample.chunks.clear()
                await gather_in_task_group(
                    ch_domain.shutdown(context),
                    replay_buffered_channel(
                        context,
                        ch_out,
                        ch_target,
                        target_sample.chunks,
                        target_metadata,
                        trace_ir=ir,
                    ),
                )
            else:
                target, domain = ir.children
                domain_indices = names_to_indices(ir.domain_on, domain.schema)
                if domain_indices != tuple(range(len(domain.schema))):
                    raise ValueError("Pushdown filter domains must contain only keys")

                execution = PrefilterExecution(context)
                ch_target_replay: Channel[TableChunk] = context.create_channel()
                ch_domain_replay: Channel[TableChunk] = context.create_channel()
                execution.add_channel(ch_target_replay)
                execution.add_channel(ch_domain_replay)
                execution.add_task(
                    replay_buffered_channel(
                        context,
                        ch_target_replay,
                        ch_target,
                        target_sample.chunks,
                        target_metadata,
                        trace_ir=ir,
                    )
                )
                execution.add_task(
                    replay_buffered_channel(
                        context,
                        ch_domain_replay,
                        ch_domain,
                        domain_sample.chunks,
                        domain_metadata,
                        trace_ir=ir,
                    )
                )
                add_prefilter(
                    execution,
                    comm,
                    spec=ir,
                    decision=decision,
                    target=target,
                    domain=domain,
                    ch_target=ch_target_replay,
                    ch_domain_keys=ch_domain_replay,
                    ch_filtered=ch_out,
                    collective_id=collective_id,
                    ir_context=ir_context,
                    trace_stats=trace_stats,
                )
                async with shutdown_on_error(
                    context,
                    *execution.channels,
                    trace_ir=ir,
                    ir_context=ir_context,
                ):
                    await gather_in_task_group(*execution.tasks)
        finally:
            for sample in collected_samples:
                sample.chunks.clear()


@generate_ir_sub_network.register(PushdownFilterHint)
def generate_pushdown_filter_subnetwork(
    ir: PushdownFilterHint, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate the actor subnetwork for a standalone filter hint."""
    target, domain = ir.children
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    actors[ir] = [
        pushdown_filter_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[target].reserve_output_slot(),
            channels[domain].reserve_output_slot(),
            rec.state["config_options"].executor,
            collective_id,
        )
    ]
    return actors, channels
