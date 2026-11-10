# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming actor for sortedness hints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_streaming.channel_metadata import OrderKey
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.ordering import (
    ensure_ordering_metadata,
)
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    process_children,
    shutdown_on_error,
)
from cudf_polars.streaming.hint_sorted import HintSorted
from cudf_polars.utils import sorting

if TYPE_CHECKING:
    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator


def _hint_sorted_order_keys(ir: HintSorted) -> list[OrderKey]:
    """Convert hint-sorted options to ordering keys."""
    column_names, descending, nulls_last = ir.options
    orders, null_orders = sorting.sort_order(
        descending, nulls_last=nulls_last, num_keys=len(column_names)
    )
    return [
        OrderKey(index, order, null_order)
        for index, order, null_order in zip(
            names_to_indices(column_names, ir.schema),
            orders,
            null_orders,
            strict=True,
        )
    ]


@define_actor()
async def hint_sorted_actor(
    context: Context,
    comm: Communicator,
    ir: HintSorted,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    collective_id: int,
) -> None:
    """Forward data and attach safe ordering metadata for ``hint_sorted``."""
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        ir_context = actor_scope.require_ir_context()
        await ensure_ordering_metadata(
            context,
            comm,
            ir,
            ir_context,
            ch_in,
            ch_out,
            _hint_sorted_order_keys(ir),
            collective_id,
        )


@generate_ir_sub_network.register(HintSorted)
def _(
    ir: HintSorted, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    nodes, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    nodes[ir] = [
        hint_sorted_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context_for_node(rec, ir),
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_id,
        )
    ]
    return nodes, channels
