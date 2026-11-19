# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tracing infrastructure for the RapidsMPF streaming runtime."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.streaming.actor_graph.prefilter import RuntimePrefilterStatistics


def _zero_bytes_by_tier() -> dict[MemoryType, int]:
    return dict.fromkeys(MemoryType, 0)


@dataclasses.dataclass(slots=True)
class ActorTracer:
    """
    Tracer for a single streaming actor (IR node).

    Collects execution statistics for Quent actor telemetry.

    Attributes
    ----------
    row_count
        Total row count produced by this node during execution.
        None if row counting is not available for this node.
    chunk_count
        Total chunk count produced by this node during execution.
    input_bytes
        Bytes received on boundary input channels, stratified by memory tier.
    output_bytes
        Bytes sent on the boundary output channel, stratified by memory tier.
    decision
        The algorithm decision made at runtime for this node
        (e.g., "broadcast_left", "shuffle", "tree", etc.).
    duplicated
        Whether the output rows are duplicated across ranks
        (e.g., after an allgather). Affects how rows are merged.
    """

    row_count: int | None = None
    chunk_count: int = 0
    input_bytes: dict[MemoryType, int] = dataclasses.field(
        default_factory=_zero_bytes_by_tier
    )
    output_bytes: dict[MemoryType, int] = dataclasses.field(
        default_factory=_zero_bytes_by_tier
    )
    decision: str | None = None
    duplicated: bool = False
    prefilters: list[RuntimePrefilterStatistics] = dataclasses.field(
        default_factory=list
    )

    def add_chunk(self, *, chunk: TableChunk | None = None) -> None:
        """
        Record a chunk.

        If chunk is provided, both row_count and chunk_count are updated.
        Otherwise, only chunk_count is incremented.

        Parameters
        ----------
        chunk
            The table chunk to record.
        """
        if chunk is not None:
            self.row_count = (self.row_count or 0) + chunk.shape[0]
        self.chunk_count += 1

    def set_duplicated(self, *, duplicated: bool = True) -> None:
        """Mark output rows as duplicated across ranks."""
        self.duplicated = duplicated


def record_channel_metrics(
    tracer: ActorTracer,
    *,
    chs_in: Sequence[Channel[Any]],
    chs_out: Sequence[Channel[Any]],
) -> None:
    """
    Record boundary channel byte volumes on an actor tracer.

    Parameters
    ----------
    tracer
        The actor tracer to update.
    chs_in
        Input boundary channels. ``recv_bytes`` are summed per memory tier.
    chs_out
        Output boundary channels. ``send_bytes`` are summed per memory tier.
    """
    for ch in chs_in:
        for mem_type, nbytes in ch.metrics().recv_bytes.items():
            tracer.input_bytes[mem_type] += nbytes
    for ch in chs_out:
        for mem_type, nbytes in ch.metrics().send_bytes.items():
            tracer.output_bytes[mem_type] += nbytes


async def send_chunk(
    context: Context,
    ch_out: Channel[TableChunk],
    chunk: TableChunk,
    sequence_number: int,
    *,
    tracer: ActorTracer | None,
) -> None:
    """
    Trace and send a TableChunk.

    Parameters
    ----------
    context
        The context of the streaming engine.
    ch_out
        The output channel to send the chunk to.
    chunk
        The chunk to send.
    sequence_number
        The sequence number of the chunk.
    tracer
        The tracer to use to trace the chunk.
    """
    if tracer is not None:
        tracer.add_chunk(chunk=chunk)
    await ch_out.send(context, Message(sequence_number, chunk))
