# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tracing infrastructure for the RapidsMPF streaming runtime."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.tracing import LOG_TRACES, Scope
from cudf_polars.streaming.explain import SerializablePlan

if TYPE_CHECKING:
    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.utils.config import ConfigOptions


@dataclasses.dataclass(slots=True)
class ActorTracer:
    """
    Tracer for a single streaming actor (IR node).

    Collects execution statistics and emits structured log events.

    Attributes
    ----------
    ir_id
        Stable identifier for the IR node (for tracing/logging).
    ir_type
        Type name of the IR node (e.g., "Sort", "Join").
    row_count
        Total row count produced by this node during execution.
        None if row counting is not available for this node.
    chunk_count
        Total chunk count produced by this node during execution.
    decision
        The algorithm decision made at runtime for this node
        (e.g., "broadcast_left", "shuffle", "tree", etc.).
    duplicated
        Whether the output rows are duplicated across ranks
        (e.g., after an allgather). Affects how rows are merged.
    """

    ir_id: int | None = None
    ir_type: str | None = None
    row_count: int | None = None
    chunk_count: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    decision: str | None = None
    duplicated: bool = False
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

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

    def set_extra(self, key: str, value: Any) -> None:
        """
        Attach structured metadata to the current actor trace event.

        This is useful for nested runtime decisions that do not have a
        separate IR node, but should still be logged with their parent actor.
        """
        self.extra[key] = value


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


def log_query_plan(ir: IR, config_options: ConfigOptions) -> None:
    """
    Log the IR tree structure as a structlog event.

    This should be called once on the client process after lowering,
    before distributed execution begins. The structure can be used
    by post-processing tools to reconstruct annotated plans.

    Parameters
    ----------
    ir
        The root IR node of the lowered query plan.
    config_options
        The GPU engine configuration options.

    Notes
    -----
    This function is a no-op if ``CUDF_POLARS_LOG_TRACES`` is not set.
    """
    if not LOG_TRACES:
        return

    import structlog

    dag = SerializablePlan.from_ir(ir, config_options=config_options)
    raw = dataclasses.asdict(dag)

    log = structlog.get_logger()
    log.info("Query Plan", scope=Scope.PLAN.value, plan=raw)
