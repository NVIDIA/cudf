# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tracing infrastructure for the RapidsMPF streaming runtime."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from rapidsmpf.streaming.core.message import Message

from cudf_polars.dsl.tracing import LOG_TRACES, Scope
from cudf_polars.streaming.explain import SerializablePlan

if TYPE_CHECKING:
    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR
    from cudf_polars.utils.config import ConfigOptions

T = TypeVar("T")


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


class TracingChannel(Generic[T]):
    """
    Channel proxy that records the bytes an actor reads and writes.

    Wrap an actor's channels to attribute ``input_bytes`` (from ``recv``) and
    ``output_bytes`` (from ``send``) to that actor.

    Internal channels an actor creates for its own sub-network are left
    unwrapped so intermediate traffic is not counted.
    """

    def __init__(self, channel: Channel[T], tracer: ActorTracer | None) -> None:
        self._channel = channel
        self._tracer = tracer

    async def recv(self, context: Context) -> Message[T] | None:
        """Wrapper around ``Channel.recv`` that records the input bytes."""
        message = await self._channel.recv(context)
        if message is not None and self._tracer is not None:
            self._tracer.input_bytes += _message_size(message)
        return message

    async def send(self, context: Context, message: Message[T]) -> None:
        """Wrapper around ``Channel.send`` that records the output bytes."""
        if self._tracer is not None:
            self._tracer.output_bytes += _message_size(message)
        await self._channel.send(context, message)

    # Implement the rest of the Channel interface.

    async def drain(self, context: Context) -> None:
        """Passthrough to ``Channel.drain``."""
        await self._channel.drain(context)

    async def drain_metadata(self, context: Context) -> None:
        """Passthrough to ``Channel.drain_metadata``."""
        await self._channel.drain_metadata(context)

    async def shutdown(self, context: Context) -> None:
        """Passthrough to ``Channel.shutdown``."""
        await self._channel.shutdown(context)

    async def shutdown_metadata(self, context: Context) -> None:
        """Passthrough to ``Channel.shutdown_metadata``."""
        await self._channel.shutdown_metadata(context)

    async def recv_metadata(self, context: Context) -> Message[Any] | None:
        """Passthrough to ``Channel.recv_metadata``."""
        return await self._channel.recv_metadata(context)

    async def send_metadata(self, context: Context, message: Message[Any]) -> None:
        """Passthrough to ``Channel.send_metadata``."""
        await self._channel.send_metadata(context, message)


def _message_size(message: Message[Any]) -> int:
    """Return the total data allocation size described by a message."""
    return sum(message.get_content_description().content_sizes.values())


def trace_channel(channel: Channel[T], tracer: ActorTracer | None) -> Channel[T]:
    """Wrap one of an actor's boundary channels to record the bytes crossing it."""
    return cast("Channel[T]", TracingChannel(channel, tracer))


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
