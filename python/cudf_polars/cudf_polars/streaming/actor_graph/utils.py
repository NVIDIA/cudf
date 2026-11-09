# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions and classes for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import math
import operator
import struct
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from functools import reduce
from typing import TYPE_CHECKING, Any

import pylibcudf as plc
import rmm.mr
from cudf_streaming import CardinalityEstimate
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    OrderKey,
    Partitioning,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.partitioning import NormalizedPartitioning
from cudf_polars.streaming.actor_graph.tracing import (
    ActorTracer,
    record_channel_metrics,
    send_chunk,
)
from cudf_polars.streaming.utils import _concat
from cudf_polars.utils.dtypes import make_empty_column

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Callable,
        Coroutine,
        Generator,
        Iterator,
        Sequence,
    )

    import cudf_polars_quent as _quent

    from cudf_streaming import CardinalityEstimator
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.spillable_messages import SpillableMessages
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.typing import Schema


type OrderingMetadata = dict[int, OrderKey]

# cuDF column/concatenate row limit (int32)
CUDF_ROW_LIMIT = 2**31 - 1
# Stay well below the cuDF row limit when forming a single table/partition.
MAX_ROWS_PER_PARTITION = CUDF_ROW_LIMIT // 4


class ChunkStore:
    """Ordered spillable buffer for Messages."""

    def __init__(self, ctx: Context) -> None:
        self._mids: deque[int] = deque()
        self._store = ctx.spillable_messages()

    def __len__(self) -> int:
        """Return the number of messages in the store."""
        return len(self._mids)

    def clear(self) -> None:
        """Discard all messages in the store."""
        for mid in self._mids:
            self._store.extract(mid=mid)
        self._mids.clear()

    def insert(self, msg: Message) -> None:
        """Insert a message into the store."""
        self._mids.append(self._store.insert(msg))

    def __iter__(self) -> Generator[Message, None, None]:
        """Yield messages in insertion order, draining the store."""
        while self._mids:
            yield self._store.extract(mid=self._mids.popleft())


@contextlib.contextmanager
def set_memory_resource(mr: rmm.mr.DeviceMemoryResource) -> Iterator[None]:
    """
    Context manager that temporarily sets ``mr`` as the current device resource.

    On entry, ``mr`` is installed via ``rmm.mr.set_current_device_resource(mr)``.
    On exit, the previously active resource is restored unconditionally.

    Parameters
    ----------
    mr
        The memory resource to activate for the duration of the block.
    """
    old = rmm.mr.get_current_device_resource()
    rmm.mr.set_current_device_resource(mr)
    try:
        yield
    finally:
        rmm.mr.set_current_device_resource(old)


async def gather_in_task_group(*coroutines: Coroutine[Any, Any, Any]) -> list[Any]:
    """
    asyncio.gather-like API for running tasks in a asyncio.TaskGroup.

    Parameters
    ----------
    coroutines
        Tasks to execute.

    Returns
    -------
    list[Any]
        The results of the coroutines.
    """
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(coro) for coro in coroutines]
    return [task.result() for task in tasks]


async def shutdown_channels(context: Context, *channels: Channel[Any]) -> None:
    """Shutdown data and metadata paths for all channels."""
    await gather_in_task_group(
        *itertools.chain.from_iterable(
            (ch.shutdown(context), ch.shutdown_metadata(context)) for ch in channels
        )
    )


@asynccontextmanager
async def shutdown_channels_on_error(
    context: Context, *channels: Channel[Any]
) -> AsyncIterator[None]:
    """Shutdown channels on error without actor tracing."""
    try:
        yield
    except BaseException:
        await shutdown_channels(context, *channels)
        raise


@dataclass(frozen=True)
class ActorScope:
    """Metrics and explicit execution context for one Actor lifetime."""

    tracer: ActorTracer | None
    ir_context: IRExecutionContext | None

    def require_ir_context(self) -> IRExecutionContext:
        """Return the Actor-bound context required by evaluation code."""
        assert self.ir_context is not None
        return self.ir_context


@asynccontextmanager
async def shutdown_on_error(
    context: Context,
    *,
    chs_in: Sequence[Channel[Any]] = (),
    chs_out: Sequence[Channel[Any]] = (),
    chs_aux: Sequence[Channel[Any]] = (),
    trace_ir: IR,
    ir_context: IRExecutionContext | None = None,
) -> AsyncIterator[ActorScope]:
    """
    Actor-level shutdown and Quent telemetry for rapidsmpf.

    This context manager handles actor channel cleanup on errors and records
    the outermost Quent actor lifecycle when telemetry is configured.

    Parameters
    ----------
    context
        The rapidsmpf context.
    chs_in
        Boundary input channels. Shut down on error, and used to record
        ``input_bytes`` from ``Channel.metrics().recv_bytes``.
    chs_out
        Boundary output channels. Shut down on error, and used to record
        ``output_bytes`` from ``Channel.metrics().send_bytes``.
    chs_aux
        Auxiliary channels. Shut down on error, but not included in
        byte-volume tracing.
    trace_ir
        IR node represented by this streaming actor.
    ir_context
        The IR execution context from cudf-polars.

    Yields
    ------
    ActorScope
        Actor metrics and the Actor-bound IR execution context.
    """
    channels = (*chs_in, *chs_out, *chs_aux)
    tracer = None if ir_context is None else ir_context.tracer

    # We might have nested calls to `shutdown_on_error` for a given actor.
    # We always want to trace the outermost call, so record a little state here
    # to know whether we're the outermost call and so created the quent actor.
    created_quent_actor = False
    unbound_quent_execution = (
        None if ir_context is None else ir_context.unbound_quent_ir_execution_state
    )
    if (
        ir_context is not None
        and ir_context.actor_bound_quent_ir_execution_state is None
        and unbound_quent_execution is not None
    ):
        import cudf_polars_quent as _quent

        created_quent_actor = True
        tracer = ActorTracer()
        actor_id = _quent.now_v7()
        quent_execution = unbound_quent_execution.bind_actor(actor_id)
        ir_context = replace(
            ir_context,
            tracer=tracer,
            actor_bound_quent_ir_execution_state=quent_execution,
            unbound_quent_ir_execution_state=None,
        )
        runtime = quent_execution.query_worker_state.runtime
        started = (
            runtime.session.binding_context.actor_observer()
            .handle(actor_id)
            .started(
                operator=quent_execution.operator_id,
                worker=runtime.worker_resources.worker_id,
            )
        )
        runtime.session._actors[actor_id] = started.running()

    actor_error: BaseException | None = None
    try:
        yield ActorScope(tracer=tracer, ir_context=ir_context)
    except BaseException as caught:
        actor_error = caught
        await shutdown_channels(context, *channels)
        raise
    finally:
        if (
            created_quent_actor
            and tracer is not None
            and ir_context is not None
            and (
                quent_ir_execution_state := (
                    ir_context.actor_bound_quent_ir_execution_state
                )
            )
            is not None
        ):
            record_channel_metrics(tracer, chs_in=chs_in, chs_out=chs_out)
            values: _quent.OperatorStatisticsDict = {
                "output_rows": tracer.row_count,
                "input_bytes": sum(tracer.input_bytes.values()),
                "output_bytes": sum(tracer.output_bytes.values()),
                "chunk_count": tracer.chunk_count,
                "duplicated": tracer.duplicated,
                "decision": tracer.decision,
            }
            prefilters: list[_quent.RuntimePrefilterStatisticsDict] = [
                {
                    "placement": value.placement,
                    "method": value.decision.method,
                    "reason": value.decision.reason,
                    "target_side": value.target_side,
                    "domain_side": value.domain_side,
                    "domain": value.domain,
                    "target_on": value.target_on,
                    "domain_on": value.domain_on,
                    "target_bytes": value.decision.target_bytes,
                    "domain_rows": value.decision.domain_rows,
                    "estimated_cardinality": value.decision.estimated_cardinality,
                    "bloom_bytes": value.decision.bloom_bytes,
                    "exact_bytes": value.decision.exact_bytes,
                    "input_rows": value.input_rows,
                    "output_rows": value.output_rows,
                }
                for value in tracer.prefilters
            ]
            runtime = quent_ir_execution_state.query_worker_state.runtime
            operator = runtime.session.binding_context.operator_observer().handle(
                quent_ir_execution_state.operator_id
            )
            for prefilter in prefilters:
                operator.runtime_prefilter_statistics(
                    actor=quent_ir_execution_state.actor_id,
                    values=prefilter,
                )
            if actor_error is None:
                runtime.session._actors.pop(
                    quent_ir_execution_state.actor_id
                ).completed(
                    values=values,
                )
            else:
                runtime.session._actors.pop(quent_ir_execution_state.actor_id).failed(
                    error=str(actor_error),
                    values=values,
                )
            operator.statistics(values=values)


async def send_metadata(
    ch: Channel[TableChunk], ctx: Context, metadata: ChannelMetadata
) -> None:
    """
    Send metadata and drain the metadata queue.

    Parameters
    ----------
    ch :
        The channel to send metadata on.
    ctx :
        The streaming context.
    metadata :
        The metadata to send.

    Notes
    -----
    This function copies the metadata before sending, so the caller
    retains ownership of the original metadata object.
    """
    msg = Message(
        0,
        # Copy metadata before sending since Message consumes the handle.
        # Metadata is small, so copying is cheap.
        ChannelMetadata(
            local_count=metadata.local_count,
            partitioning=metadata.partitioning,
            duplicated=metadata.duplicated,
        ),
    )
    await ch.send_metadata(ctx, msg)
    await ch.drain_metadata(ctx)


async def recv_metadata(ch: Channel[TableChunk], ctx: Context) -> ChannelMetadata:
    """
    Receive metadata from a channel's metadata queue.

    Parameters
    ----------
    ch :
        The channel to receive metadata from.
    ctx :
        The streaming context.

    Returns
    -------
    ChannelMetadata
        The received metadata.
    """
    msg = await ch.recv_metadata(ctx)
    assert msg is not None, f"Expected ChannelMetadata message, got {msg}."
    return ChannelMetadata.from_message(msg)


def _make_hash_shuffle_metadata(
    comm: Communicator,
    key_indices: tuple[int, ...],
    modulus: int,
    metadata_in: ChannelMetadata,
) -> ChannelMetadata:
    """
    Build output ChannelMetadata for a hash shuffle by key_indices.

    Parameters
    ----------
    comm
        The communicator.
    key_indices
        Column indices to hash-partition on.
    modulus
        Number of output partitions (must be >= comm.nranks).
    metadata_in
        Input channel metadata (used for duplicated flag and, on a
        single-rank run, to preserve the existing inter-rank scheme).

    Returns
    -------
    ChannelMetadata
        Ready to pass to send_metadata.
    """
    nranks = comm.nranks
    if nranks == 1:
        inter_rank_scheme = (
            None
            if metadata_in.partitioning is None
            else metadata_in.partitioning.inter_rank
        )
        local_scheme: HashScheme | str = HashScheme(
            column_indices=key_indices, modulus=modulus
        )
        local_output_count = modulus
    else:
        inter_rank_scheme = HashScheme(column_indices=key_indices, modulus=modulus)
        local_scheme = "inherit"
        local_output_count = (modulus - comm.rank + nranks - 1) // nranks
    return ChannelMetadata(
        local_count=local_output_count,
        partitioning=Partitioning(inter_rank_scheme, local_scheme),
        duplicated=metadata_in.duplicated,
    )


def _evaluate_chunk_sync(
    chunk: TableChunk,
    ir: IR,
    ir_context: IRExecutionContext,
    br: BufferResource,
    *,
    ordering_metadata: OrderingMetadata | None = None,
) -> TableChunk:
    """
    Apply an IR node's do_evaluate to a table chunk (synchronous).

    This is an internal helper. Use `evaluate_chunk` for the async version
    with memory reservation.

    Parameters
    ----------
    chunk
        The input table chunk (must be available).
    ir
        The IR node to evaluate.
    ir_context
        The IR execution context.
    br
        The buffer resource for lifetime tracking.
    ordering_metadata
        Optional precomputed ordering metadata to synthesize local DataFrame
        metadata from.

    Returns
    -------
    The resulting table chunk after evaluation.
    """
    df_in = chunk_to_frame(chunk, ir.children[0], ordering_metadata=ordering_metadata)
    df = ir.do_evaluate(
        *ir._non_child_args,
        df_in,
        context=ir_context,
    )
    return TableChunk.from_pylibcudf_table(
        df.table, df.stream, exclusive_view=True, br=br
    )


async def evaluate_chunk(
    context: Context,
    chunk: TableChunk,
    *irs: IR,
    ir_context: IRExecutionContext,
    ordering_metadata: OrderingMetadata | None = None,
    already_available: bool = False,
) -> TableChunk:
    """
    Make chunk available, reserve memory, and evaluate.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    chunk
        The input table chunk.
    irs
        The IR node(s) to evaluate. Evaluations are chained
        in order within a single memory reservation.
    ir_context
        The IR execution context.
    ordering_metadata
        Optional precomputed ordering metadata to synthesize local DataFrame
        metadata from during the first evaluation.
    already_available
        If true, ``chunk`` must already be available on device. The caller
        keeps ownership of the input chunk, so this helper only reserves memory
        for evaluation.

    Returns
    -------
    The resulting table chunk after evaluation.
    """
    assert len(irs) > 0, "Expected at least one IR node"
    if already_available:
        if not chunk.is_available():
            raise ValueError("already_available=True requires an available TableChunk")
        reservation = await context.memory(MemoryType.DEVICE).reserve_or_wait(
            chunk.data_alloc_size(), net_memory_delta=0
        )
    else:
        chunk, reservation = await make_table_chunks_available_or_wait(
            context,
            chunk,
            reserve_extra=chunk.data_alloc_size(),
            net_memory_delta=0,
        )
    with opaque_memory_usage(reservation):
        for single_ir in irs:
            chunk = await ir_context.to_thread(
                _evaluate_chunk_sync,
                chunk,
                single_ir,
                ir_context,
                context.br(),
                ordering_metadata=ordering_metadata,
            )
            ordering_metadata = None
        return chunk


async def allgather_and_reduce(
    context: Context,
    comm: Communicator,
    collective_id: int,
    local_chunk: TableChunk,
    reduce_ir: IR,
    ir_context: IRExecutionContext,
) -> TableChunk:
    """
    AllGather ``local_chunk`` across ranks and apply ``reduce_ir`` to the result.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    comm
        The communicator.
    collective_id
        Collective operation ID for the AllGather.
    local_chunk
        The locally-reduced chunk this rank contributes.
    reduce_ir
        IR node applied to the concatenated AllGather output.
    ir_context
        The IR execution context.

    Returns
    -------
    The chunk produced by evaluating ``reduce_ir`` on the gathered result.
    """
    allgather = AllGatherManager(context, comm, collective_id)
    with allgather.inserting() as inserter:
        await inserter.insert(0, local_chunk)
    stream = ir_context.get_cuda_stream()
    concat_chunk = TableChunk.from_pylibcudf_table(
        await allgather.extract_concatenated(stream, ir_context=ir_context),
        stream,
        exclusive_view=True,
        br=context.br(),
    )
    return await evaluate_chunk(context, concat_chunk, reduce_ir, ir_context=ir_context)


async def concat_batch(
    batch: list[TableChunk],
    context: Context,
    schema: Schema,
    ir_context: IRExecutionContext,
) -> TableChunk:
    """
    Concatenate a list of table chunks.

    Parameters
    ----------
    batch
        The list of table chunks to concatenate.
    context
        The rapidsmpf context.
    schema
        The schema of the table chunks.
    ir_context
        The IR execution context.

    Returns
    -------
    The table chunk after concatenation.
    """
    batch, extra = await make_table_chunks_available_or_wait(
        context,
        batch,
        reserve_extra=sum(c.data_alloc_size() for c in batch),
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        df = await ir_context.to_thread(
            _concat,
            *[
                DataFrame.from_table(
                    c.table_view(),
                    list(schema.keys()),
                    list(schema.values()),
                    c.stream,
                )
                for c in batch
            ],
            context=ir_context,
        )
        if len(batch) > 1:
            del batch
    return TableChunk.from_pylibcudf_table(
        df.table, df.stream, exclusive_view=True, br=context.br()
    )


async def evaluate_batch(
    batch: list[TableChunk],
    context: Context,
    *irs: IR,
    ir_context: IRExecutionContext,
) -> TableChunk:
    """
    Concatenate a list of table chunks and evaluate the result.

    Parameters
    ----------
    batch
        The list of table chunks to evaluate.
    context
        The rapidsmpf context.
    irs
        The IR node(s) to evaluate. Evaluations are chained
        in order within a single memory reservation.
    ir_context
        The IR execution context.

    Returns
    -------
    The table chunk after evaluation.
    """
    assert len(irs) > 0, "Expected at least one IR node"
    first_ir = irs[0]
    input_schema = first_ir.children[0].schema
    chunk = await concat_batch(batch, context, input_schema, ir_context)
    del batch
    return await evaluate_chunk(context, chunk, *irs, ir_context=ir_context)


async def chunkwise_evaluate(
    context: Context,
    ir: IR,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata: ChannelMetadata,
    *,
    input_metadata: ChannelMetadata | None = None,
    handle_empty_input: bool = False,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Apply IR evaluation chunk-by-chunk.

    Use when data is already partitioned on the relevant keys and each
    chunk can be processed independently.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    ir
        The IR node to evaluate.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    metadata
        The channel metadata to forward.
    input_metadata
        The input metadata to synthesize local DataFrame metadata from.
        Defaults to ``metadata`` for partition-preserving callers.
    handle_empty_input
        If True and no chunks are received, create an empty chunk and evaluate
        it. Use for operations like aggregations that always produce output.
    tracer
        Optional tracer for runtime metrics.
    """
    await send_metadata(ch_out, context, metadata)
    if tracer is not None and metadata.duplicated:
        tracer.set_duplicated()

    input_ordering_metadata = _leading_order_keys(
        metadata if input_metadata is None else input_metadata
    )

    received_any = False
    while (msg := await ch_in.recv(context)) is not None:
        received_any = True
        cd = msg.get_content_description()
        seq_num = msg.sequence_number
        chunk_ir_context = ir_context
        if (quent_state := ir_context.actor_bound_quent_ir_execution_state) is not None:
            chunk_ir_context = replace(
                ir_context,
                actor_bound_quent_ir_execution_state=replace(
                    quent_state,
                    sequence_number=seq_num,
                    content_sizes=tuple(cd.content_sizes),
                    spillable=cd.spillable,
                ),
            )
        result = await evaluate_chunk(
            context,
            TableChunk.from_message(msg, br=context.br()),
            ir,
            ir_context=chunk_ir_context,
            ordering_metadata=input_ordering_metadata,
        )
        del msg, cd
        await send_chunk(context, ch_out, result, seq_num, tracer=tracer)

    if handle_empty_input and not received_any:
        chunk = empty_table_chunk(ir.children[0], context, ir_context.get_cuda_stream())
        result = await evaluate_chunk(
            context,
            chunk,
            ir,
            ir_context=ir_context,
            ordering_metadata=input_ordering_metadata,
        )
        del chunk
        await send_chunk(context, ch_out, result, 0, tracer=tracer)

    await ch_out.drain(context)


@dataclass(frozen=True)
class TableSample:
    """Exact local prefix buffered while sampling a table channel."""

    chunks: ChunkStore
    """The sampled chunks/messages in replay order."""
    size: int = 0
    """Bytes in the buffered chunks."""
    rows: int = 0
    """Rows in the buffered chunks."""
    is_complete: bool = False
    """Whether the buffered prefix contains this rank's entire input."""


@dataclass(frozen=True)
class TableSizeStats:
    """Local sample and complete-table size estimates for a table channel."""

    local_sample: TableSample
    """The exact local buffered sample."""
    total_size: int = 0
    """The estimated table size in bytes for the represented scope."""
    total_rows: int = 0
    """The estimated number of rows for the represented scope."""
    total_chunks: int = 0
    """The estimated number of chunks for the represented scope."""
    is_complete: bool = False
    """Whether the sample contains the entire table for the represented scope."""
    cardinality: CardinalityEstimate | None = None
    """Global cardinality statistics for the sampled rows, when requested."""

    def distinct_count(self) -> int | None:
        """Extrapolate sampled distinct count to the estimated full row count."""
        if self.total_rows == 0:
            return 0
        if self.cardinality is None or self.cardinality.row_count == 0:
            return None
        return min(
            self.total_rows,
            math.ceil(
                self.cardinality.distinct_count
                * self.total_rows
                / self.cardinality.row_count
            ),
        )


async def aggregate_table_size_stats(
    context: Context,
    comm: Communicator,
    samples: tuple[TableSizeStats, ...],
    collective_id: int,
) -> tuple[TableSizeStats, ...]:
    """Aggregate table-size and row estimates across ranks."""
    totals = await allgather_reduce(
        context,
        comm,
        collective_id,
        *(
            value
            for sample in samples
            for value in (
                sample.total_size,
                sample.total_rows,
                sample.total_chunks,
                int(sample.is_complete),
            )
        ),
    )
    totals_iter = iter(totals)
    return tuple(
        TableSizeStats(
            local_sample=sample.local_sample,
            total_size=next(totals_iter),
            total_rows=next(totals_iter),
            total_chunks=next(totals_iter),
            is_complete=next(totals_iter) == comm.nranks,
            cardinality=sample.cardinality,
        )
        for sample in samples
    )


@dataclass(frozen=True)
class ChunkSampler:
    """Object for obtaining statistics from a channel of TableChunks."""

    context: Context
    ch_in: Channel[TableChunk]
    max_chunks: int
    max_bytes: int
    ch_in_chunk_count: int
    cardinality_estimator: CardinalityEstimator | None = None
    cardinality_columns: tuple[int, ...] = ()

    async def sample_channel(
        self,
        ch_out: Channel[TableChunk],
    ) -> tuple[bool, int, int]:
        """
        Sample a prefix from the input channel.

        Parameters
        ----------
        ch_out
            Channel to forward samples into.

        Returns
        -------
        bool
            Whether sampling exhausted the input.
        int
            Number of bytes in the sampled chunks.
        int
            Number of rows in the sampled chunks.
        """
        sampled_bytes = 0
        sampled_rows = 0
        exhausted = False
        await ch_out.shutdown_metadata(self.context)
        for _ in range(self.max_chunks):
            msg = await self.ch_in.recv(self.context)
            if msg is None:
                exhausted = True
                break
            chunk = TableChunk.from_message(msg, br=self.context.br())
            sampled_bytes += chunk.data_alloc_size()
            sampled_rows += chunk.shape[0]
            await ch_out.send(self.context, Message(msg.sequence_number, chunk))
            if sampled_bytes >= self.max_bytes:
                break
        await ch_out.drain(self.context)
        return exhausted, sampled_bytes, sampled_rows

    async def store_sample(
        self,
        ch_in: Channel[TableChunk],
    ) -> ChunkStore:
        """
        Store chunks from a channel.

        Parameters
        ----------
        ch_in
            Channel to store

        Returns
        -------
        ChunkStore containing the chunks from the channel.
        """
        chunks = ChunkStore(self.context)
        while (msg := await ch_in.recv(self.context)) is not None:
            chunk = TableChunk.from_message(msg, br=self.context.br())
            chunks.insert(Message(msg.sequence_number, chunk))
        return chunks

    async def sample(self) -> TableSizeStats:
        """Sample the configured channel and return extrapolated statistics."""
        ch_sampled = self.context.create_channel()
        temporary_channels = [ch_sampled]
        tasks: list[Coroutine[Any, Any, Any]]
        if self.cardinality_estimator is None:
            tasks = [self.sample_channel(ch_sampled), self.store_sample(ch_sampled)]
        else:
            ch_cardinality_input = self.context.create_channel()
            ch_cardinality = self.context.create_channel()
            temporary_channels.extend((ch_cardinality_input, ch_cardinality))
            # The sampled output must be consumed concurrently with estimation:
            # backpressure there can prevent the cardinality result from being
            # produced.
            tasks = [
                self.sample_channel(ch_cardinality_input),
                self.cardinality_estimator.estimate(
                    self.context,
                    ch_cardinality_input,
                    ch_cardinality,
                    ch_sampled,
                    self.cardinality_columns,
                ),
                self.store_sample(ch_sampled),
                ch_cardinality.recv(self.context),
            ]

        async with shutdown_channels_on_error(self.context, *temporary_channels):
            results = await gather_in_task_group(*tasks)
        if self.cardinality_estimator is None:
            (is_complete, sample_bytes, sample_rows), chunks = results
            cardinality = None
        else:
            (is_complete, sample_bytes, sample_rows), _, chunks, msg = results
            if msg is None:
                raise RuntimeError("Cardinality estimator did not produce an estimate")
            cardinality = CardinalityEstimate.from_message(msg)

        sample_count = len(chunks)
        if sample_count and not is_complete:
            total_size = int((sample_bytes / sample_count) * self.ch_in_chunk_count)
            total_rows = int((sample_rows / sample_count) * self.ch_in_chunk_count)
        else:
            total_size = sample_bytes
            total_rows = sample_rows
        return TableSizeStats(
            local_sample=TableSample(
                chunks=chunks,
                size=sample_bytes,
                rows=sample_rows,
                is_complete=is_complete,
            ),
            total_size=total_size,
            total_rows=total_rows,
            total_chunks=(sample_count if is_complete else self.ch_in_chunk_count),
            is_complete=is_complete,
            cardinality=cardinality,
        )


async def sample_inputs(
    context: Context,
    comm: Communicator,
    samplers: Sequence[ChunkSampler],
    collective_id: int,
) -> tuple[TableSizeStats, ...]:
    """Sample input channels concurrently and aggregate their statistics."""
    if not samplers:
        return ()
    local_samples = await gather_in_task_group(
        *(sampler.sample() for sampler in samplers)
    )
    return await aggregate_table_size_stats(
        context,
        comm,
        tuple(local_samples),
        collective_id,
    )


async def _sample_chunks(
    context: Context,
    ch: Channel[TableChunk],
    max_sample_chunks: int,
    max_sample_bytes: int,
    local_count: int,
    *,
    cardinality_estimator: CardinalityEstimator | None = None,
    cardinality_columns: Sequence[int] = (),
) -> TableSizeStats:
    """
    Sample chunks from a channel and extrapolate to a per-rank size estimate.

    Parameters
    ----------
    context
        The context.
    ch
        The channel to sample from.
    max_sample_chunks
        The maximum number of chunks to sample.
    max_sample_bytes
        The maximum number of bytes to sample.
    local_count
        The expected number of local chunks (used for extrapolation).
    cardinality_estimator
        Optional distributed cardinality estimator. When provided, cardinality
        is estimated from the same fixed chunk prefix used for size sampling.
    cardinality_columns
        Column indices whose row tuples are counted. An empty sequence selects
        all columns.

    Returns
    -------
    Sampled chunks, extrapolated size/rows, and optional global cardinality.
    """
    return await ChunkSampler(
        context=context,
        ch_in=ch,
        max_chunks=max_sample_chunks,
        max_bytes=max_sample_bytes,
        ch_in_chunk_count=local_count,
        cardinality_estimator=cardinality_estimator,
        cardinality_columns=tuple(cardinality_columns),
    ).sample()


async def replay_buffered_channel(
    context: Context,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    buffered_chunks: ChunkStore,
    metadata: ChannelMetadata,
    *,
    trace_ir: IR,
) -> None:
    """
    Replay a buffered input channel into an output channel.

    Parameters
    ----------
    context
        The context.
    ch_out
        The new output channel.
    ch_in
        The buffered input channel.
    buffered_chunks
        The buffered chunks to yield first. The store is empty when this
        coroutine exits, including on cancellation or error.
    metadata
        The metadata to send to the output channel.
    trace_ir
        The IR node to trace. Passed through to shutdown_on_error.
    """
    try:
        async with shutdown_on_error(
            context,
            chs_in=(ch_in,),
            chs_out=(ch_out,),
            trace_ir=trace_ir,
        ):
            await send_metadata(ch_out, context, metadata)
            for msg in buffered_chunks:
                await ch_out.send(context, msg)
            while (msg := await ch_in.recv(context)) is not None:
                await ch_out.send(context, msg)
            await ch_out.drain(context)
    finally:
        buffered_chunks.clear()


class ChannelManager:
    """A utility class for managing Channel objects."""

    def __init__(self, context: Context, *, count: int = 1):
        """
        Initialize the ChannelManager with a given number of channel slots.

        Parameters
        ----------
        context
            The rapidsmpf context.
        count: int
            The number of channel slots to allocate.
        """
        self._channel_slots: list[Channel[TableChunk]] = [
            context.create_channel() for _ in range(count)
        ]
        self._reserved_output_slots: int = 0
        self._reserved_input_slots: int = 0

    def reserve_input_slot(self) -> Channel[TableChunk]:
        """
        Reserve an input channel slot.

        Returns
        -------
        The reserved Channel.
        """
        if self._reserved_input_slots >= len(self._channel_slots):
            raise ValueError("No more input channel slots available")
        self._reserved_input_slots += 1
        return self._channel_slots[self._reserved_input_slots - 1]

    def reserve_output_slot(self) -> Channel[TableChunk]:
        """
        Reserve an output channel slot.

        Returns
        -------
        The reserved Channel.
        """
        if self._reserved_output_slots >= len(self._channel_slots):
            raise ValueError("No more output channel slots available")
        self._reserved_output_slots += 1
        return self._channel_slots[self._reserved_output_slots - 1]


def process_children(
    ir: IR, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """
    Process children IR nodes and aggregate their nodes and channels.

    This helper function recursively processes all children of an IR node,
    collects their streaming network nodes into a dictionary mapping IR nodes
    to their associated nodes, and merges their channel dictionaries.

    Parameters
    ----------
    ir
        The IR node whose children should be processed.
    rec
        Recursive SubNetGenerator callable.

    Returns
    -------
    nodes
        Dictionary mapping each IR node to its list of streaming network nodes.
    channels
        Dictionary mapping each child IR node to its ChannelManager.
    """
    if not ir.children:
        return {}, {}

    _nodes_list, _channels_list = zip(*(rec(c) for c in ir.children), strict=True)
    nodes: dict[IR, list[Any]] = reduce(operator.or_, _nodes_list)
    channels: dict[IR, ChannelManager] = reduce(operator.or_, _channels_list)
    return nodes, channels


def empty_table_chunk(ir: IR, context: Context, stream: Stream) -> TableChunk:
    """
    Make an empty table chunk.

    Parameters
    ----------
    ir
        The IR node to use for the schema.
    context
        The rapidsmpf context.
    stream
        The stream to use for the table chunk.

    Returns
    -------
    The empty table chunk.
    """
    empty_columns = [make_empty_column(dtype, stream) for dtype in ir.schema.values()]
    empty_table = plc.Table(empty_columns)

    return TableChunk.from_pylibcudf_table(
        empty_table,
        stream,
        exclusive_view=True,
        br=context.br(),
    )


def _leading_order_keys(metadata: ChannelMetadata | None) -> OrderingMetadata:
    """Return unambiguous leading order keys implied by channel metadata."""
    if metadata is None or metadata.partitioning is None:
        return {}

    candidates: dict[int, OrderKey | None] = {}
    for ordering in metadata.partitioning.row_orderings:
        if ordering.group_keys or not ordering.order_keys:
            continue
        key = ordering.order_keys[0]
        current = candidates.get(key.column_index, key)
        candidates[key.column_index] = key if current == key else None
    return {index: key for index, key in candidates.items() if key is not None}


def _apply_ordering_metadata(
    df: DataFrame, ordering_metadata: OrderingMetadata
) -> DataFrame:
    """Apply precomputed safe column-level sortedness metadata."""
    for index, key in ordering_metadata.items():
        if index >= df.num_columns:
            continue
        df.columns[index].set_sorted(
            is_sorted=plc.types.Sorted.YES,
            order=key.order,
            null_order=key.null_order,
        )
    return df


def chunk_to_frame(
    chunk: TableChunk,
    ir: IR,
    *,
    ordering_metadata: OrderingMetadata | None = None,
) -> DataFrame:
    """
    Convert a TableChunk to a DataFrame.

    Parameters
    ----------
    chunk
        The TableChunk to convert.
    ir
        The IR node to use for the schema.
    ordering_metadata
        Optional precomputed ordering metadata to synthesize local DataFrame
        metadata from.

    Returns
    -------
    A DataFrame.
    """
    df = DataFrame.from_table(
        chunk.table_view(),
        list(ir.schema.keys()),
        list(ir.schema.values()),
        chunk.stream,
    )
    return (
        df
        if ordering_metadata is None
        else _apply_ordering_metadata(df, ordering_metadata)
    )


def _is_already_partitioned(
    metadata: ChannelMetadata,
    columns_to_hash: tuple[int, ...],
    num_partitions: int,
    nranks: int,
) -> bool:
    """
    Check if data is already hash-partitioned for a shuffle actor.

    Returns True only when the channel carries a HashScheme on the requested
    keys with matching modulus and ``local="inherit"`` — the canonical layout
    produced by a prior global shuffle.
    """
    partitioning = NormalizedPartitioning.from_keys(
        metadata.partitioning,
        nranks,
        keys=columns_to_hash,
        allow_subset=False,
    )
    return (
        isinstance(partitioning.inter_rank_scheme, HashScheme)
        and partitioning.local_scheme == "inherit"
        and partitioning.inter_rank_scheme.modulus == num_partitions
    )


def make_spill_function(
    spillable_messages_list: list[SpillableMessages],
    context: Context,
) -> Callable[[int], int]:
    """
    Create a spill function for a list of SpillableMessages containers.

    This utility creates a spill function that can be registered with a
    SpillManager. The spill function uses a smart spilling strategy that
    prioritizes:
    1. Longest queues first (slow consumers that won't need data soon)
    2. Newest messages first (just arrived, won't be consumed soon)

    This strategy keeps "hot" data (about to be consumed) in fast memory
    while spilling "cold" data (won't be needed for a while) to slower tiers.

    Parameters
    ----------
    spillable_messages_list
        List of SpillableMessages containers to create a spill function for.
    context
        The RapidsMPF context to use for accessing the BufferResource.

    Returns
    -------
    A spill function that takes an amount (in bytes) and returns the
    actual amount spilled (in bytes).

    Notes
    -----
    The spilling strategy is particularly effective for fanout scenarios
    where different consumers may process messages at different rates. By
    prioritizing longest queues and newest messages, we maximize the time
    data can remain in slower memory before it's needed.
    """

    def spill_func(amount: int) -> int:
        """Spill messages from the buffers to free device/host memory."""
        spilled = 0

        # Collect all messages with metadata for smart spilling
        # Format: (message_id, container_idx, queue_length, sm)
        all_messages: list[tuple[int, int, int, SpillableMessages]] = []
        for container_idx, sm in enumerate(spillable_messages_list):
            content_descriptions = sm.get_content_descriptions()
            queue_length = len(content_descriptions)
            all_messages.extend(
                (message_id, container_idx, queue_length, sm)
                for message_id in content_descriptions
            )

        # Spill newest messages first from the longest queues
        # Sort by: (1) queue length descending, (2) message_id descending
        # This prioritizes:
        # - Longest queues (slow consumers that won't need data soon)
        # - Newest messages (just arrived, won't be consumed soon)
        all_messages.sort(key=lambda x: (-x[2], -x[0]))

        # Spill messages until we've freed enough memory
        for message_id, _, _, sm in all_messages:
            if spilled >= amount:
                break
            # Try to spill this message
            spilled += sm.spill(mid=message_id, br=context.br())

        return spilled

    return spill_func


async def allgather_reduce(
    context: Context,
    comm: Communicator,
    op_id: int,
    *local_values: int,
) -> tuple[int, ...]:
    """
    Allgather local scalar values and sum each across all ranks.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    op_id
        The collective operation ID for this allgather.
    *local_values
        One or more local scalar values to contribute.

    Returns
    -------
    tuple[int, ...]
        The sum of each local_value across all ranks.
    """
    if comm.nranks == 1:
        return tuple(local_values)

    n = len(local_values)
    fmt = f"<{'q' * n}"
    data = struct.pack(fmt, *local_values)
    packed = PackedData.from_host_bytes(data, context.br())

    allgather = AllGather(context, comm, op_id)
    try:
        allgather.insert(0, packed)
    finally:
        allgather.insert_finished()

    results = await allgather.extract_all(context, ordered=False)

    totals = [0] * n
    for packed_result in results:
        result_bytes = packed_result.to_host_bytes()
        values = struct.unpack(fmt, result_bytes)
        for i, v in enumerate(values):
            totals[i] += v

    return tuple(totals)
