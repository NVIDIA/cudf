# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions and classes for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import operator
import struct
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import pylibcudf as plc
import rmm.mr
from cudf_streaming import CardinalityEstimate
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
)
from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.core.message import Message

import cudf_polars.dsl.tracing
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Cast, Col, NamedExpr, TemporalFunction
from cudf_polars.dsl.ir import Filter, GroupBy, HStack, Join, Projection, Select
from cudf_polars.dsl.tracing import Scope
from cudf_polars.dsl.utils.column_domain import column_domain_bindings
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.tracing import ActorTracer, send_chunk
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

    from cudf_streaming import CardinalityEstimator
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.spillable_messages import SpillableMessages
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.expr import Expr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.typing import Schema


InterRankScheme: TypeAlias = HashScheme | OrderScheme | None
PartitioningScheme: TypeAlias = InterRankScheme | Literal["inherit"]

# cuDF column/concatenate row limit (int32)
CUDF_ROW_LIMIT = 2**31 - 1
# Stay well below the cuDF row limit when forming a single table/partition.
MAX_ROWS_PER_PARTITION = CUDF_ROW_LIMIT // 4


def _hash_keys_match(
    scheme: HashScheme, key_indices: tuple[int, ...], *, allow_subset: bool
) -> bool:
    current = scheme.column_indices
    target = key_indices[: len(current)] if allow_subset else key_indices
    return target == current


def _ordering_keys_match(
    ordering: Ordering,
    keys: Sequence[int | OrderKey],
    key_indices: tuple[int, ...],
    *,
    allow_subset: bool,
    order_based: bool,
) -> bool:
    n_keys = len(ordering.keys)
    if allow_subset:
        if n_keys > len(key_indices):
            return False
    else:
        if n_keys != len(key_indices):
            return False
        n_keys = len(key_indices)
    if order_based:
        return all(ok == k for ok, k in zip(ordering.keys, keys[:n_keys], strict=True))
    return tuple(k.column_index for k in ordering.keys) == key_indices[:n_keys]


def _matching_order_scheme(
    scheme: OrderScheme,
    keys: Sequence[int | OrderKey],
    key_indices: tuple[int, ...],
    *,
    allow_subset: bool,
    order_based: bool,
) -> OrderScheme | None:
    orderings = scheme.orderings
    matches = [
        (i, ordering)
        for i, ordering in enumerate(orderings)
        if _ordering_keys_match(
            ordering,
            keys,
            key_indices,
            allow_subset=allow_subset,
            order_based=order_based,
        )
    ]
    if matches:
        # Prefer the most specific matching ordering; equal-length ties
        # keep the original metadata order.
        i, ordering = max(matches, key=lambda match: len(match[1].keys))
        return OrderScheme(
            (
                ordering,
                *orderings[:i],
                *orderings[i + 1 :],
            )
        )
    return None


def _keys_match(
    scheme: object,
    keys: Sequence[int | OrderKey],
    key_indices: tuple[int, ...],
    *,
    allow_subset: bool,
    order_based: bool,
) -> InterRankScheme:
    if isinstance(scheme, HashScheme) and _hash_keys_match(
        scheme, key_indices, allow_subset=allow_subset
    ):
        return scheme
    if isinstance(scheme, OrderScheme):
        return _matching_order_scheme(
            scheme,
            keys,
            key_indices,
            allow_subset=allow_subset,
            order_based=order_based,
        )
    return None


class ChunkStore:
    """Ordered spillable buffer for TableChunk messages."""

    def __init__(self, ctx: Context) -> None:
        self._mids: deque[int] = deque()
        self._store = ctx.spillable_messages()

    def __len__(self) -> int:
        """Return the number of messages in the store."""
        return len(self._mids)

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


@asynccontextmanager
async def shutdown_on_error(
    context: Context,
    *channels: Channel[Any],
    trace_ir: IR,
    ir_context: IRExecutionContext | None = None,
) -> AsyncIterator[ActorTracer | None]:
    """
    Actor-level shutdown and tracing for rapidsmpf.

    This context manager handles actor channel cleanup on errors and emits
    structlog tracing events.

    Parameters
    ----------
    context
        The rapidsmpf context.
    channels
        The channels to shutdown on error.
    trace_ir
        Optional IR node to enable tracing for this streaming actor.
        When provided and LOG_TRACES is enabled, an ActorTracer
        is yielded for collecting stats, and a structlog event is
        emitted on exit.
    ir_context
        The IR execution context from cudf-polars. This is used to propagate
        the query_id to the structlog logs emitted in this context.

    Yields
    ------
    ActorTracer | None
        An actor tracer for collecting stats (if tracing enabled), else None.
    """
    # Create tracer only if LOG_TRACES is enabled and IR is provided
    tracer: ActorTracer | None = None
    contextvars: dict[str, Any] = {}

    ir_id = trace_ir.get_stable_id()
    ir_type = type(trace_ir).__name__
    tracer = ActorTracer(ir_id, ir_type)
    contextvars = {"actor_ir_id": ir_id, "actor_ir_type": ir_type}

    if ir_context is not None:
        contextvars["cudf_polars_query_id"] = str(ir_context.query_id)

    with cudf_polars.dsl.tracing.bound_contextvars(**contextvars):
        start = time.monotonic_ns()
        try:
            yield tracer
        except BaseException:
            await shutdown_channels(context, *channels)
            raise
        finally:
            stop = time.monotonic_ns()
            record: dict[str, Any] = {
                "scope": Scope.ACTOR.value,
            }
            if tracer is not None:
                record.update(
                    {
                        "chunk_count": tracer.chunk_count,
                        "duplicated": tracer.duplicated,
                    }
                )
                if tracer.row_count is not None:
                    record["row_count"] = tracer.row_count
                if tracer.decision is not None:
                    record["decision"] = tracer.decision
            cudf_polars.dsl.tracing.log(
                "Streaming Actor", start=start, stop=stop, **record
            )


def _update_ordering_indices(
    ordering: Ordering, new_indices: tuple[int, ...]
) -> Ordering:
    return ordering.with_keys(
        (
            OrderKey(idx, k.order, k.null_order)
            for k, idx in zip(ordering.keys, new_indices, strict=True)
        )
    )


def _is_truncate_transparent_cast(expr: Cast) -> bool:
    src_id = expr.children[0].dtype.id()
    dst_id = expr.dtype.id()
    if src_id == dst_id:
        return True
    return (
        src_id == plc.TypeId.INT64 and dst_id == plc.TypeId.TIMESTAMP_NANOSECONDS
    ) or (src_id == plc.TypeId.TIMESTAMP_NANOSECONDS and dst_id == plc.TypeId.INT64)


def _unwrap_truncate_transparent_casts(expr: Expr) -> Expr:
    while isinstance(expr, Cast) and _is_truncate_transparent_cast(expr):
        (expr,) = expr.children
    return expr


def _truncate_source_name(expr: Expr) -> str | None:
    expr = _unwrap_truncate_transparent_casts(expr)
    if (
        isinstance(expr, TemporalFunction)
        and expr.name is TemporalFunction.Name.Truncate
    ):
        source = _unwrap_truncate_transparent_casts(expr.children[0])
        if isinstance(source, Col):
            return source.name
    return None


def _ordering_derivation(ne: NamedExpr) -> tuple[str, bool] | None:
    """
    Return derivation metadata for supported one-column ordering derivations.

    This is intentionally narrow for now: only temporal truncation is recognized.
    """
    source_name = _truncate_source_name(ne.value)
    if source_name is None:
        return None
    # Truncated boundaries may be non-strict.
    return source_name, False


def _derived_ordering(
    ordering: Ordering,
    ne: NamedExpr,
    old_to_new_names: dict[str, dict[str, None]],
    child_schema: Schema,
    output_schema: Schema,
    context: Context | None,
) -> Ordering | None:
    """Create an ordering for a supported derivation of one key."""
    if context is None:
        return None

    derivation = _ordering_derivation(ne)
    if derivation is None:
        return None
    source_name, strict_boundaries = derivation

    old_key_names = indices_to_names(ordering.column_indices, child_schema)
    try:
        source_position = old_key_names.index(source_name)
    except ValueError:
        return None

    prefix_names = old_key_names[:source_position]
    if not set(prefix_names).issubset(set(old_to_new_names)):
        return None

    target_key_names = (
        *(
            _preferred_target_name(name, old_to_new_names[name])
            for name in prefix_names
        ),
        ne.name,
    )
    new_indices = names_to_indices(target_key_names, output_schema)

    br = context.br()
    boundary_chunk = ordering.get_boundaries(br)
    stream = boundary_chunk.stream
    boundary_df = DataFrame.from_table(
        boundary_chunk.table_view(),
        old_key_names,
        [child_schema[name] for name in old_key_names],
        stream,
    )
    column = ne.evaluate(boundary_df)
    boundary_columns = [
        *boundary_chunk.table_view().columns()[:source_position],
        column.obj,
    ]
    boundary_table = plc.concatenate.concatenate(
        [plc.Table(boundary_columns)],
        stream=stream,
        mr=br.device_mr,
    )
    boundaries = TableChunk.from_pylibcudf_table(
        boundary_table,
        stream,
        exclusive_view=True,
        br=br,
    )
    keys = tuple(
        OrderKey(idx, key.order, key.null_order)
        for idx, key in zip(
            new_indices,
            (*ordering.keys[:source_position], ordering.keys[source_position]),
            strict=True,
        )
    )
    return Ordering(
        keys,
        boundaries,
        strict_boundaries=strict_boundaries,
    )


def _select_column_targets(select: Select) -> dict[str, dict[str, None]]:
    old_to_new_names: defaultdict[str, dict[str, None]] = defaultdict(dict)
    for output_name, source in column_domain_bindings(select).items():
        old_to_new_names[source.name][output_name] = None
    return dict(old_to_new_names)


def _preferred_target_name(old_name: str, targets: dict[str, None]) -> str:
    return old_name if old_name in targets else next(iter(targets))


def _remap_scheme_select(
    select: Select, scheme: PartitioningScheme, context: Context | None
) -> PartitioningScheme:
    if isinstance(scheme, HashScheme):
        old_to_new_names = _select_column_targets(select)
        old_key_names = indices_to_names(
            scheme.column_indices, select.children[0].schema
        )
        if set(old_key_names).issubset(set(old_to_new_names)):
            new_indices = names_to_indices(
                tuple(
                    _preferred_target_name(n, old_to_new_names[n])
                    for n in old_key_names
                ),
                select.schema,
            )
            return HashScheme(new_indices, scheme.modulus)
        return None
    if isinstance(scheme, OrderScheme):
        old_to_new_names = _select_column_targets(select)
        new_orderings: list[Ordering] = []
        for ordering in scheme.orderings:
            old_key_names = indices_to_names(
                ordering.column_indices, select.children[0].schema
            )
            if set(old_key_names).issubset(set(old_to_new_names)):
                target_key_names = tuple(
                    _preferred_target_name(n, old_to_new_names[n])
                    for n in old_key_names
                )
                new_indices = names_to_indices(target_key_names, select.schema)
                new_orderings.append(_update_ordering_indices(ordering, new_indices))
                if len(old_key_names) == 1:
                    for alias in old_to_new_names[old_key_names[0]]:
                        if alias == target_key_names[0]:
                            continue
                        new_orderings.append(
                            _update_ordering_indices(
                                ordering, names_to_indices((alias,), select.schema)
                            )
                        )
            for ne in select.exprs:
                derived = _derived_ordering(
                    ordering,
                    ne,
                    old_to_new_names,
                    select.children[0].schema,
                    select.schema,
                    context,
                )
                if derived is not None:
                    new_orderings.append(derived)
        if new_orderings:
            return OrderScheme(new_orderings)
        return None
    if scheme not in (None, "inherit"):  # pragma: no cover
        return None  # Guard against future/unsupported scheme types
    return scheme


def _remap_scheme_simple(
    ir: IR, scheme: PartitioningScheme, child: IR
) -> PartitioningScheme:
    if isinstance(scheme, HashScheme):
        old_key_names = indices_to_names(scheme.column_indices, child.schema)
        try:
            new_indices = names_to_indices(old_key_names, ir.schema)
        except (ValueError, IndexError):
            return None
        return HashScheme(new_indices, scheme.modulus)
    if isinstance(scheme, OrderScheme):
        new_orderings: list[Ordering] = []
        for ordering in scheme.orderings:
            old_key_names = indices_to_names(ordering.column_indices, child.schema)
            try:
                new_indices = names_to_indices(old_key_names, ir.schema)
            except (ValueError, IndexError):
                continue
            new_orderings.append(_update_ordering_indices(ordering, new_indices))
        if new_orderings:
            return OrderScheme(new_orderings)
        return None
    return scheme  # None or "inherit" passes through unchanged


def _hstack_to_select(hstack: HStack) -> Select:
    """Translate HStack to the equivalent Select node."""
    col_map = {ne.name: ne for ne in hstack.columns}
    exprs = tuple(
        col_map[name] if name in col_map else NamedExpr(name, Col(dtype, name))
        for name, dtype in hstack.schema.items()
    )
    return Select(hstack.schema, exprs, hstack.should_broadcast, hstack.children[0])


def maybe_remap_partitioning(
    ir: IR,
    partitioning: Partitioning | None,
    *,
    child_ir: IR | None = None,
    context: Context | None = None,
) -> Partitioning | None:
    """
    Remap partitioning for simple IR nodes.

    Parameters
    ----------
    ir
        The IR node.
    partitioning
        The input partitioning.
    child_ir
        The child IR whose schema the partitioning refers to. When None,
        the first child (ir.children[0]) is used.
    context
        Runtime context used to materialize transformed boundary tables for
        derived orderings. When None, only metadata-only remapping is applied.

    Returns
    -------
    The remapped partitioning. When partition keys are not preserved,
    the corresponding scheme will be set to None. When the original
    partitioning is None, the output will also be None.

    Notes
    -----
    A Select preserves partitioning if all partition key columns are
    output as simple Col references (unchanged values). Other columns
    can be computed expressions - only the partition keys matter.
    """
    if partitioning is None:
        return None  # Nothing to preserve
    if isinstance(ir, (Select, HStack)):
        if isinstance(ir, HStack):
            # HStack is a special case of Select
            ir = _hstack_to_select(ir)
        return Partitioning(
            inter_rank=_remap_scheme_select(ir, partitioning.inter_rank, context),
            local=_remap_scheme_select(ir, partitioning.local, context),
        )
    if isinstance(ir, GroupBy):
        return Partitioning(
            inter_rank=_remap_scheme_simple(
                ir, partitioning.inter_rank, ir.children[0]
            ),
            local=_remap_scheme_simple(ir, partitioning.local, ir.children[0]),
        )
    if isinstance(ir, (Join, Projection, Filter)):
        child = child_ir if child_ir is not None else ir.children[0]
        return Partitioning(
            inter_rank=_remap_scheme_simple(ir, partitioning.inter_rank, child),
            local=_remap_scheme_simple(ir, partitioning.local, child),
        )
    return None


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

    Returns
    -------
    The resulting table chunk after evaluation.
    """
    input_schema = ir.children[0].schema
    names = list(input_schema.keys())
    dtypes = list(input_schema.values())
    df = ir.do_evaluate(
        *ir._non_child_args,
        DataFrame.from_table(chunk.table_view(), names, dtypes, chunk.stream),
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

    Returns
    -------
    The resulting table chunk after evaluation.
    """
    assert len(irs) > 0, "Expected at least one IR node"
    chunk, extra = await make_table_chunks_available_or_wait(
        context,
        chunk,
        reserve_extra=chunk.data_alloc_size(),
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        for single_ir in irs:
            chunk = await ir_context.to_thread(
                _evaluate_chunk_sync, chunk, single_ir, ir_context, context.br()
            )
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
        inserter.insert(0, local_chunk)
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
    handle_empty_input: bool = False,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Apply IR evaluation chunk-by-chunk, preserving partitioning.

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
        The channel metadata to forward (partitioning preserved).
    handle_empty_input
        If True and no chunks are received, create an empty chunk and evaluate
        it. Use for operations like aggregations that always produce output.
    tracer
        Optional tracer for runtime metrics.
    """
    await send_metadata(ch_out, context, metadata)
    if tracer is not None and metadata.duplicated:
        tracer.set_duplicated()

    received_any = False
    while (msg := await ch_in.recv(context)) is not None:
        received_any = True
        cd = msg.get_content_description()
        seq_num = msg.sequence_number
        with cudf_polars.dsl.tracing.bound_contextvars(
            content_sizes=cd.content_sizes,
            spillable=cd.spillable,
            sequence_number=msg.sequence_number,
        ):
            result = await evaluate_chunk(
                context,
                TableChunk.from_message(msg, br=context.br()),
                ir,
                ir_context=ir_context,
            )
        del msg, cd
        await send_chunk(context, ch_out, result, seq_num, tracer=tracer)

    if handle_empty_input and not received_any:
        chunk = empty_table_chunk(ir.children[0], context, ir_context.get_cuda_stream())
        result = await evaluate_chunk(context, chunk, ir, ir_context=ir_context)
        del chunk
        await send_chunk(context, ch_out, result, 0, tracer=tracer)

    await ch_out.drain(context)


def indices_to_names(indices: tuple[int, ...], schema: Schema) -> tuple[str, ...]:
    """
    Return column names for the given column indices in schema order.

    Parameters
    ----------
    indices
        The indices to get names for.
    schema
        The schema to get names from.

    Returns
    -------
    The column names for each index in schema order.
    """
    keys = list(schema.keys())
    return tuple(keys[i] for i in indices)


@dataclass(frozen=True)
class TableSizeStats:
    """Sampled chunks and aggregate size/row stats for a table channel."""

    chunks: ChunkStore
    """The sampled chunks/messages in replay order."""
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
            chunks=chunks,
            total_size=total_size,
            total_rows=total_rows,
            total_chunks=(sample_count if is_complete else self.ch_in_chunk_count),
            is_complete=is_complete,
            cardinality=cardinality,
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
        The buffered chunks to yield first.
    metadata
        The metadata to send to the output channel.
    trace_ir
        The IR node to trace. Passed through to shutdown_on_error.
    """
    async with shutdown_on_error(context, ch_out, ch_in, trace_ir=trace_ir):
        await send_metadata(ch_out, context, metadata)
        for msg in buffered_chunks:
            await ch_out.send(context, msg)
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
        await ch_out.drain(context)


@dataclass(frozen=True)
class NormalizedPartitioning:  # noqa: PLW1641 (frozen=True generates __hash__ even with custom __eq__)
    """
    Normalized view of channel partitioning for a set of key column indices.

    inter_rank_scheme is None when the channel metadata has no inter-rank
    partitioning on the requested keys. local_scheme is None when the local
    partitioning scheme does not cover the requested keys, and the string
    "inherit" when local layout follows from inter-rank.
    """

    inter_rank_scheme: InterRankScheme
    local_scheme: PartitioningScheme

    def __bool__(self) -> bool:
        """True when both inter-rank and local schemes are present."""
        return self.inter_rank_scheme is not None and self.local_scheme is not None

    def __eq__(self, other: object) -> bool:
        """True when both schemes are equal."""
        if not isinstance(other, NormalizedPartitioning):
            return NotImplemented
        return (
            self.inter_rank_scheme == other.inter_rank_scheme
            and self.local_scheme == other.local_scheme
        )

    def is_strictly_partitioned(self) -> bool:
        """True if data is strictly partitioned with no boundary straddling."""
        if not self:
            return False
        for scheme in [self.inter_rank_scheme, self.local_scheme]:
            if isinstance(scheme, OrderScheme):
                ordering = scheme.orderings[0]
                if ordering.strict_boundaries:
                    continue
                return False
        return True

    def is_strictly_sorted(self, order_keys: Sequence[OrderKey]) -> bool:
        """True if the selected ordering proves sortedness for order_keys."""
        if not self or not isinstance(self.inter_rank_scheme, OrderScheme):
            return False
        ordering = self.inter_rank_scheme.orderings[0]
        if len(ordering.keys) < len(order_keys):
            # If we are only sorted on a subset of the keys, we need strict
            # boundaries to know later keys cannot interleave across chunks.
            return ordering.strict_boundaries
        return True

    def is_aligned_with(
        self, other: NormalizedPartitioning, br: BufferResource
    ) -> bool:
        """True when both sides share identical inter-rank and local chunk layouts."""

        def _schemes_aligned(
            lhs: PartitioningScheme,
            rhs: PartitioningScheme,
        ) -> bool:
            if isinstance(lhs, OrderScheme):
                if not isinstance(rhs, OrderScheme):
                    return False
                lhs_ordering = lhs.orderings[0]
                rhs_ordering = rhs.orderings[0]
                return lhs_ordering.boundaries_aligned_with(rhs_ordering, br)
            elif isinstance(lhs, HashScheme):
                return (
                    isinstance(rhs, HashScheme)
                    and lhs.modulus == rhs.modulus
                    and len(lhs.column_indices) == len(rhs.column_indices)
                )
            return lhs == "inherit" and rhs == "inherit"

        return (
            self.is_strictly_partitioned()
            and other.is_strictly_partitioned()
            and _schemes_aligned(self.inter_rank_scheme, other.inter_rank_scheme)
            and _schemes_aligned(self.local_scheme, other.local_scheme)
        )

    @classmethod
    def from_keys(
        cls,
        partitioning_metadata: Partitioning | None,
        nranks: int,
        *,
        keys: Sequence[int | OrderKey],
        allow_subset: bool = True,
    ) -> NormalizedPartitioning:
        """
        Resolve partitioning from channel metadata and key column indices.

        Parameters
        ----------
        partitioning_metadata
            The channel partitioning metadata.
        nranks
            Number of ranks.
        keys
            Key column descriptors. Pass sequence of ``int`` values
            (column indices) to match by column index only (both
            ``HashScheme`` and ``OrderScheme``). Pass a sequence of
            ``OrderKey`` instances to match ``OrderScheme`` with full
            order/null-order semantics.
        allow_subset
            If True, the metadata keys may be a prefix of ``keys``.

        Returns
        -------
        NormalizedPartitioning
            The resolved inter-rank and local partitioning schemes.
        """
        inter_rank_scheme, local_scheme = NormalizedPartitioning._normalize_schemes(
            partitioning_metadata,
            keys,
            nranks,
            allow_subset=allow_subset,
        )
        return cls(inter_rank_scheme=inter_rank_scheme, local_scheme=local_scheme)

    @staticmethod
    def _normalize_schemes(
        partitioning_metadata: Partitioning | None,
        keys: Sequence[int | OrderKey],
        nranks: int,
        *,
        allow_subset: bool = False,
    ) -> tuple[
        InterRankScheme,
        PartitioningScheme,
    ]:
        """Translate Partitioning metadata into normalized partitioning schemes."""
        if not keys:
            return None, None
        order_based = isinstance(keys[0], OrderKey)
        if order_based:
            if not all(isinstance(k, OrderKey) for k in keys):
                raise TypeError("keys must be all int or all OrderKey")
            key_indices: tuple[int, ...] = tuple(
                k.column_index for k in cast("Sequence[OrderKey]", keys)
            )
        else:
            if not all(isinstance(k, int) for k in keys):
                raise TypeError("keys must be all int or all OrderKey")
            key_indices = tuple(keys)
        # On a single rank, hash queries are trivially partitioned into one bucket.
        # For order queries, single-rank does not imply sorted — we rely on local
        # scheme promotion below to handle that case.
        trivial = (
            HashScheme(key_indices, 1) if (nranks == 1 and not order_based) else None
        )
        if partitioning_metadata is None:
            return trivial, None

        inter_rank = partitioning_metadata.inter_rank
        strict_inter_rank = _keys_match(
            inter_rank,
            keys,
            key_indices,
            allow_subset=allow_subset,
            order_based=order_based,
        )
        inter_rank_scheme: InterRankScheme = strict_inter_rank or trivial
        if inter_rank_scheme is None and nranks > 1:
            # Partitioning is meaningless without inter-rank partitioning
            return None, None

        local = partitioning_metadata.local
        local_scheme: PartitioningScheme
        matched_local = _keys_match(
            local,
            keys,
            key_indices,
            allow_subset=allow_subset,
            order_based=order_based,
        )
        if matched_local is not None:
            local_scheme = matched_local
        elif local == "inherit":
            local_scheme = "inherit"
        else:
            local_scheme = None

        # Single-rank normalization: when there is no real inter-rank scheme,
        # local and inter-rank partitioning are equivalent. This applies to both
        # hash (trivial modulus=1) and order queries (a sorted local scheme is
        # globally sorted when nranks==1).
        if strict_inter_rank is None and nranks == 1:
            if local_scheme not in (None, "inherit"):
                # Translate: (trivial/None, Scheme) → (Scheme, "inherit")
                return local_scheme, "inherit"
            if trivial is not None:
                return trivial, None
            return None, None

        return inter_rank_scheme, local_scheme


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


def chunk_to_frame(chunk: TableChunk, ir: IR) -> DataFrame:
    """
    Convert a TableChunk to a DataFrame.

    Parameters
    ----------
    chunk
        The TableChunk to convert.
    ir
        The IR node to use for the schema.

    Returns
    -------
    A DataFrame.
    """
    return DataFrame.from_table(
        chunk.table_view(),
        list(ir.schema.keys()),
        list(ir.schema.values()),
        chunk.stream,
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
