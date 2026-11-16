# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GroupBy and Distinct logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    OrderKey,
    OrderScheme,
    Ordering,
    Partitioning,
    RowOrdering,
    RowOrderingScope,
)
from cudf_streaming.table_chunk import TableChunk, make_table_chunks_available_or_wait
from rapidsmpf.communicator.single import new_communicator as single_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.context import Context

from cudf_polars.containers import DataType
from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.dsl.ir import IR, Distinct, GroupBy, Select, Sort
from cudf_polars.dsl.utils.naming import names_to_indices, unique_names
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.collectives.ordering import adjust_ordering
from cudf_polars.streaming.actor_graph.collectives.shuffle import ShuffleManager
from cudf_polars.streaming.actor_graph.collectives.slice_exchange import SparseExchange
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.partitioning import (
    NormalizedPartitioning,
    is_row_ordered_by,
    maybe_remap_partitioning,
    remap_ordering_keys,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    MAX_ROWS_PER_PARTITION,
    ChannelManager,
    _make_hash_shuffle_metadata,
    allgather_and_reduce,
    allgather_reduce,
    chunkwise_evaluate,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    gather_in_task_group,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_channels_on_error,
    shutdown_on_error,
)
from cudf_polars.streaming.groupby import (
    _has_order_sensitive_agg,
    _has_stable_sorted_agg,
    combine,
    decompose,
)
from cudf_polars.streaming.repartition import Repartition
from cudf_polars.streaming.utils import partition_range
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.partitioning import PartitioningLevel
    from cudf_polars.streaming.actor_graph.tracing import ActorTracer
    from cudf_polars.typing import Schema


@dataclass
class DecomposedGroupBy:
    """Holds decomposed GroupBy or Distinct operations for multi-phase aggregation."""

    ir: GroupBy | Distinct
    """The original IR node."""
    piecewise_ir: GroupBy | Distinct
    """The decomposed piecewise IR node."""
    reduction_ir: GroupBy | Distinct
    """The decomposed reduction IR node. Same as piecewise_ir for Distinct."""
    select_ir: Select | None
    """The decomposed select IR node. Always None for Distinct."""
    need_preshuffle: bool
    """Whether the operation requires preshuffling."""
    output_indices: tuple[int, ...]
    """The output indices of the GroupBy keys."""
    shuffle_indices: tuple[int, ...]
    """The column indices to shuffle on."""

    @classmethod
    def from_ir(cls, ir: GroupBy | Distinct) -> DecomposedGroupBy:
        """Decompose a GroupBy IR node into multi-phase operations."""
        piecewise_ir: GroupBy | Distinct
        reduction_ir: GroupBy | Distinct
        select_ir: Select | None
        need_preshuffle: bool
        output_indices: tuple[int, ...]
        shuffle_indices: tuple[int, ...]

        if isinstance(ir, Distinct):
            piecewise_ir = reduction_ir = ir
            select_ir = None
            need_preshuffle = (
                ir.keep == plc.stream_compaction.DuplicateKeepOption.KEEP_NONE
            )
        elif isinstance(ir, GroupBy):
            name_generator = unique_names(ir.schema.keys())
            selection_exprs, piecewise_exprs, reduction_exprs, need_preshuffle = (
                combine(
                    *(
                        decompose(agg.name, agg.value, names=name_generator)
                        for agg in ir.agg_requests
                    )
                )
            )

            # Piecewise groupby schema and IR
            pwise_schema = {k.name: k.value.dtype for k in ir.keys} | {
                k.name: k.value.dtype for k in piecewise_exprs
            }
            piecewise_ir = GroupBy(
                pwise_schema,
                ir.keys,
                piecewise_exprs,
                ir.maintain_order,
                None,
                ir.children[0],
            )
            # Grouped keys for reduction and selection
            groupby_keys = tuple(
                NamedExpr(k.name, Col(k.value.dtype, k.name)) for k in ir.keys
            )

            # Reduction groupby schema and IR
            reduction_schema = {k.name: k.value.dtype for k in groupby_keys} | {
                k.name: k.value.dtype for k in reduction_exprs
            }
            reduction_ir = GroupBy(
                reduction_schema,
                groupby_keys,
                reduction_exprs,
                ir.maintain_order,
                None,
                piecewise_ir,
            )
            select_ir = Select(
                ir.schema,
                [
                    *(
                        NamedExpr(k.name, Col(k.value.dtype, k.name))
                        for k in groupby_keys
                    ),
                    *selection_exprs,
                ],
                False,  # noqa: FBT003
                reduction_ir,
            )
        else:  # pragma: no cover
            raise TypeError(f"Unsupported IR type: {type(ir)}")

        output_indices = _key_indices(ir, ir.schema)
        if isinstance(ir, Distinct):
            shuffle_indices = output_indices
        else:
            shuffle_indices = _key_indices(piecewise_ir, piecewise_ir.schema)

        return cls(
            ir=ir,
            piecewise_ir=piecewise_ir,
            reduction_ir=reduction_ir,
            select_ir=select_ir,
            need_preshuffle=need_preshuffle,
            output_indices=output_indices,
            shuffle_indices=shuffle_indices,
        )


async def _local_aggregation(
    context: Context,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    target_partition_size: int,
    *,
    allow_early_exit: bool = True,
) -> tuple[TableChunk, bool, int]:
    """
    Local groupby or distinct aggregation.

    Parameters
    ----------
    context
        The rapidsmpf context.
    decomposed
        The decomposed groupby or distinct.
    ir_context
        The IR execution context.
    ch_in
        The input channel.
    target_partition_size
        The target partition size.
    allow_early_exit
        Whether to allow early exit from the loop when
        the total size exceeds the target partition size.

    Returns
    -------
    aggregated
        The aggregated result.
    input_drained
        Whether the input channel is drained.
    chunks_received
        The number of chunks received from the input channel.
    """
    total_size = 0
    chunks_received = 0
    input_drained = False
    evaluated_chunks: list[TableChunk] = []
    while True:
        msg = await ch_in.recv(context)
        if msg is None:
            input_drained = True
            break

        chunks_received += 1
        chunk = await evaluate_chunk(
            context,
            TableChunk.from_message(msg, br=context.br()),
            decomposed.piecewise_ir,
            ir_context=ir_context,
        )
        chunk = _enforce_schema(chunk, decomposed.piecewise_ir.schema, context.br())

        total_size += chunk.data_alloc_size()
        evaluated_chunks.append(chunk)
        if total_size > target_partition_size and len(evaluated_chunks) > 1:
            evaluated_chunks = [
                await evaluate_batch(
                    evaluated_chunks,
                    context,
                    decomposed.reduction_ir,
                    ir_context=ir_context,
                )
            ]
            total_size = evaluated_chunks[0].data_alloc_size()
        if total_size > target_partition_size and allow_early_exit:
            break

    aggregated: TableChunk
    if len(evaluated_chunks) > 1:
        aggregated = await evaluate_batch(
            evaluated_chunks,
            context,
            decomposed.reduction_ir,
            ir_context=ir_context,
        )
    elif evaluated_chunks:
        aggregated = evaluated_chunks[0]
    else:
        aggregated = empty_table_chunk(
            decomposed.reduction_ir,
            context,
            ir_context.get_cuda_stream(),
        )

    return aggregated, input_drained, chunks_received


async def _tree_reduce(
    context: Context,
    comm: Communicator,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    collective_id: int,
    *,
    local: bool,
    aggregated: TableChunk,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Execute groupby or distinct using tree reduction to a single output.

    Parameters
    ----------
    context
        The rapidsmpf streaming context.
    comm
        The communicator.
    decomposed
        The decomposed groupby containing piecewise, reduction, and select IRs.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    metadata_in
        The input channel metadata.
    aggregated
        The aggregated result for already-evaluated chunks.
    collective_id
        Collective ID for allgather (used when local is False).
    local
        Whether to use a local aggregation.
    tracer
        Optional tracer for runtime metrics.
    """
    need_allgather = not local and not metadata_in.duplicated and comm.nranks > 1

    metadata_out = ChannelMetadata(
        local_count=1,
        partitioning=None,
        duplicated=True if need_allgather else metadata_in.duplicated,
    )
    await send_metadata(ch_out, context, metadata_out)

    if need_allgather:
        aggregated = await allgather_and_reduce(
            context,
            comm,
            collective_id,
            _enforce_schema(aggregated, decomposed.reduction_ir.schema, context.br()),
            decomposed.reduction_ir,
            ir_context,
        )

    if decomposed.select_ir is not None:
        aggregated = await evaluate_chunk(
            context, aggregated, decomposed.select_ir, ir_context=ir_context
        )
    await send_chunk(context, ch_out, aggregated, 0, tracer=tracer)

    await ch_out.drain(context)


async def _shuffle_reduce(
    context: Context,
    comm: Communicator,
    decomposed: DecomposedGroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    metadata_in: ChannelMetadata,
    modulus: int,
    collective_id: int,
    target_partition_size: int,
    *,
    local: bool,
    aggregated: TableChunk,
    input_drained: bool = False,
    tracer: ActorTracer | None = None,
) -> None:
    """
    Shuffle-based groupby or distinct.

    Shuffles data by keys, then applies local groupby or distinct to
    each partition. Use when the expected output is large.

    Parameters
    ----------
    context
        The rapidsmpf streaming context for channel operations.
    comm
        The communicator.
    decomposed
        The decomposed groupby or distinct containing piecewise, reduction, and select IRs.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    metadata_in
        The input channel metadata.
    modulus
        The modulus for the shuffle operation.
    collective_id
        The collective ID for the shuffle operation.
    target_partition_size
        Target partition size for local aggregation batches.
    local
        Whether to use a local shuffle operation.
    aggregated
        The aggregated result for already-evaluated chunks.
    input_drained
        Whether the input channel is drained.
    tracer
        Optional tracer for runtime metrics.
    """
    shuffle_context = context
    shuffle_comm = comm
    if local and comm.nranks > 1:
        options = Options(get_environment_variables())
        shuffle_comm = single_comm(options, comm.progress_thread)
        shuffle_context = Context(shuffle_comm.logger, context.br(), options)
    modulus = max(shuffle_comm.nranks, modulus)
    metadata_out = _make_hash_shuffle_metadata(
        shuffle_comm, decomposed.output_indices, modulus, metadata_in
    )
    await send_metadata(ch_out, context, metadata_out)

    shuffle = ShuffleManager(
        shuffle_context,
        shuffle_comm,
        modulus,
        collective_id,
    )
    async with shuffle.inserting() as inserter:
        await inserter.insert_hash(
            _enforce_schema(
                aggregated,
                decomposed.reduction_ir.schema,
                context.br(),
            ),
            decomposed.shuffle_indices,
        )
        del aggregated

        while not input_drained:
            aggregated, input_drained, _ = await _local_aggregation(
                context,
                decomposed,
                ir_context,
                ch_in,
                target_partition_size,
            )
            await inserter.insert_hash(
                _enforce_schema(
                    aggregated, decomposed.reduction_ir.schema, context.br()
                ),
                decomposed.shuffle_indices,
            )
            del aggregated
    extract_irs = [decomposed.reduction_ir] + (
        [decomposed.select_ir] if decomposed.select_ir else []
    )
    for partition_id in shuffle.local_partitions():
        stream = ir_context.get_cuda_stream()
        partition_chunk = TableChunk.from_pylibcudf_table(
            await shuffle.extract_chunk(partition_id, stream),
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        partition_chunk = await evaluate_chunk(
            context,
            partition_chunk,
            *extract_irs,
            ir_context=ir_context,
        )
        await send_chunk(context, ch_out, partition_chunk, partition_id, tracer=tracer)

    await ch_out.drain(context)


def _key_run_split(
    chunk: TableChunk, keys: tuple[OrderKey, ...], *, leading: bool
) -> int:
    """Find the split after the first or before the last key run."""
    table = chunk.table_view()
    key_table = plc.Table([table.columns()[key.column_index] for key in keys])
    row = 0 if leading else chunk.shape[0] - 1
    boundary = plc.copying.slice(key_table, [row, row + 1], stream=chunk.stream)[0]
    search = plc.search.upper_bound if leading else plc.search.lower_bound
    split = (
        search(
            key_table,
            boundary,
            [key.order for key in keys],
            [key.null_order for key in keys],
            stream=chunk.stream,
        )
        .to_scalar(stream=chunk.stream)
        .to_py(stream=chunk.stream)
    )
    assert isinstance(split, int)
    return split


async def _split_key_run(
    context: Context,
    ir_context: IRExecutionContext,
    chunk: TableChunk,
    keys: tuple[OrderKey, ...],
    *,
    leading: bool,
) -> tuple[TableChunk | None, TableChunk | None]:
    """Split off the first or last run of equal ordering keys."""
    if not (nrows := chunk.shape[0]):
        return None, None
    chunk, extra = await make_table_chunks_available_or_wait(
        context, chunk, reserve_extra=0, net_memory_delta=0
    )
    with opaque_memory_usage(extra):
        split = await ir_context.to_thread(_key_run_split, chunk, keys, leading=leading)
        if split == 0:
            return None, chunk
        if split == nrows:
            return chunk, None
    chunk, extra = await make_table_chunks_available_or_wait(
        context, chunk, reserve_extra=chunk.data_alloc_size(), net_memory_delta=0
    )
    with opaque_memory_usage(extra):
        table = chunk.table_view()
        left, right = (
            TableChunk.from_pylibcudf_table(
                view.copy(chunk.stream, context.br().device_mr),
                chunk.stream,
                exclusive_view=True,
                br=context.br(),
            )
            for view in plc.copying.split(table, [split], stream=chunk.stream)
        )
        return left, right


async def _rank_ordering(
    context: Context, ordering: Ordering, nranks: int
) -> Ordering | None:
    """Select only the boundaries between ranks from an ordered partitioning."""
    npartitions = ordering.num_boundaries + 1
    if npartitions < nranks:
        return None
    if npartitions == nranks:
        return ordering
    positions = [
        partition_range(rank, nranks, npartitions)[0] - 1 for rank in range(1, nranks)
    ]
    boundaries, extra = await make_table_chunks_available_or_wait(
        context,
        ordering.get_boundaries(context.br()),
        reserve_extra=0,
        net_memory_delta=0,
    )
    with opaque_memory_usage(extra):
        indices = plc.Column.from_iterable_of_py(
            positions, plc.DataType(plc.TypeId.INT32), stream=boundaries.stream
        )
        selected = plc.copying.gather(
            boundaries.table_view(),
            indices,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            stream=boundaries.stream,
        )
        rank_boundaries = TableChunk.from_pylibcudf_table(
            selected, boundaries.stream, exclusive_view=True, br=context.br()
        )
    return Ordering(
        ordering.keys,
        rank_boundaries,
        strict_boundaries=ordering.strict_boundaries,
    )


async def _strictify_chunkwise_groupby(
    context: Context,
    comm: Communicator,
    ir: GroupBy,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    input_ordering: Ordering,
    output_ordering: Ordering,
    collective_id: int,
    tracer: ActorTracer | None,
) -> None:
    """Make input key ranges strict, then group each owned partition once."""
    start, stop = partition_range(
        comm.rank, comm.nranks, output_ordering.num_boundaries + 1
    )
    adjusted_metadata = ChannelMetadata(
        local_count=stop - start,
        partitioning=Partitioning(
            OrderScheme([output_ordering]),
            "inherit",
            row_orderings=(RowOrdering(output_ordering.keys, RowOrderingScope.GLOBAL),),
        ),
        duplicated=False,
    )
    metadata_out = ChannelMetadata(
        local_count=adjusted_metadata.local_count,
        partitioning=maybe_remap_partitioning(
            ir,
            adjusted_metadata.partitioning,
            context=context,
            preserves_row_order=ir.preserves_output_order,
        ),
        duplicated=adjusted_metadata.duplicated,
    )
    ch_adjusted = context.create_channel()
    if tracer is not None:
        tracer.decision = "strictify_chunkwise"
    async with shutdown_channels_on_error(context, ch_adjusted):
        await gather_in_task_group(
            adjust_ordering(
                context,
                comm,
                ir.children[0],
                ir_context,
                ch_adjusted,
                ch_in,
                input_ordering,
                output_ordering,
                input_rows_ordered=True,
                collective_id=collective_id,
            ),
            chunkwise_evaluate(
                context,
                ir,
                ir_context,
                ch_out,
                ch_adjusted,
                metadata_out,
                input_metadata=adjusted_metadata,
                tracer=tracer,
            ),
        )


@dataclass
class _OrderedGroupCursor:
    """Carry and emit state for one ordered groupby stream."""

    initial: TableChunk | None = None
    carry: TableChunk | None = None
    sequence_number: int = 0
    send_first: bool = False

    async def emit(self, groupby: _OrderedGroupBy, chunk: TableChunk | None) -> None:
        if chunk is None or chunk.shape[0] == 0:
            return
        if groupby.decomposed.select_ir is not None:
            chunk = await evaluate_chunk(
                groupby.context,
                chunk,
                groupby.decomposed.select_ir,
                ir_context=groupby.ir_context,
            )
        await send_chunk(
            groupby.context,
            groupby.ch_out,
            chunk,
            self.sequence_number,
            tracer=groupby.tracer,
        )
        self.sequence_number += 1

    async def emit_complete(
        self,
        groupby: _OrderedGroupBy,
        chunk: TableChunk,
        exchange: SparseExchange,
        destination: int,
    ) -> None:
        if self.send_first:
            first, chunk = await _split_key_run(
                groupby.context,
                groupby.ir_context,
                chunk,
                groupby.keys,
                leading=True,
            )
            assert first is not None
            await exchange.insert_chunk(destination, first)
            self.send_first = False
            exchange.start_finish()
        await self.emit(groupby, chunk)


@dataclass(frozen=True)
class _OrderedGroupBy:
    """
    Stream a groupby whose keys have an ordered prefix.

    After initial local aggregation and strategy selection:

    1. Sort partial results on the prefix if aggregation did not maintain order.
    2. Carry the trailing prefix run between batches and emit completed runs.
    3. Send the leading run to its lower rank when boundaries overlap, then
       merge incoming runs with the final local run. Without boundaries,
       gathered first keys choose a conservative owner.

    Output stays ordered on the prefix. Known inter-rank boundaries become
    strict, but local chunk boundaries need not be. A prefix run may exceed
    the target partition size or cuDF row limit. Splitting it would require
    combining equal full keys across the pieces.
    """

    context: Context
    comm: Communicator
    decomposed: DecomposedGroupBy
    ir_context: IRExecutionContext
    ch_out: Channel[TableChunk]
    ch_in: Channel[TableChunk]
    local_count: int
    first_key_id: int
    exchange_id: int
    target_partition_size: int
    keys: tuple[OrderKey, ...]
    inter_rank_ordering: Ordering | None
    tracer: ActorTracer | None

    @staticmethod
    def ordered_prefix(
        ir: GroupBy | Distinct,
        metadata: ChannelMetadata,
        partitioning: NormalizedPartitioning,
        group_keys: tuple[int, ...],
        inter_rank_ordering: Ordering | None,
        nranks: int,
    ) -> tuple[OrderKey, ...] | None:
        """Find the longest ordered prefix eligible for cursor reduction."""
        if not isinstance(ir, GroupBy) or not group_keys or metadata.duplicated:
            return None
        if nranks > 1 and isinstance(partitioning.inter_rank_scheme, HashScheme):
            return None
        if _has_stable_sorted_agg(ir.agg_requests):
            return None

        row_orderings = (
            metadata.partitioning.row_orderings if metadata.partitioning else ()
        )
        best: tuple[OrderKey, ...] | None = None
        for row in row_orderings:
            if row.scope != RowOrderingScope.GLOBAL or row.group_keys:
                continue
            prefix = row.order_keys[: len(group_keys)]
            if not prefix:
                continue
            if tuple(key.column_index for key in prefix) != group_keys[: len(prefix)]:
                continue
            if best is None or len(prefix) > len(best):
                best = prefix
        if inter_rank_ordering is None:
            return best
        keys = inter_rank_ordering.keys
        if tuple(key.column_index for key in keys) != group_keys[: len(keys)]:
            return best
        if (best is None or len(keys) > len(best)) and is_row_ordered_by(
            metadata.partitioning, keys, scope=RowOrderingScope.RANK_LOCAL
        ):
            return keys
        return best

    async def run(self, initial: TableChunk, *, input_drained: bool) -> None:
        """Prepare rank ownership and metadata, then stream the reduction."""
        initial = _enforce_schema(
            initial, self.decomposed.reduction_ir.schema, self.context.br()
        )
        sort_ir = None
        if (
            isinstance(self.decomposed.ir, GroupBy)
            and not self.decomposed.ir.maintain_order
        ):
            schema_items = tuple(self.decomposed.reduction_ir.schema.items())
            by = []
            for key in self.keys:
                name, dtype = schema_items[key.column_index]
                by.append(NamedExpr(name, Col(dtype, name)))
            sort_ir = Sort(
                self.decomposed.reduction_ir.schema,
                by,
                [key.order for key in self.keys],
                [key.null_order for key in self.keys],
                False,  # noqa: FBT003
                None,
                self.decomposed.reduction_ir,
            )

        initial = await self._sort(initial, sort_ir)
        if self.comm.nranks > 1 and (
            self.inter_rank_ordering is None
            or not self.inter_rank_ordering.strict_boundaries
        ):
            routes, initial = await self._first_key_routes(initial)
        else:
            routes = {}

        output_keys = tuple(
            OrderKey(index, key.order, key.null_order)
            for index, key in zip(
                self.decomposed.output_indices[: len(self.keys)],
                self.keys,
                strict=True,
            )
        )
        scheme = None
        if self.inter_rank_ordering is not None:
            strict = Ordering(
                self.inter_rank_ordering.keys,
                self.inter_rank_ordering.get_boundaries(self.context.br()),
                strict_boundaries=True,
            )
            scheme = OrderScheme(
                [
                    remap_ordering_keys(
                        strict, self.decomposed.output_indices[: len(strict.keys)]
                    )
                ]
            )
        partitioning = Partitioning(
            scheme,
            None,
            row_orderings=(RowOrdering(output_keys, RowOrderingScope.GLOBAL),),
        )
        await send_metadata(
            self.ch_out,
            self.context,
            ChannelMetadata(
                local_count=self.local_count,
                partitioning=partitioning,
                duplicated=False,
            ),
        )
        if self.tracer is not None:
            self.tracer.decision = "ordered"
        cursor = _OrderedGroupCursor(initial=initial)
        del initial
        await self._run_cursor(cursor, sort_ir, routes, input_drained=input_drained)

    @staticmethod
    def _route_first_keys(
        gathered: plc.Table,
        keys: tuple[OrderKey, ...],
        stream: Stream,
        boundaries: TableChunk | None,
        context: Context,
    ) -> dict[int, int]:
        """Route leading runs using gathered keys and optional rank boundaries."""
        first_keys = plc.Table(gathered.columns()[1:])
        orders = [key.order for key in keys]
        null_orders = [key.null_order for key in keys]
        if boundaries is None:
            lower_column = plc.search.lower_bound(
                first_keys, first_keys, orders, null_orders, stream=stream
            )
            stream.synchronize()
            ranks = gathered.columns()[0].to_pylist()
            lower = lower_column.to_pylist()
            return {
                source: ranks[position - 1] if position else ranks[0]
                for source, position in zip(ranks, lower, strict=True)
            }

        with stream_ordered_after(
            context.br().stream_pool.get_stream,
            upstreams=(stream, boundaries.stream),
        ) as search_stream:
            boundary_table = boundaries.table_view()
            lower_column = plc.search.lower_bound(
                boundary_table, first_keys, orders, null_orders, stream=search_stream
            )
            upper_column = plc.search.upper_bound(
                boundary_table, first_keys, orders, null_orders, stream=search_stream
            )
            search_stream.synchronize()
            ranks = gathered.columns()[0].to_pylist()
            lower = lower_column.to_pylist()
            upper = upper_column.to_pylist()
        return {
            source: first_boundary
            if first_boundary <= source - 1 < after_last
            else source
            for source, first_boundary, after_last in zip(
                ranks, lower, upper, strict=True
            )
        }

    async def _first_key_routes(
        self, initial: TableChunk
    ) -> tuple[dict[int, int], TableChunk]:
        """Find the lower owner of each rank's leading prefix run."""
        stream = self.ir_context.get_cuda_stream()
        chunk, extra = await make_table_chunks_available_or_wait(
            self.context, initial, reserve_extra=0, net_memory_delta=0
        )
        with opaque_memory_usage(extra):
            table = chunk.table_view()
            key_table = plc.Table(
                [table.columns()[key.column_index] for key in self.keys]
            )
            first = (
                plc.copying.slice(key_table, [0, 1], stream=chunk.stream)[0]
                if chunk.shape[0]
                else key_table
            )
            rank_type = plc.DataType(plc.TypeId.INT32)
            rank = plc.Column.from_scalar(
                plc.Scalar.from_py(self.comm.rank, rank_type, stream=chunk.stream),
                first.num_rows(),
                stream=chunk.stream,
            )
            first_chunk = TableChunk.from_pylibcudf_table(
                plc.Table([rank, *first.columns()]),
                chunk.stream,
                exclusive_view=False,
                br=self.context.br(),
            )
        gather = AllGatherManager(self.context, self.comm, self.first_key_id)
        with gather.inserting() as inserter:
            await inserter.insert(0, first_chunk)
        gathered = await gather.extract_concatenated(stream, ir_context=self.ir_context)
        if gathered.num_rows() == 0:
            return {}, chunk
        if self.inter_rank_ordering is None:
            routes = await self.ir_context.to_thread(
                self._route_first_keys, gathered, self.keys, stream, None, self.context
            )
            return routes, chunk

        boundaries, extra = await make_table_chunks_available_or_wait(
            self.context,
            self.inter_rank_ordering.get_boundaries(self.context.br()),
            reserve_extra=0,
            net_memory_delta=0,
        )
        with opaque_memory_usage(extra):
            routes = await self.ir_context.to_thread(
                self._route_first_keys,
                gathered,
                self.keys,
                stream,
                boundaries,
                self.context,
            )
        return routes, chunk

    async def _sort(self, chunk: TableChunk, sort_ir: Sort | None) -> TableChunk:
        return (
            await evaluate_chunk(
                self.context, chunk, sort_ir, ir_context=self.ir_context
            )
            if sort_ir is not None
            else chunk
        )

    async def _consume_local(
        self,
        cursor: _OrderedGroupCursor,
        sort_ir: Sort | None,
        exchange: SparseExchange,
        destination: int,
        *,
        input_drained: bool,
    ) -> None:
        current, cursor.initial = cursor.initial, None
        assert current is not None
        drained = input_drained
        while True:
            if current.shape[0]:
                if cursor.carry is not None:
                    first, current = await _split_key_run(
                        self.context, self.ir_context, current, self.keys, leading=True
                    )
                    assert first is not None
                    head = await evaluate_batch(
                        [cursor.carry, first],
                        self.context,
                        self.decomposed.reduction_ir,
                        ir_context=self.ir_context,
                    )
                    head = await self._sort(head, sort_ir)
                    cursor.carry = None
                    if current is not None:
                        await cursor.emit_complete(self, head, exchange, destination)
                    else:
                        current = head
                if current is not None:
                    complete, cursor.carry = await _split_key_run(
                        self.context, self.ir_context, current, self.keys, leading=False
                    )
                    if complete is not None:
                        await cursor.emit_complete(
                            self, complete, exchange, destination
                        )
            if drained:
                break
            current, drained, _ = await _local_aggregation(
                self.context,
                self.decomposed,
                self.ir_context,
                self.ch_in,
                self.target_partition_size,
            )
            current = _enforce_schema(
                current, self.decomposed.reduction_ir.schema, self.context.br()
            )
            current = await self._sort(current, sort_ir)

    async def _run_cursor(
        self,
        cursor: _OrderedGroupCursor,
        sort_ir: Sort | None,
        routes: dict[int, int],
        *,
        input_drained: bool,
    ) -> None:
        destination = routes.get(self.comm.rank, self.comm.rank)
        sources = tuple(
            source
            for source, owner in routes.items()
            if owner == self.comm.rank and source != self.comm.rank
        )
        destinations = (destination,) if destination != self.comm.rank else ()
        cursor.send_first = bool(destinations)

        async with SparseExchange(
            self.context,
            self.comm,
            self.exchange_id,
            sources=sources,
            destinations=destinations,
        ) as exchange:
            if not cursor.send_first:
                exchange.start_finish()
            await self._consume_local(
                cursor,
                sort_ir,
                exchange,
                destination,
                input_drained=input_drained,
            )
            if cursor.send_first and cursor.carry is not None:
                await exchange.insert_chunk(destination, cursor.carry)
                cursor.carry = None
                exchange.start_finish()
            await exchange.finish()
            for source in sources:
                remote = exchange.extract(source)
                try:
                    for message in remote:
                        piece = TableChunk.from_message(message, br=self.context.br())
                        cursor.carry = (
                            await evaluate_batch(
                                [cursor.carry, piece],
                                self.context,
                                self.decomposed.reduction_ir,
                                ir_context=self.ir_context,
                            )
                            if cursor.carry is not None
                            else piece
                        )
                finally:
                    remote.clear()
            if cursor.carry is not None and sources:
                cursor.carry = await self._sort(cursor.carry, sort_ir)
            await cursor.emit(self, cursor.carry)
        await self.ch_out.drain(self.context)


def _enforce_schema(
    chunk: TableChunk,
    canonical_schema: dict[str, Any],
    br: BufferResource,
) -> TableChunk:
    """Enforce the canonical schema of a TableChunk."""
    tbl = chunk.table_view()
    cols = tbl.columns()
    names = list(canonical_schema.keys())
    if len(cols) != len(names):  # pragma: no cover
        raise ValueError(
            f"Column count ({len(cols)}) does not match schema ({len(names)})"
        )

    target_plcs = []
    needs_cast = False
    for col, name in zip(cols, names, strict=True):
        dt = canonical_schema[name]
        target_plc = (dt if isinstance(dt, DataType) else DataType(dt)).plc_type
        target_plcs.append(target_plc)
        if col.type().id() != target_plc.id():
            needs_cast = True
    if not needs_cast:
        return chunk
    new_columns = [
        plc.unary.cast(col, target_plc, stream=chunk.stream)
        if col.type().id() != target_plc.id()
        else col
        for col, target_plc in zip(cols, target_plcs, strict=True)
    ]
    return TableChunk.from_pylibcudf_table(
        plc.Table(new_columns), chunk.stream, exclusive_view=True, br=br
    )


def _key_indices(
    ir: GroupBy | Distinct, schema: Schema, *, concrete_prefix: bool = False
) -> tuple[int, ...]:
    schema_keys = {n: i for i, n in enumerate(schema.keys())}
    if isinstance(ir, GroupBy):
        if concrete_prefix:
            return names_to_indices(ir.keys, schema, concrete_prefix=True)
        groupby_key_names = tuple(ne.name for ne in ir.keys)
        if not all(k in schema_keys for k in groupby_key_names):
            return ()
        return tuple(schema_keys[k] for k in groupby_key_names)
    else:
        subset = ir.subset or frozenset(ir.schema)
        return tuple(schema_keys[k] for k in schema if k in subset)


def _maintain_order(ir: GroupBy | Distinct) -> bool:
    if isinstance(ir, GroupBy):
        return (
            ir.maintain_order
            or _has_stable_sorted_agg(ir.agg_requests)
            or _has_order_sensitive_agg(ir.agg_requests)
        )
    else:
        return ir.stable or ir.keep in (
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
        )


async def _choose_strategy(
    context: Context,
    comm: Communicator,
    local_count: int,
    aggregated: TableChunk,
    chunks_received: int,
    input_drained: bool,  # noqa: FBT001
    collective_ids: list[int],
    target_partition_size: int,
    skip_global_comm: bool,  # noqa: FBT001
    maintain_order: bool,  # noqa: FBT001
    tracer: ActorTracer | None,
) -> int:
    """
    Select the best algorithm for the given context and metadata.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    local_count
        The local count of the input channel.
    aggregated
        The aggregated result for already-evaluated chunks.
    chunks_received
        The number of chunks received from the input channel.
    input_drained
        Whether the input channel is drained.
    collective_ids
        The collective IDs.
    target_partition_size
        The target partition size.
    skip_global_comm
        Whether to skip the global communication.
    maintain_order
        Whether the operation should maintain input ordering semantics.
    tracer
        Optional tracer for runtime metrics.

    Returns
    -------
    The output count.
    """
    aggregated_size = aggregated.data_alloc_size()
    sample_count = max(1, chunks_received)
    local_estimated_size = (aggregated_size // sample_count) * local_count
    if input_drained:
        local_estimated_rows = aggregated.shape[0]
    else:
        local_estimated_rows = (
            aggregated.shape[0] * local_count + sample_count - 1
        ) // sample_count

    if skip_global_comm:
        total_estimated_size = local_estimated_size
        total_estimated_rows = local_estimated_rows
        total_chunk_count = local_count
        total_need_shuffle = int(not input_drained)
    else:
        (
            total_estimated_size,
            total_estimated_rows,
            total_chunk_count,
            total_need_shuffle,
        ) = await allgather_reduce(
            context,
            comm,
            collective_ids[-1],
            local_estimated_size,
            local_estimated_rows,
            local_count,
            int(not input_drained),
        )

    min_row_limit_count = 1
    if total_estimated_rows > 0:
        min_row_limit_count = (
            total_estimated_rows + MAX_ROWS_PER_PARTITION - 1
        ) // MAX_ROWS_PER_PARTITION

    ideal_count = 1
    use_tree = maintain_order or (
        total_need_shuffle == 0
        and (skip_global_comm or total_estimated_rows < MAX_ROWS_PER_PARTITION)
    )
    if not use_tree:
        ideal_count = max(
            2, total_estimated_size // target_partition_size, min_row_limit_count
        )

    output_count_limit = local_count if skip_global_comm else total_chunk_count
    output_count = min(ideal_count, output_count_limit)
    if not use_tree:
        output_count = max(2, output_count, min_row_limit_count)
    if tracer is not None:
        tracer.decision = (
            "tree_local"
            if skip_global_comm and use_tree
            else "shuffle_local"
            if skip_global_comm
            else "tree_allgather"
            if use_tree
            else "shuffle"
        )

    return output_count


@define_actor()
async def groupby_actor(
    context: Context,
    comm: Communicator,
    ir: GroupBy | Distinct,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    target_partition_size: int,
    collective_ids: list[int],
) -> None:
    """
    Dynamic GroupBy or Distinct actor that selects the best strategy at runtime.

    Strategy selection uses input metadata and observed data:
    - Chunk-wise: Data already partitioned on the necessary keys
    - Strictify: Ordered input with flat boundaries
    - Tree reduction: Small estimated output (< target_partition_size)
    - Ordered cursor: Large output with an ordered grouping-key prefix
    - Shuffle: Large estimated output requiring redistribution

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The IR node to evaluate.
    ir_context
        The IR execution context.
    ch_out
        The output channel.
    ch_in
        The input channel.
    target_partition_size
        The target partition size.
    collective_ids
        The collective IDs.
    """
    async with shutdown_on_error(
        context,
        chs_in=(ch_in,),
        chs_out=(ch_out,),
        trace_ir=ir,
        ir_context=ir_context,
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        metadata_in = await recv_metadata(ch_in, context)

        nranks = comm.nranks
        group_keys = _key_indices(ir, ir.children[0].schema, concrete_prefix=True)
        partitioning = NormalizedPartitioning.from_keys(
            metadata_in.partitioning,
            nranks,
            keys=group_keys,
        )
        partitioning_level: PartitioningLevel = (
            "local" if metadata_in.duplicated else "flat"
        )
        maintain_order = _maintain_order(ir)
        preserves_output_order = ir.preserves_output_order
        fully_partitioned = partitioning.is_strictly_partitioned(
            level=partitioning_level,
        )
        inter_rank_ordering = partitioning.get_ordering(level="inter_rank")
        ordered_keys = _OrderedGroupBy.ordered_prefix(
            ir, metadata_in, partitioning, group_keys, inter_rank_ordering, nranks
        )
        fallback_case = (
            # NOTE: This criteria means that we fell back
            # to one partition at lowering time.
            metadata_in.local_count == 1
            and (metadata_in.duplicated or nranks == 1)
            and isinstance(ir.children[0], Repartition)
        )

        if fully_partitioned or fallback_case:
            if tracer is not None:
                tracer.decision = "chunkwise"
            output_partitioning = maybe_remap_partitioning(
                ir,
                metadata_in.partitioning,
                context=context,
                preserves_row_order=preserves_output_order,
            )
            metadata_out = ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=output_partitioning,
                duplicated=metadata_in.duplicated,
            )
            await chunkwise_evaluate(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_out,
                input_metadata=metadata_in,
                tracer=tracer,
            )
            return

        flat_ordering = None if metadata_in.duplicated else partitioning.get_ordering()
        if (
            isinstance(ir, GroupBy)
            and flat_ordering is not None
            and is_row_ordered_by(
                metadata_in.partitioning,
                flat_ordering.keys,
                scope=RowOrderingScope.RANK_LOCAL,
            )
        ):
            strict_ordering = await ir_context.to_thread(
                flat_ordering.as_strict, context.br()
            )
            # Collapsed boundaries could require assembling many raw chunks at once.
            if strict_ordering.num_boundaries == flat_ordering.num_boundaries:
                await _strictify_chunkwise_groupby(
                    context,
                    comm,
                    ir,
                    ir_context,
                    ch_out,
                    ch_in,
                    flat_ordering,
                    strict_ordering,
                    collective_ids.pop(),
                    tracer,
                )
                return

        decomposed = DecomposedGroupBy.from_ir(ir)
        assert not decomposed.need_preshuffle, "Should already be shuffled."

        aggregated, input_drained, chunks_received = await _local_aggregation(
            context,
            decomposed,
            ir_context,
            ch_in,
            target_partition_size,
            allow_early_exit=not maintain_order or ordered_keys is not None,
        )

        skip_global_comm = metadata_in.duplicated or isinstance(
            partitioning.inter_rank_scheme, HashScheme
        )
        output_count = await _choose_strategy(
            context,
            comm,
            metadata_in.local_count,
            aggregated,
            chunks_received,
            input_drained,
            collective_ids,
            target_partition_size,
            skip_global_comm,
            maintain_order and ordered_keys is None,
            tracer,
        )

        # RapidsMPF permits ID reuse after AllGather.extract_all completes.
        # Any strategy-selection AllGather finishes before reduction starts.
        if output_count == 1:
            await _tree_reduce(
                context,
                comm,
                decomposed,
                ir_context,
                ch_out,
                metadata_in,
                collective_ids.pop(),
                local=skip_global_comm,
                aggregated=aggregated,
                tracer=tracer,
            )
        elif ordered_keys is not None:
            local_count = (
                min(2, max(1, aggregated.shape[0]))
                if input_drained
                else max(1, metadata_in.local_count)
            )
            partial_keys = tuple(
                OrderKey(index, key.order, key.null_order)
                for index, key in zip(
                    decomposed.shuffle_indices[: len(ordered_keys)],
                    ordered_keys,
                    strict=True,
                )
            )
            compatible_inter_rank = (
                await _rank_ordering(context, inter_rank_ordering, nranks)
                if inter_rank_ordering is not None
                and inter_rank_ordering.keys == ordered_keys
                and nranks > 1
                else None
            )
            ordered = _OrderedGroupBy(
                context=context,
                comm=comm,
                decomposed=decomposed,
                ir_context=ir_context,
                ch_out=ch_out,
                ch_in=ch_in,
                local_count=local_count,
                first_key_id=collective_ids[-1],
                exchange_id=collective_ids[0],
                target_partition_size=target_partition_size,
                keys=partial_keys,
                inter_rank_ordering=compatible_inter_rank,
                tracer=tracer,
            )
            ordered_run = ordered.run(aggregated, input_drained=input_drained)
            del aggregated
            await ordered_run
        else:
            await _shuffle_reduce(
                context,
                comm,
                decomposed,
                ir_context,
                ch_out,
                ch_in,
                metadata_in,
                output_count,
                collective_ids.pop(),
                target_partition_size,
                local=skip_global_comm,
                aggregated=aggregated,
                input_drained=input_drained,
                tracer=tracer,
            )


@generate_ir_sub_network.register(GroupBy)
@generate_ir_sub_network.register(Distinct)
def _(
    ir: GroupBy | Distinct, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate sub-network for GroupBy or Distinct operation."""
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming"

    if config_options.executor.dynamic_planning is None:
        # Fall back to the default IR handler (bypass GroupBy/Distinct dispatch)
        return generate_ir_sub_network.dispatch(IR)(ir, rec)

    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    collective_ids = list(rec.state["collective_id_map"].get(ir, []))
    ir_context = ir_context_for_node(rec, ir)
    assert len(collective_ids) == 2, (
        f"{type(ir).__name__} requires 2 collective IDs, got {len(collective_ids)}"
    )
    actors[ir] = [
        groupby_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context,
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            config_options.executor.target_partition_size,
            collective_ids,
        )
    ]

    return actors, channels
