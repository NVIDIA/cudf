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
)
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.communicator.single import new_communicator as single_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.context import Context

from cudf_polars.containers import DataType
from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.dsl.ir import IR, Distinct, GroupBy, Select
from cudf_polars.dsl.utils.naming import names_to_indices, unique_names
from cudf_polars.streaming.actor_graph.collectives.shuffle import ShuffleManager
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    MAX_ROWS_PER_PARTITION,
    ChannelManager,
    NormalizedPartitioning,
    _make_hash_shuffle_metadata,
    allgather_and_reduce,
    allgather_reduce,
    chunkwise_evaluate,
    empty_table_chunk,
    evaluate_batch,
    evaluate_chunk,
    maybe_remap_partitioning,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)
from cudf_polars.streaming.groupby import combine, decompose
from cudf_polars.streaming.repartition import Repartition

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.channel import Channel

    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
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
        inserter.insert_hash(
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
            inserter.insert_hash(
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
            shuffle.extract_chunk(partition_id, stream),
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
        return ir.maintain_order
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
            collective_ids.pop(),
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

    Strategy selection based on observed data:
    - Chunk-wise: Data already partitioned on the necessary keys
    - Tree reduction: Small estimated output (< target_partition_size)
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
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ) as tracer:
        metadata_in = await recv_metadata(ch_in, context)

        nranks = comm.nranks
        partitioning = NormalizedPartitioning.from_keys(
            metadata_in.partitioning,
            nranks,
            keys=_key_indices(ir, ir.children[0].schema, concrete_prefix=True),
        )
        maintain_order = _maintain_order(ir)
        fully_partitioned = partitioning.is_strictly_partitioned()
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
            metadata_out = ChannelMetadata(
                local_count=metadata_in.local_count,
                partitioning=maybe_remap_partitioning(
                    ir,
                    metadata_in.partitioning,
                    child_ir=ir.children[0],
                    context=context,
                ),
                duplicated=metadata_in.duplicated,
            )
            await chunkwise_evaluate(
                context,
                ir,
                ir_context,
                ch_out,
                ch_in,
                metadata_out,
                tracer=tracer,
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
            allow_early_exit=not maintain_order,
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
            maintain_order,
            tracer,
        )

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
    assert len(collective_ids) == 2, (
        f"{type(ir).__name__} requires 2 collective IDs, got {len(collective_ids)}"
    )
    actors[ir] = [
        groupby_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            config_options.executor.target_partition_size,
            collective_ids,
        )
    ]

    return actors, channels
