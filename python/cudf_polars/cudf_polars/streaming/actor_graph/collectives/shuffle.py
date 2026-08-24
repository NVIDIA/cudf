# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shuffle logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pylibcudf as plc
import pylibcudf.partitioning
from cudf_streaming.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from cudf_streaming.partition_utils import (
    partition_and_pack as py_partition_and_pack,
    split_and_pack as py_split_and_pack,
    unpack_and_concat as py_unpack_and_concat,
)
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.communicator.single import new_communicator as single_comm
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync
from rapidsmpf.streaming.core.actor import define_actor
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.traversal import traversal
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
)
from cudf_polars.streaming.actor_graph.nodes import shutdown_on_error
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    _is_already_partitioned,
    recv_metadata,
    send_metadata,
)
from cudf_polars.streaming.shuffle import Shuffle
from cudf_polars.utils.cuda_stream import stream_ordered_after

if TYPE_CHECKING:
    from collections.abc import Generator

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.packed_data import PackedData
    from rapidsmpf.streaming.core.channel import Channel
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.expr import NamedExpr
    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.core import SubNetGenerator
    from cudf_polars.typing import Schema


class ShuffleManager:
    """
    ShufflerAsync manager.

    Parameters
    ----------
    context
        The streaming context.
    comm
        The communicator.
    num_partitions
        The number of partitions to shuffle into.
    collective_id
        The collective ID.
    partition_assignment, optional
        How to assign partition IDs to ranks: ROUND_ROBIN (default) or
        CONTIGUOUS. Use CONTIGUOUS for sort so each rank gets adjacent
        partition IDs and concatenation order matches global order.
    """

    class Inserter:
        """
        Context manager for the insert phase of a shuffle operation.

        Obtained via :meth:`ShuffleManager.inserting`. On exit, signals
        the end of insertion to all ranks by calling ``insert_finished()``.

        Parameters
        ----------
        manager
            The shuffle manager to insert into.
        """

        def __init__(self, manager: ShuffleManager):
            self._manager = manager

        def insert_hash(
            self, chunk: TableChunk, columns_to_hash: tuple[int, ...]
        ) -> None:
            """Partition chunk by hash and insert into the shuffler."""
            self._manager.shuffler.insert(
                py_partition_and_pack(
                    table=chunk.table_view(),
                    columns_to_hash=columns_to_hash,
                    num_partitions=self._manager.num_partitions,
                    stream=chunk.stream,
                    br=self._manager.context.br(),
                )
            )

        def insert_hash_keys(self, chunk: TableChunk, key_table: plc.Table) -> None:
            """Partition chunk by hash of a row-aligned key table and insert."""
            partitioned_table, offsets = plc.partitioning.hash_partition(
                chunk.table_view(),
                key_table,
                self._manager.num_partitions,
                stream=chunk.stream,
            )
            self._manager.shuffler.insert(
                py_split_and_pack(
                    table=partitioned_table,
                    splits=list(offsets[1:-1]),
                    stream=chunk.stream,
                    br=self._manager.context.br(),
                )
            )

        def insert_split(self, chunk: TableChunk, splits: list[int]) -> None:
            """Split chunk at the given indices and insert into the shuffler."""
            self._manager.shuffler.insert(
                py_split_and_pack(
                    table=chunk.table_view(),
                    splits=splits,
                    stream=chunk.stream,
                    br=self._manager.context.br(),
                )
            )

        def insert_index(self, chunk: TableChunk, partition_map: TableChunk) -> None:
            """
            Partition chunk by a separate single-column partition-map and insert.

            Parameters
            ----------
            chunk
                The payload chunk to partition. Its schema is preserved
                unchanged in the shuffler output.
            partition_map
                Single-column ``TableChunk`` whose integer values give the
                target partition ID for each row. Must be row-aligned with
                ``chunk``.
            """
            with stream_ordered_after(
                self._manager.context.br().stream_pool.get_stream,
                upstreams=(chunk.stream, partition_map.stream),
            ) as stream:
                partition_map_col = partition_map.table_view().columns()[0]
                reordered, offsets = plc.partitioning.partition(
                    chunk.table_view(),
                    partition_map_col,
                    self._manager.num_partitions,
                    stream=stream,
                )
                self._manager.shuffler.insert(
                    py_split_and_pack(
                        table=reordered,
                        splits=list(offsets[1:-1]),
                        stream=stream,
                        br=self._manager.context.br(),
                    )
                )

        async def __aenter__(self) -> ShuffleManager.Inserter:
            """Enter the context manager."""
            return self

        async def __aexit__(self, *args: object) -> None:
            """Exit the context manager, calling ``insert_finished()``."""
            await self._manager.shuffler.insert_finished(self._manager.context)

    def __init__(
        self,
        context: Context,
        comm: Communicator,
        num_partitions: int,
        collective_id: int,
        *,
        partition_assignment: PartitionAssignment = PartitionAssignment.ROUND_ROBIN,
    ):
        self.context = context
        self.comm = comm
        self.num_partitions = num_partitions
        self.collective_id = collective_id
        self.shuffler = ShufflerAsync(
            context,
            comm,
            collective_id,
            num_partitions,
            partition_assignment=partition_assignment,
        )

    def inserting(self) -> ShuffleManager.Inserter:
        """Return a context manager for the insert phase."""
        return ShuffleManager.Inserter(self)

    def local_partitions(self) -> list[int]:
        """Get the local partition IDs for this rank."""
        return self.shuffler.local_partitions()

    def extract_chunk(self, partition_id: int, stream: Stream) -> plc.Table:
        """
        Extract a chunk from the ShuffleManager.

        Parameters
        ----------
        partition_id
            The partition ID of the chunk to extract.
        stream
            The stream to use for chunk extraction.

        Returns
        -------
        The extracted table.
        """
        return py_unpack_and_concat(
            partitions=self.shuffler.extract(partition_id),
            stream=stream,
            br=self.context.br(),
        )

    def extract_pieces(self, partition_id: int) -> list[PackedData]:
        """
        Extract raw packed items for a partition without unpacking.

        Parameters
        ----------
        partition_id
            The partition ID to extract.

        Returns
        -------
        list[PackedData]
            Raw packed items for the partition.
        """
        return self.shuffler.extract(partition_id)


class LocalRepartitioner:
    """
    Local re-partitioner that wraps a completed :class:`ShuffleManager`.

    Parameters
    ----------
    shuffle
        Completed inter-rank :class:`ShuffleManager` (insertion phase done).
        The repartitioner consumes whatever local partitions this rank owns.
    local_count
        Number of local output partitions to produce.
    """

    def __init__(self, shuffle: ShuffleManager, local_count: int) -> None:
        self._global_shuffle = shuffle
        self._br = shuffle.context.br()
        options = Options(get_environment_variables())
        local_comm = single_comm(options, shuffle.comm.progress_thread)
        local_ctx = Context(local_comm.logger, self._br, options)
        self._local_shuffle = ShuffleManager(
            local_ctx,
            local_comm,
            local_count,
            shuffle.collective_id,
        )

    def _iter_chunks(self, stream: Stream) -> Generator[plc.Table, None, None]:
        for partition_id in self._global_shuffle.local_partitions():
            for piece in self._global_shuffle.extract_pieces(partition_id):
                # TODO: batch pieces up to target_partition_size before unpacking
                table = py_unpack_and_concat([piece], stream=stream, br=self._br)
                if table.num_rows() > 0:
                    yield table

    async def repartition_by_hash(
        self, *, columns_to_hash: tuple[int, ...], stream: Stream
    ) -> None:
        """
        Re-partition items by hash of the given columns.

        Parameters
        ----------
        columns_to_hash
            Tuple of column indices to use for hashing.
        stream
            CUDA stream for the unpack operation.
        """
        async with self._local_shuffle.inserting() as inserter:
            for table in self._iter_chunks(stream):
                inserter.insert_hash(
                    TableChunk.from_pylibcudf_table(
                        table, stream, exclusive_view=True, br=self._br
                    ),
                    columns_to_hash,
                )

    async def repartition_by_index(
        self,
        *,
        partition_col: int,
        stream: Stream,
        drop_partition_col: bool = True,
    ) -> None:
        """
        Re-partition items by a pre-computed integer column in the received data.

        Parameters
        ----------
        partition_col
            Index of the integer column whose values give the target
            local partition ID for each row.
        stream
            CUDA stream for the unpack operation.
        drop_partition_col
            If ``True`` (default), the partition column is stripped from the
            payload before inserting. If ``False``, it is kept in the output.
        """
        async with self._local_shuffle.inserting() as inserter:
            for table in self._iter_chunks(stream):
                cols = table.columns()
                payload = plc.Table(
                    [
                        c
                        for i, c in enumerate(cols)
                        if not drop_partition_col or i != partition_col
                    ]
                )
                partition_map = plc.Table([cols[partition_col]])
                inserter.insert_index(
                    TableChunk.from_pylibcudf_table(
                        payload, stream, exclusive_view=True, br=self._br
                    ),
                    TableChunk.from_pylibcudf_table(
                        partition_map, stream, exclusive_view=True, br=self._br
                    ),
                )

    def local_partitions(self) -> list[int]:
        """Return the local partition IDs."""
        return self._local_shuffle.local_partitions()

    def extract_chunk(self, partition_id: int, stream: Stream) -> plc.Table:
        """
        Extract the table for *partition_id* from the local shuffle.

        Parameters
        ----------
        partition_id
            The local partition to extract.
        stream
            CUDA stream for the unpack operation.
        """
        return self._local_shuffle.extract_chunk(partition_id, stream)


def _key_column_indices(
    keys: tuple[NamedExpr, ...], schema: Schema
) -> tuple[int, ...] | None:
    """Return column indices for simple column keys, or None for expressions."""
    columns = list(schema)
    indices: list[int] = []
    for key in keys:
        if not isinstance(key.value, Col):
            return None
        indices.append(columns.index(key.value.name))
    return tuple(indices)


def _ensure_pointwise_keys(keys: tuple[NamedExpr, ...]) -> None:
    if not all(expr.is_pointwise for expr in traversal([key.value for key in keys])):
        raise NotImplementedError("Shuffle requires pointwise key expressions.")


def _evaluate_key_table(
    chunk: TableChunk, keys: tuple[NamedExpr, ...], schema: Schema
) -> plc.Table:
    df = DataFrame.from_table(
        chunk.table_view(),
        list(schema.keys()),
        list(schema.values()),
        chunk.stream,
    )
    return plc.Table([key.evaluate(df).obj for key in keys])


async def _global_shuffle(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    keys_to_hash: tuple[NamedExpr, ...],
    input_schema: Schema,
    num_partitions: int,
    collective_id: int,
) -> None:
    """
    Global shuffle implementation.

    Parameters
    ----------
    context
        The streaming context.
    comm
        The communicator.
    ir_context
        The execution context for the IR node.
    ch_out
        Output Channel[TableChunk] with metadata and data channels.
    ch_in
        Input Channel[TableChunk] with metadata and data channels.
    keys_to_hash
        Tuple of expressions to use for hashing.
    input_schema
        Schema of incoming chunks used to evaluate ``keys_to_hash``.
    num_partitions
        Number of partitions to shuffle into.
    collective_id
        The collective ID.
    """
    _ensure_pointwise_keys(keys_to_hash)
    columns_to_hash = _key_column_indices(keys_to_hash, input_schema)
    metadata_in = await recv_metadata(ch_in, context)

    # Check if we can skip the shuffle (already partitioned correctly)
    if columns_to_hash is not None and _is_already_partitioned(
        metadata_in, columns_to_hash, num_partitions, comm.nranks
    ):
        # Forward metadata and data unchanged
        await send_metadata(ch_out, context, metadata_in)
        while (msg := await ch_in.recv(context)) is not None:
            await ch_out.send(context, msg)
        await ch_out.drain(context)
        return

    # Normal shuffle path
    output_metadata = ChannelMetadata(
        local_count=max(1, num_partitions // comm.nranks),
        partitioning=(
            Partitioning(
                inter_rank=HashScheme(columns_to_hash, num_partitions),
                local="inherit",
            )
            if columns_to_hash is not None
            else None
        ),
    )
    await send_metadata(ch_out, context, output_metadata)

    # When input is duplicated, only rank 0 should contribute data.
    # Other ranks still participate in the shuffle protocol.
    skip_insert = metadata_in.duplicated and comm.rank != 0

    shuffle = ShuffleManager(context, comm, num_partitions, collective_id)
    async with shuffle.inserting() as inserter:
        while (msg := await ch_in.recv(context)) is not None:
            if not skip_insert:
                chunk = TableChunk.from_message(
                    msg, br=context.br()
                ).make_available_and_spill(context.br(), allow_overbooking=True)
                if columns_to_hash is None:
                    inserter.insert_hash_keys(
                        chunk, _evaluate_key_table(chunk, keys_to_hash, input_schema)
                    )
                else:
                    inserter.insert_hash(
                        chunk,
                        columns_to_hash,
                    )

    for partition_id in shuffle.local_partitions():
        stream = ir_context.get_cuda_stream()
        await ch_out.send(
            context,
            Message(
                partition_id,
                TableChunk.from_pylibcudf_table(
                    table=shuffle.extract_chunk(partition_id, stream),
                    stream=stream,
                    exclusive_view=True,
                    br=context.br(),
                ),
            ),
        )

    await ch_out.drain(context)


@define_actor()
async def shuffle_actor(
    context: Context,
    comm: Communicator,
    ir: Shuffle,
    ir_context: IRExecutionContext,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    keys_to_hash: tuple[NamedExpr, ...],
    input_schema: Schema,
    num_partitions: int,
    collective_id: int,
) -> None:
    """
    Execute a global shuffle pipeline within a single node.

    This node combines partition_and_pack, shuffler, and unpack_and_concat
    into a single Python node using rapidsmpf.shuffler.Shuffler and utilities
    from cudf_streaming.partition_utils.

    Parameters
    ----------
    context
        The rapidsmpf context.
    comm
        The communicator.
    ir
        The Shuffle IR node.
    ir_context
        The execution context for the IR node.
    ch_in
        Input Channel[TableChunk] with metadata and data channels.
    ch_out
        Output Channel[TableChunk] with metadata and data channels.
    keys_to_hash
        Tuple of expressions to use for hashing.
    input_schema
        Schema of incoming chunks used to evaluate ``keys_to_hash``.
    num_partitions
        Number of partitions to shuffle into.
    collective_id
        The collective ID.
    """
    async with shutdown_on_error(
        context, ch_in, ch_out, trace_ir=ir, ir_context=ir_context
    ):
        await _global_shuffle(
            context,
            comm,
            ir_context,
            ch_out,
            ch_in,
            keys_to_hash,
            input_schema,
            num_partitions,
            collective_id,
        )


@generate_ir_sub_network.register(Shuffle)
def _(
    ir: Shuffle, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    # Local shuffle operation.

    # Process children
    (child,) = ir.children
    nodes, channels = rec(child)

    _ensure_pointwise_keys(ir.keys)

    context = rec.state["context"]
    num_partitions = rec.state["partition_info"][ir].count

    # Look up the reserved collective ID for this operation
    collective_id = rec.state["collective_id_map"][ir][0]

    # Create output ChannelManager
    channels[ir] = ChannelManager(rec.state["context"])

    # Complete shuffle node
    nodes[ir] = [
        shuffle_actor(
            context,
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            ch_in=channels[child].reserve_output_slot(),
            ch_out=channels[ir].reserve_input_slot(),
            keys_to_hash=ir.keys,
            input_schema=child.schema,
            num_partitions=num_partitions,
            collective_id=collective_id,
        )
    ]

    return nodes, channels
