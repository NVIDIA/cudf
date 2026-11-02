# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the RapidsMPF streaming sort."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import OrderKey, OrderScheme, Ordering
from cudf_streaming.table_chunk import TableChunk

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl.expr import Col
from cudf_polars.dsl.ir import Empty
from cudf_polars.dsl.utils.naming import names_to_indices, unique_names
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.utils import (
    chunk_to_frame,
    concat_batch,
    empty_table_chunk,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.dsl.ir import IRExecutionContext, Sort
    from cudf_polars.typing import Schema


def boundary_schema(by: list[str], by_dtypes: list[DataType]) -> Schema:
    """
    Return the schema of a sort boundaries table.

    Parameters
    ----------
    by
        Names of the sort key columns.
    by_dtypes
        Data types of the sort key columns, in the order of ``by``.

    Returns
    -------
    The sort key columns followed by two ``UInt32`` tie-breaker columns, the
    source partition (chunk sequence number) and the local row number of each
    split candidate, under names that do not clash with ``by``.
    """
    name_gen = unique_names(by)
    part_id_dtype = DataType(pl.UInt32())
    return dict(
        zip(
            [*by, next(name_gen), next(name_gen)],
            [*by_dtypes, part_id_dtype, part_id_dtype],
            strict=True,
        )
    )


def find_sort_splits(
    tbl: plc.Table,
    sort_boundaries: plc.Table,
    my_part_id: int,
    column_order: Sequence[plc.types.Order],
    null_order: Sequence[plc.types.NullOrder],
    stream: Stream,
    *,
    chunk_relative: bool = False,
) -> list[int]:
    """
    Find local sort splits given all (global) split candidates.

    When multiple rows share a boundary value, they go to the later partition.
    The ``part_id`` and ``local_row`` columns in ``sort_boundaries`` are used
    for tiebreaking when ``chunk_relative=False``.

    Parameters
    ----------
    tbl
        Locally sorted table only containing sort columns.
    sort_boundaries
        Sorted table containing the global sort/split boundaries.  Compared to `tbl`
        must contain additional partition_id and local_row_number columns.
    my_part_id
        The partition id of the local node (as the `split_candidates` column).
    column_order
        The order in which tbl is sorted.
    null_order
        The null order in which tbl is sorted.
    stream
        CUDA stream used for device memory operations and kernel launches.
        The values in both ``tbl`` and ``sort_boundaries`` must be valid on
        ``stream``.
    chunk_relative
        If True, when the boundary belongs to this partition (part_id == my_part_id)
        use the position of the boundary value in ``tbl`` (``first``) instead of
        ``local_row``. Use True when ``tbl`` is a chunk of the partition (e.g.
        multiple chunks per rank); False when ``tbl`` is the full partition.

    Returns
    -------
    The split points for the local partition.
    """
    column_order = list(column_order)
    null_order = list(null_order)

    *boundary_cols, split_part_id, split_local_row = sort_boundaries.columns()
    sort_boundaries = plc.Table(boundary_cols)
    split_first_col = plc.search.lower_bound(
        tbl,
        sort_boundaries,
        column_order,
        null_order,
        stream=stream,
    )
    # Use DataFrame.to_polars() for a stream-aware D→H transfer.
    # lower_bound returns size_type (INT32); part_id/local_row are UInt32.
    _u32 = DataType(pl.UInt32())
    df = DataFrame.from_table(
        plc.Table([split_first_col, split_part_id, split_local_row]),
        ["first", "part_id", "local_row"],
        [DataType(pl.Int32()), _u32, _u32],
        stream=stream,
    ).to_polars()
    out = (
        pl.col("first")
        if chunk_relative
        else pl.when(pl.col("part_id") == my_part_id)
        .then(pl.col("local_row"))
        .otherwise(pl.col("first"))
    )
    cap = tbl.num_rows()
    return df.select(out.clip(0, cap).sort()).to_series().to_list()


def select_local_split_candidates(
    df: DataFrame,
    by: Sequence[str],
    num_partitions: int,
    my_part_id: int,
) -> DataFrame:
    """
    Select evenly spaced split candidates from a locally sorted chunk.

    Parameters
    ----------
    df
        Locally sorted chunk.
    by
        Names of the sort key columns.
    num_partitions
        Number of output partitions; one candidate is taken per partition.
    my_part_id
        Identifier of the chunk, stored with each candidate for tiebreaking.

    Returns
    -------
    The candidates' ``by`` columns followed by two ``UInt32`` columns: the
    chunk identifier (``my_part_id``) and each candidate's row number in the
    chunk. Empty, with the same schema, when ``df`` has no rows.
    """
    df = df.select(by)
    name_gen = unique_names(df.column_names)
    part_id_dtype = DataType(pl.UInt32())
    if df.num_rows == 0:
        # Return empty DataFrame with the correct column names and dtypes
        return DataFrame(
            [
                *df.columns,
                Column(
                    plc.column_factories.make_empty_column(
                        part_id_dtype.plc_type, stream=df.stream
                    ),
                    dtype=part_id_dtype,
                    name=next(name_gen),
                ),
                Column(
                    plc.column_factories.make_empty_column(
                        part_id_dtype.plc_type, stream=df.stream
                    ),
                    dtype=part_id_dtype,
                    name=next(name_gen),
                ),
            ],
            stream=df.stream,
        )

    candidates = [i * df.num_rows // num_partitions for i in range(num_partitions)]
    row_id = plc.Column.from_iterable_of_py(
        candidates, part_id_dtype.plc_type, stream=df.stream
    )

    res = plc.copying.gather(
        df.table, row_id, plc.copying.OutOfBoundsPolicy.DONT_CHECK, stream=df.stream
    )
    part_id = plc.Column.from_scalar(
        plc.Scalar.from_py(my_part_id, part_id_dtype.plc_type, stream=df.stream),
        len(candidates),
        stream=df.stream,
    )

    return DataFrame.from_table(
        plc.Table([*res.columns(), part_id, row_id]),
        [*df.column_names, next(name_gen), next(name_gen)],
        [*df.dtypes, part_id_dtype, part_id_dtype],
        stream=df.stream,
    )


def get_final_sort_boundaries(
    sort_boundaries_candidates: DataFrame,
    column_order: Sequence[plc.types.Order],
    null_order: Sequence[plc.types.NullOrder],
    num_partitions: int,
) -> DataFrame:
    """
    Find the global sort split boundaries from all gathered split candidates.

    Parameters
    ----------
    sort_boundaries_candidates
        All gathered split candidates.
    column_order
        The order in which the split candidates are sorted.
    null_order
        The null order in which the split candidates are sorted.
    num_partitions
        The number of partitions to split the data into.

    Returns
    -------
    sort_boundaries
        Same schema as input (sort keys plus ``partition_id``,
        ``local_row_number``). Empty when ``num_partitions <= 1``
        or the candidate list is empty. Otherwise, the DataFrame
        will contain ``num_partitions - 1`` rows of split
        boundaries in global sort order.
    """
    column_order = list(column_order)
    null_order = list(null_order)

    if num_partitions <= 1 or sort_boundaries_candidates.table.num_rows() == 0:
        return sort_boundaries_candidates.slice((0, 0))  # pragma: no cover

    # The global split candidates need to be stable sorted to find the correct
    # final split points.
    # NOTE: This could be a merge if done earlier (but it should be small data).
    sorted_candidates = plc.sorting.stable_sort(
        sort_boundaries_candidates.table,
        # split candidates has the additional partition_id and row_number columns
        column_order + [plc.types.Order.ASCENDING] * 2,
        null_order + [plc.types.NullOrder.AFTER] * 2,
        stream=sort_boundaries_candidates.stream,
    )
    selected_candidates = plc.Column.from_iterable_of_py(
        [
            i * sorted_candidates.num_rows() // num_partitions
            for i in range(1, num_partitions)
        ],
        stream=sort_boundaries_candidates.stream,
    )
    # Get the actual values at which we will split the data
    sort_boundaries = plc.copying.gather(
        sorted_candidates,
        selected_candidates,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=sort_boundaries_candidates.stream,
    )

    return DataFrame.from_table(
        sort_boundaries,
        sort_boundaries_candidates.column_names,
        sort_boundaries_candidates.dtypes,
        stream=sort_boundaries_candidates.stream,
    )


async def compute_sort_boundaries(
    context: Context,
    comm: Communicator,
    ir_context: IRExecutionContext,
    local_candidates_list: list[TableChunk],
    ir: Sort,
    by: list[str],
    num_partitions: int,
    allgather_id: int | None,
) -> DataFrame:
    """
    Compute the range-partition boundaries for a global sort.

    Parameters
    ----------
    context
        The RapidsMPF context.
    comm
        The RapidsMPF communicator.
    ir_context
        The IR execution context.
    local_candidates_list
        This rank's split candidates, one chunk per input chunk, each with the
        schema of :func:`boundary_schema`.
    ir
        The ``Sort`` node providing the key orders and dtypes.
    by
        Names of the sort key columns.
    num_partitions
        Number of output partitions.
    allgather_id
        Collective ID for allgathering the local boundaries across ranks, or
        ``None`` to use this rank's boundaries only (single rank or duplicated
        input).

    Returns
    -------
    The boundaries, with the schema of :func:`boundary_schema` and
    ``num_partitions - 1`` rows in global sort order (empty when
    ``num_partitions <= 1`` or there are no candidates). Every rank returns the
    same boundaries when ``allgather_id`` is given.
    """
    column_order = list(ir.order)
    null_order = list(ir.null_order)
    by_dtypes = [ir.schema[b] for b in by]
    boundary_ir = Empty(boundary_schema(by, by_dtypes))
    local_boundaries_df = get_final_sort_boundaries(
        chunk_to_frame(
            await concat_batch(
                local_candidates_list,
                context,
                boundary_ir.schema,
                ir_context,
            )
            if local_candidates_list
            else empty_table_chunk(
                boundary_ir,
                context,
                ir_context.get_cuda_stream(),
            ),
            boundary_ir,
        ),
        column_order,
        null_order,
        num_partitions,
    )
    stream = local_boundaries_df.stream

    if allgather_id is not None:
        chunk = TableChunk.from_pylibcudf_table(
            local_boundaries_df.table,
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        allgather = AllGatherManager(context, comm, allgather_id)
        with allgather.inserting() as inserter:
            await inserter.insert(comm.rank, chunk)
        concat_table = await allgather.extract_concatenated(
            stream, ordered=True, ir_context=ir_context
        )
        return get_final_sort_boundaries(
            DataFrame.from_table(
                concat_table,
                list(boundary_ir.schema.keys()),
                list(boundary_ir.schema.values()),
                stream=stream,
            ),
            column_order,
            null_order,
            num_partitions,
        )
    else:
        return local_boundaries_df


def sort_by_column_names(ir: Sort) -> list[str]:
    """
    Resolve the underlying column names for a ``Sort`` node's keys.

    A sort key's ``NamedExpr.name`` reflects its output alias, which may differ from
    the referenced column after upstream renames. E.g., a join dedup may leave
    ``ORDER BY df2.text`` represented as ``Col('text:df2').alias('text')``. Resolve
    through the underlying ``Col`` so schema lookups use the actual column name.

    Parameters
    ----------
    ir
        The ``Sort`` node.

    Returns
    -------
    The referenced column names, in sort key order.

    Raises
    ------
    NotImplementedError
        If any sort key is not a bare column reference.
    """
    by = [ne.value.name for ne in ir.by if isinstance(ne.value, Col)]
    if len(by) != len(ir.by):
        raise NotImplementedError("Sorting columns must be column names.")
    return by


def sort_to_order_keys(ir: Sort) -> list[OrderKey]:
    """
    Convert a ``Sort`` node's keys to order keys.

    Parameters
    ----------
    ir
        The ``Sort`` node.

    Returns
    -------
    One ``OrderKey`` per sort key: the key's column index in ``ir.schema`` with
    its sort order and null order.
    """
    return [
        OrderKey(index, order, null_order)
        for index, order, null_order in zip(
            names_to_indices(tuple(sort_by_column_names(ir)), ir.schema),
            ir.order,
            ir.null_order,
            strict=False,
        )
    ]


def build_order_scheme(
    context: Context,
    order_keys: list[OrderKey],
    sort_boundaries_df: DataFrame,
) -> OrderScheme:
    """
    Build the ``OrderScheme`` metadata describing a range-partitioned sort output.

    Parameters
    ----------
    context
        The RapidsMPF context.
    order_keys
        The output's order keys, as returned by :func:`sort_to_order_keys`.
    sort_boundaries_df
        The boundaries from :func:`compute_sort_boundaries`. Only the leading
        ``len(order_keys)`` key columns are used; the tie-breaker columns are
        dropped.

    Returns
    -------
    A single-``Ordering`` scheme over the boundary keys, with strict boundaries
    when there are no boundaries or no two boundaries share the same key.
    """
    n_keys = len(order_keys)
    stream = sort_boundaries_df.stream
    # sort_boundaries_df will contain a tie-breaker column
    by_table = plc.Table(sort_boundaries_df.table.columns()[:n_keys])
    n_rows = by_table.num_rows()

    strict_boundaries = (
        n_rows == 0
        # TODO: Use unique_count_table
        # Requires https://github.com/NVIDIA/cudf/pull/22487
        or plc.stream_compaction.unique(
            by_table,
            list(range(n_keys)),
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            stream=stream,
        ).num_rows()
        == n_rows
    )

    boundaries_chunk = TableChunk.from_pylibcudf_table(
        by_table, stream, exclusive_view=False, br=context.br()
    )
    return OrderScheme(
        [Ordering(order_keys, boundaries_chunk, strict_boundaries=strict_boundaries)]
    )
