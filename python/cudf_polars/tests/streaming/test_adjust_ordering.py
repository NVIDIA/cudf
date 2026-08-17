# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.channel_metadata import (
    OrderKey,
    Ordering,
)
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.ir import Empty, IRExecutionContext
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.ordering import (
    adjust_ordering,
)
from cudf_polars.streaming.actor_graph.utils import gather_in_task_group

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.engine.spmd import SPMDEngine

_SCHEMA = {"key": DataType(pl.Int32()), "val": DataType(pl.Int32())}
_NAMES = list(_SCHEMA)
_DTYPES = list(_SCHEMA.values())
_Boundary = int | tuple[int, ...]
_ExpectedPartitions = dict[int, list[int]]
_ExpectedByRank = dict[int, _ExpectedPartitions]


def _boundary_value(boundary: _Boundary, index: int) -> int:
    return boundary[index] if isinstance(boundary, tuple) else boundary


def _make_ordering(
    context: Context,
    boundary: _Boundary | list[_Boundary],
    *,
    key_indices: tuple[int, ...] = (0,),
    order: plc.types.Order = plc.types.Order.ASCENDING,
    null_order: plc.types.NullOrder = plc.types.NullOrder.BEFORE,
    strict: bool = True,
    stream: Stream,
) -> Ordering:
    boundary_rows: list[_Boundary] = (
        boundary if isinstance(boundary, list) else [boundary]
    )
    boundary_df = DataFrame.from_polars(
        pl.DataFrame(
            {
                f"k{i}": pl.Series(
                    [_boundary_value(value, i) for value in boundary_rows],
                    dtype=pl.Int32(),
                )
                for i in range(len(key_indices))
            }
        ),
        stream,
    )
    return Ordering(
        [
            OrderKey(
                index,
                order,
                null_order,
            )
            for index in key_indices
        ],
        TableChunk.from_pylibcudf_table(
            boundary_df.table,
            stream,
            exclusive_view=True,
            br=context.br(),
        ),
        strict_boundaries=strict,
    )


def _payload_value(key: int) -> int:
    return key * 10 + 1


def _frame(values: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "key": pl.Series(values, dtype=pl.Int32()),
            "val": pl.Series([_payload_value(v) for v in values], dtype=pl.Int32()),
        }
    )


def _chunk_to_polars(chunk: TableChunk) -> pl.DataFrame:
    return DataFrame.from_table(
        chunk.table_view(),
        _NAMES,
        _DTYPES,
        chunk.stream,
    ).to_polars()


async def _adjust_and_collect(
    context: Context,
    comm: Communicator,
    input_df: pl.DataFrame | list[pl.DataFrame],
    input_ordering: Ordering,
    output_ordering: Ordering,
    *,
    collective_id: int,
) -> dict[int, pl.DataFrame]:
    """Run adjustment and collect output chunks by partition ID."""
    ch_in = context.create_channel()
    ch_out = context.create_channel()
    stream = context.br().stream_pool.get_stream()
    output: dict[int, pl.DataFrame] = {}

    async def _produce() -> None:
        input_dfs = input_df if isinstance(input_df, list) else [input_df]
        for sequence_number, df in enumerate(input_dfs):
            cudf_df = DataFrame.from_polars(df, stream)
            await ch_in.send(
                context,
                Message(
                    sequence_number,
                    TableChunk.from_pylibcudf_table(
                        cudf_df.table,
                        stream,
                        exclusive_view=True,
                        br=context.br(),
                    ),
                ),
            )
        await ch_in.drain(context)

    async def _consume() -> None:
        while (msg := await ch_out.recv(context)) is not None:
            output[msg.sequence_number] = _chunk_to_polars(
                TableChunk.from_message(msg, br=context.br())
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        ir_context = IRExecutionContext(
            executor, get_cuda_stream=context.br().stream_pool.get_stream
        )
        await gather_in_task_group(
            _produce(),
            adjust_ordering(
                context,
                comm,
                Empty(_SCHEMA),
                ir_context,
                ch_out,
                ch_in,
                input_ordering,
                output_ordering,
                collective_id=collective_id,
            ),
            _consume(),
        )

    return output


def _assert_partition_output(
    output: dict[int, pl.DataFrame], expected: dict[int, list[int]]
) -> None:
    """Assert output partition IDs and per-partition row order."""
    assert set(output) == set(expected)
    for pid, keys in expected.items():
        assert output[pid]["key"].to_list() == keys
        assert output[pid]["val"].to_list() == [_payload_value(key) for key in keys]


async def _adjust_direct(
    context: Context,
    comm: Communicator,
    input_ordering: Ordering,
    output_ordering: Ordering,
    *,
    collective_id: int,
) -> None:
    ch_in = context.create_channel()
    ch_out = context.create_channel()
    with ThreadPoolExecutor(max_workers=1) as executor:
        ir_context = IRExecutionContext(
            executor, get_cuda_stream=context.br().stream_pool.get_stream
        )
        await adjust_ordering(
            context,
            comm,
            Empty(_SCHEMA),
            ir_context,
            ch_out,
            ch_in,
            input_ordering,
            output_ordering,
            collective_id=collective_id,
        )


@pytest.mark.spmd
@pytest.mark.parametrize(
    "input_keys,output_keys,strict,err,match",
    [
        ((1,), (0,), True, NotImplementedError, "prefix"),
        ((0,), (0,), False, ValueError, "strict output"),
    ],
)
def test_adjust_ordering_rejects_invalid_orderings(
    spmd_engine: SPMDEngine,
    input_keys: tuple[int, ...],
    output_keys: tuple[int, ...],
    strict: bool,  # noqa: FBT001
    err: type[Exception],
    match: str,
) -> None:
    context = spmd_engine.context
    stream = context.br().stream_pool.get_stream()
    input_ordering = _make_ordering(context, 4, key_indices=input_keys, stream=stream)
    output_ordering = _make_ordering(
        context,
        4,
        key_indices=output_keys,
        strict=strict,
        stream=stream,
    )

    with pytest.raises(err, match=match), reserve_op_id() as op_id:
        asyncio.run(
            _adjust_direct(
                context,
                spmd_engine.comm,
                input_ordering,
                output_ordering,
                collective_id=op_id,
            )
        )


@pytest.mark.spmd
@pytest.mark.parametrize(
    "target_boundary,expected",
    [
        (3, {0: {0: [0, 1, 2]}, 1: {1: [3, 4, 5, 6, 7]}}),
        (5, {0: {0: [0, 1, 2, 3, 4]}, 1: {1: [5, 6, 7]}}),
    ],
)
@pytest.mark.parametrize("input_strict", [True, False])
def test_adjust_ordering_sparse_boundary_shift(
    spmd_engine: SPMDEngine,
    target_boundary: int,
    expected: _ExpectedByRank,
    input_strict: bool,  # noqa: FBT001
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    keys = list(range(4)) if comm.rank == 0 else list(range(4, 8))
    stream = context.br().stream_pool.get_stream()
    # Input sorted on (key, val) is also sorted on the target key prefix.
    input_ordering = _make_ordering(
        context,
        (4, 4),
        key_indices=(0, 1),
        strict=input_strict,
        stream=stream,
    )
    output_ordering = _make_ordering(context, target_boundary, stream=stream)

    with reserve_op_id() as op_id:
        output = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                _frame(keys),
                input_ordering,
                output_ordering,
                collective_id=op_id,
            )
        )

    _assert_partition_output(output, expected[comm.rank])


@pytest.mark.spmd
def test_adjust_ordering_descending_sparse_boundary_shift(
    spmd_engine: SPMDEngine,
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    keys = list(range(7, 3, -1)) if comm.rank == 0 else list(range(3, -1, -1))
    stream = context.br().stream_pool.get_stream()
    input_ordering = _make_ordering(
        context,
        3,
        order=plc.types.Order.DESCENDING,
        stream=stream,
    )
    output_ordering = _make_ordering(
        context,
        5,
        order=plc.types.Order.DESCENDING,
        stream=stream,
    )

    with reserve_op_id() as op_id:
        output = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                _frame(keys),
                input_ordering,
                output_ordering,
                collective_id=op_id,
            )
        )

    expected = {
        0: {0: [7, 6]},
        1: {1: [5, 4, 3, 2, 1, 0]},
    }[comm.rank]
    _assert_partition_output(output, expected)


@pytest.mark.spmd
def test_adjust_ordering_emits_empty_owned_partitions(
    spmd_engine: SPMDEngine,
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 2:
        pytest.skip("This test expects exactly two ranks.")

    keys = [0, 1, 2] if comm.rank == 0 else [5, 8]
    stream = context.br().stream_pool.get_stream()
    input_ordering = _make_ordering(context, 5, stream=stream)
    output_ordering = _make_ordering(context, [3, 5, 7], stream=stream)

    with reserve_op_id() as op_id:
        output = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                _frame(keys),
                input_ordering,
                output_ordering,
                collective_id=op_id,
            )
        )

    expected = {
        0: {0: [0, 1, 2], 1: []},
        1: {2: [5], 3: [8]},
    }[comm.rank]
    _assert_partition_output(output, expected)


@pytest.mark.spmd
def test_adjust_ordering_middle_rank_buffers_only_as_needed(
    spmd_engine: SPMDEngine,
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 3:
        pytest.skip("This test expects exactly three ranks.")

    keys = {
        0: [0, 5, 9],
        1: [10, 15, 19],
        2: [20, 25],
    }[comm.rank]
    stream = context.br().stream_pool.get_stream()
    input_ordering = _make_ordering(context, [10, 20], stream=stream)
    output_ordering = _make_ordering(context, [5, 15], stream=stream)

    with reserve_op_id() as op_id:
        output = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                _frame(keys),
                input_ordering,
                output_ordering,
                collective_id=op_id,
            )
        )

    expected = {
        0: {0: [0]},
        1: {1: [5, 9, 10]},
        2: {2: [15, 19, 20, 25]},
    }[comm.rank]
    _assert_partition_output(output, expected)


@pytest.mark.spmd
def test_adjust_ordering_empty_rank_window(spmd_engine: SPMDEngine) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 3:
        pytest.skip("This test expects exactly three ranks.")

    keys = {0: [0, 1], 1: [5, 6], 2: [9, 10]}[comm.rank]
    stream = context.br().stream_pool.get_stream()
    input_ordering = _make_ordering(context, [5, 9], stream=stream)
    output_ordering = _make_ordering(context, 5, stream=stream)

    with reserve_op_id() as op_id:
        output = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                _frame(keys),
                input_ordering,
                output_ordering,
                collective_id=op_id,
            )
        )

    expected = {
        0: {0: [0, 1]},
        1: {1: [5, 6, 9, 10]},
        2: {},
    }[comm.rank]
    _assert_partition_output(output, expected)


@pytest.mark.spmd
def test_adjust_ordering_all_empty_input(spmd_engine: SPMDEngine) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    stream = context.br().stream_pool.get_stream()
    input_ordering = _make_ordering(context, 5, stream=stream)
    output_ordering = _make_ordering(context, [3, 5, 7], stream=stream)
    output_npartitions = output_ordering.num_boundaries + 1
    expected: _ExpectedPartitions = {
        pid: []
        for pid in range(output_npartitions)
        if pid * comm.nranks // output_npartitions == comm.rank
    }

    with reserve_op_id() as op_id:
        output = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                _frame([]),
                input_ordering,
                output_ordering,
                collective_id=op_id,
            )
        )

    _assert_partition_output(output, expected)


@pytest.mark.spmd
@pytest.mark.parametrize(
    "target_boundary,expected",
    [
        (3, {0: [0, 1, 2], 1: [3, 4, 5, 6, 7]}),
        (0, {0: [], 1: list(range(8))}),
    ],
)
def test_adjust_ordering_single_rank(
    spmd_engine: SPMDEngine,
    target_boundary: int,
    expected: _ExpectedPartitions,
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 1:
        pytest.skip("This test covers the single-rank path.")

    stream = context.br().stream_pool.get_stream()
    input_ordering = _make_ordering(context, 4, stream=stream)
    output_ordering = _make_ordering(context, target_boundary, stream=stream)
    with reserve_op_id() as op_id:
        output_by_pid = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                _frame(list(range(8))),
                input_ordering,
                output_ordering,
                collective_id=op_id,
            )
        )

    _assert_partition_output(output_by_pid, expected)


@pytest.mark.spmd
def test_adjust_ordering_multi_chunk_input(spmd_engine: SPMDEngine) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 1:
        pytest.skip("This test covers local chunk accumulation.")

    stream = context.br().stream_pool.get_stream()
    input_ordering = _make_ordering(context, 4, stream=stream)
    output_ordering = _make_ordering(context, 4, stream=stream)
    with reserve_op_id() as op_id:
        output = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                [_frame([0, 1]), _frame([2, 3, 4, 5]), _frame([6, 7]), _frame([])],
                input_ordering,
                output_ordering,
                collective_id=op_id,
            )
        )

    _assert_partition_output(output, {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]})


@pytest.mark.spmd
@pytest.mark.parametrize(
    "order,null_order,keys,expected",
    [
        (
            plc.types.Order.DESCENDING,
            plc.types.NullOrder.BEFORE,
            list(range(7, -1, -1)),
            {0: [7, 6, 5], 1: [4, 3, 2, 1, 0]},
        ),
        (
            plc.types.Order.ASCENDING,
            plc.types.NullOrder.AFTER,
            list(range(8)),
            {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
        ),
    ],
)
def test_adjust_ordering_respects_order_key_metadata(
    spmd_engine: SPMDEngine,
    order: plc.types.Order,
    null_order: plc.types.NullOrder,
    keys: list[int],
    expected: _ExpectedPartitions,
) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm
    if comm.nranks != 1:
        pytest.skip("This test covers local order-key metadata variants.")

    stream = context.br().stream_pool.get_stream()
    input_ordering = _make_ordering(
        context,
        4,
        order=order,
        null_order=null_order,
        stream=stream,
    )
    output_ordering = _make_ordering(
        context,
        4,
        order=order,
        null_order=null_order,
        stream=stream,
    )
    with reserve_op_id() as op_id:
        output_by_pid = asyncio.run(
            _adjust_and_collect(
                context,
                comm,
                _frame(keys),
                input_ordering,
                output_ordering,
                collective_id=op_id,
            )
        )

    _assert_partition_output(output_by_pid, expected)
