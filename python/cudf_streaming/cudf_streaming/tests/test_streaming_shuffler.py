# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pylibcudf as plc
import pytest

from cudf_streaming.partition import (
    partition_and_pack,
    unpack_and_concat as streaming_unpack_and_concat,
)
from cudf_streaming.partition_utils import (
    split_and_pack,
    unpack_and_concat,
)
from cudf_streaming.table_chunk import TableChunk
from cudf_streaming.testing import assert_eq
from rapidsmpf.shuffler import PartitionAssignment
from rapidsmpf.streaming.coll.shuffler import (
    ShufflerAsync,
    shuffler,
)
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import (
    pull_from_channel,
    push_to_channel,
)
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.chunks.partition import (
        PartitionMapChunk,
        PartitionVectorChunk,
    )
    from rapidsmpf.streaming.core.actor import CppActor
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_single_rank_shuffler(
    context: Context, comm: Communicator, stream: Stream, num_partitions: int
) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")

    num_rows = 1000
    num_chunks = 5
    chunk_size = num_rows // num_chunks
    op_id = 0
    # We start a full dataframe.
    df = plc.Table(
        [
            plc.Column.from_array(cp.arange(num_rows, dtype=cp.int32)),
            plc.Column.from_array(
                cp.random.randint(0, 10, size=num_rows, dtype=cp.int32)
            ),
        ]
    )

    # That we slice into chunks and wrap as TableChunk (sequence_number=i).
    input_chunks: list[Message[TableChunk]] = []
    for i in range(num_chunks):
        lo = i * chunk_size
        hi = (i + 1) * chunk_size
        df_chunk = plc.copying.slice(df, [lo, hi])[0]
        chunk = TableChunk.from_pylibcudf_table(
            table=df_chunk,
            stream=stream,
            exclusive_view=False,
            br=context.br(),
        )
        input_chunks.append(Message(i, chunk))

    # Build the streaming pipeline:
    #   push -> partition/pack -> shuffle -> unpack/concat -> pull.
    actors = []

    ch1: Channel[TableChunk] = context.create_channel()
    actors.append(push_to_channel(context, ch1, input_chunks))

    ch2: Channel[PartitionMapChunk] = context.create_channel()
    actors.append(
        partition_and_pack(
            context,
            ch_in=ch1,
            ch_out=ch2,
            columns_to_hash=(1,),
            num_partitions=num_partitions,
        )
    )

    ch3: Channel[PartitionVectorChunk] = context.create_channel()
    actors.append(
        shuffler(
            context,
            comm,
            ch_in=ch2,
            ch_out=ch3,
            op_id=op_id,
            total_num_partitions=num_partitions,
        )
    )

    ch4: Channel[TableChunk] = context.create_channel()
    actors.append(streaming_unpack_and_concat(context, ch_in=ch3, ch_out=ch4))

    pull_actor, out_messages = pull_from_channel(context, ch_in=ch4)
    actors.append(pull_actor)

    run_actor_network(context, actors=actors)

    output_chunks = [
        TableChunk.from_message(msg, br=context.br())
        for msg in out_messages.release()
    ]

    result = plc.concatenate.concatenate(
        [chunk.table_view() for chunk in output_chunks]
    )
    assert_eq(result, df, sort_rows=0)


@define_actor()
async def generate_inputs(
    context: Context, ch: Channel[TableChunk], num_rows: int, num_chunks: int
) -> None:
    for i in range(num_chunks):
        stream = context.br().stream_pool.get_stream()
        table = plc.Table(
            [
                plc.Column.from_array(
                    np.arange(num_rows, dtype=np.int32) + i * num_rows,
                    stream=stream,
                )
            ]
        )
        msg = Message(
            i,
            TableChunk.from_pylibcudf_table(
                table, stream, exclusive_view=True, br=context.br()
            ),
        )
        await ch.send(context, msg)
    await ch.drain(context)


@define_actor()
async def do_shuffle(
    context: Context,
    comm: Communicator,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    op_id: int,
    num_partitions: int,
    *,
    partition_assignment: PartitionAssignment = PartitionAssignment.ROUND_ROBIN,
) -> None:
    shuffle = ShufflerAsync(
        context,
        comm,
        op_id,
        num_partitions,
        partition_assignment=partition_assignment,
    )
    while (msg := await ch_in.recv(context)) is not None:
        chunk = TableChunk.from_message(msg, br=context.br())
        num_rows = chunk.table_view().num_rows()
        part_size = num_rows // num_partitions + (num_rows % num_partitions)
        splits = range(part_size, num_rows, part_size)
        shuffle.insert(
            split_and_pack(
                chunk.table_view(), splits, chunk.stream, context.br()
            )
        )
    await shuffle.insert_finished(context)
    for pid in shuffle.local_partitions():
        data = shuffle.extract(pid)
        stream = context.br().stream_pool.get_stream()
        unpacked = TableChunk.from_pylibcudf_table(
            unpack_and_concat(data, stream, context.br()),
            stream,
            exclusive_view=True,
            br=context.br(),
        )
        await ch_out.send(context, Message(pid, unpacked))
    await ch_out.drain(context)


@pytest.mark.parametrize("num_partitions", [4, 8])
def test_shuffler_runtime_obeys_contiguous_assignment(
    context: Context,
    comm: Communicator,
    num_partitions: int,
) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")

    actors: list[CppActor | Awaitable[None]] = []

    num_rows = 200
    num_chunks = 3
    op_id = 0
    ch_in: Channel[TableChunk] = context.create_channel()
    actors.append(generate_inputs(context, ch_in, num_rows, num_chunks))
    ch_shuffled: Channel[TableChunk] = context.create_channel()
    actors.append(
        do_shuffle(
            context,
            comm,
            ch_in,
            ch_shuffled,
            op_id,
            num_partitions,
            partition_assignment=PartitionAssignment.CONTIGUOUS,
        )
    )
    actor, deferred = pull_from_channel(context, ch_shuffled)
    actors.append(actor)

    run_actor_network(context, actors=actors)
    messages = deferred.release()
    received_pids = [msg.sequence_number for msg in messages]

    nranks = comm.nranks
    rank = comm.rank
    expected_local = list(
        range(
            rank * num_partitions // nranks,
            (rank + 1) * num_partitions // nranks,
        )
    )
    assert set(received_pids) == set(expected_local)
    assert len(received_pids) == len(expected_local)


def test_shuffler_object_interface(
    context: Context,
    comm: Communicator,
) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")
    actors: list[CppActor | Awaitable[None]] = []

    num_partitions = 5
    num_rows = 100
    num_chunks = 4
    op_id = 0
    ch_in: Channel[TableChunk] = context.create_channel()
    actors.append(generate_inputs(context, ch_in, num_rows, num_chunks))
    ch_shuffled: Channel[TableChunk] = context.create_channel()
    actors.append(
        do_shuffle(
            context,
            comm,
            ch_in,
            ch_shuffled,
            op_id,
            num_partitions,
        )
    )
    actor, deferred = pull_from_channel(context, ch_shuffled)
    actors.append(actor)

    run_actor_network(context, actors=actors)
    messages = deferred.release()
    # TODO: single rank only assertions
    assert len(messages) == 5
    assert [msg.sequence_number for msg in messages] == list(
        range(num_partitions)
    )
    chunks = [
        (msg.sequence_number, TableChunk.from_message(msg, br=context.br()))
        for msg in messages
    ]

    full_column = np.arange(num_rows * num_chunks, dtype=np.int32)
    part_size = num_rows // num_partitions + (num_rows % num_partitions)
    splits = [*range(0, num_rows, part_size), num_rows]
    for pid, table in chunks:
        expect = plc.Column.from_array(
            np.concat(
                [
                    full_column[i * num_rows : (i + 1) * num_rows][
                        splits[pid] : splits[pid + 1]
                    ]
                    for i in range(num_chunks)
                ]
            ),
            stream=table.stream,
        )
        got = table.table_view()
        table.stream.synchronize()
        assert_eq(plc.Table([expect]), got, sort_rows=0)
