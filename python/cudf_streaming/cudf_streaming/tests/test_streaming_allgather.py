# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import pylibcudf as plc
import pytest

from cudf_streaming.partition_utils import (
    packed_data_from_cudf_packed_columns,
    unpack_and_concat,
)
from cudf_streaming.table_chunk import TableChunk
from cudf_streaming.testing import assert_eq
from rapidsmpf.streaming.chunks.packed_data import PackedDataChunk
from rapidsmpf.streaming.coll.allgather import AllGather, allgather
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import (
    pull_from_channel,
    push_to_channel,
)
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.actor import CppActor
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


def test_allgather_actor(context: Context, comm: Communicator) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")

    num_rows = 1000
    op_id = 0
    stream = context.br().stream_pool.get_stream()
    input_tables = [
        plc.Table(
            [
                plc.Column.from_array(
                    np.arange(num_rows, dtype=np.int32) + i * num_rows,
                    stream=stream,
                )
            ]
        )
        for i in range(3)
    ]
    inputs = [
        PackedDataChunk.from_packed_data(
            packed_data_from_cudf_packed_columns(
                plc.contiguous_split.pack(table, stream=stream),
                stream,
                context.br(),
            ),
            br=context.br(),
        )
        for table in input_tables
    ]
    actors = []

    ch1: Channel[PackedDataChunk] = context.create_channel()
    actors.append(
        push_to_channel(
            context, ch1, [Message(i, chunk) for i, chunk in enumerate(inputs)]
        )
    )

    ch2: Channel[PackedDataChunk] = context.create_channel()
    actors.append(allgather(context, comm, ch1, ch2, op_id, ordered=True))

    actor, deferred = pull_from_channel(context, ch2)
    actors.append(actor)
    run_actor_network(context, actors=actors)

    result = unpack_and_concat(
        (
            PackedDataChunk.from_message(msg, br=context.br()).to_packed_data()
            for msg in deferred.release()
        ),
        stream,
        context.br(),
    )

    expect = plc.concatenate.concatenate(input_tables, stream=stream)
    stream.synchronize()
    assert_eq(result, expect)


@define_actor()
async def generate_inputs(
    context: Context,
    ch: Channel[PackedDataChunk],
    num_rows: int,
    num_chunks: int,
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
            PackedDataChunk.from_packed_data(
                packed_data_from_cudf_packed_columns(
                    plc.contiguous_split.pack(table, stream=stream),
                    stream,
                    context.br(),
                ),
                br=context.br(),
            ),
        )
        await ch.send(context, msg)
    await ch.drain(context)


@define_actor()
async def allgather_and_concat(
    context: Context,
    comm: Communicator,
    ch_in: Channel[PackedDataChunk],
    ch_out: Channel[TableChunk],
    op_id: int,
    use_context_manager: bool,
) -> None:
    gather = AllGather(context, comm, op_id)
    cm = gather if use_context_manager else nullcontext(gather)
    with cm as ag:
        while (msg := await ch_in.recv(context)) is not None:
            chunk = PackedDataChunk.from_message(
                msg, br=context.br()
            ).to_packed_data()
            ag.insert(msg.sequence_number, chunk)
    if not use_context_manager:
        gather.insert_finished()
    gathered = await gather.extract_all(context, ordered=True)
    stream = context.br().stream_pool.get_stream()
    table = unpack_and_concat(gathered, stream, context.br())
    to_send = TableChunk.from_pylibcudf_table(
        table, stream, exclusive_view=True, br=context.br()
    )
    await ch_out.send(context, Message(0, to_send))
    await ch_out.drain(context)


@pytest.mark.parametrize(
    "use_context_manager", [True, False], ids=["context", "non-context"]
)
def test_allgather_object_interface(
    context: Context,
    comm: Communicator,
    use_context_manager: bool,
) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")

    ch_in: Channel[PackedDataChunk] = context.create_channel()
    ch_out: Channel[TableChunk] = context.create_channel()
    actors: list[CppActor | Awaitable[None]] = []
    num_rows = 100
    num_chunks = 10
    op_id = 0
    actors.append(generate_inputs(context, ch_in, num_rows, num_chunks))
    actors.append(
        allgather_and_concat(
            context, comm, ch_in, ch_out, op_id, use_context_manager
        )
    )

    actor, deferred = pull_from_channel(context, ch_out)
    actors.append(actor)

    run_actor_network(context, actors=actors)
    (result_msg,) = deferred.release()
    result = TableChunk.from_message(result_msg, br=context.br())
    expect = plc.Table(
        [
            plc.Column.from_array(
                np.arange(num_rows * num_chunks, dtype=np.int32),
                stream=result.stream,
            )
        ]
    )
    got = result.table_view()
    result.stream.synchronize()
    assert_eq(expect, got)
