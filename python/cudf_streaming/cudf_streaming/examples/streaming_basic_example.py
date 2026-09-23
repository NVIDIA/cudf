# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Basic streaming example."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf

import rmm.mr
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.communicator.single import (
    new_communicator as single_process_comm,
)
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.streaming.core.actor import (
    define_actor,
    run_actor_network,
)
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.leaf_actor import (
    pull_from_channel,
    push_to_channel,
)
from rapidsmpf.streaming.core.message import Message
from rmm.pylibrmm.stream import DEFAULT_STREAM

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from rapidsmpf.streaming.core.actor import CppActor
    from rapidsmpf.streaming.core.channel import Channel


def main() -> int:
    """Basic example of a streaming graph."""
    # Initialize configuration options from environment variables.
    options = Options(get_environment_variables())

    # Create a communicator and context that will be used by all streaming actors.
    comm = single_process_comm(options, ProgressThread())
    ctx = Context(
        logger=comm.logger,
        br=BufferResource(rmm.mr.get_current_device_resource()),
        options=options,
    )

    # Create some pylibcudf tables as input to the streaming graph.
    tables = [
        pylibcudf.Table(
            [
                pylibcudf.Column.from_iterable_of_py(
                    [1 * seq, 2 * seq, 3 * seq],
                    pylibcudf.DataType(pylibcudf.TypeId.INT64),
                )
            ]
        )
        for seq in range(10)
    ]

    # Wrap tables in TableChunk objects before sending them into the graph.
    # A TableChunk contains a pylibcudf table, a sequence number, and a CUDA stream.
    table_chunks = [
        Message(
            seq,
            TableChunk.from_pylibcudf_table(
                expect, DEFAULT_STREAM, exclusive_view=False, br=ctx.br()
            ),
        )
        for seq, expect in enumerate(tables)
    ]

    # Create input and output channels for table chunks.
    ch1: Channel[TableChunk] = ctx.create_channel()
    ch2: Channel[TableChunk] = ctx.create_channel()

    # Actor 1: producer that pushes messages into the graph.
    # This is a native C++ actor that runs as a coroutine with minimal Python overhead.
    actor1: CppActor = push_to_channel(ctx, ch_out=ch1, messages=table_chunks)

    # Actor 2: Python actor that counts the total number of rows.
    # Runs as a Python coroutine (asyncio), which comes with overhead,
    # but releases the GIL on `await` and when calling into C++ APIs.
    @define_actor()
    async def count_num_rows(
        ctx: Context,
        ch_in: Channel,
        ch_out: Channel,
        total_num_rows: list[int],
    ) -> None:
        assert len(total_num_rows) == 1, "should be a scalar"
        msg: Message[TableChunk] | None
        while (msg := await ch_in.recv(ctx)) is not None:
            # Convert the message back into a table chunk (releases the message).
            table = TableChunk.from_message(msg, br=ctx.br())

            # Accumulate the number of rows.
            total_num_rows[0] += table.table_view().num_rows()

            # The message is now empty since it was released.
            assert msg.empty()

            # Wrap the table chunk in a new message.
            msg = Message(msg.sequence_number, table)

            # Forward the message to the output channel.
            await ch_out.send(ctx, msg)

        # `msg == None` indicates the channel is closed, i.e. we are done.
        # Before exiting, drain the output channel to close it gracefully.
        await ch_out.drain(ctx)

    # Actors return None, so if we want an "output" value we can use either a closure
    # or an output parameter like `total_num_rows`.
    total_num_rows = [0]  # Wrap scalar in a list to make it mutable in-place.
    actor2: Awaitable[None] = count_num_rows(
        ctx, ch_in=ch1, ch_out=ch2, total_num_rows=total_num_rows
    )

    # Actor 3: consumer that pulls messages from the graph.
    # Like push_to_channel(), it returns a CppActor. It also returns a placeholder
    # object that will be populated with the pulled messages after execution.
    actor3, out_messages = pull_from_channel(ctx, ch_in=ch2)

    # Run all actors. This blocks until every actor has completed.
    run_actor_network(
        ctx,
        actors=(
            actor1,
            actor2,
            actor3,
        ),
    )

    # Collect and verify results.
    expect = 0
    for msg in out_messages.release():
        table = TableChunk.from_message(msg, br=ctx.br()).table_view()
        expect += table.num_rows()
    assert total_num_rows[0] == expect

    # Shut down the context explicitly to ensure it happens on the same thread that
    # created it. Alternatively, use `with Context(...) as ctx:` to shut it down
    # automatically.
    ctx.shutdown()

    return total_num_rows[0]


if __name__ == "__main__":
    print(f"total_num_rows: {main()}")  # noqa: T201
