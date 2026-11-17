# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc
import pytest

from cudf_streaming.table_chunk import TableChunk
from cudf_streaming.testing import assert_eq
from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import (
    pull_from_channel,
    push_to_channel,
)
from rapidsmpf.streaming.core.message import Message


@pytest.fixture
def expects() -> list[plc.Table]:
    return [
        plc.Table(
            [
                plc.Column.from_iterable_of_py(
                    [1 * seq, 2 * seq, 3 * seq], plc.DataType(plc.TypeId.INT64)
                )
            ]
        )
        for seq in range(10)
    ]


if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


def test_send_table_chunks(
    context: Context, stream: Stream, expects: list[plc.Table]
) -> None:
    ch1: Channel[TableChunk] = context.create_channel()

    # The actor access `ch1` both through the `ch_out` parameter and the closure.
    @define_actor(extra_channels=(ch1,))
    async def actor1(ctx: Context, /, ch_out: Channel) -> None:
        for seq, chunk in enumerate(expects):
            await ch1.send(
                context,
                Message(
                    seq,
                    TableChunk.from_pylibcudf_table(
                        table=chunk,
                        stream=stream,
                        exclusive_view=False,
                        br=context.br(),
                    ),
                ),
            )
        await ch_out.drain(context)

    actor2, output = pull_from_channel(context, ch_in=ch1)

    run_actor_network(
        context,
        actors=[
            actor1(context, ch_out=ch1),
            actor2,
        ],
    )

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result, br=context.br())
        assert_eq(tbl.table_view(), expect)


def test_shutdown(context: Context) -> None:
    @define_actor()
    async def actor1(ctx: Context, ch_out: Channel[TableChunk]) -> None:
        await ch_out.shutdown(ctx)
        # Calling shutdown multiple times is allowed.
        await ch_out.shutdown(ctx)

    ch1: Channel[TableChunk] = context.create_channel()
    actor2, output = pull_from_channel(context, ch_in=ch1)

    run_actor_network(
        context,
        actors=[
            actor1(context, ch_out=ch1),
            actor2,
        ],
    )

    assert output.release() == []


def test_send_error(context: Context) -> None:
    @define_actor()
    async def actor1(ctx: Context, ch_out: Channel[TableChunk]) -> None:
        raise RuntimeError("MyError")

    ch1: Channel[TableChunk] = context.create_channel()
    actor2, output = pull_from_channel(context, ch_in=ch1)

    with pytest.RaisesGroup(
        pytest.RaisesExc(
            RuntimeError,
            match="MyError",
        )
    ):
        run_actor_network(
            context,
            actors=[
                actor1(context, ch_out=ch1),
                actor2,
            ],
        )

    assert output.release() == []


def test_recv_table_chunks(
    context: Context, stream: Stream, expects: list[plc.Table]
) -> None:
    table_chunks = [
        Message(
            seq,
            TableChunk.from_pylibcudf_table(
                expect, stream, exclusive_view=False, br=context.br()
            ),
        )
        for seq, expect in enumerate(expects)
    ]

    results: list[Message[TableChunk]] = []

    @define_actor()
    async def actor1(ctx: Context, ch_in: Channel[TableChunk]) -> None:
        while True:
            chunk = await ch_in.recv(context)
            if chunk is None:
                break
            results.append(chunk)

    ch1: Channel[TableChunk] = context.create_channel()

    run_actor_network(
        context,
        actors=[
            push_to_channel(context, ch_out=ch1, messages=table_chunks),
            actor1(context, ch_in=ch1),
        ],
    )

    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result, br=context.br())
        assert_eq(tbl.table_view(), expect)


@pytest.mark.filterwarnings("error")
def test_unawaited_actor_closed_coroutines_no_warning(
    context: Context,
) -> None:
    ch: Channel[ArbitraryChunk[int]] = context.create_channel()

    @define_actor()
    async def my_actor(
        ctx: Context, ch_out: Channel[ArbitraryChunk[int]]
    ) -> None:
        await ch_out.send(ctx, Message(0, ArbitraryChunk(42)))
        await ch_out.drain(ctx)

    # Never awaited, just verifying no RuntimeWarning is emitted
    actor = my_actor(context, ch_out=ch)
    del actor
