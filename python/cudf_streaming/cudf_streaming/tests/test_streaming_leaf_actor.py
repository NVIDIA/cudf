# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf_streaming.table_chunk import TableChunk
from cudf_streaming.testing import assert_eq
from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.core.leaf_actor import (
    pull_from_channel,
    push_to_channel,
)
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


def test_roundtrip(context: Context, stream: Stream) -> None:
    expects = [
        plc.Table(
            [
                plc.Column.from_iterable_of_py(
                    [1 * seq, 2 * seq, 3 * seq], plc.DataType(plc.TypeId.INT64)
                )
            ]
        )
        for seq in range(10)
    ]
    table_chunks = [
        Message(
            seq,
            TableChunk.from_pylibcudf_table(
                expect, stream, exclusive_view=False, br=context.br()
            ),
        )
        for seq, expect in enumerate(expects)
    ]
    ch1: Channel[TableChunk] = context.create_channel()
    actor1 = push_to_channel(context, ch_out=ch1, messages=table_chunks)
    actor2, output = pull_from_channel(context, ch_in=ch1)
    run_actor_network(context, actors=(actor1, actor2))

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result, br=context.br())
        assert_eq(tbl.table_view(), expect)
