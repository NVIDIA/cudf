# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc
import pytest

from cudf_streaming.partition import (
    partition_and_pack,
    unpack_and_concat,
)
from cudf_streaming.table_chunk import TableChunk
from cudf_streaming.testing import assert_eq
from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.core.leaf_actor import (
    pull_from_channel,
    push_to_channel,
)
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from rapidsmpf.streaming.chunks.partition import PartitionMapChunk
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_partition_and_pack_unpack(
    context: Context, stream: Stream, num_partitions: int
) -> None:
    expects = [
        plc.Table(
            [
                plc.Column.from_iterable_of_py(
                    [1, 2, 3], plc.DataType(plc.TypeId.INT64)
                ),
                plc.Column.from_iterable_of_py(
                    [2, 1, 1], plc.DataType(plc.TypeId.INT64)
                ),
            ]
        ),
        plc.Table(
            [
                plc.Column.from_iterable_of_py(
                    [], plc.DataType(plc.TypeId.INT64)
                ),
                plc.Column.from_iterable_of_py(
                    [], plc.DataType(plc.TypeId.INT64)
                ),
            ]
        ),
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

    ch2: Channel[PartitionMapChunk] = context.create_channel()
    actor2 = partition_and_pack(
        context,
        ch_in=ch1,
        ch_out=ch2,
        columns_to_hash=(1,),
        num_partitions=num_partitions,
    )

    ch3: Channel[TableChunk] = context.create_channel()
    actor3 = unpack_and_concat(
        context,
        ch_in=ch2,
        ch_out=ch3,
    )

    actor4, output = pull_from_channel(context, ch_in=ch3)
    run_actor_network(context, actors=(actor1, actor2, actor3, actor4))

    results = output.release()
    for seq, (result, expect) in enumerate(zip(results, expects, strict=True)):
        assert result.sequence_number == seq
        tbl = TableChunk.from_message(result, br=context.br())
        assert_eq(tbl.table_view(), expect, sort_rows=0)
