# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for streaming fanout actor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc
import pytest

from cudf_streaming.table_chunk import TableChunk
from cudf_streaming.testing import assert_eq
from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.core.fanout import FanoutPolicy, fanout
from rapidsmpf.streaming.core.leaf_actor import (
    pull_from_channel,
    push_to_channel,
)
from rapidsmpf.streaming.core.message import Message

_INT64 = plc.DataType(plc.TypeId.INT64)


def _ab_table(i: int) -> plc.Table:
    return plc.Table(
        [
            plc.Column.from_iterable_of_py(
                [i, i + 1, i + 2], plc.DataType(plc.TypeId.INT64)
            ),
            plc.Column.from_iterable_of_py(
                [i * 10, i * 10 + 1, i * 10 + 2],
                plc.DataType(plc.TypeId.INT64),
            ),
        ]
    )


if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


@pytest.mark.parametrize(
    "policy", [FanoutPolicy.BOUNDED, FanoutPolicy.UNBOUNDED]
)
def test_fanout_basic(
    context: Context, stream: Stream, policy: FanoutPolicy
) -> None:
    """Test basic fanout functionality with multiple output channels."""
    # Create channels
    ch_in: Channel[TableChunk] = context.create_channel()
    ch_out1: Channel[TableChunk] = context.create_channel()
    ch_out2: Channel[TableChunk] = context.create_channel()

    # Create test messages
    messages = []
    for i in range(5):
        chunk = TableChunk.from_pylibcudf_table(
            _ab_table(i), stream, exclusive_view=False, br=context.br()
        )
        messages.append(Message(i, chunk))

    # Create actors
    push_actor = push_to_channel(context, ch_in, messages)
    fanout_actor = fanout(context, ch_in, [ch_out1, ch_out2], policy)
    pull_actor1, output1 = pull_from_channel(context, ch_out1)
    pull_actor2, output2 = pull_from_channel(context, ch_out2)

    # Run pipeline
    run_actor_network(
        context,
        actors=[push_actor, fanout_actor, pull_actor1, pull_actor2],
    )

    # Verify results
    results1 = output1.release()
    results2 = output2.release()

    assert len(results1) == 5, (
        f"Expected 5 messages in output1, got {len(results1)}"
    )
    assert len(results2) == 5, (
        f"Expected 5 messages in output2, got {len(results2)}"
    )

    # Check that both outputs received the same sequence numbers and data
    for i in range(5):
        assert results1[i].sequence_number == i
        assert results2[i].sequence_number == i

        chunk1 = TableChunk.from_message(results1[i], br=context.br())
        chunk2 = TableChunk.from_message(results2[i], br=context.br())

        # Verify data is correct
        expected_table = _ab_table(i)
        assert_eq(chunk1.table_view(), expected_table)
        assert_eq(chunk2.table_view(), expected_table)


@pytest.mark.parametrize("num_outputs", [1, 3, 5])
@pytest.mark.parametrize(
    "policy", [FanoutPolicy.BOUNDED, FanoutPolicy.UNBOUNDED]
)
def test_fanout_multiple_outputs(
    context: Context, stream: Stream, num_outputs: int, policy: FanoutPolicy
) -> None:
    """Test fanout with varying numbers of output channels."""
    # Create channels
    ch_in: Channel[TableChunk] = context.create_channel()
    chs_out: list[Channel[TableChunk]] = [
        context.create_channel() for _ in range(num_outputs)
    ]

    if num_outputs == 1:
        with pytest.raises(ValueError):
            fanout(context, ch_in, chs_out, policy)
        return

    # Create test messages
    messages = []
    for i in range(3):
        table = plc.Table(
            [
                plc.Column.from_iterable_of_py(
                    [i * 10, i * 10 + 1], plc.DataType(plc.TypeId.INT64)
                ),
            ]
        )
        chunk = TableChunk.from_pylibcudf_table(
            table, stream, exclusive_view=False, br=context.br()
        )
        messages.append(Message(i, chunk))

    # Create actors
    push_actor = push_to_channel(context, ch_in, messages)
    fanout_actor = fanout(context, ch_in, chs_out, policy)
    pull_actors = []
    outputs = []
    for ch_out in chs_out:
        pull_actor, output = pull_from_channel(context, ch_out)
        pull_actors.append(pull_actor)
        outputs.append(output)

    # Run pipeline
    run_actor_network(
        context,
        actors=[push_actor, fanout_actor, *pull_actors],
    )

    # Verify all outputs received the messages
    for output_idx, output in enumerate(outputs):
        results = output.release()
        assert len(results) == 3, (
            f"Output {output_idx}: Expected 3 messages, got {len(results)}"
        )
        for i in range(3):
            assert results[i].sequence_number == i


def test_fanout_empty_outputs(context: Context, stream: Stream) -> None:
    """Test fanout with empty output list raises value error."""
    ch_in: Channel[TableChunk] = context.create_channel()
    with pytest.raises(ValueError):
        fanout(context, ch_in, [], FanoutPolicy.BOUNDED)


def test_fanout_policy_enum() -> None:
    """Test that FanoutPolicy enum has correct values."""
    assert FanoutPolicy.BOUNDED == 0
    assert FanoutPolicy.UNBOUNDED == 1
    assert len(FanoutPolicy) == 2
