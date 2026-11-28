# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the runtime prefilter actors."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

from cudf_streaming.channel_metadata import ChannelMetadata, HashScheme
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.message import Message

from cudf_polars.containers import DataFrame
from cudf_polars.streaming.actor_graph.join import (
    ShuffleJoinStrategy,
    _resized_shuffle_decision,
)
from cudf_polars.streaming.actor_graph.partitioning import NormalizedPartitioning
from cudf_polars.streaming.actor_graph.prefilter import (
    PrefilterSampleMeasurement,
    measure_prefilter_sample,
)
from cudf_polars.streaming.actor_graph.utils import (
    gather_in_task_group,
    recv_metadata,
    send_metadata,
)

if TYPE_CHECKING:
    from cudf_polars.engine.spmd import SPMDEngine


@pytest.mark.parametrize("sample_chunks", [0, 2, 6])
def test_prefilter_sample_is_measured_by_position(
    spmd_engine: SPMDEngine, sample_chunks: int
) -> None:
    """Sequence numbers may repeat, so the sample is the first chunks."""
    context = spmd_engine.context
    chunk_rows = [1, 2, 3, 4]
    measurement = PrefilterSampleMeasurement(
        sample_chunks=sample_chunks, sample_rows=0, sample_size=0
    )
    sent_sizes = []

    async def produce(ch: Any) -> None:
        await send_metadata(ch, context, ChannelMetadata(local_count=1))
        stream = context.br().stream_pool.get_stream()
        for rows in chunk_rows:
            table = DataFrame.from_polars(
                pl.DataFrame({"key": range(rows)}), stream
            ).table
            chunk = TableChunk.from_pylibcudf_table(
                table, stream, exclusive_view=True, br=context.br()
            )
            sent_sizes.append(chunk.data_alloc_size())
            # Partition identifiers, not chunk identifiers: all chunks share one.
            await ch.send(context, Message(0, chunk))
        await ch.drain(context)

    async def consume(ch: Any) -> list[tuple[int, int]]:
        await recv_metadata(ch, context)
        # The consumer must be able to rely on the measurement.
        assert measurement.done.is_set()
        received = []
        while (msg := await ch.recv(context)) is not None:
            chunk = TableChunk.from_message(msg, br=context.br())
            received.append((msg.sequence_number, chunk.shape[0]))
        return received

    async def run() -> list[tuple[int, int]]:
        ch_in = context.create_channel()
        ch_out = context.create_channel()
        _, _, received = await gather_in_task_group(
            produce(ch_in),
            measure_prefilter_sample(context, ch_in, ch_out, measurement),
            consume(ch_out),
        )
        return received

    received = asyncio.run(run())

    assert received == [(0, rows) for rows in chunk_rows]
    assert measurement.output_rows == sum(chunk_rows[:sample_chunks])
    assert measurement.output_size == sum(sent_sizes[:sample_chunks])


@pytest.mark.parametrize(
    "decision, expected",
    [
        ("shuffle_right", "shuffle"),
        ("broadcast_left_rejected_shuffle_right", "broadcast_left_rejected_shuffle"),
    ],
)
def test_resized_shuffle_decision_for_prepartitioned_input(
    decision: str, expected: str
) -> None:
    """Reducing the modulus of a prepartitioned input means shuffling it too."""
    # The left input is already hash partitioned on the key with modulus 16.
    left = NormalizedPartitioning(HashScheme((0,), 16), "inherit")
    right = NormalizedPartitioning(None, None)
    original = ShuffleJoinStrategy(
        shuffle_modulus=16, left_indices=(0,), right_indices=(0,)
    )
    resized = ShuffleJoinStrategy(
        shuffle_modulus=2, left_indices=(0,), right_indices=(0,)
    )
    assert _resized_shuffle_decision(decision, original, resized, left, right) == (
        expected
    )
    # The modulus the input is partitioned with keeps it from being shuffled.
    assert _resized_shuffle_decision("shuffle", resized, original, left, right) == (
        "shuffle_right"
    )
