# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RapidsMPF spilling functionality."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

import polars as pl

import pylibcudf as plc
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.pinned_memory_resource import is_pinned_memory_resources_supported
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.spillable_messages import SpillableMessages

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph.utils import (
    make_spill_function,
)

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream


def create_test_table(nbytes: int, stream: Stream) -> plc.Table:
    """Create a test table with specified size in bytes."""
    assert nbytes % 4 == 0, "nbytes must be divisible by 4 for float32"
    # Create a simple table with one column of random float32 data
    num_elements = nbytes // 4
    rng = random.Random(42)
    pl_data = pl.Series([rng.random() for _ in range(num_elements)], dtype=pl.Float32())
    return plc.Table([plc.Column.from_arrow(pl_data, stream=stream)])


@pytest.mark.parametrize(
    "pinned_memory,spilled_host_mem_type",
    [
        pytest.param(
            True,
            MemoryType.PINNED_HOST,
            marks=pytest.mark.skipif(
                not is_pinned_memory_resources_supported(),
                reason="Pinned memory requires CUDA 12.6+ driver and runtime",
            ),
        ),
        (False, MemoryType.HOST),
    ],
)
def test_make_spill_function(
    spmd_engine_factory,
    *,
    pinned_memory: bool,
    spilled_host_mem_type: MemoryType,
) -> None:
    """Test that spilling prioritizes longest queues and newest messages."""
    try:
        engine = spmd_engine_factory(
            StreamingOptions(
                pinned_memory=pinned_memory,
                pinned_initial_pool_size=3 * 1024 * 1024,
            )
        )
    except RuntimeError as e:
        pytest.skip(f"Error creating engine: {e}")
    context = engine.context

    if spilled_host_mem_type == MemoryType.PINNED_HOST:
        assert context.br().pinned_mr is not None
        other_host_mem_type = MemoryType.HOST
    else:
        assert context.br().pinned_mr is None
        other_host_mem_type = MemoryType.PINNED_HOST

    # Create 3 spillable message containers simulating fanout buffers
    # Buffer 0: Fast consumer (2 messages)
    # Buffer 1: Slow consumer (5 messages) <- should spill from here first
    # Buffer 2: Medium consumer (3 messages)
    buffers = [SpillableMessages(context.br()) for _ in range(3)]
    messages_per_buffer = [2, 5, 3]

    # Track message IDs for each buffer
    message_ids: dict[int, list[int]] = {}

    # Populate buffers with messages
    stream = context.br().stream_pool.get_stream()
    for buffer_idx, (sm, count) in enumerate(
        zip(buffers, messages_per_buffer, strict=False)
    ):
        message_ids[buffer_idx] = []
        for msg_idx in range(count):
            # Create 1MB messages
            table = create_test_table(1024 * 1024, stream)
            chunk = TableChunk.from_pylibcudf_table(
                table, stream, exclusive_view=True, br=context.br()
            )
            msg = Message(msg_idx, chunk)
            mid = sm.insert(msg)
            message_ids[buffer_idx].append(mid)

    assert context.br().spill_manager.spill(1 * 1024 * 1024) == 0, (
        "No messages should be spilled initially"
    )

    # Register spill function
    spill_func = make_spill_function(buffers, context)
    func_id = context.br().spill_manager.add_spill_function(spill_func, priority=0)

    try:
        # Manually trigger spilling of 3MB
        # Expected: Buffer 1 (longest) should spill newest messages first
        amount_to_spill = 3 * 1024 * 1024
        actual_spilled = context.br().spill_manager.spill(amount_to_spill)

        # Allow some tolerance
        assert actual_spilled >= amount_to_spill * 0.95
        assert actual_spilled <= amount_to_spill * 1.05

        # Verify Buffer 1 (longest queue): newest 3 messages should be spilled
        buffer_1_descs = buffers[1].get_content_descriptions()
        for i in range(3, 5):  # Messages 3, 4 (newest)
            mid = message_ids[1][i]
            desc = buffer_1_descs[mid]
            # Should be in HOST memory (spilled)
            assert desc.content_sizes[spilled_host_mem_type] > 0
            assert desc.content_sizes[other_host_mem_type] == 0
            assert desc.content_sizes[MemoryType.DEVICE] == 0

        # Buffer 1: oldest messages should still be in device
        for i in range(2):  # Messages 0, 1 (oldest)
            mid = message_ids[1][i]
            desc = buffer_1_descs[mid]
            # Should still be in DEVICE memory
            assert desc.content_sizes[MemoryType.DEVICE] > 0
            assert desc.content_sizes[spilled_host_mem_type] == 0
            assert desc.content_sizes[other_host_mem_type] == 0

        # Buffer 0 (shortest queue): all messages should still be on device
        buffer_0_descs = buffers[0].get_content_descriptions()
        for mid in message_ids[0]:
            desc = buffer_0_descs[mid]
            assert desc.content_sizes[MemoryType.DEVICE] > 0
            assert desc.content_sizes[spilled_host_mem_type] == 0
            assert desc.content_sizes[other_host_mem_type] == 0

        # Verify we can extract and make available a spilled message
        spilled_mid = message_ids[1][4]  # Newest message from longest queue
        spilled_msg = buffers[1].extract(mid=spilled_mid)

        chunk = TableChunk.from_message(spilled_msg, br=context.br())
        assert not chunk.is_available()  # Should be on host

        # Make it available should bring it back to device
        cost = chunk.make_available_cost()
        assert cost > 0
        res, _ = context.br().reserve(MemoryType.DEVICE, cost, allow_overbooking=True)
        chunk_available = chunk.make_available(res)

        assert chunk_available.is_available()
        # Verify we got a valid table back
        assert chunk_available.table_view().num_rows() > 0

    finally:
        context.br().spill_manager.remove_spill_function(func_id)
