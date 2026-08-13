# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pylibcudf as plc
import pytest

from cudf_streaming import ChannelMetadata
from cudf_streaming.bloom_filter import BloomFilter
from cudf_streaming.table_chunk import TableChunk
from cudf_streaming.testing import assert_eq
from rapidsmpf.streaming.core.actor import define_actor, run_actor_network
from rapidsmpf.streaming.core.leaf_actor import (
    pull_from_channel,
    push_to_channel,
)
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from cudf_streaming.bloom_filter import BloomFilterChunk
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.actor import CppActor
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rmm.pylibrmm.stream import Stream


def make_table(
    values: np.ndarray, stream: Stream, br: BufferResource
) -> TableChunk:
    table = plc.Table([plc.Column.from_array(values, stream=stream)])
    return TableChunk.from_pylibcudf_table(
        table, stream, exclusive_view=True, br=br
    )


def test_aligned_size() -> None:
    assert BloomFilter.aligned_size(31) == 0
    assert BloomFilter.aligned_size(32) == 32
    assert BloomFilter.aligned_size(65) == 64


def test_requires_aligned_size(context: Context, comm: Communicator) -> None:
    with pytest.raises(RuntimeError, match="must be a multiple"):
        BloomFilter(context, comm, seed=0, filter_size=65)


@define_actor()
async def add_metadata(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    await ch_out.send_metadata(ctx, Message(0, ChannelMetadata(1)))
    await ch_out.drain_metadata(ctx)
    while (msg := await ch_in.recv(ctx)) is not None:
        await ch_out.send(ctx, msg)
    await ch_out.drain(ctx)


@define_actor()
async def receive_metadata(
    ctx: Context, ch_in: Channel[TableChunk], ch_out: Channel[TableChunk]
) -> None:
    m = await ch_in.recv_metadata(ctx)
    assert m is not None
    meta = ChannelMetadata.from_message(m)
    assert meta.local_count == 1
    while (msg := await ch_in.recv(ctx)) is not None:
        await ch_out.send(ctx, msg)
    await ch_out.drain(ctx)


@define_actor()
async def bloom_pipeline(
    ctx: Context,
    bloom: BloomFilter,
    ch_build: Channel[TableChunk],
    ch_probe: Channel[TableChunk],
    ch_out: Channel[TableChunk],
) -> None:
    ch_filter: Channel[BloomFilterChunk] = ctx.create_channel()
    await asyncio.gather(
        bloom.build(ctx, ch_in=ch_build, ch_out=ch_filter, tag=0),
        bloom.apply(
            ctx,
            bloom_filter=ch_filter,
            ch_in=ch_probe,
            ch_out=ch_out,
            keys=(0,),
        ),
    )


def run_bloom_filter_pipeline(
    context: Context,
    comm: Communicator,
    build_table: TableChunk,
    probe_table: TableChunk,
    *,
    seed: int = 42,
    filter_size: int = 1 << 20,
) -> list[Message]:
    bloom = BloomFilter(
        context,
        comm,
        seed=seed,
        filter_size=filter_size,
    )

    build_msg = Message(0, build_table)
    probe_msg = Message(0, probe_table)

    ch_build: Channel[TableChunk] = context.create_channel()
    ch_probe: Channel[TableChunk] = context.create_channel()
    ch_probe_meta: Channel[TableChunk] = context.create_channel()
    ch_out_meta: Channel[TableChunk] = context.create_channel()
    ch_out: Channel[TableChunk] = context.create_channel()

    actors: list[CppActor | Awaitable[None]] = [
        push_to_channel(context, ch_build, [build_msg]),
        push_to_channel(context, ch_probe, [probe_msg]),
        add_metadata(context, ch_probe, ch_probe_meta),
        bloom_pipeline(context, bloom, ch_build, ch_probe_meta, ch_out_meta),
        receive_metadata(context, ch_out_meta, ch_out),
    ]
    pull_actor, deferred = pull_from_channel(context, ch_out)
    actors.append(pull_actor)
    run_actor_network(context, actors=actors)
    return deferred.release()


def test_bloom_filter_roundtrip(context: Context, comm: Communicator) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")

    stream = context.br().stream_pool.get_stream()
    values = np.arange(10, dtype=np.int32)
    build_table = make_table(values, stream=stream, br=context.br())
    probe_table = make_table(values, stream=stream, br=context.br())
    messages = run_bloom_filter_pipeline(
        context, comm, build_table, probe_table
    )
    assert len(messages) == 1

    result = TableChunk.from_message(messages[0], br=context.br())
    expected = plc.Table([plc.Column.from_array(values, stream=result.stream)])
    result.stream.synchronize()
    assert_eq(result.table_view(), expected)


def test_bloom_filter_empty_build_filters_all(
    context: Context, comm: Communicator
) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")

    stream = context.br().stream_pool.get_stream()
    build_table = make_table(
        np.array([], dtype=np.int32), stream=stream, br=context.br()
    )
    probe_table = make_table(
        np.arange(5, dtype=np.int32), stream=stream, br=context.br()
    )
    messages = run_bloom_filter_pipeline(
        context, comm, build_table, probe_table
    )
    assert len(messages) == 1

    result = TableChunk.from_message(messages[0], br=context.br())
    expected = plc.Table(
        [
            plc.Column.from_array(
                np.array([], dtype=np.int32), stream=result.stream
            )
        ]
    )
    result.stream.synchronize()
    assert_eq(result.table_view(), expected)


def test_bloom_filter_build_exception_no_shutdown(
    context: Context, comm: Communicator
) -> None:
    if comm.nranks != 1:
        pytest.skip("Only support single-rank runs")

    stream = context.br().stream_pool.get_stream()
    bloom = BloomFilter(
        context,
        comm,
        seed=42,
        filter_size=(1 << 20),
    )
    ch_in: Channel[TableChunk] = context.create_channel()
    ch_out: Channel[BloomFilterChunk] = context.create_channel()
    messages = [
        Message(
            sequence_number,
            make_table(
                np.arange(10, dtype=np.int32),
                stream=stream,
                br=context.br(),
            ),
        )
        for sequence_number in range(3)
    ]

    async def recv_then_raise(
        context: Context, ch_in: Channel[BloomFilterChunk]
    ):
        await ch_in.recv(context)
        raise RuntimeError("Raising but didn't shutdown channel")

    # With no consumer for ch_out, the bloom-filter build blocks while draining
    # its output channel. pytest-timeout interrupts run_actor_network, which
    # must cancel and drain the worker before propagating the timeout failure.
    with pytest.RaisesGroup(
        pytest.RaisesExc(RuntimeError, match="didn't shutdown channel")
    ):
        run_actor_network(
            context,
            actors=[
                push_to_channel(context, ch_in, messages),
                bloom.build(context, ch_in=ch_in, ch_out=ch_out, tag=0),
                recv_then_raise(context, ch_out),
            ],
        )

    async def recv_after_cancellation() -> Message | None:
        return await asyncio.wait_for(ch_out.recv(context), timeout=1)

    # Cancellation should close the output channel. Without shutting down the
    # channels inside the bloom filter if we get a cancellation, this
    # receive picks up the message that is still in the channel.
    assert asyncio.run(recv_after_cancellation()) is None
