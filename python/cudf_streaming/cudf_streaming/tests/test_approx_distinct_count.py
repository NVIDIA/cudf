# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pylibcudf as plc
import pytest

from cudf_streaming import CardinalityEstimate, CardinalityEstimator
from cudf_streaming.table_chunk import TableChunk
from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.core.leaf_actor import (
    pull_from_channel,
    push_to_channel,
)
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context


def make_table(values: np.ndarray, context: Context) -> TableChunk:
    stream = context.br().stream_pool.get_stream()
    table = plc.Table([plc.Column.from_array(values, stream=stream)])
    return TableChunk.from_pylibcudf_table(
        table, stream, exclusive_view=True, br=context.br()
    )


def estimate(
    context: Context,
    comm: Communicator,
    chunks: list[TableChunk],
    *,
    precision: int = 14,
) -> CardinalityEstimate:
    estimator = CardinalityEstimator(context, comm, tag=0, precision=precision)
    ch_in = context.create_channel()
    ch_out = context.create_channel()
    inputs = [Message(i, chunk) for i, chunk in enumerate(chunks)]
    pull_actor, deferred = pull_from_channel(context, ch_out)
    run_actor_network(
        context,
        actors=[
            push_to_channel(context, ch_in, inputs),
            estimator.estimate(context, ch_in, ch_out),
            pull_actor,
        ],
    )
    outputs = deferred.release()
    assert len(outputs) == 1
    return CardinalityEstimate.from_message(outputs[0])


def sample(
    context: Context,
    comm: Communicator,
    chunks: list[TableChunk],
    *,
    max_chunks: int,
    column_indices: tuple[int, ...] = (),
) -> tuple[list[Message], CardinalityEstimate]:
    estimator = CardinalityEstimator(context, comm, tag=0)
    ch_in = context.create_channel()
    ch_sampled = context.create_channel()
    ch_out = context.create_channel()
    inputs = [Message(i, chunk) for i, chunk in enumerate(chunks[:max_chunks])]

    sampled_actor, sampled_deferred = pull_from_channel(context, ch_sampled)
    estimate_actor, estimate_deferred = pull_from_channel(context, ch_out)
    run_actor_network(
        context,
        actors=[
            push_to_channel(context, ch_in, inputs),
            estimator.estimate(
                context,
                ch_in,
                ch_out,
                ch_sampled,
                column_indices,
            ),
            sampled_actor,
            estimate_actor,
        ],
    )
    estimates = estimate_deferred.release()
    assert len(estimates) == 1
    return (
        sampled_deferred.release(),
        CardinalityEstimate.from_message(estimates[0]),
    )


def test_estimate_global_cardinality(
    context: Context, comm: Communicator
) -> None:
    distinct_count = 10_000
    values = np.tile(np.arange(distinct_count, dtype=np.int64), 2)
    result = estimate(
        context,
        comm,
        [make_table(part, context) for part in np.array_split(values, 3)],
    )

    assert result.row_count == values.size * comm.nranks
    assert result.distinct_count == pytest.approx(distinct_count, rel=0.05)


def test_estimate_empty_input(context: Context, comm: Communicator) -> None:
    result = estimate(context, comm, [])

    assert result.row_count == 0
    assert result.distinct_count == 0


def test_estimate_forwards_input(context: Context, comm: Communicator) -> None:
    values = np.arange(100, dtype=np.int64)
    sampled, result = sample(
        context,
        comm,
        [make_table(values, context)],
        max_chunks=2,
    )

    assert len(sampled) == 1
    assert result.row_count == values.size * comm.nranks
    assert result.distinct_count == pytest.approx(values.size, rel=0.05)


def test_precision(context: Context, comm: Communicator) -> None:
    assert CardinalityEstimator(context, comm, tag=7).tag == 7
    assert CardinalityEstimator(context, comm, tag=7).precision == 14
    assert (
        CardinalityEstimator(context, comm, tag=7, precision=12).precision
        == 12
    )


@pytest.mark.parametrize("precision", [3, 19])
def test_invalid_precision(
    context: Context, comm: Communicator, precision: int
) -> None:
    with pytest.raises(
        ValueError, match=r"precision must be in range \[4, 18\]"
    ):
        CardinalityEstimator(context, comm, tag=0, precision=precision)
