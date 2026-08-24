# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pylibcudf as plc
import pytest

from cudf_streaming.partition_utils import (
    packed_data_from_cudf_packed_columns,
    unpack_and_concat,
)
from cudf_streaming.testing import assert_eq
from rapidsmpf.streaming.coll.sparse_alltoall import SparseAlltoall

if TYPE_CHECKING:
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.packed_data import PackedData
    from rapidsmpf.streaming.core.context import Context


def make_packed_data(context: Context, values: np.ndarray) -> PackedData:
    stream = context.br().stream_pool.get_stream()
    table = plc.Table([plc.Column.from_array(values, stream=stream)])
    return packed_data_from_cudf_packed_columns(
        plc.contiguous_split.pack(table, stream=stream),
        stream,
        context.br(),
    )


def unpack_table(context: Context, packed_data: PackedData) -> plc.Table:
    stream = context.br().stream_pool.get_stream()
    return unpack_and_concat([packed_data], stream, context.br())


def test_sparse_alltoall_non_participating_ranks(
    context: Context,
    comm: Communicator,
) -> None:
    if comm.nranks < 2:
        pytest.skip("Need at least two ranks")
    if comm.rank == 0:
        srcs = []
        dsts = [1]
    elif comm.rank == 1:
        srcs = [0]
        dsts = []
    else:
        srcs = []
        dsts = []

    exchange = SparseAlltoall(
        context,
        comm,
        0,
        srcs,
        dsts,
    )

    if comm.rank == 0:
        exchange.insert(
            1, make_packed_data(context, np.array([11], dtype=np.int32))
        )
        exchange.insert(
            1, make_packed_data(context, np.array([29], dtype=np.int32))
        )

    asyncio.run(exchange.insert_finished(context))

    if comm.rank == 1:
        results = exchange.extract(0)
        assert len(results) == 2
        stream = context.br().stream_pool.get_stream()
        assert_eq(
            unpack_table(context, results[0]),
            plc.Table(
                [
                    plc.Column.from_array(
                        np.array([11], dtype=np.int32), stream=stream
                    )
                ]
            ),
        )
        assert_eq(
            unpack_table(context, results[1]),
            plc.Table(
                [
                    plc.Column.from_array(
                        np.array([29], dtype=np.int32), stream=stream
                    )
                ]
            ),
        )
