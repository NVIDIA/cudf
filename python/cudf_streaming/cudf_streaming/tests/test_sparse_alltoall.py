# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SparseAlltoall functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pylibcudf as plc
import pytest

from cudf_streaming.partition_utils import (
    packed_data_from_cudf_packed_columns,
    unpack_and_concat,
)
from cudf_streaming.testing import assert_eq
from rapidsmpf.coll.sparse_alltoall import SparseAlltoall
from rapidsmpf.memory.buffer_resource import BufferResource

if TYPE_CHECKING:
    import rmm.mr
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.packed_data import PackedData
    from rmm.pylibrmm.stream import Stream


def generate_packed_data(
    n_elements: int, offset: int, stream: Stream, br: BufferResource
) -> PackedData:
    """Generate packed integer data with a predictable payload."""
    values = np.arange(offset, offset + n_elements, dtype=np.int32)
    table = plc.Table([plc.Column.from_array(values, stream=stream)])
    packed_columns = plc.contiguous_split.pack(table, stream=stream)
    return packed_data_from_cudf_packed_columns(packed_columns, stream, br)


def unpack_table(
    packed_data: PackedData,
    stream: Stream,
    br: BufferResource,
) -> plc.Table:
    """Unpack a PackedData payload into a single-column table."""
    return unpack_and_concat([packed_data], stream, br)


def expected_peers(comm: Communicator) -> tuple[list[int], list[int]]:
    """Return the immediate non-self neighbors in the communicator."""
    peers = []
    if comm.rank > 0:
        peers.append(comm.rank - 1)
    if comm.rank + 1 < comm.nranks:
        peers.append(comm.rank + 1)
    return peers, peers


def make_offset(src: int, dst: int, ordinal: int) -> int:
    """Encode the sender, receiver, and per-destination ordinal into payload data."""
    return src * 1000 + dst * 100 + ordinal * 10


@pytest.mark.parametrize("n_inserts", [0, 1, 3])
def test_basic(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
    stream: Stream,
    n_inserts: int,
) -> None:
    """
    Test a simple sparse exchange with immediate neighbors.

    Each rank sends `n_inserts` messages to each adjacent rank and then checks
    that extraction by source returns the expected number of payloads.
    """
    br = BufferResource(device_mr)
    srcs, dsts = expected_peers(comm)
    sparse_alltoall = SparseAlltoall(
        comm=comm,
        op_id=0,
        br=br,
        srcs=srcs,
        dsts=dsts,
    )

    for dst in dsts:
        for ordinal in range(n_inserts):
            sparse_alltoall.insert(
                dst,
                generate_packed_data(
                    n_elements=4,
                    offset=make_offset(comm.rank, dst, ordinal),
                    stream=stream,
                    br=br,
                ),
            )

    sparse_alltoall.insert_finished()
    sparse_alltoall.wait()

    for src in srcs:
        results = sparse_alltoall.extract(src)
        assert len(results) == n_inserts
        for ordinal, result in enumerate(results):
            expected = plc.Table(
                [
                    plc.Column.from_array(
                        np.arange(
                            make_offset(src, comm.rank, ordinal),
                            make_offset(src, comm.rank, ordinal) + 4,
                            dtype=np.int32,
                        ),
                        stream=stream,
                    )
                ]
            )
            assert_eq(unpack_table(result, stream, br), expected)


def test_non_participating_ranks(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
    stream: Stream,
) -> None:
    if comm.nranks < 2:
        pytest.skip("Need at least two ranks")

    br = BufferResource(device_mr)

    if comm.rank == 0:
        srcs = []
        dsts = [1]
    elif comm.rank == 1:
        srcs = [0]
        dsts = []
    else:
        srcs = []
        dsts = []

    sparse_alltoall = SparseAlltoall(
        comm=comm,
        op_id=1,
        br=br,
        srcs=srcs,
        dsts=dsts,
    )

    if comm.rank == 0:
        sparse_alltoall.insert(1, generate_packed_data(1, 11, stream, br))
        sparse_alltoall.insert(1, generate_packed_data(1, 29, stream, br))

    sparse_alltoall.insert_finished()
    sparse_alltoall.wait()

    if comm.rank == 1:
        results = sparse_alltoall.extract(0)
        assert len(results) == 2
        assert_eq(
            unpack_table(results[0], stream, br),
            plc.Table(
                [
                    plc.Column.from_array(
                        np.array([11], dtype=np.int32), stream=stream
                    )
                ]
            ),
        )
        assert_eq(
            unpack_table(results[1], stream, br),
            plc.Table(
                [
                    plc.Column.from_array(
                        np.array([29], dtype=np.int32), stream=stream
                    )
                ]
            ),
        )


def test_invalid_peers_raise(
    comm: Communicator,
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    rank = comm.rank
    size = comm.nranks
    br = BufferResource(device_mr)
    for src, dst in [([], [rank]), ([rank], []), ([], [size]), ([size], [])]:
        with pytest.raises(
            IndexError,
            match=r"SparseAlltoall invalid (source|destination) rank",
        ):
            SparseAlltoall(comm, 1, br=br, srcs=src, dsts=dst)
    if size > 1:
        peer = (rank + 1) % size
        for src, dst in [([], [peer, peer]), ([peer, peer], [])]:
            with pytest.raises(
                ValueError,
                match=r"SparseAlltoall (source|destination) rank list must be unique",
            ):
                SparseAlltoall(comm, 1, br=br, srcs=src, dsts=dst)
