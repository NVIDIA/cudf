# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pylibcudf as plc
import pytest

from cudf_streaming.partition_utils import (
    partition_and_pack,
    partition_and_pack_cost,
    split_and_pack,
    split_and_pack_cost,
    unpack_and_concat,
    unpack_and_concat_cost,
)
from cudf_streaming.testing import assert_eq
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.reservation_events import ReservationContextCreator
from rapidsmpf.memory.spill import spill_partitions, unspill_partitions
from rmm.pylibrmm.stream import DEFAULT_STREAM

if TYPE_CHECKING:
    import rmm.mr


def _make_table(cols: list[list[int]]) -> plc.Table:
    # Assigns empty column inputs as int64
    return plc.Table(
        [
            plc.Column.from_iterable_of_py(col, plc.DataType(plc.TypeId.INT64))
            for col in cols
        ]
    )


@pytest.mark.parametrize("cols", [[[1, 2, 3], [2, 2, 1]], [[], []]])
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_partition_and_pack_unpack(
    device_mr: rmm.mr.CudaMemoryResource,
    cols: list[list[int]],
    num_partitions: int,
) -> None:
    br = BufferResource(device_mr)
    expect = _make_table(cols)
    partitions = partition_and_pack(
        expect,
        columns_to_hash=(1,),
        num_partitions=num_partitions,
        br=br,
        stream=DEFAULT_STREAM,
    )
    got = unpack_and_concat(
        tuple(partitions.values()),
        br=br,
        stream=DEFAULT_STREAM,
    )
    # Since the row order isn't preserved, we sort the rows by the first column.
    assert_eq(expect, got, sort_rows=0)


@pytest.mark.parametrize(
    "cols",
    [
        [[1, 2, 3], [2, 2, 1]],
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        [[], []],
    ],
)
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_split_and_pack_unpack(
    device_mr: rmm.mr.CudaMemoryResource,
    cols: list[list[int]],
    num_partitions: int,
) -> None:
    br = BufferResource(device_mr)
    expect = _make_table(cols)
    splits = np.linspace(0, expect.num_rows(), num_partitions, endpoint=False)[
        1:
    ].astype(int)
    partitions = split_and_pack(
        expect,
        splits=splits,
        br=br,
        stream=DEFAULT_STREAM,
    )
    got = unpack_and_concat(
        tuple(partitions[i] for i in range(num_partitions)),
        br=br,
        stream=DEFAULT_STREAM,
    )

    assert_eq(expect, got)


@pytest.mark.parametrize("cols", [[[1, 2, 3], [2, 2, 1]], [[], []]])
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_split_and_pack_unpack_out_of_range(
    device_mr: rmm.mr.CudaMemoryResource,
    cols: list[list[int]],
    num_partitions: int,
) -> None:
    br = BufferResource(device_mr)
    expect = _make_table(cols)
    with pytest.raises(IndexError):
        split_and_pack(
            expect,
            splits=[100],
            br=br,
            stream=DEFAULT_STREAM,
        )


@pytest.mark.parametrize("cols", [[[1, 2, 3], [2, 2, 1]], [[], []]])
@pytest.mark.parametrize("num_partitions", [1, 2, 3, 10])
def test_spill_unspill_roundtrip(
    device_mr: rmm.mr.CudaMemoryResource,
    cols: list[list[int]],
    num_partitions: int,
) -> None:
    br = BufferResource(device_mr)
    expect = _make_table(cols)
    partitions = partition_and_pack(
        expect,
        columns_to_hash=(1,),
        num_partitions=num_partitions,
        br=br,
        stream=DEFAULT_STREAM,
    )

    # Spill roundtrip
    spilled = spill_partitions(partitions.values(), br=br)
    unspilled = unspill_partitions(spilled, br=br, allow_overbooking=False)

    got = unpack_and_concat(
        unspilled,
        br=br,
        stream=DEFAULT_STREAM,
    )
    # Since the row order isn't preserved, we sort the rows by the first column.
    assert_eq(expect, got, sort_rows=0)


def test_unpack_and_concat_cost_does_not_consume(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    br = BufferResource(device_mr)
    stream = DEFAULT_STREAM
    expect = _make_table([[1, 2, 3, 4], [5, 6, 7, 8]])
    packed = list(
        partition_and_pack(
            expect,
            columns_to_hash=(0,),
            num_partitions=2,
            stream=stream,
            br=br,
        ).values()
    )

    # Calling it twice, and from a generator, must leave the partitions usable.
    cost = unpack_and_concat_cost(packed)
    assert cost > 0
    assert unpack_and_concat_cost(p for p in packed) == cost

    got = unpack_and_concat(packed, stream, br)
    assert_eq(expect, got, sort_rows=0)

    # They are consumed now, so using them again must raise rather than crash.
    with pytest.raises(ValueError):
        unpack_and_concat_cost(packed)
    with pytest.raises(ValueError):
        unpack_and_concat(packed, stream, br)


def test_round_trip_with_reservations(
    device_mr: rmm.mr.CudaMemoryResource,
) -> None:
    """Each function consumes a reservation sized by its cost function."""
    br = BufferResource(device_mr)
    stream = DEFAULT_STREAM
    expect = _make_table([[1, 2, 3, 4], [5, 6, 7, 8]])

    pack_res = br.reserve_or_fail(
        partition_and_pack_cost(expect, stream, br), [MemoryType.DEVICE]
    )
    packed = partition_and_pack(
        expect,
        columns_to_hash=(0,),
        num_partitions=2,
        stream=stream,
        br=br,
        reservation=pack_res,
    )
    assert pack_res.size == 0

    split_res = br.reserve_or_fail(
        split_and_pack_cost(expect, stream, br), [MemoryType.DEVICE]
    )
    split = split_and_pack(expect, [2], stream, br, reservation=split_res)
    assert split_res.size == 0

    for chunks in (packed, split):
        parts = list(chunks.values())
        concat_res = br.reserve_or_fail(
            unpack_and_concat_cost(parts), [MemoryType.DEVICE]
        )
        got = unpack_and_concat(parts, stream, br, reservation=concat_res)
        assert concat_res.size == 0
        assert_eq(expect, got, sort_rows=0)


@pytest.mark.parametrize(
    "operation", ["partition_and_pack", "split_and_pack", "unpack_and_concat"]
)
@pytest.mark.parametrize("with_context", [True, False])
def test_reservation_context(
    device_mr: rmm.mr.CudaMemoryResource,
    operation: str,
    *,
    with_context: bool,
) -> None:
    """The internal reservation carries the given reservation context."""
    br = BufferResource(device_mr)
    stream = DEFAULT_STREAM
    expect = _make_table([[1, 2, 3, 4], [5, 6, 7, 8]])
    reservation_context = ReservationContextCreator().create()
    passed_context = reservation_context if with_context else None
    # Packed up front, so that unpack_and_concat has something to consume.
    packed = list(
        partition_and_pack(
            expect,
            columns_to_hash=(0,),
            num_partitions=2,
            stream=stream,
            br=br,
        ).values()
    )

    br.start_reservation_event_recording()
    if operation == "partition_and_pack":
        partition_and_pack(
            expect,
            columns_to_hash=(0,),
            num_partitions=2,
            stream=stream,
            br=br,
            reservation_context=passed_context,
        )
    elif operation == "split_and_pack":
        split_and_pack(
            expect,
            splits=[2],
            stream=stream,
            br=br,
            reservation_context=passed_context,
        )
    else:
        got = unpack_and_concat(
            packed,
            stream=stream,
            br=br,
            reservation_context=passed_context,
        )
        assert_eq(expect, got, sort_rows=0)
    events = br.stop_reservation_event_recording().events

    assert events
    if with_context:
        assert any(
            event.reservation_context == reservation_context
            and event.byte_delta > 0
            for event in events
        )
    else:
        assert not any(
            event.reservation_context.is_attributed for event in events
        )
