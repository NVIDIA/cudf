# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pylibcudf as plc
import pytest

from cudf_streaming.partition_utils import (
    partition_and_pack,
    split_and_pack,
    unpack_and_concat,
)
from cudf_streaming.testing import assert_eq
from rapidsmpf.memory.buffer_resource import BufferResource
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
