# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_table_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def partitioning_data():
    data = {"a": [1, 2, 3], "b": [1, 2, 5], "c": [1, 2, 10]}
    pa_table = pa.table(data)
    plc_table = plc.Table.from_arrow(pa_table)
    return data, plc_table


def test_partition(partitioning_data):
    raw_data, plc_table = partitioning_data
    got, offsets = plc.partitioning.partition(
        plc_table,
        plc.Column.from_arrow(pa.array([0, 0, 0])),
        1,
    )
    expect = pa.table(
        list(raw_data.values()),
        schema=pa.schema([pa.field("", pa.int64(), nullable=False)] * 3),
    )
    assert_table_eq(expect, got)
    assert offsets == [0, 3]


@pytest.mark.parametrize("use_key_table", [False, True])
def test_hash_partition(partitioning_data, use_key_table):
    raw_data, plc_table = partitioning_data
    key_cols = [0, 1]
    if use_key_table:
        keys = plc.Table([plc_table.columns()[i] for i in key_cols])
    else:
        keys = key_cols
    got, offsets = plc.partitioning.hash_partition(plc_table, keys, 1)
    expect = pa.table(
        list(raw_data.values()),
        schema=pa.schema([pa.field("", pa.int64(), nullable=False)] * 3),
    )
    assert_table_eq(expect, got)
    # Should return num_partitions + 1 offsets: [0, 3] for 1 partition with 3 rows
    assert offsets == [0, 3]


def _hash_partition_indices_groups(keys, num_partitions, order):
    gather_map, offsets = plc.partitioning.hash_partition_indices(
        keys, num_partitions, order
    )
    assert gather_map.type().id() == plc.TypeId.INT32
    assert len(offsets) == num_partitions + 1
    assert offsets[0] == 0
    assert offsets[-1] == keys.num_rows()
    indices = gather_map.to_arrow().to_pylist()
    return [
        indices[offsets[i] : offsets[i + 1]] for i in range(num_partitions)
    ]


@pytest.mark.parametrize(
    "order",
    [
        plc.partitioning.PartitionOrder.UNSTABLE,
        plc.partitioning.PartitionOrder.STABLE,
    ],
)
@pytest.mark.parametrize("num_partitions", [1, 3, 7])
def test_hash_partition_indices(order, num_partitions):
    pa_keys = pa.table(
        {"a": list(range(100)), "b": [i % 7 for i in range(100)]}
    )
    keys = plc.Table.from_arrow(pa_keys)
    groups = _hash_partition_indices_groups(keys, num_partitions, order)
    assert sorted(i for g in groups for i in g) == list(range(100))
    if order == plc.partitioning.PartitionOrder.STABLE:
        assert all(g == sorted(g) for g in groups)

    expect, expect_offsets = plc.partitioning.hash_partition(
        keys, keys, num_partitions
    )
    assert [len(g) for g in groups] == [
        expect_offsets[i + 1] - expect_offsets[i]
        for i in range(num_partitions)
    ]
    expect_rows = list(
        zip(*(c.to_pylist() for c in expect.to_arrow().columns), strict=True)
    )
    for i, g in enumerate(groups):
        got = pa_keys.take(pa.array(g, pa.int32()))
        got_rows = list(
            zip(*(c.to_pylist() for c in got.columns), strict=True)
        )
        expect_part = expect_rows[expect_offsets[i] : expect_offsets[i + 1]]
        assert sorted(got_rows) == sorted(expect_part)


def test_hash_partition_indices_empty_input():
    keys = plc.Table.from_arrow(pa.table({"a": pa.array([], pa.int64())}))
    gather_map, offsets = plc.partitioning.hash_partition_indices(
        keys, 3, plc.partitioning.PartitionOrder.STABLE
    )
    assert gather_map.type().id() == plc.TypeId.INT32
    assert gather_map.size() == 0
    assert offsets == [0, 0, 0, 0]


def test_hash_partition_indices_zero_partitions(partitioning_data):
    _, plc_table = partitioning_data
    gather_map, offsets = plc.partitioning.hash_partition_indices(
        plc_table, 0, plc.partitioning.PartitionOrder.UNSTABLE
    )
    assert gather_map.type().id() == plc.TypeId.INT32
    assert gather_map.size() == 0
    assert offsets == [0]


def test_round_robin_partition(partitioning_data):
    raw_data, plc_table = partitioning_data
    got, offsets = plc.partitioning.round_robin_partition(plc_table, 1, 0)
    expect = pa.table(
        list(raw_data.values()),
        schema=pa.schema([pa.field("", pa.int64(), nullable=False)] * 3),
    )
    assert_table_eq(expect, got)
    # Should return num_partitions + 1 offsets: [0, 3] for 1 partition with 3 rows
    assert offsets == [0, 3]


def test_round_robin_partition_zero_columns_preserves_num_rows():
    result, offsets = plc.partitioning.round_robin_partition(
        plc.Table([], num_rows=7), 3
    )
    assert result.num_columns() == 0
    assert result.num_rows() == 7
    # 7 rows over 3 partitions deals sizes {3, 2, 2}: boundaries hold with no columns.
    assert offsets == [0, 3, 5, 7]
