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
