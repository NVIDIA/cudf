# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq, assert_table_eq

import pylibcudf as plc


@pytest.fixture
def lists_column_and_mask():
    pa_input = pa.array(
        [[0, 1, 2, 3], [4, 5], [6, 7, 8, 9]], type=pa.list_(pa.int32())
    )
    pa_mask = pa.array(
        [
            [True, False, True, False],
            [True, False],
            [True, False, True, False],
        ],
        type=pa.list_(pa.bool_()),
    )
    return pa_input, pa_mask


def test_lists_apply_boolean_mask(lists_column_and_mask):
    pa_input, pa_mask = lists_column_and_mask
    result = plc.lists.apply_boolean_mask(
        plc.Column.from_arrow(pa_input), plc.Column.from_arrow(pa_mask)
    )
    expected = pa.array([[0, 2], [4], [6, 8]], type=pa.list_(pa.int32()))
    assert_column_eq(expected, result)


def test_lists_apply_deletion_mask(lists_column_and_mask):
    pa_input, pa_mask = lists_column_and_mask
    result = plc.lists.apply_deletion_mask(
        plc.Column.from_arrow(pa_input), plc.Column.from_arrow(pa_mask)
    )
    expected = pa.array([[1, 3], [5], [7, 9]], type=pa.list_(pa.int32()))
    assert_column_eq(expected, result)


def test_apply_boolean_mask():
    pa_table = pa.table(
        {
            "a": pa.array([10, 40, 70, 5, 2, 10], type=pa.int32()),
            "b": pa.array([10, 40, 70, 5, 2, 10], type=pa.float64()),
        }
    )
    pa_mask = pa.array(
        [True, False, True, False, True, False], type=pa.bool_()
    )
    result = plc.stream_compaction.apply_boolean_mask(
        plc.Table.from_arrow(pa_table), plc.Column.from_arrow(pa_mask)
    )
    expected = pa_table.filter(pa_mask)
    assert_table_eq(expected, result)


def test_apply_deletion_mask():
    pa_table = pa.table({"a": pa.array([1, 2, 3, 4, 5], type=pa.int32())})
    pa_mask = pa.array([True, False, True, False, True], type=pa.bool_())
    result = plc.stream_compaction.apply_deletion_mask(
        plc.Table.from_arrow(pa_table), plc.Column.from_arrow(pa_mask)
    )
    expected = pa.table({"a": pa.array([2, 4], type=pa.int32())})
    assert_table_eq(expected, result)


def test_apply_boolean_mask_zero_columns_preserves_num_rows():
    source = plc.Table([], num_rows=4)
    mask = plc.Column.from_arrow(pa.array([True, False, True, True]))
    result = plc.stream_compaction.apply_boolean_mask(source, mask)
    assert result.num_columns() == 0
    assert result.num_rows() == 3
