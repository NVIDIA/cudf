# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_table_eq

from rmm.pylibrmm.stream import Stream

import pylibcudf as plc


@pytest.mark.parametrize(
    "arrow_tbl",
    [
        pa.table([]),
        pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        pa.table({"a": [1, 2, 3]}),
        pa.table({"a": [1], "b": [2], "c": [3]}),
    ],
)
def test_table_shape(arrow_tbl):
    plc_tbl = plc.Table.from_arrow(arrow_tbl)

    assert plc_tbl.shape() == arrow_tbl.shape


@pytest.mark.parametrize("stream", [None, Stream()])
def test_table_to_arrow(table_data, stream):
    plc_tbl, _ = table_data
    expect = plc_tbl.tbl
    got = expect.to_arrow(stream=stream)
    # The order of `got` and `expect` is reversed here
    # because in almost all pylibcudf tests the `expect`
    # is a pyarrow object while `got` is a pylibcudf object,
    # whereas in this case those types are reversed.
    assert_table_eq(got, expect)


def test_table_copy(table_data):
    plc_tbl, _ = table_data
    original = plc_tbl.tbl

    copied = original.copy()

    assert copied.shape() == original.shape()
    assert copied.num_columns() == original.num_columns()
    assert copied.num_rows() == original.num_rows()

    for orig_col, copy_col in zip(
        original.columns(), copied.columns(), strict=True
    ):
        assert orig_col is not copy_col

    assert_table_eq(original.to_arrow(), copied)


@pytest.mark.parametrize(
    "kwargs, expected_rows", [({}, 0), ({"num_rows": 5}, 5)]
)
def test_zero_column_table_num_rows(kwargs, expected_rows):
    tbl = plc.Table([], **kwargs)
    assert tbl.num_columns() == 0
    assert tbl.num_rows() == expected_rows
    assert tbl.shape() == (expected_rows, 0)
    # The row count is preserved by copy().
    assert tbl.copy().num_rows() == expected_rows


@pytest.mark.parametrize(
    "num_rows, exc",
    [(-1, ValueError), (2**31, OverflowError), (3.5, TypeError)],
)
def test_zero_column_table_invalid_num_rows_raises(num_rows, exc):
    with pytest.raises(exc):
        plc.Table([], num_rows=num_rows)


def test_table_num_rows_mismatch_raises():
    col3 = plc.Column.from_arrow(pa.array([1, 2, 3]))
    col4 = plc.Column.from_arrow(pa.array([1, 2, 3, 4]))
    # num_rows does not match the single column.
    with pytest.raises(ValueError):
        plc.Table([col3], num_rows=4)
    # num_rows matches the first column but not the second; both must agree.
    with pytest.raises(ValueError):
        plc.Table([col3, col4], num_rows=3)


def test_zero_column_table_concatenate_sums_rows():
    result = plc.concatenate.concatenate(
        [plc.Table([], num_rows=7), plc.Table([], num_rows=5)]
    )
    assert result.num_columns() == 0
    assert result.num_rows() == 12


def test_from_arrow_zero_column_preserves_num_rows():
    batch = pa.RecordBatch.from_struct_array(
        pa.array([{}] * 5, type=pa.struct([]))
    )
    arrow_tbl = pa.Table.from_batches([batch])
    assert arrow_tbl.shape == (5, 0)

    tbl = plc.Table.from_arrow(arrow_tbl)
    assert tbl.num_columns() == 0
    assert tbl.num_rows() == 5


def test_to_arrow_zero_column_preserves_num_rows():
    arrow = plc.Table([], num_rows=5).to_arrow()
    assert arrow.num_columns == 0
    assert arrow.num_rows == 5
