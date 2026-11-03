# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import pylibcudf as plc


def test_tables_equal_true():
    left = plc.Table.from_arrow(
        pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    )
    right = plc.Table.from_arrow(
        pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    )
    assert plc.table_equality.tables_equal(left, right)


def test_tables_equal_false_values():
    left = plc.Table.from_arrow(pa.table({"a": [1, 2, 3]}))
    right = plc.Table.from_arrow(pa.table({"a": [1, 2, 4]}))
    assert not plc.table_equality.tables_equal(left, right)


def test_tables_equal_false_num_rows():
    left = plc.Table.from_arrow(pa.table({"a": [1, 2, 3]}))
    right = plc.Table.from_arrow(pa.table({"a": [1, 2]}))
    assert not plc.table_equality.tables_equal(left, right)


def test_tables_equal_false_num_columns():
    left = plc.Table.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    right = plc.Table.from_arrow(pa.table({"a": [1, 2, 3]}))
    assert not plc.table_equality.tables_equal(left, right)


def test_tables_equal_false_dtype():
    left = plc.Table.from_arrow(
        pa.table({"a": pa.array([1, 2, 3], type=pa.int32())})
    )
    right = plc.Table.from_arrow(
        pa.table({"a": pa.array([1, 2, 3], type=pa.int64())})
    )
    assert not plc.table_equality.tables_equal(left, right)


def test_tables_equal_empty():
    left = plc.Table.from_arrow(pa.table({"a": pa.array([], type=pa.int64())}))
    right = plc.Table.from_arrow(
        pa.table({"a": pa.array([], type=pa.int64())})
    )
    assert plc.table_equality.tables_equal(left, right)


@pytest.mark.parametrize(
    "nulls_equal,expected",
    [
        (plc.types.NullEquality.EQUAL, True),
        (plc.types.NullEquality.UNEQUAL, False),
    ],
)
def test_tables_equal_nulls(nulls_equal, expected):
    left = plc.Table.from_arrow(pa.table({"a": pa.array([1, None, 3])}))
    right = plc.Table.from_arrow(pa.table({"a": pa.array([1, None, 3])}))
    assert (
        plc.table_equality.tables_equal(left, right, nulls_equal) is expected
    )
