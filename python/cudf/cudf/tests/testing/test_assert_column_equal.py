# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import cudf
from cudf.core.column.column import as_column
from cudf.testing._utils import (
    assert_column_memory_eq,
    assert_column_memory_ne,
)
from cudf.testing.testing import assert_column_equal


@pytest.fixture(
    scope="module",
    params=[
        range(10),
        ["hello", "world", "rapids", "AI"],
        [[1, 2, 3], [4, 5], [6], [], [7]],
        [{"f0": "hello", "f1": 42}, {"f0": "world", "f1": 3}],
    ],
)
def arrow_arrays(request):
    return pa.array(request.param)


def test_assert_column_memory_basic(arrow_arrays):
    left = cudf.core.column.ColumnBase.from_arrow(arrow_arrays)
    right = cudf.core.column.ColumnBase.from_arrow(arrow_arrays)

    with pytest.raises(AssertionError):
        assert_column_memory_eq(left, right)
    assert_column_memory_ne(left, right)


def test_assert_column_memory_slice(arrow_arrays):
    col = cudf.core.column.ColumnBase.from_arrow(arrow_arrays)
    left = col.slice(0, 1)
    right = col.slice(1, 2)

    with pytest.raises(AssertionError):
        assert_column_memory_eq(left, right)
    assert_column_memory_ne(left, right)

    with pytest.raises(AssertionError):
        assert_column_memory_eq(left, col)
    assert_column_memory_ne(left, col)

    with pytest.raises(AssertionError):
        assert_column_memory_eq(right, col)
    assert_column_memory_ne(right, col)


def test_assert_column_memory_basic_same(arrow_arrays):
    data = cudf.core.column.ColumnBase.from_arrow(arrow_arrays)
    plc_col = data.plc_column

    # Create two references to same underlying data
    left = cudf.core.column.ColumnBase.create(plc_col, dtype=data.dtype)
    right = cudf.core.column.ColumnBase.create(plc_col, dtype=data.dtype)

    assert_column_memory_eq(left, right)
    with pytest.raises(AssertionError):
        assert_column_memory_ne(left, right)


@pytest.mark.parametrize(
    "other_data",
    [
        ["1", "2", "3"],
        [[1], [2], [3]],
        [{"a": 1}, {"a": 2}, {"a": 3}],
    ],
)
def test_assert_column_equal_dtype_edge_cases(other_data):
    # string series should be 100% different
    # even when the elements are the same
    base = as_column([1, 2, 3])
    other = as_column(other_data)

    # for these dtypes, the diff should always be 100% regardless of the values
    with pytest.raises(
        AssertionError, match=r".*values are different \(100.0 %\).*"
    ):
        assert_column_equal(base, other, check_dtype=False)

    # the exceptions are the empty and all null cases
    assert_column_equal(base.slice(0, 0), other.slice(0, 0), check_dtype=False)
    assert_column_equal(other.slice(0, 0), base.slice(0, 0), check_dtype=False)

    base = as_column(cudf.NA, length=len(base), dtype=base.dtype)
    other = as_column(cudf.NA, length=len(other), dtype=other.dtype)

    assert_column_equal(base, other, check_dtype=False)
    assert_column_equal(other, base, check_dtype=False)
