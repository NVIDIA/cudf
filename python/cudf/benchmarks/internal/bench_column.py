# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of Column methods."""

import pytest
import pytest_cases
from utils import (
    benchmark_with_object,
    make_boolean_mask_column,
    make_gather_map,
)


@benchmark_with_object(cls="column", dtype="float")
def bench_apply_retention_mask(benchmark, column):
    mask = make_boolean_mask_column(column.size)
    benchmark(column.apply_retention_mask, mask)


@benchmark_with_object(cls="column", dtype="float")
def bench_dropna(benchmark, column):
    benchmark(column.dropna)


@benchmark_with_object(cls="column", dtype="float")
def bench_unique_single_column(benchmark, column):
    benchmark(column.unique)


@benchmark_with_object(cls="column", dtype="float")
@pytest.mark.parametrize("nullify", [True, False])
@pytest.mark.parametrize("gather_how", ["sequence", "reverse", "random"])
def bench_take(benchmark, column, gather_how, nullify):
    gather_map = make_gather_map(
        column.size * 0.4, column.size, gather_how
    )._column
    benchmark(column.take, gather_map, nullify=nullify)


@benchmark_with_object(cls="column", dtype="int", nulls=False)
def setitem_case_stride_1_slice_scalar(column):
    return column, slice(None, None, 1), 42


@benchmark_with_object(cls="column", dtype="int", nulls=False)
def setitem_case_stride_2_slice_scalar(column):
    return column, slice(None, None, 2), 42


@benchmark_with_object(cls="column", dtype="int", nulls=False)
def setitem_case_boolean_column_scalar(column):
    column = column
    return column, [True, False] * (len(column) // 2), 42


@benchmark_with_object(cls="column", dtype="int", nulls=False)
def setitem_case_int_column_scalar(column):
    column = column
    return column, list(range(len(column))), 42


@benchmark_with_object(cls="column", dtype="int", nulls=False)
def setitem_case_stride_1_slice_align_to_key_size(
    column,
):
    column = column
    key = slice(None, None, 1)
    start, stop, stride = key.indices(len(column))
    materialized_key_size = len(column.slice(start, stop, stride))
    return column, key, [42] * materialized_key_size


@benchmark_with_object(cls="column", dtype="int", nulls=False)
def setitem_case_stride_2_slice_align_to_key_size(
    column,
):
    column = column
    key = slice(None, None, 2)
    start, stop, stride = key.indices(len(column))
    materialized_key_size = len(column.slice(start, stop, stride))
    return column, key, [42] * materialized_key_size


@benchmark_with_object(cls="column", dtype="int", nulls=False)
def setitem_case_boolean_column_align_to_col_size(
    column,
):
    column = column
    size = len(column)
    return column, [True, False] * (size // 2), [42] * size


@benchmark_with_object(cls="column", dtype="int", nulls=False)
def setitem_case_int_column_align_to_col_size(column):
    column = column
    size = len(column)
    return column, list(range(size)), [42] * size


# Benchmark Grid
# key:  slice == 1  (fill or copy_range shortcut),
#       slice != 1  (scatter),
#       column(bool)    (boolean_mask_scatter),
#       column(int) (scatter)
# value:    scalar,
#           column (len(val) == len(key)),
#           column (len(val) != len(key) and len == num_true)


@pytest_cases.parametrize_with_cases(
    "column,key,value", cases=".", prefix="setitem"
)
def bench_setitem(benchmark, column, key, value):
    benchmark(column.__setitem__, key, value)
