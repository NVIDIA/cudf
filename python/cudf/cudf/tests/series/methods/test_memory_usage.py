# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
from cudf.testing import assert_eq


def test_memory_usage_list():
    s1 = cudf.Series([[1, 2], [3, 4]])
    assert s1.memory_usage() == 44
    assert s1[1:].memory_usage() == 24
    s2 = cudf.Series([[[[1, 2]]], [[[3, 4]]]])
    assert s2.memory_usage() == 68
    assert s2[1:].memory_usage() == 40
    s3 = cudf.Series([[{"b": 1, "a": 10}, {"b": 2, "a": 100}]])
    assert s3.memory_usage() == 40


def test_empty_nested_list_uninitialized_offsets_memory_usage():
    ser = cudf.Series(
        [[[1, 2], [3]], []], dtype=cudf.ListDtype(cudf.ListDtype("int64"))
    )
    # Unlike in Arrow a 0-size list column produced by libcudf could have no offsets
    # allocated and thus be truly empty (https://github.com/NVIDIA/cudf/issues/16164)
    assert ser.iloc[:0].memory_usage() == 0


def test_series_memory_usage():
    sr = cudf.Series([1, 2, 3, 4], dtype="int64")
    assert sr.memory_usage() == 32

    # Sliced series reports actual data size, not the base buffer size.
    # This is important to make correct decisions about data transfers, for
    # example in Dask.
    sliced_sr = sr[2:]
    assert sliced_sr.memory_usage() == 16

    sliced_sr[3] = None
    assert sliced_sr.memory_usage() == 80

    sr = cudf.Series(["hello world", "rapids ai", "abc", "z"])
    assert sr.memory_usage() == 44


def test_struct_with_null_memory_usage():
    df = cudf.DataFrame(
        {
            "a": cudf.Series([1, 2, -1, -1, 3], dtype="int64"),
            "b": cudf.Series([10, 20, -1, -1, 30], dtype="int64"),
        }
    )
    s = df.to_struct()
    assert s.memory_usage() == 80

    s[2:4] = None
    assert s.memory_usage() == 272


def test_struct_memory_usage():
    s = cudf.Series([{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}])
    df = s.struct.explode()

    assert_eq(s.memory_usage(), df.memory_usage().sum())
    assert s[1:].memory_usage() == 32
