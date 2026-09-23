# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
from cudf.core.udf.groupby_utils import GroupByApplyKernel
from cudf.core.udf.row_function import DataFrameApplyKernel
from cudf.core.udf.scalar_function import SeriesApplyKernel
from cudf.core.udf.utils import UDF_SHIM_FILE


def test_numeric_apply_kernels_do_not_link_shim():
    def series_func(value):
        return value + 1

    def dataframe_func(row):
        return row["a"] + row["b"]

    series_kernel = SeriesApplyKernel(cudf.Series([1, 2]), series_func, ())
    dataframe_kernel = DataFrameApplyKernel(
        cudf.DataFrame({"a": [1, 2], "b": [3, 4]}), dataframe_func, ()
    )

    assert series_kernel._get_link_files(nrt=False) == []
    assert dataframe_kernel._get_link_files(nrt=False) == []

    assert series_kernel.frame.apply(series_func).to_arrow().to_pylist() == [
        2,
        3,
    ]
    assert dataframe_kernel.frame.apply(
        dataframe_func, axis=1
    ).to_arrow().to_pylist() == [4, 6]


def test_string_apply_kernel_links_shim():
    def func(value):
        return value.upper()

    kernel = SeriesApplyKernel(cudf.Series(["a", "b"]), func, ())

    assert kernel._get_link_files(nrt=False) == [UDF_SHIM_FILE]


def test_nrt_kernel_links_shim():
    def func(value):
        return value + 1

    kernel = SeriesApplyKernel(cudf.Series([1, 2]), func, ())

    assert kernel._get_link_files(nrt=True) == [UDF_SHIM_FILE]


def test_groupby_apply_kernel_links_shim():
    def func(group):
        return group["value"].sum()

    kernel = GroupByApplyKernel(cudf.DataFrame({"value": [1, 2]}), func, ())

    assert kernel._get_link_files(nrt=False) == [UDF_SHIM_FILE]
