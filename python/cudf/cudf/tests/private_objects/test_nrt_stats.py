# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from numba.cuda import config
from numba.cuda.memory_management.nrt import rtsys

import cudf
from cudf._lib import strings_udf
from cudf.core.column import ColumnBase, as_column
from cudf.core.udf.scalar_function import SeriesApplyKernel
from cudf.core.udf.utils import (
    _get_input_args_from_frame,
    _make_free_string_kernel,
    _return_arr_from_dtype,
)
from cudf.utils._numba import _CUDFNumbaConfig


@pytest.mark.skip(reason="https://github.com/NVIDIA/cudf/issues/19880")
def test_string_udf_basic(monkeypatch):
    monkeypatch.setattr(config, "CUDA_NRT_STATS", True)

    def double(st):
        return st + st

    sr = cudf.Series(["a", "b", "c"])

    sr.apply(double)

    stats = rtsys.get_allocation_stats()

    # one meminfo for each string that is later freed
    assert stats.mi_alloc - stats.mi_free == 0

    # one NRT_Allocate call for each string (string heap copy)
    # and later its matching free
    assert stats.alloc - stats.free == 0


@pytest.mark.skip(reason="https://github.com/NVIDIA/cudf/issues/19880")
def test_string_udf_conditional_allocations(monkeypatch):
    monkeypatch.setattr(config, "CUDA_NRT_STATS", True)

    # One thread allocates an intermediate string
    # but the others do not
    def double(st):
        if st == "b":
            return st + st == "BB"
        return st == "a" or st == "c"

    sr = cudf.Series(["a", "b", "c"])

    before_stats = rtsys.get_allocation_stats()
    sr.apply(double)
    after_stats = rtsys.get_allocation_stats()

    assert after_stats.mi_alloc - before_stats.mi_free == 1
    assert after_stats.alloc - before_stats.free == 1


@pytest.mark.skip(reason="https://github.com/NVIDIA/cudf/issues/19880")
def test_string_udf_free_kernel(monkeypatch):
    monkeypatch.setattr(config, "CUDA_NRT_STATS", True)

    def double(st):
        return st + st

    sr = cudf.Series(["a", "b", "c"])

    kernel, retty = SeriesApplyKernel(sr, double, ()).get_kernel()

    ans_col = _return_arr_from_dtype(retty, len(sr))
    ans_mask = as_column(True, length=len(sr), dtype="bool")
    output_args = [(ans_col, ans_mask), len(sr)]
    input_args = _get_input_args_from_frame(sr)
    launch_args = output_args + input_args

    with _CUDFNumbaConfig():
        kernel.forall(len(sr))(*launch_args)
    col = ColumnBase.create(
        strings_udf.column_from_managed_udf_string_array(ans_col),
        dtype=np.dtype(object),
    )

    # MemInfos that own the strings should still be alive
    # and in turn, so should the heap strings
    stats = rtsys.get_allocation_stats()
    assert stats.mi_alloc - stats.mi_free == len(sr)
    assert stats.alloc - stats.free == len(sr)

    # free kernel should equalize all allocations
    free_kernel = _make_free_string_kernel()
    with _CUDFNumbaConfig():
        free_kernel.forall(len(col))(ans_col, len(col))

    stats = rtsys.get_allocation_stats()

    assert stats.mi_alloc - stats.mi_free == 0
    assert stats.alloc - stats.free == 0
