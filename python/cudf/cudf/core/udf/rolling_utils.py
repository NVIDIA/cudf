# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from numba_cuda_mlir import compiler, cuda
from numba_cuda_mlir.numba_cuda.core import config as _mlir_config
from numba_cuda_mlir.numba_cuda.np import numpy_support

from cudf.core.column.column import ColumnBase, as_column
from cudf.core.udf.utils import UDFError
from cudf.utils.performance_tracking import _performance_tracking

if TYPE_CHECKING:
    from collections.abc import Callable


class _MLIRNumbaCudaConfig:
    """Silence numba_cuda_mlir low-occupancy warnings during launch."""

    def __enter__(self) -> None:
        self._low_occupancy_warnings = _mlir_config.CUDA_LOW_OCCUPANCY_WARNINGS
        _mlir_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _mlir_config.CUDA_LOW_OCCUPANCY_WARNINGS = self._low_occupancy_warnings


def _get_udf_return_type(func: Callable, value_dtype: np.dtype) -> np.dtype:
    """Compile ``func`` for a 1D window of ``value_dtype`` to infer its
    output dtype.
    """
    nb_value_type = numpy_support.from_dtype(value_dtype)
    signature = (nb_value_type[::1],)
    try:
        _, return_type = compiler.compile(
            func, signature, device=True, output="ptx"
        )
    except Exception as e:
        raise UDFError(str(e)) from e
    return np.dtype(numpy_support.as_dtype(return_type))


def _make_rolling_kernel(device_func):
    @cuda.jit
    def _kernel(data, start, end, out, valid, min_periods):
        i = cuda.grid(1)
        if i < out.size:
            begin = start[i]
            stop = end[i]
            count = stop - begin
            if count >= min_periods:
                out[i] = device_func(data[begin:stop])
                valid[i] = True
            else:
                valid[i] = False

    return _kernel


@functools.lru_cache(maxsize=32)
def _compile_or_get_kernel(func: Callable, value_dtype: np.dtype):
    return_dtype = _get_udf_return_type(func, value_dtype)
    device_func = cuda.jit(device=True)(func)
    kernel = _make_rolling_kernel(device_func)
    return kernel, return_dtype


@_performance_tracking
def jit_rolling_apply(
    source_column: ColumnBase,
    start: cp.ndarray,
    end: cp.ndarray,
    min_periods: int,
    func: Callable,
) -> ColumnBase:
    """Apply a user-defined function to each rolling window using a custom
    CUDA kernel compiled with ``numba_cuda_mlir``.

    Parameters
    ----------
    source_column : ColumnBase
        The (non-null) numeric column the windows are drawn from.
    start, end : cupy.ndarray
        ``size_type`` arrays giving the absolute ``[start, end)`` row
        indices of each row's window.
    min_periods : int
        Minimum number of observations in a window required to produce a
        non-null result.
    func : callable
        The user-defined function. Receives a 1D array (the window) and
        returns a scalar.
    """
    value_dtype = source_column.dtype
    kernel, return_dtype = _compile_or_get_kernel(func, value_dtype)

    n = len(source_column)
    if n == 0:
        return as_column(cp.empty(0, dtype=return_dtype))

    data = source_column.values
    out = cp.empty(n, dtype=return_dtype)
    valid = cp.zeros(n, dtype=np.bool_)

    with _MLIRNumbaCudaConfig():
        kernel.forall(n)(data, start, end, out, valid, min_periods)

    result = as_column(out)
    valid_col = as_column(valid)
    mask, null_count = valid_col.as_mask()
    return result.set_mask(mask, null_count)
