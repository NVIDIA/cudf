# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda.experimental import consteval
from numba_cuda_mlir.numba_cuda.np import numpy_support

from cudf.api.types import is_string_dtype as _is_string_dtype
from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.mlir_backend.masked_typing import MaskedType
from cudf.core.udf.mlir_backend.strings_typing import mlir_string
from cudf.core.udf.mlir_backend.udf_kernel_base import ApplyKernelBase
from cudf.core.udf.utils import (
    _mask_get,
)


class SeriesApplyKernel(ApplyKernelBase):
    """
    Class representing a kernel that computes the result of
    a Series.apply operation. Expects that the user passed
    a function that operates on an single element of the Series,
    for example

    def f(x):
        return x + 1
    """

    @property
    def kernel_type(self):
        return "series_apply"

    def _get_frame_type(self):
        if _is_string_dtype(self.frame.dtype):
            return MaskedType(mlir_string)
        return MaskedType(numpy_support.from_dtype(self.frame.dtype))

    def _make_kernel(self):
        """Build the Series.apply kernel by metaprogramming (``consteval``).

        The single input column is ``columns[0]`` and ``*consteval(extra)``
        splats the UDF's extra arguments. No string templates or ``exec``.
        """
        device_func = self.device_func

        def _kernel(retval, size, columns, extra):
            i = cuda.grid(1)
            ret_data_arr, ret_mask_arr = retval
            if i < size:
                d, m, off = columns[0]
                if m.size == 0:
                    valid = True
                else:
                    valid = _mask_get(m, i + off)
                masked = Masked(d[i + off], valid)
                ret = device_func(masked, *consteval(extra))
                ret_masked = pack_return(ret)
                ret_data_arr[i] = ret_masked.value
                ret_mask_arr[i] = ret_masked.valid

        return _kernel
