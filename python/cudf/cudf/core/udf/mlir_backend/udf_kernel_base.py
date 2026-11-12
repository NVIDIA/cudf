# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import numpy as np
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.np import numpy_support
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof
from numba_cuda_mlir.types import (
    CPointer,
    Poison,
    Tuple,
    int8,
    int64,
    void,
)

from cudf.api.types import is_scalar
from cudf.core.udf.mlir_backend.masked_typing import MaskedType
from cudf.core.udf.mlir_backend.strings_typing import mlir_string_arg_handler
from cudf.core.udf.nrt_utils import CaptureNRTUsage
from cudf.core.udf.utils import (
    _generate_cache_key,
    _mask_get,
    _masked_array_type_from_col,
    _supported_cols_from_frame,
    precompiled as kernel_cache,
)
from cudf.utils.performance_tracking import _performance_tracking

__all__ = ["ApplyKernelBase"]


class ApplyKernelBase(ABC):
    """
    Base class for kernels computing the result of `.apply`
    operations on Series or DataFrame objects.
    """

    def __init__(self, frame, func, args):
        self.frame = frame
        self.func = func

        if not all(is_scalar(arg) for arg in args):
            raise TypeError("only scalar valued args are supported by apply")
        self.args = args
        self.nrt = False

        self.frame_type = self._get_frame_type()
        self.device_func = cuda.jit(device=True)(self.func)

    @property
    @abstractmethod
    def kernel_type(self):
        """
        API type launching the kernel, used to break
        degenerecies in the cache.
        """

    @abstractmethod
    def _get_frame_type(self):
        """
        Numba type of the frame being passed to the kernel.
        """

    @abstractmethod
    def _get_kernel_string(self):
        """
        Generate executable string of python that defines a
        kernel we may retrieve from the context and compile
        with numba.
        """

    @abstractmethod
    def _get_kernel_string_exec_context(self):
        """
        Get a dict of globals needed to exec the kernel
        string.
        """

    @staticmethod
    def _format_arg_list(prefix, count):
        """Build a comma-separated parameter list like 'prefix_0, prefix_1, ...'."""
        return ", ".join(f"{prefix}_{i}" for i in range(count))

    def _construct_signature(self, return_type):
        """
        Build the signature of numba types that will be used to
        actually JIT the kernel itself later, accounting for types
        and offsets. Skips columns with unsupported dtypes.
        """
        if not return_type.is_internal:
            return_type = CPointer(return_type)
        else:
            return_type = return_type[::1]
        # Tuple of arrays, first the output data array, then the mask.
        # int8 (not boolean): released numba-cuda-mlir's array marshaller has
        # no bitwidth for its Boolean type; int8 has identical byte layout.
        return_type = Tuple((return_type, int8[::1]))
        supported_cols = _supported_cols_from_frame(self.frame)
        offsets = [int64] * len(supported_cols)
        sig = (
            [return_type, int64]
            + [
                _masked_array_type_from_col(col)
                for col in supported_cols.values()
            ]
            + offsets
            + [typeof(arg) for arg in self.args]
        )
        return void(*sig)

    @_performance_tracking
    def _get_udf_return_type(self):
        compile_sig = (self.frame_type, *(typeof(arg) for arg in self.args))
        cres = self.device_func.compile(compile_sig)
        output_type = cres.signature.return_type
        if isinstance(output_type, MaskedType):
            result = output_type.value_type
        else:
            if not isinstance(output_type, types.Type):
                result = numpy_support.from_dtype(np.dtype(output_type))
            else:
                result = output_type

        result = result if result.is_internal else result.return_as

        # _get_udf_return_type will throw a TypingError if the user tries to use
        # a field in the row containing an unsupported dtype, except in the
        # edge case where all the function does is return that element:

        # def f(row):
        #    return row[<bad dtype key>]
        # In this case numba is happy to return MaskedType(<bad dtype key>)
        # because it relies on not finding overloaded operators for types to raise
        # the exception, so we have to explicitly check for that case.
        if isinstance(result, Poison):
            raise TypingError(str(result))

        return result

    def compile_kernel(self):
        """
        Compile the kernel and return it as well as the
        return type.
        """

        # First compilation pass compiles the UDF alone
        # gets us the return type to allocate the output
        # and determines if NRT must be enabled
        capture_nrt_usage = CaptureNRTUsage()
        with capture_nrt_usage:
            return_type = self._get_udf_return_type()

        self.sig = self._construct_signature(return_type)
        kernel_string = self._get_kernel_string()
        kernel = self.compile_kernel_string(
            kernel_string, nrt=capture_nrt_usage.use_nrt
        )

        return kernel, return_type

    def _extra_link(self):
        """LinkableCode objects to link into the kernel.

        Defaults to none; the groupby kernel overrides this to link the
        cooperative reduction PTX shim.
        """
        return []

    def compile_kernel_string(self, kernel_string, nrt=False):
        global_exec_context = self._get_kernel_string_exec_context()
        global_exec_context["f_"] = self.device_func
        global_exec_context.update({"cuda": cuda, "_mask_get": _mask_get})
        exec(kernel_string, global_exec_context)
        _kernel = global_exec_context["_kernel"]
        return cuda.jit(
            self.sig,
            extensions=[mlir_string_arg_handler],
            link=self._extra_link(),
        )(_kernel)

    def get_kernel(self):
        return self._compile_or_get_kernel()

    def _compile_or_get_kernel(self):
        """
        Check the cache for a kernel corresponding to this
        function, frame, arguments, and udf type. Else, compile
        the kernel and store it in the cache.
        """

        cache_key = _generate_cache_key(
            self.frame, self.func, self.args, suffix=self.kernel_type
        )

        if kernel_cache.get(cache_key) is not None:
            kernel, masked_or_scalar = kernel_cache[cache_key]
            return kernel, masked_or_scalar

        kernel, scalar_return_type = self.compile_kernel()

        np_return_type = (
            numpy_support.as_dtype(scalar_return_type)
            if scalar_return_type.is_internal
            else scalar_return_type.np_dtype
        )

        kernel_cache[cache_key] = (kernel, np_return_type)
        return kernel, np_return_type
