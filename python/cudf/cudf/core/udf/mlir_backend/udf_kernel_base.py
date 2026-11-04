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
    LIBCUDF_BITMASK_TYPE,
    _column_data_type,
    _generate_cache_key,
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
    def _make_kernel(self):
        """Build and return the kernel function to JIT.

        Returns a plain Python function (a closure capturing the device UDF and
        the column names). It is metaprogrammed with ``consteval`` loop unrolling
        and star-splat (no string templates / ``exec``), and compiled with
        ``experimental_ast_transforms=True``.
        """

    def _construct_signature(self, return_type):
        """
        Build the numba signature of the kernel: ``void(retval, size, columns,
        extra)``.

        ``retval`` is ``(out_data, out_mask)``; ``columns`` is a tuple with one
        ``(data, mask, offset)`` entry per supported column (``mask`` is an empty
        bitmask array for columns without a validity mask); ``extra`` is a tuple
        of the UDF's additional scalar arguments.
        """
        if not return_type.is_internal:
            return_type = CPointer(return_type)
        else:
            return_type = return_type[::1]
        # Tuple of arrays, first the output data array, then the mask.
        # int8 (not boolean): released numba-cuda-mlir's array marshaller has
        # no bitwidth for its Boolean type; int8 has identical byte layout.
        return_type = Tuple((return_type, int8[::1]))
        columns_ty = Tuple(
            tuple(
                Tuple((_column_data_type(col), LIBCUDF_BITMASK_TYPE[::1], int64))
                for col in _supported_cols_from_frame(self.frame).values()
            )
        )
        extra_ty = Tuple(tuple(typeof(arg) for arg in self.args))
        return void(return_type, int64, columns_ty, extra_ty)

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
        kernel = cuda.jit(
            self.sig,
            extensions=[mlir_string_arg_handler],
            link=self._extra_link(),
            experimental_ast_transforms=True,
        )(self._make_kernel())

        return kernel, return_type

    def _extra_link(self):
        """LinkableCode objects to link into the kernel.

        Defaults to none; the groupby kernel overrides this to link the
        cooperative reduction PTX shim.
        """
        return []

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
