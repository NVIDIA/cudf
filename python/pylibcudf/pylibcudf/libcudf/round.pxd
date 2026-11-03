# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/round.hpp" namespace "cudf" nogil:

    cpdef enum class rounding_method(int32_t):
        HALF_UP
        HALF_EVEN

    cdef unique_ptr[column] round (
        const column_view& input,
        int32_t decimal_places,
        rounding_method method,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] round_decimal (
        const column_view& input,
        int32_t decimal_places,
        rounding_method method,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
