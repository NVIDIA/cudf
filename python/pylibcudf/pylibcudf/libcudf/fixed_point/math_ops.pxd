# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.round cimport rounding_method
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport data_type

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/fixed_point/math_ops.hpp" namespace "cudf" nogil:
    cpdef enum class decimal_intermediate(int32_t):
        STANDARD
        WIDE

    cdef unique_ptr[column] divide_decimal(
        const column_view& lhs,
        const column_view& rhs,
        data_type output_type,
        rounding_method method,
        decimal_intermediate intermediate,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] divide_decimal(
        const column_view& lhs,
        const scalar& rhs,
        data_type output_type,
        rounding_method method,
        decimal_intermediate intermediate,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] divide_decimal(
        const scalar& lhs,
        const column_view& rhs,
        data_type output_type,
        rounding_method method,
        decimal_intermediate intermediate,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
