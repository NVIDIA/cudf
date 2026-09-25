# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport data_type

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/binaryop.hpp" namespace "cudf" nogil:
    cpdef enum class binary_operator(int32_t):
        ADD
        SUB
        MUL
        DIV
        TRUE_DIV
        FLOOR_DIV
        MOD
        PMOD
        PYMOD
        POW
        INT_POW
        LOG_BASE
        ATAN2
        SHIFT_LEFT
        SHIFT_RIGHT
        SHIFT_RIGHT_UNSIGNED
        BITWISE_AND
        BITWISE_OR
        BITWISE_XOR
        LOGICAL_AND
        LOGICAL_OR
        EQUAL
        NOT_EQUAL
        LESS
        GREATER
        LESS_EQUAL
        GREATER_EQUAL
        NULL_EQUALS
        NULL_NOT_EQUALS
        NULL_MAX
        NULL_MIN
        GENERIC_BINARY
        NULL_LOGICAL_AND
        NULL_LOGICAL_OR
        ADD_OVERFLOW
        SUB_OVERFLOW
        MUL_OVERFLOW
        DIV_OVERFLOW
        MOD_OVERFLOW
        INVALID_BINARY

    cdef unique_ptr[column] binary_operation (
        const scalar& lhs,
        const column_view& rhs,
        binary_operator op,
        data_type output_type,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] binary_operation (
        const column_view& lhs,
        const scalar& rhs,
        binary_operator op,
        data_type output_type,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] binary_operation (
        const column_view& lhs,
        const column_view& rhs,
        binary_operator op,
        data_type output_type,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] binary_operation (
        const column_view& lhs,
        const column_view& rhs,
        const string& op,
        data_type output_type,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

cdef extern from "cudf/binaryop.hpp" namespace "cudf::binops" nogil:
    cdef bool is_supported_operation(
        data_type output_type,
        data_type lhs_type,
        data_type rhs_type,
        binary_operator op
    ) except +libcudf_exception_handler
