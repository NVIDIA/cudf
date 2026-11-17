# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from pylibcudf.libcudf.scalar.scalar cimport scalar
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/replace.hpp" namespace "cudf" nogil:

    cpdef enum class replace_policy(bool):
        PRECEDING
        FOLLOWING

    cdef unique_ptr[column] replace_nulls(
        column_view source_column,
        column_view replacement_column,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] replace_nulls(
        column_view source_column,
        scalar replacement,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] replace_nulls(
        column_view source_column,
        replace_policy replace_policy,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] find_and_replace_all(
        column_view source_column,
        column_view values_to_replace,
        column_view replacement_values,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] clamp(
        column_view source_column,
        scalar lo, scalar lo_replace,
        scalar hi, scalar hi_replace,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] clamp(
        column_view source_column,
        scalar lo, scalar hi,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] normalize_nans_and_zeros(
        column_view source_column,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef void normalize_nans_and_zeros(
        mutable_column_view source_column,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler
