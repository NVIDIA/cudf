# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t, int64_t, uint8_t
from libcpp cimport bool
from libcpp.functional cimport reference_wrapper
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type

from rmm.librmm.device_buffer cimport device_buffer
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref

ctypedef const scalar constscalar

cdef extern from "cudf/copying.hpp" namespace "cudf" nogil:
    cpdef enum class out_of_bounds_policy(bool):
        NULLIFY
        DONT_CHECK

    cdef unique_ptr[table] gather (
        const table_view& source_table,
        const column_view& gather_map,
        out_of_bounds_policy policy,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] shift(
        const column_view& input,
        size_type offset,
        const scalar& fill_values,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] reverse(
        const table_view& source_table,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] reverse(
        const column_view& source_column,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] scatter (
        const table_view& source_table,
        const column_view& scatter_map,
        const table_view& target_table,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] scatter (
        const vector[reference_wrapper[constscalar]]& source_scalars,
        const column_view& indices,
        const table_view& target,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cpdef enum class mask_allocation_policy(int32_t):
        NEVER
        RETAIN
        ALWAYS

    cdef unique_ptr[column] empty_like (
        const column_view& input_column
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] allocate_like (
        const column_view& input_column,
        mask_allocation_policy policy,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] allocate_like (
        const column_view& input_column,
        size_type size,
        mask_allocation_policy policy,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] empty_like (
        const table_view& input_table
    ) except +libcudf_exception_handler

    cdef void copy_range_in_place (
        const column_view& input_column,
        mutable_column_view& target_column,
        size_type input_begin,
        size_type input_end,
        size_type target_begin,
        cudaStream_t stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] copy_range (
        const column_view& input_column,
        const column_view& target_column,
        size_type input_begin,
        size_type input_end,
        size_type target_begin,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef vector[column_view] slice (
        const column_view& input_column,
        vector[size_type] indices,
        cudaStream_t stream
    ) except +libcudf_exception_handler

    cdef vector[table_view] slice (
        const table_view& input_table,
        vector[size_type] indices,
        cudaStream_t stream
    ) except +libcudf_exception_handler

    cdef vector[column_view] split (
        const column_view& input_column,
        vector[size_type] splits,
        cudaStream_t stream
    ) except +libcudf_exception_handler

    cdef vector[table_view] split (
        const table_view& input_table,
        vector[size_type] splits,
        cudaStream_t stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] copy_if_else (
        const column_view& lhs,
        const column_view& rhs,
        const column_view& boolean_mask,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] copy_if_else (
        const scalar& lhs,
        const column_view& rhs,
        const column_view& boolean_mask,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] copy_if_else (
        const column_view& lhs,
        const scalar& rhs,
        const column_view boolean_mask,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] copy_if_else (
        const scalar& lhs,
        const scalar& rhs,
        const column_view boolean_mask,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] boolean_mask_scatter (
        const table_view& input,
        const table_view& target,
        const column_view& boolean_mask,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] boolean_mask_scatter (
        const vector[reference_wrapper[constscalar]]& input,
        const table_view& target,
        const column_view& boolean_mask,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[scalar] get_element (
        const column_view& input,
        size_type index,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cpdef enum class sample_with_replacement(bool):
        FALSE
        TRUE
