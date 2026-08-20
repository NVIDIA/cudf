# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport (
    nan_equality,
    null_equality,
    size_type,
)
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/stream_compaction.hpp" namespace "cudf" nogil:
    cpdef enum class duplicate_keep_option:
        KEEP_ANY
        KEEP_FIRST
        KEEP_LAST
        KEEP_NONE

    cdef unique_ptr[table] drop_nulls(
        table_view source_table,
        vector[size_type] keys,
        size_type keep_threshold,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] drop_nans(
        table_view source_table,
        vector[size_type] keys,
        size_type keep_threshold,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] apply_retention_mask(
        table_view source_table,
        column_view retention_mask,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] apply_boolean_mask(
        table_view source_table,
        column_view boolean_mask,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] apply_deletion_mask(
        table_view source_table,
        column_view deletion_mask,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] unique(
        table_view input,
        vector[size_type] keys,
        duplicate_keep_option keep,
        null_equality nulls_equal,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] distinct(
        table_view input,
        vector[size_type] keys,
        duplicate_keep_option keep,
        null_equality nulls_equal,
        nan_equality nans_equals,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] distinct_indices(
        table_view input,
        duplicate_keep_option keep,
        null_equality nulls_equal,
        nan_equality nans_equal,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] stable_distinct(
        table_view input,
        vector[size_type] keys,
        duplicate_keep_option keep,
        null_equality nulls_equal,
        nan_equality nans_equal,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] filter(
        table_view predicate_table,
        const expression& predicate_expr,
        table_view filter_table,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
