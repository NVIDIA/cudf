# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
cimport pylibcudf.libcudf.types as libcudf_types
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/search.hpp" namespace "cudf" nogil:

    cdef unique_ptr[column] lower_bound(
        table_view haystack,
        table_view needles,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] upper_bound(
        table_view haystack,
        table_view needles,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] contains(
        column_view haystack,
        column_view needles,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
