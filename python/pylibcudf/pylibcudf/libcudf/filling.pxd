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
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/filling.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] fill(
        const column_view & input,
        size_type begin,
        size_type end,
        const scalar & value,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef void fill_in_place(
        const mutable_column_view & destination,
        size_type begin,
        size_type end,
        const scalar & value,
        cudaStream_t stream
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] repeat(
        const table_view & input,
        const column_view & count,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] repeat(
        const table_view & input,
        size_type count,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] sequence(
        size_type size,
        const scalar & init,
        const scalar & step,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] calendrical_month_sequence(
        size_type n,
        const scalar& init,
        size_type months,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
