# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport (
    interpolation,
    null_order,
    order,
    order_info,
    sorted,
)
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/quantiles.hpp" namespace "cudf" nogil:

    cdef unique_ptr[column] quantile (
        column_view input,
        vector[double] q,
        interpolation interp,
        column_view ordered_indices,
        bool exact,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] quantiles (
        table_view source_table,
        vector[double] q,
        interpolation interp,
        sorted is_input_sorted,
        vector[order] column_order,
        vector[null_order] null_precedence,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
