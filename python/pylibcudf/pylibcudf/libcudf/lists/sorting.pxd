# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.types cimport null_order, order
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/lists/sorting.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] sort_lists(
        const lists_column_view source_column,
        order column_order,
        null_order null_precedence,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] stable_sort_lists(
        const lists_column_view source_column,
        order column_order,
        null_order null_precedence,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
