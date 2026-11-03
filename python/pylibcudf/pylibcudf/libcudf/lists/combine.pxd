# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/lists/combine.hpp" namespace \
        "cudf::lists" nogil:

    cpdef enum class concatenate_null_policy(int32_t):
        IGNORE
        NULLIFY_OUTPUT_ROW

    cdef unique_ptr[column] concatenate_rows(
        const table_view input_table,
        concatenate_null_policy null_policy,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] concatenate_list_elements(
        const table_view input_table,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] concatenate_list_elements(
        const column_view input_table,
        concatenate_null_policy null_policy,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
