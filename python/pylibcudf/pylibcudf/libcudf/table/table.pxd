# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table_view cimport mutable_table_view, table_view
from pylibcudf.libcudf.types cimport size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/table/table.hpp" namespace "cudf" nogil:
    cdef cppclass table:
        table(
            const table&,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler
        table(
            table_view,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler
        size_type num_columns() except +libcudf_exception_handler
        size_type num_rows() except +libcudf_exception_handler
        table_view view() except +libcudf_exception_handler
        mutable_table_view mutable_view() except +libcudf_exception_handler
        vector[unique_ptr[column]] release() except +libcudf_exception_handler
