# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type, data_type
from pylibcudf.libcudf.utilities.span cimport device_span

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref

cdef extern from "cuda/functional" namespace "cuda::std":
    cdef cppclass byte:
        pass


cdef extern from "cudf/reshape.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] interleave_columns(
        table_view source_table,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[table] tile(
        table_view source_table,
        size_type count,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef void table_to_array(
        table_view input_table,
        device_span[byte] output,
        cudaStream_t stream
    ) except +libcudf_exception_handler
