# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.types cimport size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/lists/extract.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] extract_list_element(
        const lists_column_view&,
        size_type,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[column] extract_list_element(
        const lists_column_view&,
        const column_view&,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
