# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.table.table cimport table, table_view
from pylibcudf.libcudf.utilities.span cimport host_span

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/concatenate.hpp" namespace "cudf" nogil:
    # The versions of concatenate taking vectors don't exist in libcudf
    # C++, but passing a vector works because a host_span is implicitly
    # constructable from a vector. In case they are needed in the future,
    # host_span versions can be added, e.g:
    #
    # cdef unique_ptr[column] concatenate(
    #    host_span[column_view] columns
    # ) except +libcudf_exception_handler

    cdef unique_ptr[column] concatenate(
        const vector[column_view] columns,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef unique_ptr[table] concatenate(
        const vector[table_view] tables,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
