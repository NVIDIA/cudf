# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/strings/case.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] to_lower(
        const column_view & strings,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] to_upper(
        const column_view & strings,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] swapcase(
        const column_view & strings,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler
