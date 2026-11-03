# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/strings/replace.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] replace_slice(
        column_view source_strings,
        string_scalar repl,
        size_type start,
        size_type stop,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] replace(
        column_view source_strings,
        string_scalar target,
        string_scalar repl,
        int32_t maxrepl,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] replace_multiple(
        column_view source_strings,
        column_view target_strings,
        column_view repl_strings,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler
