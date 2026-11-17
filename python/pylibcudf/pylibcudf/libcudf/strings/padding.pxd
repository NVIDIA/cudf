# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.side_type cimport side_type
from pylibcudf.libcudf.types cimport size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/strings/padding.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] pad(
        column_view input,
        size_type width,
        side_type side,
        string fill_char,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] zfill(
        column_view input,
        size_type width,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] zfill_by_widths(
        column_view input,
        column_view widths,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler
