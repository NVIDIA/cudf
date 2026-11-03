# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.char_types cimport string_character_types
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/strings/capitalize.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] capitalize(
        const column_view & strings,
        const string_scalar & delimiters,
        cudaStream_t stream,
        device_async_resource_ref mr
        ) except +libcudf_exception_handler

    cdef unique_ptr[column] title(
        const column_view & strings,
        string_character_types sequence_type,
        cudaStream_t stream,
        device_async_resource_ref mr
        ) except +libcudf_exception_handler

    cdef unique_ptr[column] is_title(
        const column_view & strings,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler
