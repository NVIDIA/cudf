# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cuda.bindings.cyruntime cimport cudaStream_t
from pylibcudf.libcudf.utilities.memory_resource cimport device_resource_ref


cdef extern from "cudf/strings/char_types/char_types.hpp" \
        namespace "cudf::strings" nogil:

    cpdef enum class string_character_types(uint32_t):
        DECIMAL
        NUMERIC
        DIGIT
        ALPHA
        SPACE
        UPPER
        LOWER
        ALPHANUM
        CASE_TYPES
        ALL_TYPES

    cdef unique_ptr[column] all_characters_of_type(
        column_view source_strings,
        string_character_types types,
        string_character_types verify_types,
        cudaStream_t stream,
        device_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] filter_characters_of_type(
        column_view source_strings,
        string_character_types types_to_remove,
        string_scalar replacement,
        string_character_types types_to_keep,
        cudaStream_t stream,
        device_resource_ref mr) except +libcudf_exception_handler
