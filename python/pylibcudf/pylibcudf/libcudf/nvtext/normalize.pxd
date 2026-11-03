# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "nvtext/normalize.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] normalize_spaces(
        const column_view & strings,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef struct character_normalizer:
        pass

    cdef unique_ptr[character_normalizer] create_character_normalizer(
        bool do_lower_case,
        const column_view & strings,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cpdef enum class normalize_flags:
        NONE
        STRIP_ACCENTS
        PAD_PUNCTUATION

    cdef unique_ptr[column] normalize_characters(
        const column_view & strings,
        const character_normalizer & normalizer,
        normalize_flags flags,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

ctypedef uint32_t underlying_type_t_normalize_flags
