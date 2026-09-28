# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "nvtext/unicode_normalize.hpp" namespace "nvtext" nogil:

    cpdef enum class unicode_normalization_form(int32_t):
        NFD
        NFC
        NFKD
        NFKC

    cdef struct unicode_normalizer:
        pass

    cdef unique_ptr[unicode_normalizer] create_unicode_normalizer(
        const table_view &unicode_data,
        unicode_normalization_form form,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] normalize_unicode(
        const column_view &input,
        const unicode_normalizer &normalizer,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
