# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport numeric_scalar
from pylibcudf.libcudf.types cimport size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "nvtext/minhash.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] minhash(
        const column_view &strings,
        const uint32_t seed,
        const column_view &a,
        const column_view &b,
        const size_type width,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] minhash64(
        const column_view &strings,
        const uint64_t seed,
        const column_view &a,
        const column_view &b,
        const size_type width,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] minhash_ngrams(
        const column_view &strings,
        const size_type ngrams,
        const uint32_t seed,
        const column_view &a,
        const column_view &b,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] minhash64_ngrams(
        const column_view &strings,
        const size_type ngrams,
        const uint64_t seed,
        const column_view &a,
        const column_view &b,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
