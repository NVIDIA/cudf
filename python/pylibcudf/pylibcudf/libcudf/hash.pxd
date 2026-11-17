# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/hashing.hpp" namespace "cudf::hashing" nogil:
    cdef unique_ptr[column] murmurhash3_x86_32(
        const table_view& input,
        const uint32_t seed,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[table] murmurhash3_x64_128(
        const table_view& input,
        const uint64_t seed,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] md5(
        const table_view& input,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] sha1(
        const table_view& input,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] sha224(
        const table_view& input,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] sha256(
        const table_view& input,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] sha384(
        const table_view& input,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] sha512(
        const table_view& input,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] xxhash_32(
        const table_view& input,
        const uint32_t seed,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] xxhash_64(
        const table_view& input,
        const uint64_t seed,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

cdef extern from "cudf/hashing.hpp" namespace "cudf" nogil:
    cdef uint32_t DEFAULT_HASH_SEED
