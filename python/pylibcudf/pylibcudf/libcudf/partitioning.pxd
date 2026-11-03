# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
cimport pylibcudf.libcudf.types as libcudf_types
from libc.stdint cimport int32_t, uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.hash cimport DEFAULT_HASH_SEED
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref

cdef extern from "cudf/partitioning.hpp" namespace "cudf" nogil:
    cpdef enum class hash_id(int32_t):
        HASH_IDENTITY "cudf::hash_id::HASH_IDENTITY"
        HASH_MURMUR3 "cudf::hash_id::HASH_MURMUR3"


cdef extern from "cudf/partitioning.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] \
        hash_partition "cudf::hash_partition" (
        const table_view& input,
        const vector[libcudf_types.size_type]& columns_to_hash,
        int num_partitions,
        hash_id hash_function,
        uint32_t seed,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] \
        hash_partition "cudf::hash_partition" (
        const table_view& input,
        const table_view& keys,
        int num_partitions,
        hash_id hash_function,
        uint32_t seed,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] \
        partition "cudf::partition" (
        const table_view& t,
        const column_view& partition_map,
        int num_partitions,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[table], vector[libcudf_types.size_type]] \
        round_robin_partition "cudf::round_robin_partition" (
        const table_view& input,
        int num_partitions,
        int start_partition,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
