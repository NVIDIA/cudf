# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.functional cimport reference_wrapper
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.utility cimport pair
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.aggregation cimport reduce_aggregation, scan_aggregation, Kind
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport data_type, nan_policy, null_policy
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport (
    any_resource,
    device_accessible,
    device_async_resource_ref,
)

ctypedef const scalar constscalar

cdef extern from "cudf/reduction.hpp" namespace "cudf" nogil:
    cdef unique_ptr[scalar] reduce(
        const column_view& col,
        const reduce_aggregation& agg,
        data_type output_type,
        optional[reference_wrapper[constscalar]] init,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cpdef enum class scan_type(bool):
        INCLUSIVE
        EXCLUSIVE

    cdef unique_ptr[column] scan(
        const column_view& col,
        const scan_aggregation& agg,
        scan_type inclusive,
        null_policy null_handling,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[scalar], unique_ptr[scalar]] minmax(
        const column_view& col,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler


cdef extern from "cudf/reduction.hpp" namespace "cudf::reduction" nogil:
    bool is_valid_aggregation(
        data_type source, Kind kind
    ) noexcept


cdef extern from "cudf/reduction/approx_distinct_count.hpp" namespace "cudf" nogil:
    cdef cppclass approx_distinct_count:
        approx_distinct_count(
            const table_view& input,
            int32_t precision,
            null_policy null_handling,
            nan_policy nan_handling,
            cudaStream_t stream,
            any_resource[device_accessible] mr,
        ) except +libcudf_exception_handler
        void add(
            const table_view& input, cudaStream_t stream,
        ) except +libcudf_exception_handler
        void merge(
            const approx_distinct_count& other, cudaStream_t stream
        ) except +libcudf_exception_handler
        size_t estimate(cudaStream_t stream) except +libcudf_exception_handler
        null_policy null_handling() noexcept
        nan_policy nan_handling() noexcept
        int32_t precision() noexcept
        double standard_error() noexcept

        @staticmethod
        size_t sketch_bytes(int32_t precision) except +libcudf_exception_handler

        @staticmethod
        size_t sketch_alignment() except +libcudf_exception_handler
