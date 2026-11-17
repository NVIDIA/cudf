# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.aggregation cimport Kind, rolling_aggregation
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport data_type, null_order, order, size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/rolling.hpp" namespace "cudf" nogil:
    cdef cppclass rolling_request:
        column_view values
        size_type min_periods
        unique_ptr[rolling_aggregation] aggregation
    # This inheritance is a lie, a range_window_type is actually a
    # std::variant of the concrete window types. However, we can't
    # construct an instance of a variant in Cython so lie here.
    # This is only used in type-checking the cython, so that's fine.
    cdef cppclass range_window_type:
        pass
    cdef cppclass bounded_closed(range_window_type):
        bounded_closed(const scalar&) noexcept
    cdef cppclass bounded_open(range_window_type):
        bounded_open(const scalar&) noexcept
    cdef cppclass unbounded(range_window_type):
        unbounded() noexcept
    cdef cppclass current_row(range_window_type):
        current_row() noexcept

    cdef unique_ptr[table] grouped_range_rolling_window(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        range_window_type preceding,
        range_window_type following,
        vector[rolling_request]& requests,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] rolling_window(
        column_view source,
        column_view preceding_window,
        column_view following_window,
        size_type min_periods,
        rolling_aggregation& agg,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef unique_ptr[column] rolling_window(
        column_view source,
        size_type preceding_window,
        size_type following_window,
        size_type min_periods,
        rolling_aggregation& agg,
        cudaStream_t stream,
        device_async_resource_ref mr) except +libcudf_exception_handler

    cdef pair[unique_ptr[column], unique_ptr[column]] make_range_windows(
        const table_view& group_keys,
        const column_view& orderby,
        order order,
        null_order null_order,
        range_window_type preceding,
        range_window_type following,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    bool is_valid_rolling_aggregation(
        data_type source, Kind kind
    ) noexcept
