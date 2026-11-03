# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libcpp cimport bool
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.io cimport types as cudf_io_types
from pylibcudf.variant cimport monostate, variant
from cuda.bindings.cyruntime cimport cudaStream_t


cdef extern from "cudf/io/orc_metadata.hpp" \
        namespace "cudf::io" nogil:

    ctypedef monostate no_statistics

    cdef cppclass minmax_statistics[T]:
        optional[T] minimum
        optional[T] maximum

    cdef cppclass sum_statistics[T]:
        optional[T] sum

    cdef cppclass integer_statistics(
        minmax_statistics[int64_t], sum_statistics[int64_t]
    ):
        pass

    cdef cppclass double_statistics(
        minmax_statistics[double], sum_statistics[double]
    ):
        pass

    cdef cppclass string_statistics(
        minmax_statistics[string], sum_statistics[int64_t]
    ):
        pass

    cdef cppclass bucket_statistics:
        vector[int64_t] count

    cdef cppclass decimal_statistics(
        minmax_statistics[string], sum_statistics[string]
    ):
        pass

    ctypedef minmax_statistics[int32_t] date_statistics

    ctypedef sum_statistics[int64_t] binary_statistics

    cdef cppclass timestamp_statistics(minmax_statistics[int64_t]):
        optional[int64_t] minimum_utc
        optional[int64_t] maximum_utc
        optional[uint32_t] minimum_nanos
        optional[uint32_t] maximum_nanos

    # This is a std::variant of all the statistics types
    ctypedef variant statistics_type

    cdef cppclass column_statistics:
        optional[uint64_t] number_of_values
        optional[bool] has_null
        statistics_type type_specific_stats

    cdef cppclass parsed_orc_statistics:
        vector[string] column_names
        vector[column_statistics] file_stats
        vector[vector[column_statistics]] stripes_stats

    cdef parsed_orc_statistics read_parsed_orc_statistics(
        const cudf_io_types.source_info& src_info,
        cudaStream_t stream
    ) except +libcudf_exception_handler
