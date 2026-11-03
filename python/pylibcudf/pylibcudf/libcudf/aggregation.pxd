# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stddef cimport size_t
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport (
    data_type,
    interpolation,
    nan_equality,
    null_equality,
    null_order,
    null_policy,
    order,
    size_type,
)


cdef extern from "cudf/aggregation.hpp" namespace "cudf" nogil:

    # Cython doesn't appear to support enum class nested inside a class, so
    # have to namespace it manually
    cpdef enum class Kind "cudf::aggregation::Kind":
        SUM
        PRODUCT
        MIN
        MAX
        COUNT_VALID
        COUNT_ALL
        ANY
        ALL
        SUM_OF_SQUARES
        MEAN
        M2
        VARIANCE
        STD
        MEDIAN
        QUANTILE
        ARGMAX
        ARGMIN
        NUNIQUE
        NTH_ELEMENT
        ROW_NUMBER
        EWMA
        RANK
        COLLECT_LIST
        COLLECT_SET
        LEAD
        LAG
        HOST_UDF
        MERGE_LISTS
        MERGE_SETS
        MERGE_M2
        COVARIANCE
        CORRELATION
        TDIGEST
        MERGE_TDIGEST
        HISTOGRAM
        MERGE_HISTOGRAM
        BITWISE_AGG

    cdef cppclass aggregation:
        Kind kind
        unique_ptr[aggregation] clone()
        size_t do_hash() noexcept
        bool is_equal(const aggregation const) noexcept

    cdef cppclass rolling_aggregation(aggregation):
        pass

    cdef cppclass groupby_aggregation(aggregation):
        pass

    cdef cppclass groupby_scan_aggregation(aggregation):
        pass

    cdef cppclass reduce_aggregation(aggregation):
        pass

    cdef cppclass scan_aggregation(aggregation):
        pass

    cpdef enum class correlation_type(int32_t):
        PEARSON
        KENDALL
        SPEARMAN

    cpdef enum class ewm_history(int32_t):
        INFINITE
        FINITE

    cpdef enum class rank_method(int32_t):
        FIRST
        AVERAGE
        MIN
        MAX
        DENSE

    cpdef enum class rank_percentage(int32_t):
        NONE
        ZERO_NORMALIZED
        ONE_NORMALIZED

    cpdef enum class bitwise_op(int32_t):
        AND
        OR
        XOR

    cdef unique_ptr[T] make_sum_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_product_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_min_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_max_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_count_aggregation[T](
        null_policy null_handling
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_any_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_all_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_sum_of_squares_aggregation[T]()\
        except +libcudf_exception_handler

    cdef unique_ptr[T] make_mean_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_variance_aggregation[T](
        size_type ddof) except +libcudf_exception_handler

    cdef unique_ptr[T] make_std_aggregation[T](
        size_type ddof
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_median_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_quantile_aggregation[T](
        vector[double] q, interpolation i) except +libcudf_exception_handler

    cdef unique_ptr[T] make_argmax_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_argmin_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_nunique_aggregation[T](
        null_policy null_handling
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_nth_element_aggregation[T](
        size_type n,
        null_policy null_handling
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_collect_list_aggregation[T](
        null_policy null_handling
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_collect_set_aggregation[T](
        null_policy null_handling, null_equality nulls_equal, nan_equality nans_equal
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_ewma_aggregation[T](
        double com, ewm_history adjust
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_correlation_aggregation[T](
        correlation_type type, size_type min_periods) except +libcudf_exception_handler

    cdef unique_ptr[T] make_covariance_aggregation[T](
        size_type min_periods, size_type ddof) except +libcudf_exception_handler

    cdef unique_ptr[T] make_rank_aggregation[T](
        rank_method method,
        order column_order,
        null_policy null_handling,
        null_order null_precedence,
        rank_percentage percentage) except +libcudf_exception_handler

    cdef unique_ptr[T] make_tdigest_aggregation[T](
        int max_centroids
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_merge_tdigest_aggregation[T](
        int max_centroids
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_histogram_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_merge_histogram_aggregation[T](
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_merge_lists_aggregation[T](
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_merge_sets_aggregation[T](
        null_equality nulls_equal,
        nan_equality nans_equal,
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_merge_m2_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_m2_aggregation[T]() except +libcudf_exception_handler

    cdef unique_ptr[T] make_bitwise_aggregation[T](
        bitwise_op op
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_lag_aggregation[T](
        size_type offset
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_lead_aggregation[T](
        size_type offset
    ) except +libcudf_exception_handler

    cdef unique_ptr[T] make_row_number_aggregation[T](
    ) except +libcudf_exception_handler

    bool is_valid_aggregation(
        data_type source, Kind kind
    ) noexcept
