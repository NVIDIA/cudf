# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.aggregation cimport (
    Kind as kind_t,
    aggregation,
    correlation_type,
    ewm_history,
    groupby_aggregation,
    groupby_scan_aggregation,
    rank_method,
    rank_percentage,
    reduce_aggregation,
    rolling_aggregation,
    scan_aggregation,
)
from pylibcudf.libcudf.types cimport (
    interpolation,
    nan_equality,
    null_equality,
    null_order,
    null_policy,
    order,
    size_type,
)

from .types cimport DataType

# workaround for https://github.com/cython/cython/issues/3885
ctypedef groupby_aggregation * gba_ptr
ctypedef groupby_scan_aggregation * gbsa_ptr
ctypedef reduce_aggregation * ra_ptr
ctypedef scan_aggregation * sa_ptr
ctypedef rolling_aggregation * roa_ptr


cdef class Aggregation:
    cdef unique_ptr[aggregation] c_obj
    cpdef kind_t kind(self)
    cdef void _unsupported_agg_error(self, str alg)
    cdef unique_ptr[groupby_aggregation] clone_underlying_as_groupby(self) except *
    cdef unique_ptr[groupby_scan_aggregation] clone_underlying_as_groupby_scan(
        self
    ) except *
    cdef unique_ptr[rolling_aggregation] clone_underlying_as_rolling(self) except *
    cdef const reduce_aggregation* view_underlying_as_reduce(self) except *
    cdef const scan_aggregation* view_underlying_as_scan(self) except *
    cdef const rolling_aggregation* view_underlying_as_rolling(self) except *

    @staticmethod
    cdef Aggregation from_libcudf(unique_ptr[aggregation] agg)


cpdef Aggregation sum()

cpdef Aggregation product()

cpdef Aggregation min()

cpdef Aggregation max()

cpdef Aggregation count(null_policy null_handling = *)

cpdef Aggregation any()

cpdef Aggregation all()

cpdef Aggregation sum_of_squares()

cpdef Aggregation mean()

cpdef Aggregation variance(size_type ddof = *)

cpdef Aggregation std(size_type ddof = *)

cpdef Aggregation median()

cpdef Aggregation quantile(list quantiles, interpolation interp = *)

cpdef Aggregation argmax()

cpdef Aggregation argmin()

cpdef Aggregation ewma(float center_of_mass, ewm_history history)

cpdef Aggregation nunique(null_policy null_handling = *)

cpdef Aggregation nth_element(size_type n, null_policy null_handling = *)

cpdef Aggregation collect_list(null_policy null_handling = *)

cpdef Aggregation collect_set(null_handling = *, nulls_equal = *, nans_equal = *)

cpdef Aggregation udf(str operation, DataType output_type)

cpdef Aggregation correlation(correlation_type type, size_type min_periods)

cpdef Aggregation covariance(size_type min_periods, size_type ddof)

cpdef Aggregation rank(
    rank_method method,
    order column_order = *,
    null_policy null_handling = *,
    null_order null_precedence = *,
    rank_percentage percentage = *,
)

cpdef bool is_valid_aggregation(DataType source, Aggregation agg)
