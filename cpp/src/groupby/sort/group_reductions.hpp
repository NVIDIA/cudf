/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

/** @internal @file Internal API in this file are mostly segmented reduction operations on column,
 * which are used in sort-based groupby aggregations.
 *
 */
namespace cudf {
namespace groupby {
namespace detail {
/**
 * @brief Internal API to calculate groupwise sum
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups   = 4
 *
 * group_sum    = [7, -3, 4, <NA>]
 * @endcode
 *
 * @param values Grouped values to get sum of
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_sum(column_view const& values,
                                  size_type num_groups,
                                  cudf::device_span<size_type const> group_labels,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate groupwise sum with overflow detection.
 *
 * Returns a STRUCT column with two children: the (wrapping) sum and a BOOL8 overflow flag that is
 * true when the true sum does not fit in the value type. On overflow the sum value is unspecified;
 * the flag is the meaningful output. A group is null only when all of its values are null.
 *
 * @param values Grouped values to sum
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<column> group_sum_overflow(
  column_view const& values,
  size_type num_groups,
  cudf::device_span<size_type const> group_labels,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate groupwise product
 *
 * @code{.pseudo}
 * values        = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels  = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups    = 4
 *
 * group_product = [6, 2, 4, <NA>]
 * @endcode
 *
 * @param values Grouped values to get product of
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_product(column_view const& values,
                                      size_type num_groups,
                                      cudf::device_span<size_type const> group_labels,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate groupwise minimum value
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups   = 4
 *
 * group_min    = [1, -2, 4, <NA>]
 * @endcode
 *
 * @param values Grouped values to get minimum from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_min(column_view const& values,
                                  size_type num_groups,
                                  cudf::device_span<size_type const> group_labels,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate groupwise maximum value
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups   = 4
 *
 * group_max    = [4, -1, 4, <NA>]
 * @endcode
 *
 * @param values Grouped values to get maximum from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_max(column_view const& values,
                                  size_type num_groups,
                                  cudf::device_span<size_type const> group_labels,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate group-wise indices of maximum values.
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups   = 4
 *
 * group_max    = [2, 0, 0, <NA>]
 * @endcode
 *
 * @param values Grouped values to get maximum value's index from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param key_sort_order Indices indicating sort order of groupby keys
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_argmax(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& key_sort_order,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate group-wise indices of minimum values.
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups   = 4
 *
 * group_max    = [1, 1, 0, <NA>]
 * @endcode
 *
 * @param values Grouped values to get minimum value's index from
 * @param num_groups Number of groups
 * @param group_labels ID of group that the corresponding value belongs to
 * @param key_sort_order Indices indicating sort order of groupby keys
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_argmin(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& key_sort_order,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate number of non-null values in each group of
 *  @p values
 *
 * @code{.pseudo}
 * values            = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels      = [0, 0, 0,  1,  1,    2, 2,    3]
 * num_groups        = 4
 *
 * group_count_valid = [3, 2, 1, 0]
 * @endcode
 *
 * @param values Grouped values to get valid count of
 * @param group_labels ID of group that the corresponding value belongs to
 * @param num_groups Number of groups ( unique values in @p group_labels )
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_count_valid(column_view const& values,
                                          cudf::device_span<size_type const> group_labels,
                                          size_type num_groups,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate number of values in each group of @p values
 *
 * @code{.pseudo}
 * group_offsets = [0, 3, 5, 7, 8]
 * num_groups    = 4
 *
 * group_count_all = [3, 2, 2, 1]
 * @endcode
 *
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param num_groups Number of groups ( unique values in @p group_labels )
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_count_all(cudf::device_span<size_type const> group_offsets,
                                        size_type num_groups,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);
/**
 * @brief Internal API to compute histogram for each group in @p values.
 *
 * The returned column is a lists column, each list corresponds to one input group and stores the
 * histogram of the distinct elements in that group in the form of `STRUCT<value, count>`.
 *
 * Note that the order of distinct elements in each output list is not specified.
 *
 * @code{.pseudo}
 * values       = [2, 1, 1, 3, 5, 2, 2, 3, 1, 4]
 * group_labels = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
 * num_groups   = 3
 *
 * output = [[<1, 2>, <2, 1>], [<2, 2>, <3, 2>, <5, 1>], [<1, 1>, <4, 1>]]
 * @endcode
 *
 * @param values Grouped values to compute histogram
 * @param group_labels ID of group that the corresponding value belongs to
 * @param num_groups Number of groups
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_histogram(column_view const& values,
                                        cudf::device_span<size_type const> group_labels,
                                        size_type num_groups,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate sum of squares of differences from means.
 *
 * If there are only nulls in the group, the output value of that group will be null.
 *
 * @code{.pseudo}
 * values        = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels  = [0, 0, 0,  1,  1,    2, 2,    3]
 * group_means   = [2.333333, -1.5, 4.0, <NA>]
 * group_m2(...) = [4.666666,  1.0, 0.0, <NA>]
 * @endcode
 *
 * @param values Grouped values to compute M2 values
 * @param group_means Pre-computed groupwise MEAN
 * @param group_labels ID of group corresponding value in @p values belongs to
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_m2(column_view const& values,
                                 column_view const& group_means,
                                 cudf::device_span<size_type const> group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate groupwise variance
 *
 * @code{.pseudo}
 * values       = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * group_means  = [2.333333, -1.5, 4.0, <NA>]
 * group_sizes  = [3, 2, 2, 1]
 * ddof         = 1
 *
 * group_var    = [2.333333, 0.5, <NA>, <NA>]
 * @endcode
 *
 * @param values Grouped values to get variance of
 * @param group_means Pre-calculated groupwise MEAN
 * @param group_sizes Number of valid elements per group
 * @param group_labels ID of group corresponding value in @p values belongs to
 * @param ddof Delta degrees of freedom. The divisor used in calculation of
 *             `var` is `N - ddof`, where `N` is the group size.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_var(column_view const& values,
                                  column_view const& group_means,
                                  column_view const& group_sizes,
                                  cudf::device_span<size_type const> group_labels,
                                  size_type ddof,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate groupwise quantiles
 *
 * @code{.pseudo}
 * values       = [1, 2, 4, -2, -1, <NA>, 4, <NA>]
 * group_labels = [0, 0, 0,  1,  1,    2, 2,    3]
 * group_sizes  = [3, 2, 2, 1]
 * num_groups   = 4
 * quantiles    = [0.25, 0.5]
 *
 * group_quantiles = [1.5, 2, -1.75, -1.5,  4,  4, <NA>, <NA>]
 * @endcode
 *
 * @param values Grouped and sorted (within group) values to get quantiles from
 * @param group_sizes Number of valid elements per group
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param num_groups Number of groups
 * @param quantiles List of quantiles q where q lies in [0,1]
 * @param interp Method to use when desired value lies between data points
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_quantiles(column_view const& values,
                                        column_view const& group_sizes,
                                        cudf::device_span<size_type const> group_offsets,
                                        size_type const num_groups,
                                        std::vector<double> const& quantiles,
                                        interpolation interp,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate number of unique values in each group of
 *  @p values
 *
 * @code{.pseudo}
 * values        = [2, 4, 4, -1, -2, <NA>, 4, <NA>]
 * group_labels  = [0, 0, 0,  1,  1,    2, 2,    3]
 * group_offsets = [0,        3,        5,       7, 8]
 * num_groups    = 4
 *
 * group_nunique(null_policy::EXCLUDE) = [2, 2, 1, 0]
 * group_nunique(null_policy::INCLUDE) = [2, 2, 2, 1]
 * @endcode
 *
 * @param values Grouped and sorted (within group) values to get unique count of
 * @param group_labels ID of group that the corresponding value belongs to
 * @param num_groups Number of groups ( unique values in @p group_labels )
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param null_handling Exclude nulls while counting if null_policy::EXCLUDE,
 *  Include nulls if null_policy::INCLUDE.
 *  Nulls are treated equal.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_nunique(column_view const& values,
                                      cudf::device_span<size_type const> group_labels,
                                      size_type const num_groups,
                                      cudf::device_span<size_type const> group_offsets,
                                      null_policy null_handling,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate nth values in each group of  @p values
 *
 * @code{.pseudo}
 * values        = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_sizes   = [3,        2,        2,       1]
 * group_labels  = [0, 0, 0,  1,  1,    2, 2,    3]
 * group_offsets = [0,        3,        5,       7, 8]
 * num_groups    = 4
 *
 * group_nth_element(n=0, null_policy::EXCLUDE) = [2, -1, 4, <NA>]
 * group_nth_element(n=0, null_policy::INCLUDE) = [2, -1, <NA>, <NA>]
 * @endcode
 *
 * @param values Grouped values to get nth value of
 * @param group_sizes Number of elements per group
 * @param group_labels ID of group that the corresponding value belongs to
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param num_groups Number of groups ( unique values in @p group_labels )
 * @param n nth element to choose from each group of @p values
 * @param null_handling Exclude nulls while counting if null_policy::EXCLUDE,
 *  Include nulls if null_policy::INCLUDE.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_nth_element(column_view const& values,
                                          column_view const& group_sizes,
                                          cudf::device_span<size_type const> group_labels,
                                          cudf::device_span<size_type const> group_offsets,
                                          size_type num_groups,
                                          size_type n,
                                          null_policy null_handling,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);
/**
 * @brief Internal API to collect grouped values into a lists column
 *
 * @code{.pseudo}
 * values        = [2, 1, 4, -1, -2, <NA>, 4, <NA>]
 * group_offsets = [0,        3,        5,       7, 8]
 * num_groups    = 4
 *
 * group_collect(...) = [[2, 1, 4], [-1, -2], [<NA>, 4], [<NA>]]
 * @endcode
 *
 * @param values Grouped values to collect.
 * @param group_offsets Offsets of groups' starting points within @p values.
 * @param num_groups Number of groups.
 * @param null_handling Exclude nulls while counting if null_policy::EXCLUDE,
 *        include nulls if null_policy::INCLUDE.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<column> group_collect(column_view const& values,
                                      cudf::device_span<size_type const> group_offsets,
                                      size_type num_groups,
                                      null_policy null_handling,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to merge grouped lists into one list.
 *
 * @code{.pseudo}
 * values        = [[2, 1], [], [4, -1, -2], [], [<NA>, 4, <NA>]]
 * group_offsets = [0,                        3,                  5]
 * num_groups    = 2
 *
 * group_merge_lists(...) = [[2, 1, 4, -1, -2], [<NA>, 4, <NA>]]
 * @endcode
 *
 * @param values Grouped values (lists column) to collect.
 * @param group_offsets Offsets of groups' starting points within @p values.
 * @param num_groups Number of groups.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<column> group_merge_lists(column_view const& values,
                                          cudf::device_span<size_type const> group_offsets,
                                          size_type num_groups,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to merge grouped M2 values corresponding to the same key.
 *
 * The values of M2 are merged following the parallel algorithm described here:
 * `https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm`
 *
 * Merging M2 values require accessing to partial M2 values, means, and valid counts. Thus, the
 * input to this aggregation need to be a structs column containing tuples of 3 values
 * `(valid_count, mean, M2)`.
 *
 * This aggregation not only merges the partial results of `M2` but also merged all the partial
 * results of input aggregations (`COUNT_VALID`, `MEAN`, and `M2`). As such, the output will be a
 * structs column containing children columns of merged `COUNT_VALID`, `MEAN`, and `M2` values.
 *
 * @param values Grouped values (tuples of values `(valid_count, mean, M2)`) to merge.
 * @param group_offsets Offsets of groups' starting points within @p values.
 * @param num_groups Number of groups.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_merge_m2(column_view const& values,
                                       cudf::device_span<size_type const> group_offsets,
                                       size_type num_groups,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to merge multiple output of HISTOGRAM aggregation.
 *
 * The input values column should be given as a lists column in the form of
 * `LIST<STRUCT<value, count>>`.
 * After merging, the order of distinct elements in each output list is not specified.
 *
 * @code{.pseudo}
 * values        = [ [<1, 2>, <2, 1>], [<2, 2>], [<3, 2>, <2, 1>], [<1, 1>, <2, 1>] ]
 * group_offsets = [ 0,                          2,                                 4]
 * num_groups    = 2
 *
 * output = [[<1, 2>, <2, 3>], [<1, 1>, <2, 2>, <3, 2>]]]
 * @endcode
 *
 * @param values Grouped values to get valid count of
 * @param group_offsets Offsets of groups' starting points within @p values
 * @param num_groups Number of groups
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_merge_histogram(column_view const& values,
                                              cudf::device_span<size_type const> group_offsets,
                                              size_type num_groups,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to find covariance of child columns of a non-nullable struct column.
 *
 * @param values_0 The first grouped values column to compute covariance
 * @param values_1 The second grouped values column to compute covariance
 * @param group_labels ID of group that the corresponding value belongs to
 * @param num_groups Number of groups.
 * @param count The count of valid rows of the grouped values of both columns
 * @param mean_0 The mean of the first grouped values column
 * @param mean_1 The mean of the second grouped values column
 * @param min_periods The minimum number of non-null rows required to consider the covariance
 * @param ddof The delta degrees of freedom used in the calculation of the variance
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_covariance(column_view const& values_0,
                                         column_view const& values_1,
                                         cudf::device_span<size_type const> group_labels,
                                         size_type num_groups,
                                         column_view const& count,
                                         column_view const& mean_0,
                                         column_view const& mean_1,
                                         size_type min_periods,
                                         size_type ddof,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to find correlation from covariance and standard deviation.
 *
 * @param covariance The covariance of two grouped values columns
 * @param stddev_0 The standard deviation of the first grouped values column
 * @param stddev_1 The standard deviation of the second grouped values column
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_correlation(column_view const& covariance,
                                          column_view const& stddev_0,
                                          column_view const& stddev_1,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to calculate bitwise operation on grouped values ignoring nulls.
 *
 * @param bit_op Bitwise operation to perform on the input
 * @param grouped_values Grouped values to perform bitwise operation on
 * @param group_labels ID of group that the corresponding value belongs to
 * @param num_groups Number of groups
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_bitwise(bitwise_op bit_op,
                                      column_view const& grouped_values,
                                      device_span<size_type const> group_labels,
                                      size_type num_groups,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Internal API to find top k elements in each group of grouped values
 *
 * @param k Number of top elements to find in each group
 * @param topk_order Identifies ascending or descending for selecting the values
 * @param values Grouped values to find top k elements within
 * @param group_offsets Offsets to identify each group in values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> group_top_k(size_type k,
                                    order topk_order,
                                    column_view const& values,
                                    device_span<size_type const> group_offsets,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
