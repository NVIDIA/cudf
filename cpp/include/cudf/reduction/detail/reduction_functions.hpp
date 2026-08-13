/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <optional>

namespace cudf {
namespace reduction::detail {
/**
 * @brief Computes sum of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`
 * @throw cudf::logic_error if `output_dtype` is not an arithmetic type
 *
 * @param col input column to compute sum
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the sum
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Sum as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> sum(column_view const& col,
                            data_type const output_dtype,
                            std::optional<std::reference_wrapper<scalar const>> init,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

/**
 * @brief Computes sum with overflow detection of signed integer or decimal elements in input column
 *
 * Returns a struct scalar with {sum: same type as input, overflow: bool} fields.
 * Supported input types: signed integers (int8/16/32/64) and decimals (decimal32/64/128).
 *
 * @throw std::invalid_argument if input column type is not a supported signed integer or decimal
 * @throw std::invalid_argument if `output_type` is not STRUCT
 *
 * @param col input column to compute sum with overflow detection
 * @param output_type data type of return type (must be STRUCT)
 * @param init initial value of the sum
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Struct scalar with sum and overflow flag
 */
std::unique_ptr<scalar> sum_overflow(column_view const& col,
                                     data_type const output_type,
                                     std::optional<std::reference_wrapper<scalar const>> init,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @brief Computes minimum of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not the same as `output_dtype`
 *
 * @param col input column to compute minimum
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the minimum
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Minimum element as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> min(column_view const& col,
                            data_type const output_dtype,
                            std::optional<std::reference_wrapper<scalar const>> init,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

/**
 * @brief Computes maximum of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not the same as `output_dtype`
 *
 * @param col input column to compute maximum
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the maximum
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Maximum element as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> max(column_view const& col,
                            data_type const output_dtype,
                            std::optional<std::reference_wrapper<scalar const>> init,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

/**
 * @brief Computes index of the minimum element in the input column.
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @param col input column to compute reduction
 * @param dispatch_type The type to dispatch on. For dictionary columns this must be the keys type;
 *                      for all other columns it must equal `col.type()`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Index of the minimum element as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> argmin(column_view const& col,
                               data_type dispatch_type,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

/**
 * @brief Computes index of the maximum element in the input column.
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @param col input column to compute reduction
 * @param dispatch_type The type to dispatch on. For dictionary columns this must be the keys type;
 *                      for all other columns it must equal `col.type()`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Index of the maximum element as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> argmax(column_view const& col,
                               data_type dispatch_type,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

/**
 * @brief Computes the minimum and maximum values of the input column
 *
 * @param col input column to compute minmax
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A pair consisting of the minimum value and the maximum value
 */
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(
  cudf::column_view const& col, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

/**
 * @brief Computes any of elements in input column is true when typecasted to bool
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to bool
 * @throw cudf::logic_error if `output_dtype` is not bool
 *
 * @param col input column to compute any
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the any
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return bool scalar if any of elements is true when typecasted to bool
 */
std::unique_ptr<scalar> any(column_view const& col,
                            data_type const output_dtype,
                            std::optional<std::reference_wrapper<scalar const>> init,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

/**
 * @brief Computes all of elements in input column is true when typecasted to bool
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to bool
 * @throw cudf::logic_error if `output_dtype` is not bool
 *
 * @param col input column to compute all
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the all
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return bool scalar if all of elements is true when typecasted to bool
 */
std::unique_ptr<scalar> all(column_view const& col,
                            data_type const output_dtype,
                            std::optional<std::reference_wrapper<scalar const>> init,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

/**
 * @brief Compute frequency for each unique element in the input column.
 *
 * The result histogram is stored in structs column having two children. The first child contains
 * unique elements from the input, and the second child contains their corresponding frequencies.
 *
 * @param input The column to compute histogram
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return A list_scalar storing a structs column as the result histogram
 */
std::unique_ptr<scalar> histogram(column_view const& input,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @brief Merge multiple histograms together.
 *
 * @param input The input given as multiple histograms concatenated together
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return A list_scalar storing the result histogram
 */
std::unique_ptr<scalar> merge_histogram(column_view const& input,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

/**
 * @brief Computes product of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`
 * @throw cudf::logic_error if `output_dtype` is not an arithmetic type
 *
 * @param col input column to compute product
 * @param output_dtype data type of return type and typecast elements of input column
 * @param init initial value of the product
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Product as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> product(column_view const& col,
                                data_type const output_dtype,
                                std::optional<std::reference_wrapper<scalar const>> init,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr);

/**
 * @brief Computes sum of squares of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`
 * @throw cudf::logic_error if `output_dtype` is not an arithmetic type
 *
 * @param col input column to compute sum of squares
 * @param output_dtype data type of return type and typecast elements of input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Sum of squares as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> sum_of_squares(column_view const& col,
                                       data_type const output_dtype,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @brief Computes mean of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col input column to compute mean
 * @param output_dtype data type of return type and typecast elements of input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Mean as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> mean(column_view const& col,
                             data_type const output_dtype,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr);

/**
 * @brief Computes variance of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col input column to compute variance
 * @param output_dtype data type of return type and typecast elements of input column
 * @param ddof Delta degrees of freedom. The divisor used is N - ddof, where N represents the number
 * of elements.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Variance as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> variance(column_view const& col,
                                 data_type const output_dtype,
                                 size_type ddof,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @brief Computes standard deviation of elements in input column
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col input column to compute standard deviation
 * @param output_dtype data type of return type and typecast elements of input column
 * @param ddof Delta degrees of freedom. The divisor used is N - ddof, where N represents the number
 * of elements.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Standard deviation as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> standard_deviation(column_view const& col,
                                           data_type const output_dtype,
                                           size_type ddof,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @brief Returns nth element in input column
 *
 * A negative value `n` is interpreted as `n+count`, where `count` is the number of valid
 * elements in the input column if `null_handling` is `null_policy::EXCLUDE`, else `col.size()`.
 *
 * If all elements in input column are null, output scalar is null.
 *
 * @warning This function is expensive (invokes a kernel launch). So, it is not
 * recommended to be used in performance sensitive code or inside a loop.
 * It takes O(`col.size()`) time and space complexity for nullable column with
 * `null_policy::EXCLUDE` as input.
 *
 * @throw cudf::logic_error if n falls outside the range `[-count, count)` where `count` is the
 * number of valid * elements in the input column if `null_handling` is `null_policy::EXCLUDE`,
 * else `col.size()`.
 *
 * @param col input column to get nth element from
 * @param n index of element to get
 * @param null_handling Indicates if null values will be counted while indexing
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return nth element as scalar
 */
std::unique_ptr<scalar> nth_element(column_view const& col,
                                    size_type n,
                                    null_policy null_handling,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @brief Collect input column into a (list) scalar
 *
 * @param col input column to collect from
 * @param null_handling Indicates if null values will be counted while collecting
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return collected list as scalar
 */
std::unique_ptr<scalar> collect_list(column_view const& col,
                                     null_policy null_handling,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @brief Merge a bunch of list scalars into single list scalar
 *
 * @param col input list column representing numbers of list scalars to be merged
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return merged list as scalar
 */
std::unique_ptr<scalar> merge_lists(lists_column_view const& col,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @brief Collect input column into a (list) scalar without duplicated elements
 *
 * @param col input column to collect from
 * @param null_handling Indicates if null values will be counted while collecting
 * @param nulls_equal Indicates if null values will be considered as equal values
 * @param nans_equal Indicates if nan values will be considered as equal values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return collected list with unique elements as scalar
 */
std::unique_ptr<scalar> collect_set(column_view const& col,
                                    null_policy null_handling,
                                    null_equality nulls_equal,
                                    nan_equality nans_equal,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @brief Merge a bunch of list scalars into single list scalar then drop duplicated elements
 *
 * @param col input list column representing numbers of list scalars to be merged
 * @param nulls_equal Indicates if null values will be considered as equal values
 * @param nans_equal Indicates if nan values will be considered as equal values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return collected list with unique elements as scalar
 */
std::unique_ptr<scalar> merge_sets(lists_column_view const& col,
                                   null_equality nulls_equal,
                                   nan_equality nans_equal,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @brief Performs bitwise reduction on the input column, ignoring nulls.
 *
 * @param bit_op Bitwise operation to perform on the input
 * @param col input column to perform bitwise reduction on
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Scalar containing the result of bitwise operation on all elements of the input
 */
std::unique_ptr<scalar> bitwise_reduction(bitwise_op bit_op,
                                          column_view const& col,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @brief Computes quantile value of the elements in the input column
 *
 * @see cudf::quantile for additional details
 *
 * @throw std::invalid_argument if the input column type is not an arithmetic type
 *
 * @param col Input column to compute quantile
 * @param quantile_value Quantile value in range [0,1]
 * @param interpolation Interpolation method
 * @param output_type Data type of return type
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Quantile as scalar of type `output_type`
 */
std::unique_ptr<cudf::scalar> quantile(column_view const& col,
                                       double quantile_value,
                                       cudf::interpolation interpolation,
                                       cudf::data_type const output_type,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @brief Computes the number of unique elements in the input column
 *
 * @param col Input column to compute the number of unique elements
 * @param null_handling Indicates if null values will be counted while computing the number of
 * unique elements
 * @param output_dtype Data type of return type
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Number of unique elements as scalar of type `output_dtype`
 */
std::unique_ptr<scalar> nunique(column_view const& col,
                                null_policy null_handling,
                                data_type const output_dtype,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr);

/**
 * @brief Returns the number of elements in the input column
 *
 * Returns `col.size()` or `col.size() - col.null_count()` depending on `null_handling`
 *
 * @param col Input column to compute the number of elements
 * @param null_handling Indicates if null values will be included in the count
 * @param output_type Data type of return type
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @return Number of elements as scalar of type `output_type`
 */
std::unique_ptr<scalar> count(column_view const& col,
                              null_policy null_handling,
                              data_type const output_type,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

}  // namespace reduction::detail
}  // namespace cudf
