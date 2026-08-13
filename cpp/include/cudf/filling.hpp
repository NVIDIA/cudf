/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

/**
 * @file
 * @brief Column APIs for fill, repeat, and sequence
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup transformation_fill
 * @{
 */

/**
 * @brief Fills a range of elements in-place in a column with a scalar value.
 *
 * Fills N elements of @p destination starting at @p begin with @p value, where
 * N = (@p end - @p begin).
 *
 * Overwrites the range of elements in @p destination indicated by the indices
 * [@p begin, @p end) with @p value. Use the out-of-place fill function
 * returning std::unique_ptr<column> for use cases requiring memory
 * reallocation.
 *
 * @throws cudf::logic_error if memory reallocation is required (e.g. for
 * variable width types).
 * @throws cudf::logic_error for invalid range (if @p begin < 0,
 * @p begin > @p end, or @p end > @p destination.size()).
 * @throws cudf::logic_error if @p destination and @p value have different
 * types.
 * @throws cudf::logic_error if @p value is invalid but @p destination is not
 * nullable.
 *
 * @param destination The preallocated column to fill into
 * @param begin The starting index of the fill range (inclusive)
 * @param end The index of the last element in the fill range (exclusive)
 * @param value The scalar value to fill
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void fill_in_place(mutable_column_view& destination,
                   size_type begin,
                   size_type end,
                   scalar const& value,
                   rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Fills a range of elements in a column out-of-place with a scalar
 * value.
 *
 * Creates a new column as-if an in-place fill was performed into @p input;
 * i.e. it is as if a copy of @p input was created first and then the elements
 * indicated by the indices [@p begin, @p end) were overwritten by @p value.
 *
 * @throws cudf::logic_error for invalid range (if @p begin < 0,
 * @p begin > @p end, or @p end > @p destination.size()).
 * @throws cudf::logic_error if @p destination and @p value have different
 * types.
 *
 * @param input The input column used to create a new column. The new column
 * is created by replacing the values of @p input in the specified range with
 * @p value.
 * @param begin The starting index of the fill range (inclusive)
 * @param end The index of the last element in the fill range (exclusive)
 * @param value The scalar value to fill
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The result output column
 */
std::unique_ptr<column> fill(
  column_view const& input,
  size_type begin,
  size_type end,
  scalar const& value,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Repeat rows of a Table.
 *
 * Creates a new table by repeating the rows of @p input_table. The number of
 * repetitions of each element is defined by the value at the corresponding
 * index of @p count
 * Example:
 * ```
 * in = [4,5,6]
 * count = [1,2,3]
 * return = [4,5,5,6,6,6]
 * ```
 * @p count should not have null values; should not contain negative values;
 * and the sum of count elements should not overflow the size_type's limit.
 * The behavior of this function is undefined if @p count has negative values
 * or the sum overflows.
 *
 * @throws cudf::logic_error if the data type of @p count is not size_type.
 * @throws cudf::logic_error if @p input_table and @p count have different
 * number of rows.
 * @throws cudf::logic_error if @p count has null values.
 *
 * @param input_table Input table
 * @param count Non-nullable column of an integral type
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return The result table containing the repetitions
 */
std::unique_ptr<table> repeat(
  table_view const& input_table,
  column_view const& count,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Repeat rows of a Table.
 *
 * Creates a new table by repeating @p count times the rows of @p input_table.
 * Example:
 * ```
 * in = [4,5,6]
 * count = 2
 * return = [4,4,5,5,6,6]
 * ```
 * @throws cudf::logic_error if @p count is negative.
 * @throws std::overflow_error if @p input_table.num_rows() * @p count overflows size_type.
 *
 * @param input_table Input table
 * @param count Number of repetitions
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return The result table containing the repetitions
 */
std::unique_ptr<table> repeat(
  table_view const& input_table,
  size_type count,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Fills a column with a sequence of value specified by an initial value and a step.
 *
 * Creates a new column and fills with @p size values starting at @p init and
 * incrementing by @p step, generating the sequence
 * [ init, init+step, init+2*step, ... init + (size - 1)*step]
 *
 * ```
 * size = 3
 * init = 0
 * step = 2
 * return = [0, 2, 4]
 * ```
 *
 * @throws std::invalid_argument if @p init and @p step are not the same type.
 * @throws std::invalid_argument if scalar types are not numeric or invalid
 * @throws std::invalid_argument if @p size is < 0.
 *
 * @param size Size of the output column
 * @param init First value in the sequence
 * @param step Increment value
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The result column containing the generated sequence
 */
std::unique_ptr<column> sequence(
  size_type size,
  scalar const& init,
  scalar const& step,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Fills a column with a sequence of value specified by an initial value and a step of 1.
 *
 * Creates a new column and fills with @p size values starting at @p init and
 * incrementing by 1, generating the sequence
 * [ init, init+1, init+2, ... init + (size - 1)]
 *
 * ```
 * size = 3
 * init = 0
 * return = [0, 1, 2]
 * ```
 *
 * @throws std::invalid_argument if @p init is not numeric or invalid
 * @throws std::invalid_argument if @p size is < 0
 *
 * @param size Size of the output column
 * @param init First value in the sequence
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return The result column containing the generated sequence
 */
std::unique_ptr<column> sequence(
  size_type size,
  scalar const& init,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate a sequence of timestamps beginning at `init` and incrementing by `months` for
 * each successive element, i.e., `output[i] = init + i * months` for `i` in `[0, size)`.
 *
 * If a given date is invalid, the date is scaled back to the last available day of that month.
 *
 * Example:
 * ```
 * size = 3
 * init = 2020-01-31 08:00:00
 * months = 1
 * return = [2020-01-31 08:00:00, 2020-02-29 08:00:00, 2020-03-31 08:00:00]
 * ```
 *
 * @throw cudf::logic_error if input datatype is not a TIMESTAMP
 *
 * @param size Number of timestamps to generate
 * @param init The initial timestamp
 * @param months Months to increment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @return Timestamps column with sequences of months
 */
std::unique_ptr<cudf::column> calendrical_month_sequence(
  size_type size,
  scalar const& init,
  size_type months,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
