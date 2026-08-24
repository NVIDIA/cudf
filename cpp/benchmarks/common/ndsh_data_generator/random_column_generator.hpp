/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <string>

namespace cudf::datagen {

/**
 * @brief Generate a column of random strings
 *
 * @param lower The lower bound of the length of the strings
 * @param upper The upper bound of the length of the strings
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> generate_random_string_column(
  cudf::size_type lower,
  cudf::size_type upper,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate a column of random numbers
 *
 * Example:
 *
 * lower = 10
 * upper = 15
 * num_rows = 10
 * result = [10, 11, 14, 14, 13, 12, 11, 11, 12, 14]

 *
 * @param lower The lower bound of the random numbers
 * @param upper The upper bound of the random numbers
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <typename T>
std::unique_ptr<cudf::column> generate_random_numeric_column(
  T lower,
  T upper,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate a primary key column
 *
 * Example:
 *
 * start = 1
 * num_rows = 10
 * result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
 *
 * @param start The starting value of the primary key
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> generate_primary_key_column(
  cudf::scalar const& start,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate a column where all the rows have the same string value
 *
 * Example:
 *
 * value = "abc"
 * num_rows = 5
 * result = ["abc", "abc", "abc", "abc", "abc"]
 *
 * @param value The string value to fill the column with
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> generate_repeat_string_column(
  std::string const& value,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate a column by randomly choosing from set of strings
 *
 * Example:
 *
 * set = {"s1", "s2", "s3"}
 * num_rows = 10
 * result = ["s1", "s2", "s2", "s1", "s3", "s3", "s3", "s2", "s1", "s1"]
 *
 * @param set The set of strings to choose from
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> generate_random_string_column_from_set(
  cudf::host_span<char const* const> set,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Generate a column consisting of a repeating sequence of integers
 *
 * Example:
 *
 * seq_length = 3
 * zero_indexed = false
 * num_rows = 10
 * result = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
 *
 * @param seq_length The length of the repeating sequence
 * @param zero_indexed Whether the sequence is zero or one indexed
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <typename T>
std::unique_ptr<cudf::column> generate_repeat_sequence_column(
  T seq_length,
  bool zero_indexed,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace cudf::datagen
