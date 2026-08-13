/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

//! NVText APIs
namespace CUDF_EXPORT nvtext {
/**
 * @addtogroup nvtext_dedup
 * @{
 * @file
 * @brief APIs for identifying and removing duplicate substrings in a strings column.
 */

/**
 * @brief Builds a suffix array for the input strings column
 *
 * The internal implementation creates a suffix array of the input which
 * requires ~4x the input size for temporary memory. The output is an additional
 * 4x of the input size.
 *
 * @throw std::invalid_argument If `min_width` is greater than the input chars size
 * @throw std::invalid_argument If the `input` chars size is greater than 2GB
 *
 * @param input Strings column to build suffix array for
 * @param min_width Minimum number of bytes that must match to identify a duplicate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Sorted suffix array and corresponding sizes
 */
std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_suffix_array(
  cudf::strings_column_view const& input,
  cudf::size_type min_width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns duplicate strings found in the given input
 *
 * The output includes any strings of at least `min_width` bytes that
 * appear more than once in the entire input.
 *
 * The result is undefined if the indices were not created on the same input
 * provided here.
 *
 * @throw If `min_width` <= 8
 * @throw If `min_width` is greater than the input chars size
 * @throw If the `input` chars size is greater than 2GB
 *
 * @param input Strings column for indices
 * @param indices Suffix array from nvtext::build_suffix_array
 * @param min_width Minimum number of bytes that must match to identify a duplicate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with updated strings
 */
std::unique_ptr<cudf::column> resolve_duplicates(
  cudf::strings_column_view const& input,
  cudf::device_span<cudf::size_type const> indices,
  cudf::size_type min_width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns duplicate strings found from input1 found in the given input2
 *
 * The output includes any strings of at least `min_width` bytes that
 * appear more than once between input1 and input2.
 *
 * The result is undefined if the indices1 were not created on the input1 and
 * indices2 were not created on input2.
 *
 * @throw If `min_width` <= 8
 * @throw If `min_width` is greater than the input chars size
 * @throw If the `input` chars size is greater than 2GB
 *
 * @param input1 Strings column for indices1
 * @param indices1 Suffix array from nvtext::build_suffix_array for input1
 * @param input2 Strings column for indices2
 * @param indices2 Suffix array from nvtext::build_suffix_array for input2
 * @param min_width Minimum number of bytes that must match to identify a duplicate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with updated strings
 */
std::unique_ptr<cudf::column> resolve_duplicates_pair(
  cudf::strings_column_view const& input1,
  cudf::device_span<cudf::size_type const> indices1,
  cudf::strings_column_view const& input2,
  cudf::device_span<cudf::size_type const> indices2,
  cudf::size_type min_width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
