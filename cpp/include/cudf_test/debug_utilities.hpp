/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/default_stream.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace test {

/**
 * @brief Formats a column view as a string
 *
 * @param col The input column view
 * @param delimiter The delimiter to put between strings
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 */
std::string to_string(cudf::column_view const& col,
                      std::string const& delimiter,
                      rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
                      cudf::memory_resources mr    = cudf::get_current_device_resource_ref());

/**
 * @brief Convert column values to a host vector of strings
 *
 * @param col The input column view
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 */
std::vector<std::string> to_strings(
  cudf::column_view const& col,
  rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
  cudf::memory_resources mr    = cudf::get_current_device_resource_ref());

/**
 * @brief Print a column view to an ostream
 *
 * @param col The input column view
 * @param os The output stream
 * @param stream CUDA stream used for device memory operations
 * @param mr Memory resources used for temporary device allocations
 */
void print(cudf::column_view const& col,
           std::ostream& os             = std::cout,
           rmm::cuda_stream_view stream = cudf::test::get_default_stream(),
           cudf::memory_resources mr    = cudf::get_current_device_resource_ref());

}  // namespace test
}  // namespace CUDF_EXPORT cudf
