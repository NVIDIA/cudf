/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::compute_column
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> compute_column(table_view const& table,
                                       ast::expression const& expr,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::bools_to_mask
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> bools_to_mask(
  column_view const& input, cuda::stream_ref stream, rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::encode
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> encode(
  cudf::table_view const& input, cuda::stream_ref stream, rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::one_hot_encode
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<column>, table_view> one_hot_encode(column_view const& input,
                                                              column_view const& categories,
                                                              cuda::stream_ref stream,
                                                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::mask_to_bools
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> mask_to_bools(bitmask_type const* null_mask,
                                      size_type begin_bit,
                                      size_type end_bit,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::row_bit_count
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> row_bit_count(table_view const& t,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::segmented_row_bit_count
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> segmented_row_bit_count(table_view const& t,
                                                size_type segment_length,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
