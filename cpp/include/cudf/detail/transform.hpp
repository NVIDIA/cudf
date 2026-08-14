/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <tuple>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::compute_column
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> compute_column(table_view const& table,
                                       ast::expression const& expr,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::bools_to_mask
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> bools_to_mask(
  column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::encode
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> encode(
  cudf::table_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::one_hot_encode
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::pair<std::unique_ptr<column>, table_view> one_hot_encode(column_view const& input,
                                                              column_view const& categories,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::mask_to_bools
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> mask_to_bools(bitmask_type const* null_mask,
                                      size_type begin_bit,
                                      size_type end_bit,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::row_bit_count
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> row_bit_count(table_view const& t,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::segmented_row_bit_count
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> segmented_row_bit_count(table_view const& t,
                                                size_type segment_length,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);

/**
 * @brief Owns the allocations and device views for a transform operation.
 */
class CUDF_EXPORT prepared_transform {
 public:
  /**
   * @brief Validates and prepares a CUDA-source transform operation.
   */
  prepared_transform(null_aware is_null_aware,
                     std::span<transform_input const> inputs,
                     std::span<transform_output const> outputs,
                     std::vector<std::unique_ptr<column>>&& string_offsets,
                     std::optional<size_type> row_size,
                     rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                     rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  prepared_transform(prepared_transform const&)            = delete;
  prepared_transform& operator=(prepared_transform const&) = delete;
  prepared_transform(prepared_transform&&) noexcept;
  prepared_transform& operator=(prepared_transform&&) noexcept;
  ~prepared_transform();

  /**
   * @brief Returns the device arguments for this prepared transform.
   */
  std::tuple<size_type,
             bitmask_type const*,
             column_device_view_core const*,
             mutable_column_device_view_core const*,
             int32_t*>
  kernel_arguments();

  /**
   * @brief Checks the transform error state and constructs the output table.
   */
  std::unique_ptr<table> finalize() &&;

 private:
  struct impl;
  std::unique_ptr<impl> _state;
};

}  // namespace detail
}  // namespace cudf
