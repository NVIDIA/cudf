/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/groupby/direct_groupby.hpp>

namespace CUDF_EXPORT cudf {
namespace groupby::detail {

/**
 * @brief Selects the aggregation strategy used by `direct_aggregate`
 */
enum class direct_aggregate_path {
  AUTO,           ///< Choose based on capacity, input size and available shared memory
  SHARED_MEMORY,  ///< Per-block shared memory accumulation; throws if the slots do not fit
  GLOBAL_MEMORY   ///< Global memory atomics directly into the aggregation slots
};

/**
 * @copydoc cudf::groupby::direct_aggregate
 *
 * @param path The aggregation strategy to use
 */
[[nodiscard]] std::pair<std::unique_ptr<column>, std::vector<aggregation_result>> direct_aggregate(
  column_view const& keys,
  std::span<aggregation_request const> requests,
  std::size_t capacity,
  direct_aggregate_path path,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace groupby::detail
}  // namespace CUDF_EXPORT cudf
