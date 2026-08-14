/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace cudf::groupby::detail::hash {
template <typename SetType>
std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>> compute_global_memory_aggs(
  bitmask_type const* row_bitmask,
  table_view const& values,
  SetType const& key_set,
  host_span<aggregation::Kind const> h_agg_kinds,
  device_span<aggregation::Kind const> d_agg_kinds,
  std::span<int8_t const> is_agg_intermediate,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
