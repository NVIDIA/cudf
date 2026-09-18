/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_single_pass_aggs.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {

constexpr bool is_fusable_sum(aggregation::Kind kind)
{
  return kind == aggregation::SUM || kind == aggregation::SUM_OF_SQUARES ||
         kind == aggregation::COUNT_VALID;
}

constexpr bool is_fusable_minmax_sum(aggregation::Kind kind)
{
  return kind == aggregation::MIN || kind == aggregation::MAX || kind == aggregation::SUM;
}

template <typename T>
struct value_accessor;

struct reduction_context {
  column_view const& values;
  column_device_view const& d_values;
  data_type values_type;  ///< Type of the values, or of the keys for dictionary values
  grouped_rows const& grouped;
  size_type num_groups;
  bool nullable;  ///< Whether the result carries a null mask

  template <typename T>
  value_accessor<T> accessor() const;
};

// Shared host helpers are defined only in the frontend, keeping their reduction kernels unique.
std::pair<rmm::device_buffer, size_type> reduce_group_validity(reduction_context const& ctx,
                                                               cuda::stream_ref stream,
                                                               cudf::memory_resources mr);
void set_group_null_mask(column& result,
                         reduction_context const& ctx,
                         cuda::stream_ref stream,
                         cudf::memory_resources mr);
std::unique_ptr<column> make_size_type_column(reduction_context const& ctx,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr);

std::unique_ptr<column> count_groups(reduction_context const& ctx,
                                     bool valid_only,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr);

// Kind-specific TUs explicitly instantiate this bridge; the frontend needs no reducer definition.
template <aggregation::Kind K>
std::unique_ptr<column> compute_reduction(reduction_context const& ctx,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr);

// Suppress implicit instantiation in the frontend and other reducer translation units.
extern template std::unique_ptr<column> compute_reduction<aggregation::SUM>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);
extern template std::unique_ptr<column> compute_reduction<aggregation::PRODUCT>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);
extern template std::unique_ptr<column> compute_reduction<aggregation::SUM_OF_SQUARES>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);
extern template std::unique_ptr<column> compute_reduction<aggregation::MIN>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);
extern template std::unique_ptr<column> compute_reduction<aggregation::MAX>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);
extern template std::unique_ptr<column> compute_reduction<aggregation::ARGMIN>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);
extern template std::unique_ptr<column> compute_reduction<aggregation::ARGMAX>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);
extern template std::unique_ptr<column> compute_reduction<aggregation::SUM_OVERFLOW>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);

template <aggregation::Kind K>
std::vector<std::unique_ptr<column>> compute_reductions(host_span<reduction_context const> contexts,
                                                        std::span<int8_t const> is_intermediate,
                                                        cuda::stream_ref stream,
                                                        cudf::memory_resources mr);

extern template std::vector<std::unique_ptr<column>> compute_reductions<aggregation::SUM>(
  host_span<reduction_context const>,
  std::span<int8_t const>,
  cuda::stream_ref,
  cudf::memory_resources);
extern template std::vector<std::unique_ptr<column>>
  compute_reductions<aggregation::SUM_OF_SQUARES>(host_span<reduction_context const>,
                                                  std::span<int8_t const>,
                                                  cuda::stream_ref,
                                                  cudf::memory_resources);
extern template std::vector<std::unique_ptr<column>> compute_reductions<aggregation::PRODUCT>(
  host_span<reduction_context const>,
  std::span<int8_t const>,
  cuda::stream_ref,
  cudf::memory_resources);
extern template std::vector<std::unique_ptr<column>> compute_reductions<aggregation::MIN>(
  host_span<reduction_context const>,
  std::span<int8_t const>,
  cuda::stream_ref,
  cudf::memory_resources);
extern template std::vector<std::unique_ptr<column>> compute_reductions<aggregation::MAX>(
  host_span<reduction_context const>,
  std::span<int8_t const>,
  cuda::stream_ref,
  cudf::memory_resources);

std::vector<std::unique_ptr<column>> compute_fused_sums(reduction_context const& ctx,
                                                        host_span<aggregation::Kind const> kinds,
                                                        std::span<int8_t const> is_intermediate,
                                                        cuda::stream_ref stream,
                                                        cudf::memory_resources mr);

std::vector<std::unique_ptr<column>> compute_fused_minmax_sum(
  reduction_context const& ctx,
  host_span<aggregation::Kind const> kinds,
  std::span<int8_t const> is_intermediate,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash
