/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_global_memory_aggs.cuh"
#include "compute_global_memory_aggs_null.hpp"
#include "compute_global_memory_aggs_null_kernels.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace cudf::groupby::detail::hash {

std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs_null_dense(bitmask_type const* row_bitmask,
                                      table_view const& values,
                                      nullable_global_set_t const& key_set,
                                      host_span<aggregation::Kind const> h_agg_kinds,
                                      device_span<aggregation::Kind const> d_agg_kinds,
                                      std::span<int8_t const> is_agg_intermediate,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto const num_rows                = values.num_rows();
  auto [unique_keys, target_indices] = [&] {
    auto matching_keys =
      compute_matching_keys(row_bitmask, key_set.ref(cuco::op::insert_and_find), num_rows, stream);
    auto unique_keys       = extract_populated_keys(key_set, num_rows, stream, mr);
    auto key_transform_map = compute_key_transform_map(
      num_rows, unique_keys, stream, cudf::get_current_device_resource_ref());
    auto target_indices = compute_target_indices(
      matching_keys, key_transform_map, stream, cudf::get_current_device_resource_ref());
    return std::pair{std::move(unique_keys), std::move(target_indices)};
  }();

  auto const d_values  = table_device_view::create(values, stream);
  auto agg_results     = create_results_table(static_cast<size_type>(unique_keys.size()),
                                          values,
                                          h_agg_kinds,
                                          is_agg_intermediate,
                                          stream,
                                          mr);
  auto d_results       = mutable_table_device_view::create(*agg_results, stream);
  auto const num_items = num_rows * static_cast<int64_t>(h_agg_kinds.size());

  auto const has_dictionary = std::any_of(
    values.begin(), values.end(), [](column_view const& col) { return is_dictionary(col.type()); });
  auto const has_non_dictionary =
    std::any_of(values.begin(), values.end(), [](column_view const& col) {
      return not is_dictionary(col.type());
    });

  if (has_non_dictionary) {
    launch_null_dense_non_dictionary(
      target_indices.data(), d_agg_kinds.data(), *d_values, *d_results, num_items, stream);
  }
  if (has_dictionary) {
    launch_null_dense_dictionary(
      target_indices.data(), d_agg_kinds.data(), *d_values, *d_results, num_items, stream);
  }

  return {std::move(agg_results), std::move(unique_keys)};
}

}  // namespace cudf::groupby::detail::hash
