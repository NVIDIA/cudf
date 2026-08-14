/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_global_memory_aggs_null.hpp"
#include "compute_global_memory_aggs_null_kernels.hpp"
#include "output_utils.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace cudf::groupby::detail::hash {

std::pair<std::unique_ptr<table>, rmm::device_uvector<size_type>>
compute_global_memory_aggs_null_sparse(bitmask_type const* row_bitmask,
                                       table_view const& values,
                                       nullable_global_set_t const& key_set,
                                       host_span<aggregation::Kind const> h_agg_kinds,
                                       device_span<aggregation::Kind const> d_agg_kinds,
                                       std::span<int8_t const> is_agg_intermediate,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const num_rows = values.num_rows();
  auto const d_values = table_device_view::create(values, stream);
  auto agg_results =
    create_results_table(num_rows, values, h_agg_kinds, is_agg_intermediate, stream, mr);
  auto d_results = mutable_table_device_view::create(*agg_results, stream);

  auto const has_dictionary = std::any_of(
    values.begin(), values.end(), [](column_view const& col) { return is_dictionary(col.type()); });
  auto const has_non_dictionary =
    std::any_of(values.begin(), values.end(), [](column_view const& col) {
      return not is_dictionary(col.type());
    });
  auto const set_ref = key_set.ref(cuco::op::insert_and_find);

  if (has_non_dictionary) {
    launch_null_sparse_non_dictionary(
      set_ref, row_bitmask, d_agg_kinds.data(), *d_values, *d_results, num_rows, stream);
  }
  if (has_dictionary) {
    launch_null_sparse_dictionary(
      set_ref, row_bitmask, d_agg_kinds.data(), *d_values, *d_results, num_rows, stream);
  }

  auto unique_keys   = extract_populated_keys(key_set, num_rows, stream, mr);
  auto dense_results = cudf::detail::gather(agg_results->view(),
                                            unique_keys,
                                            out_of_bounds_policy::DONT_CHECK,
                                            cudf::negative_index_policy::NOT_ALLOWED,
                                            stream,
                                            mr);
  return {std::move(dense_results), std::move(unique_keys)};
}

}  // namespace cudf::groupby::detail::hash
