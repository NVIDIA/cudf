/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "groupby/hash/single_pass_functors.cuh"

#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace cudf::groupby {

struct replicated_aggs_fn {
  size_type const* target_indices;
  aggregation::Kind const* aggs;
  table_device_view input_values;
  mutable_table_device_view output_values;
  mutable_table_device_view replica_values;
  size_type replica_rows;

  __device__ void operator()(int64_t idx) const
  {
    auto const num_rows       = input_values.num_rows();
    auto const source_row_idx = static_cast<size_type>(idx % num_rows);
    auto const target_row_idx = target_indices[source_row_idx];
    if (target_row_idx == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
      return;
    }

    auto const col_idx     = static_cast<size_type>(idx / num_rows);
    auto const& source_col = input_values.column(col_idx);
    auto const replicated  = target_row_idx < replica_rows;
    auto const& target_col = replicated ? replica_values.column(col_idx) : output_values.column(col_idx);
    auto const row = replicated ? (source_row_idx % agg_replica_count) * replica_rows + target_row_idx : target_row_idx;
    cudf::detail::dispatch_type_and_aggregation(source_col.type(),
                                                aggs[col_idx],
                                                cudf::detail::element_aggregator{},
                                                target_col,
                                                row,
                                                source_col,
                                                source_row_idx);
  }
};

void streaming_groupby::impl::do_aggregate(table_view const& data, cuda::stream_ref stream)
{
  ensure_not_invalidated();

  auto const batch_size = data.num_rows();
  if (batch_size == 0) { return; }

  CUDF_EXPECTS(batch_size <= _max_distinct_keys,
               "Batch size (" + std::to_string(batch_size) + ") exceeds max_distinct_keys (" +
                 std::to_string(_max_distinct_keys) + ").",
               std::invalid_argument);

  CUDF_EXPECTS(static_cast<int64_t>(_max_distinct_keys) + static_cast<int64_t>(batch_size) <=
                 static_cast<int64_t>(std::numeric_limits<size_type>::max()),
               "Transient key encoding (max_distinct_keys + batch_size) would overflow size_type.",
               std::invalid_argument);

  // The transient key encoding is only valid while a single insertion is in flight, so
  // insertion is serialized across concurrent callers on the host and, via the event, on the
  // device.  The aggregation below is per-group atomic and runs unserialized.
  auto const [result, d_replicas] = [&] {
    std::lock_guard const lock{_insert_mutex};

    // Re-check under the lock: another caller may have invalidated the object since the
    // fail-fast check above.
    ensure_not_invalidated();

    if (!_initialized) { initialize(data, stream); }

    auto const batch_keys = data.select(_key_indices);

    update_nullable_state(batch_keys);

    if (!_key_set) { create_key_set(stream); }

    _insert_done.wait(stream);
    auto inserted = probe_and_insert(batch_keys, stream);

    mutable_table_device_view const* replicas = nullptr;
    if (_distinct_keys.load() <= _replica_rows) {
      if (!_agg_replicas) {
        create_agg_replicas(stream);
      }
      replicas = _d_agg_replicas.get();
    }
    _insert_done.record(stream);
    return std::pair{std::move(inserted), replicas};
  }();

  auto const values_view = data.select(_value_col_indices);
  auto const d_values    = table_device_view::create(values_view, stream);

  auto const temp_mr      = cudf::get_current_device_resource_ref();
  auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());
  auto const num_items    = static_cast<int64_t>(batch_size) * num_agg_cols;
  if (d_replicas != nullptr) {
    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<int64_t>(0),
                       num_items,
                       replicated_aggs_fn{result.target_indices.begin(),
                                          _d_agg_kinds->data(),
                                          *d_values,
                                          *_d_agg_results,
                                          *d_replicas,
                                          _replica_rows});
  } else {
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, temp_mr),
      cuda::counting_iterator<int64_t>(0),
      num_items,
      detail::hash::compute_single_pass_aggs_dense_output_fn{
        result.target_indices.begin(), _d_agg_kinds->data(), *d_values, *_d_agg_results});
  }
}

}  // namespace cudf::groupby
