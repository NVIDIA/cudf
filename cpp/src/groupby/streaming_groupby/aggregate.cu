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

#include <cstdint>
#include <limits>
#include <mutex>
#include <string>

namespace cudf::groupby {

namespace {

/*
 * Picks the partial result copy for a row.  XOR-folding the row index in 5-bit chunks keeps
 * the 32 lanes of a warp on distinct copies (sorted or runs of equal keys) while also mixing in
 * the higher bits, so that cyclic keys such as `key[i] = i % 128` do not send every row of a
 * group to the same copy.
 */
__device__ size_type partial_agg_copy(size_type row) noexcept
{
  auto r = static_cast<uint32_t>(row);
  r ^= r >> 20;
  r ^= r >> 10;
  r ^= r >> 5;
  return static_cast<size_type>(r % num_partial_agg_results);
}

struct partial_aggs_fn {
  size_type const* target_indices;
  aggregation::Kind const* aggs;
  table_device_view input_values;
  mutable_table_device_view output_values;
  mutable_table_device_view partial_values;
  size_type partial_rows;

  __device__ void operator()(int64_t idx) const
  {
    auto const num_rows       = input_values.num_rows();
    auto const source_row_idx = static_cast<size_type>(idx % num_rows);
    auto const target_row_idx = target_indices[source_row_idx];
    if (target_row_idx == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) { return; }

    auto const col_idx     = static_cast<size_type>(idx / num_rows);
    auto const& source_col = input_values.column(col_idx);
    auto const is_partial  = target_row_idx < partial_rows;
    auto const& target_col =
      is_partial ? partial_values.column(col_idx) : output_values.column(col_idx);
    auto const row = is_partial
                       ? partial_agg_copy(source_row_idx) * partial_rows + target_row_idx
                       : target_row_idx;
    cudf::detail::dispatch_type_and_aggregation(source_col.type(),
                                                aggs[col_idx],
                                                cudf::detail::element_aggregator{},
                                                target_col,
                                                row,
                                                source_col,
                                                source_row_idx);
  }
};

}  // namespace

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
  // device.  The aggregation below is per-group atomic and runs unserialized, except while
  // partial results exist (see `_partial_agg_results`).
  std::unique_lock lock{_insert_mutex};

  // Re-check under the lock: another caller may have invalidated the object since the
  // fail-fast check above.
  ensure_not_invalidated();

  if (!_initialized) { initialize(data, stream); }

  auto const batch_keys = data.select(_key_indices);

  update_nullable_state(batch_keys);

  if (!_key_set) { create_key_set(stream); }

  _insert_done.wait(stream);
  auto const result = probe_and_insert(batch_keys, stream);
  update_partial_agg_results(batch_size, stream);

  // Batches below the contention thresholds update the dense results directly even while
  // partial results exist.
  auto const has_partials = _partial_agg_results != nullptr;
  auto const use_partials =
    has_partials &&
    use_partial_agg_results(batch_size, _distinct_keys.load(std::memory_order_relaxed));
  if (!has_partials) {
    _insert_done.record(stream);
    lock.unlock();
  }

  auto const values_view = data.select(_value_col_indices);
  auto const d_values    = table_device_view::create(values_view, stream);

  auto const temp_mr   = cudf::get_current_device_resource_ref();
  auto const num_items = static_cast<int64_t>(batch_size) * static_cast<int64_t>(_agg_kinds.size());
  if (use_partials) {
    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<int64_t>(0),
                       num_items,
                       partial_aggs_fn{result.target_indices.begin(),
                                       _d_agg_kinds->data(),
                                       *d_values,
                                       *_d_agg_results,
                                       *_d_partial_agg_results,
                                       _partial_agg_rows});
  } else {
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, temp_mr),
      cuda::counting_iterator<int64_t>(0),
      num_items,
      detail::hash::compute_single_pass_aggs_dense_output_fn{
        result.target_indices.begin(), _d_agg_kinds->data(), *d_values, *_d_agg_results});
  }

  if (has_partials) { _insert_done.record(stream); }
}

}  // namespace cudf::groupby
