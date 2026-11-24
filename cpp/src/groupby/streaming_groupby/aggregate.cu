/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "warp_aggregate.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>

namespace cudf::groupby {

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

  // Keep host bookkeeping locked until every result use is enqueued and registered.
  std::lock_guard const lock{_insert_mutex};
  ensure_not_invalidated();
  bool insertion_started = false;
  try {
    if (!_initialized) { initialize(data, stream); }
    // An initial tiny batch must not permanently limit later batches to one partial. Increasing
    // the budget only after doubling bounds how often stable groups require a reshape.
    auto const batch_budget = std::max<size_type>(1, batch_size / partial_aggregation_block_size);
    if (_partial_budget == 0 || static_cast<int64_t>(batch_budget) >= 2LL * _partial_budget) {
      _partial_budget = batch_budget;
    }
    auto const batch_keys = data.select(_key_indices);
    update_nullable_state(batch_keys);
    if (!_key_set) { create_key_set(stream); }
    _insert_done.wait(stream);
    insertion_started  = true;
    auto const targets = probe_and_insert(batch_keys, true, stream);
    _insert_done.record(stream);
    _results_read_done.wait(stream);

    compute_warp_aggregates(data.select(_value_col_indices),
                            _agg_kinds,
                            {targets.data(), targets.size()},
                            _agg_results->mutable_view(),
                            stream,
                            cudf::get_current_device_resource_ref());
    record_aggregate_completion(stream);
  } catch (...) {
    if (insertion_started) { _invalidated = true; }
    // Do not let partially enqueued writes escape tracking before a subsequent resize/read.
    synchronize_on_error(stream);
    throw;
  }
}

}  // namespace cudf::groupby
