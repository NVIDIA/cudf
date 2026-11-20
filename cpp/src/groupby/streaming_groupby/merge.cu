/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "partial_aggregates.cuh"

#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <mutex>
#include <string>

namespace cudf::groupby {

void streaming_groupby::impl::do_merge(impl const& other, cuda::stream_ref stream)
{
  // Register the source read while both objects' host bookkeeping is stable. std::lock
  // avoids opposite-direction merge deadlocks; self-merge acquires its mutex only once.
  std::unique_lock lock{_insert_mutex, std::defer_lock};
  std::unique_lock source_lock{other._insert_mutex, std::defer_lock};
  if (this == &other) {
    lock.lock();
  } else {
    std::lock(lock, source_lock);
  }

  ensure_not_invalidated();
  CUDF_EXPECTS(!other._invalidated.load(std::memory_order_relaxed),
               "Cannot merge from an invalidated streaming_groupby.");

  auto const other_distinct_keys = other._distinct_keys.load(std::memory_order_relaxed);
  if (!other._initialized || other_distinct_keys == 0) { return; }
  CUDF_EXPECTS(_initialized,
               "Cannot merge into an uninitialized streaming_groupby. "
               "Call aggregate() at least once before merge().");
  CUDF_EXPECTS(other_distinct_keys <= _max_distinct_keys,
               "Merge source distinct keys (" + std::to_string(other_distinct_keys) +
                 ") exceeds max_distinct_keys (" + std::to_string(_max_distinct_keys) + ").",
               std::invalid_argument);
  CUDF_EXPECTS(other._agg_kinds == _agg_kinds,
               "Cannot merge streaming_groupby objects with different aggregation schemas.",
               std::invalid_argument);
  CUDF_EXPECTS(other._key_indices == _key_indices,
               "Cannot merge streaming_groupby objects with different key column indices.",
               std::invalid_argument);
  CUDF_EXPECTS(other._null_handling == _null_handling,
               "Cannot merge streaming_groupby objects with different null handling policies.",
               std::invalid_argument);

  auto const mr          = cudf::get_current_device_resource_ref();
  bool insertion_started = false;
  try {
    // Source writes and earlier source reads must precede this snapshot. A later source
    // writer/resize observes record_result_read below before these locks are released.
    other._aggregate_done.wait(stream);
    auto other_keys           = other.gather_distinct_keys(stream, mr);
    auto const other_key_view = other_keys->view();
    update_nullable_state(other_key_view);
    if (!_key_set) { create_key_set(stream); }
    _insert_done.wait(stream);
    insertion_started  = true;
    auto const targets = probe_and_insert(other_key_view, false, stream);
    _insert_done.record(stream);
    _results_read_done.wait(stream);

    std::unique_ptr<table> source_snapshot;
    auto source_partials_per_group = other._partials_per_group;
    auto source_capacity           = other._aggregate_capacity;
    auto const other_aggs_view     = [&] {
      if (this == &other) {
        // Partials collapse into the first partial. A self-merge must snapshot first so its first
        // partial cannot observe contributions concurrently arriving from other partials.
        source_snapshot           = other.gather_agg_results(stream, mr);
        source_partials_per_group = 1;
        source_capacity           = other_distinct_keys;
        return source_snapshot->view();
      }
      return other._agg_results->view();
    }();
    auto const d_source = table_device_view::create(other_aggs_view, stream);
    auto const d_target = mutable_table_device_view::create(_agg_results->mutable_view(), stream);
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, mr),
      cuda::counting_iterator<int64_t>(0),
      static_cast<int64_t>(other_distinct_keys) * source_partials_per_group * _agg_kinds.size(),
      merge_partial_aggs_fn{targets.data(),
                            source_capacity,
                            other_distinct_keys,
                            source_partials_per_group,
                            _d_agg_kinds->data(),
                            *d_source,
                            *d_target});
    record_aggregate_completion(stream);
    other.record_result_read(stream);
  } catch (...) {
    if (insertion_started) { _invalidated = true; }
    // Source reads must finish before unlocking source bookkeeping even on failure.
    synchronize_on_error(stream);
    throw;
  }
}

}  // namespace cudf::groupby
