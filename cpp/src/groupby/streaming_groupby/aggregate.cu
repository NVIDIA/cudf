/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "groupby/hash/single_pass_functors.cuh"

#include <cudf/detail/copy.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <algorithm>
#include <mutex>

namespace cudf::groupby {

void streaming_groupby::impl::do_aggregate(table_view const& data, cuda::stream_ref stream)
{
  ensure_not_invalidated();

  auto const batch_size = data.num_rows();
  if (batch_size == 0) { return; }

  // The transient key encoding is only valid while a single insertion is in flight, and
  // growing the state replaces what every phase works on, so the calls are serialized on the
  // host, and their insertion phases, via the event, on the device.  The aggregation phase of
  // each chunk is per-group atomic, so its kernels overlap freely across streams.
  std::lock_guard const lock{_insert_mutex};

  // Re-check under the lock: another caller may have invalidated the object since the
  // fail-fast check above.
  ensure_not_invalidated();

  if (!_initialized) { initialize(data, stream); }
  if (!_key_set) { create_key_set(stream); }

  auto const temp_mr      = cudf::get_current_device_resource_ref();
  auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());

  // A batch goes in by chunks that the state has room for even were every row of the chunk a
  // new key, so the state is sized by the groups it holds rather than by the rows of a batch.
  for (size_type offset = 0; offset < batch_size;) {
    _insert_done.wait(stream);
    ensure_room(std::min(batch_size - offset, min_chunk_rows), stream);

    auto const rows  = std::min(batch_size - offset, room());
    auto const chunk = rows == batch_size
                         ? data
                         : cudf::detail::slice(data, {offset, offset + rows}, stream).front();

    auto const batch_keys = chunk.select(_key_indices);
    update_nullable_state(batch_keys);

    auto const result = probe_and_insert(batch_keys, stream);
    _insert_done.record(stream);

    auto const values_view = chunk.select(_value_col_indices);
    auto const d_values    = table_device_view::create(values_view, stream);
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, temp_mr),
      cuda::counting_iterator<int64_t>(0),
      static_cast<int64_t>(rows) * num_agg_cols,
      detail::hash::compute_single_pass_aggs_dense_output_fn{
        result.target_indices.begin(), _d_agg_kinds->data(), *d_values, *_d_agg_results});
    record_aggregation(stream);

    offset += rows;
  }
}

}  // namespace cudf::groupby
