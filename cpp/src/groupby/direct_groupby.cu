/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/streaming_groupby/common.cuh"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf::groupby {

struct direct_groupby::impl : aggregate_state {
  size_type _num_groups;

  impl(size_type num_groups,
       host_span<streaming_aggregation_request const> requests,
       cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  void do_aggregate(device_span<size_type const> group_ids,
                    table_view const& values,
                    cuda::stream_ref stream,
                    cudf::memory_resources mr);
  [[nodiscard]] std::vector<aggregation_result> do_finalize(cuda::stream_ref stream,
                                                            cudf::memory_resources mr) const;
};

direct_groupby::impl::impl(size_type num_groups,
                           host_span<streaming_aggregation_request const> requests,
                           cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : aggregate_state{requests, std::move(mr)}, _num_groups{num_groups}
{
  CUDF_EXPECTS(num_groups > 0, "num_groups must be positive.", std::invalid_argument);
}

void direct_groupby::impl::do_aggregate(device_span<size_type const> group_ids,
                                        table_view const& values,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  ensure_not_invalidated();

  CUDF_EXPECTS(std::cmp_equal(group_ids.size(), values.num_rows()),
               "group_ids and values must have the same number of rows.",
               std::invalid_argument);
  if (group_ids.empty()) { return; }

  // Keep host bookkeeping locked until every result use is enqueued and registered.
  std::lock_guard const lock{_insert_mutex};
  ensure_not_invalidated();
  bool reshape_started = false;
  try {
    auto const state_mr = cudf::memory_resources{_mr, mr.get_temporary_mr()};
    if (!_initialized) { initialize_aggregates(values, stream, state_mr); }
    update_partial_budget(values.num_rows());
    _insert_done.wait(stream);
    reshape_started = true;
    reshape_aggregate_results(_num_groups, _num_groups, stream, state_mr);
    _insert_done.record(stream);
    // Only now does finalize() have storage to read.
    _initialized = true;
    aggregate_values(values, group_ids, stream, mr);
  } catch (...) {
    if (reshape_started) { _invalidated = true; }
    // Do not let partially enqueued writes escape tracking before a subsequent resize/read.
    synchronize_on_error(stream);
    throw;
  }
}

std::vector<aggregation_result> direct_groupby::impl::do_finalize(cuda::stream_ref stream,
                                                                  cudf::memory_resources mr) const
{
  std::lock_guard const lock{_insert_mutex};
  CUDF_EXPECTS(_initialized, "Cannot finalize direct_groupby with no accumulated data.");
  _aggregate_done.wait(stream);
  try {
    auto results = finalize_gathered(gather_agg_results(_num_groups, stream, mr), stream, mr);
    record_result_read(stream);
    return results;
  } catch (...) {
    // A failed read must not leave untracked accesses before a subsequent resize.
    synchronize_on_error(stream);
    throw;
  }
}

direct_groupby::direct_groupby(size_type num_groups,
                               host_span<streaming_aggregation_request const> requests,
                               cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<impl>(num_groups, requests, std::move(mr))}
{
}

direct_groupby::~direct_groupby() = default;

direct_groupby::direct_groupby(direct_groupby&&) noexcept = default;

direct_groupby& direct_groupby::operator=(direct_groupby&&) noexcept = default;

void direct_groupby::aggregate(device_span<size_type const> group_ids,
                               table_view const& values,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  _impl->do_aggregate(group_ids, values, stream, mr);
}

std::vector<aggregation_result> direct_groupby::finalize(cuda::stream_ref stream,
                                                         cudf::memory_resources mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->do_finalize(stream, mr);
}

}  // namespace cudf::groupby
