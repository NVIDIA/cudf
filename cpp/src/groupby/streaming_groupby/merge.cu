/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

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

#include <string>

namespace cudf::groupby {

namespace {

/**
 * @brief Element aggregator for merging intermediate results.
 */
struct merge_element_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    if constexpr (!cudf::detail::is_valid_aggregation<Source, k>()) {
      return;
    } else {
      if constexpr (k != aggregation::COUNT_ALL) {
        if (source.is_null(source_index)) { return; }
      }
      if constexpr (!(k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL)) {
        if (target.is_null(target_index)) { target.set_valid(target_index); }
      }

      if constexpr (k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL) {
        using Target = cudf::detail::target_type_t<Source, k>;
        cudf::detail::atomic_add(&target.element<Target>(target_index),
                                 source.element<Target>(source_index));
      } else if constexpr (k == aggregation::SUM_OF_SQUARES) {
        using Target = cudf::detail::target_type_t<Source, k>;
        cudf::detail::atomic_add(&target.element<Target>(target_index),
                                 static_cast<Target>(source.element<Source>(source_index)));
      } else {
        cudf::detail::update_target_element<Source, k>{}(
          target, target_index, source, source_index);
      }
    }
  }
};

struct merge_single_pass_aggs_fn {
  size_type const* target_indices;
  aggregation::Kind const* aggs;
  table_device_view source_values;
  mutable_table_device_view target_values;

  __device__ void operator()(int64_t idx) const
  {
    auto const num_rows       = source_values.num_rows();
    auto const source_row_idx = static_cast<size_type>(idx % num_rows);
    if (auto const target_row_idx = target_indices[source_row_idx];
        target_row_idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
      auto const col_idx     = static_cast<size_type>(idx / num_rows);
      auto const& source_col = source_values.column(col_idx);
      auto const& target_col = target_values.column(col_idx);
      cudf::detail::dispatch_type_and_aggregation(source_col.type(),
                                                  aggs[col_idx],
                                                  merge_element_aggregator{},
                                                  target_col,
                                                  target_row_idx,
                                                  source_col,
                                                  source_row_idx);
    }
  }
};

}  // namespace

void streaming_groupby::impl::do_merge(impl const& other, cuda::stream_ref stream)
{
  CUDF_EXPECTS(!_invalidated,
               "streaming_groupby is in an invalidated state from a prior failure; "
               "no further aggregate()/merge() is allowed.  finalize() may still be called.");
  CUDF_EXPECTS(!other._invalidated, "Cannot merge from an invalidated streaming_groupby.");

  if (!other._initialized || !other.has_state()) { return; }
  CUDF_EXPECTS(_initialized,
               "Cannot merge into an uninitialized streaming_groupby. "
               "Call aggregate() at least once before merge().");
  CUDF_EXPECTS(other._distinct_keys <= _max_distinct_keys,
               "Merge source distinct keys (" + std::to_string(other._distinct_keys) +
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

  auto const mr = cudf::get_current_device_resource_ref();

  auto other_keys                = other.gather_distinct_keys(stream, mr);
  auto const other_key_view      = other_keys->view();
  auto const other_distinct_keys = other._distinct_keys;
  if (other_distinct_keys == 0) { return; }

  update_nullable_state(other_key_view);

  if (!_key_set) { create_key_set(stream); }

  auto result = probe_and_insert(other_key_view, stream);

  // Merge aggregation values using dense target indices.  We only read from
  // `other._agg_results`; no need to deep-copy the source rows like keys.
  auto const other_aggs_view =
    cudf::detail::slice(other._agg_results->view(), {0, other_distinct_keys}, stream).front();
  auto const d_source = table_device_view::create(other_aggs_view, stream);

  auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, mr),
    cuda::counting_iterator<int64_t>(0),
    static_cast<int64_t>(other_distinct_keys) * num_agg_cols,
    merge_single_pass_aggs_fn{
      result.target_indices.begin(), _d_agg_kinds.data(), *d_source, *_d_agg_results});
}

}  // namespace cudf::groupby
