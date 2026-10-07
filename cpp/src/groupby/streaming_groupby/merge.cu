/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
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

/**
 * @brief Element aggregator that folds every partial result copy of one row into the target.
 *
 * One thread owns each target element, so the copies are reduced in registers and stored once
 * without atomics.  Rows that were never updated still hold the aggregation identity, so they
 * can be folded unconditionally.
 */
struct partial_results_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view partials,
                             size_type partial_rows) const noexcept
  {
    // The partial results already hold the target type, for which `Source == Target` except
    // for COUNT on non-integer sources, which never reach this aggregator.
    using Target = cudf::detail::target_type_t<Source, k>;
    using T      = cudf::device_storage_type_t<Target>;
    constexpr bool is_additive = k == aggregation::SUM || k == aggregation::SUM_OF_SQUARES ||
                                 k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL;
    constexpr bool is_reducible =
      (is_additive && requires(T v) { v + v; }) ||
      (k == aggregation::PRODUCT && requires(T v) { v * v; }) ||
      ((k == aggregation::MIN || k == aggregation::MAX) && requires(T v) { v < v; });
    if constexpr (!cudf::detail::is_valid_aggregation<Source, k>() ||
                  !cudf::is_fixed_width<Target>() || !is_reducible) {
      return;
    } else {
      auto value  = target.element<T>(target_index);
      auto valid  = target.is_valid(target_index);
      auto reduce = [](T lhs, T rhs) -> T {
        if constexpr (k == aggregation::MIN) {
          return cudf::DeviceMin{}(lhs, rhs);
        } else if constexpr (k == aggregation::MAX) {
          return cudf::DeviceMax{}(lhs, rhs);
        } else if constexpr (k == aggregation::PRODUCT) {
          return static_cast<T>(lhs * rhs);
        } else {
          return static_cast<T>(lhs + rhs);
        }
      };
      for (size_type copy = 0; copy < num_partial_agg_results; ++copy) {
        auto const row = copy * partial_rows + target_index;
        value          = reduce(value, partials.element<T>(row));
        valid          = valid || partials.is_valid(row);
      }
      target.element<T>(target_index) = value;
      if (valid && target.nullable()) { target.set_valid(target_index); }
    }
  }
};

struct merge_partial_results_fn {
  aggregation::Kind const* aggs;
  table_device_view partial_values;
  mutable_table_device_view target_values;
  size_type partial_rows;
  size_type num_keys;

  __device__ void operator()(int64_t idx) const
  {
    auto const row     = static_cast<size_type>(idx % num_keys);
    auto const col_idx = static_cast<size_type>(idx / num_keys);
    auto const& col    = partial_values.column(col_idx);
    cudf::detail::dispatch_type_and_aggregation(col.type(),
                                                aggs[col_idx],
                                                partial_results_aggregator{},
                                                target_values.column(col_idx),
                                                row,
                                                col,
                                                partial_rows);
  }
};

}  // namespace

void streaming_groupby::impl::merge_partial_agg_results(mutable_table_device_view const& target,
                                                        size_type num_keys,
                                                        cuda::stream_ref stream,
                                                        cudf::memory_resources mr) const
{
  num_keys = std::min(num_keys, _partial_agg_rows);
  if (!_partial_agg_results || num_keys == 0) { return; }
  auto const d_partials = table_device_view::create(_partial_agg_results->view(), stream);
  thrust::for_each_n(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                     cuda::counting_iterator<int64_t>(0),
                     static_cast<int64_t>(num_keys) * static_cast<int64_t>(_agg_kinds.size()),
                     merge_partial_results_fn{
                       _d_agg_kinds->data(), *d_partials, target, _partial_agg_rows, num_keys});
}

void streaming_groupby::impl::do_merge(impl const& other, cuda::stream_ref stream)
{
  // `other` is only read from, so a single lock on this object's insertion state is enough.
  std::lock_guard const lock{_insert_mutex};

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

  auto const mr = cudf::get_current_device_resource_ref();

  auto other_keys           = other.gather_distinct_keys(stream, mr);
  auto const other_key_view = other_keys->view();

  update_nullable_state(other_key_view);

  if (!_key_set) { create_key_set(stream); }

  _insert_done.wait(stream);
  auto result = probe_and_insert(other_key_view, stream);
  update_partial_agg_results(0, stream);
  // While partial results exist, the merge below must also be ordered by `_insert_done`.
  auto const has_partials = _partial_agg_results != nullptr;
  if (!has_partials) { _insert_done.record(stream); }

  // Merge aggregation values using dense target indices.  We only read from
  // `other._agg_results`; no need to deep-copy the source rows like keys, unless `other` has
  // partial results to fold in first.
  auto const other_gathered =
    other._partial_agg_results ? other.gather_agg_results(stream, mr) : nullptr;
  auto const other_aggs_view =
    other_gathered
      ? other_gathered->view()
      : cudf::detail::slice(other._agg_results->view(), {0, other_distinct_keys}, stream).front();
  auto const d_source = table_device_view::create(other_aggs_view, stream);

  auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, mr),
    cuda::counting_iterator<int64_t>(0),
    static_cast<int64_t>(other_distinct_keys) * num_agg_cols,
    merge_single_pass_aggs_fn{
      result.target_indices.begin(), _d_agg_kinds->data(), *d_source, *_d_agg_results});

  if (has_partials) { _insert_done.record(stream); }
}

}  // namespace cudf::groupby
