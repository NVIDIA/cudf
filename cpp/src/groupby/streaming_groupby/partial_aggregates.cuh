/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <cstdint>

namespace cudf::groupby {

/**
 * @brief Adapts existing element aggregation to intermediate counts and squared sums.
 */
struct merge_element_aggregator {
  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type target_index,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    // Input is an aggregate result column: dictionary keys have been decoded and
    // variable-width inputs can only produce fixed-width counts.
    if constexpr (!cudf::is_fixed_width<Source>()) {
      return;
    } else if constexpr (k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL) {
      // Partial counts already contain the number of input rows. COUNT's ordinary
      // element aggregator would increment by one and SUM would promote INT32.
      if constexpr (k == aggregation::COUNT_VALID) {
        if (source.is_null(source_index)) { return; }
      }
      cudf::detail::atomic_add(&target.element<size_type>(target_index),
                               source.element<size_type>(source_index));
    } else if constexpr ((k == aggregation::SUM || k == aggregation::SUM_OF_SQUARES ||
                          k == aggregation::PRODUCT || k == aggregation::MIN ||
                          k == aggregation::MAX) &&
                         cudf::detail::is_valid_aggregation<Source, k>()) {
      // SUM_OF_SQUARES partials are already squared and promoted to their result type.
      constexpr auto merge_kind = k == aggregation::SUM_OF_SQUARES ? aggregation::SUM : k;
      cudf::detail::element_aggregator{}.template operator()<Source, merge_kind>(
        target, target_index, source, source_index);
    }
  }
};

/**
 * @brief Merges partial aggregate states into a destination group's first partial.
 *
 * A null group map denotes the identity mapping. The scalar merge operation is shared with
 * streaming merge, so counts are summed and already-squared SUM_OF_SQUARES values are not squared.
 */
struct merge_partial_aggs_fn {
  size_type const* group_targets;
  size_type source_capacity;
  size_type source_groups;
  size_type source_partials_per_group;
  aggregation::Kind const* aggs;
  table_device_view source_values;
  mutable_table_device_view target_values;

  __device__ void operator()(int64_t idx) const
  {
    auto const num_rows = static_cast<int64_t>(source_groups) * source_partials_per_group;
    auto const position = idx % num_rows;
    auto const group    = static_cast<size_type>(position % source_groups);
    auto const partial  = static_cast<size_type>(position / source_groups);
    // Capacity is the physical pitch; iterate only live groups in every partial.
    auto const source_row = partial * source_capacity + group;
    auto const target     = group_targets == nullptr ? group : group_targets[group];
    if (target == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) { return; }
    auto const column  = static_cast<size_type>(idx / num_rows);
    auto const& values = source_values.column(column);
    cudf::detail::dispatch_type_and_aggregation(values.type(),
                                                aggs[column],
                                                merge_element_aggregator{},
                                                target_values.column(column),
                                                target,
                                                values,
                                                source_row);
  }
};

}  // namespace cudf::groupby
