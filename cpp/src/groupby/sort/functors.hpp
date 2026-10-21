/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>
#include <optional>

namespace cudf {
namespace groupby {
namespace detail {
/**
 * @brief Functor to dispatch aggregation with
 *
 * This functor is to be used with `aggregation_dispatcher` to compute the
 * appropriate aggregation. If the values on which to run the aggregation are
 * unchanged, then this functor should be re-used. This is because it stores
 * memoised sorted and/or grouped values and re-using will save on computation
 * of these values.
 */
struct store_result_functor {
  store_result_functor(column_view const& values,
                       sort::sort_groupby_helper& helper,
                       cudf::detail::result_cache& cache,
                       cuda::stream_ref stream,
                       cudf::memory_resources mr,
                       sorted keys_are_sorted = sorted::NO)
    : helper(helper),
      cache(cache),
      values(values),
      stream(stream),
      mr(std::move(mr)),
      keys_are_sorted(keys_are_sorted)
  {
  }

 protected:
  /**
   * @brief Check if the groupby keys are presorted
   */
  [[nodiscard]] bool is_presorted() const { return keys_are_sorted == sorted::YES; }

  /**
   * @brief Borrow grouped values in their original order within each group.
   *
   * Computes the grouped values from @p values on first invocation and returns
   * the stored result on subsequent invocations, including after ownership has
   * been transferred to a cached result by release_grouped_values().
   */
  column_view get_grouped_values()
  {
    if (is_presorted()) { return values; }
    if (grouped_values_view) { return *grouped_values_view; }

    // Positional aggregations and host UDFs need input order within every group,
    // even if an earlier aggregation requested a value-sorted view.
    return grouped_values
             ? grouped_values->view()
             : (grouped_values = helper.grouped_values(values, stream, mr.get_output_mr()))->view();
  }

  /**
   * @brief Transfer grouped values to a result, retaining a view for later aggregations.
   *
   * Moves the owned column to the caller and retains a non-owning view so later
   * aggregations can reuse the same storage without gathering the values again.
   * The caller must keep the unmodified column alive in the result cache while
   * later aggregations use that view. For presorted keys, copies the borrowed
   * input values instead.
   */
  std::unique_ptr<column> release_grouped_values()
  {
    if (is_presorted()) { return std::make_unique<column>(values, stream, mr.get_output_mr()); }
    if (!grouped_values) {
      grouped_values = helper.grouped_values(values, stream, mr.get_output_mr());
    }
    grouped_values_view = grouped_values->view();
    return std::move(grouped_values);
  }

  /**
   * @brief Borrow grouped values without requiring a particular order within each group.
   *
   * Both cached layouts keep groups in key order with the same group boundaries,
   * so either can be used with helper.group_labels() and helper.group_offsets().
   * Only the value order within each group may differ: original input order or
   * sorted value order.
   */
  column_view get_unordered_grouped_values()
  {
    if (is_presorted()) { return values; }
    if (grouped_values_view) { return *grouped_values_view; }
    if (grouped_values) { return grouped_values->view(); }
    if (sorted_values) { return sorted_values->view(); }
    return get_grouped_values();
  }

  /**
   * @brief Get the grouped and sorted values
   *
   * Computes the grouped and sorted (within each group) values from @p values
   * on first invocation and returns the stored result on subsequent invocations.
   * This layout can coexist with the original-order grouped values: for example,
   * NUNIQUE needs sorted values, while a later NTH_ELEMENT needs input order.
   */
  column_view get_sorted_values()
  {
    return sorted_values
             ? sorted_values->view()
             : (sorted_values = helper.sorted_values(values, stream, mr.get_temporary_mr()))
                 ->view();
  };

 protected:
  sort::sort_groupby_helper& helper;  ///< Sort helper
  cudf::detail::result_cache& cache;  ///< cache of results to store into
  column_view const& values;          ///< Column of values to group and aggregate

  cuda::stream_ref stream;    ///< CUDA stream on which to execute kernels
  cudf::memory_resources mr;  ///< Resources for results and temporary storage

  sorted keys_are_sorted;                          ///< Whether the keys are sorted
  std::unique_ptr<column> sorted_values;           ///< Memoised grouped and sorted values
  std::unique_ptr<column> grouped_values;          ///< Memoised grouped values
  std::optional<column_view> grouped_values_view;  ///< Values now owned by a cached result
};
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
