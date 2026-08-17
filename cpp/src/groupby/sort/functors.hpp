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
                       rmm::device_async_resource_ref mr,
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
   * @brief Get the grouped values
   *
   * Computes the grouped values from @p values on first invocation and returns
   * the stored result on subsequent invocation
   */
  column_view get_grouped_values()
  {
    if (is_presorted()) { return values; }

    // TODO (dm): After implementing single pass multi-agg, explore making a
    //            cache of all grouped value columns rather than one at a time
    if (grouped_values)
      return grouped_values->view();
    else if (sorted_values)
      // In scan, it wouldn't be ok to return sorted values when asked for grouped values.
      // It's overridden in scan implementation.
      return sorted_values->view();
    else
      return (grouped_values = helper.grouped_values(values, stream, mr))->view();
  };

  /**
   * @brief Get the grouped and sorted values
   *
   * Computes the grouped and sorted (within each group) values from @p values
   * on first invocation and returns the stored result on subsequent invocation
   */
  column_view get_sorted_values()
  {
    return sorted_values ? sorted_values->view()
                         : (sorted_values = helper.sorted_values(values, stream, mr))->view();
  };

 protected:
  sort::sort_groupby_helper& helper;  ///< Sort helper
  cudf::detail::result_cache& cache;  ///< cache of results to store into
  column_view const& values;          ///< Column of values to group and aggregate

  cuda::stream_ref stream;            ///< CUDA stream on which to execute kernels
  rmm::device_async_resource_ref mr;  ///< Memory resource to allocate space for results

  sorted keys_are_sorted;                  ///< Whether the keys are sorted
  std::unique_ptr<column> sorted_values;   ///< Memoised grouped and sorted values
  std::unique_ptr<column> grouped_values;  ///< Memoised grouped values
};
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
