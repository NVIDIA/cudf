/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/groupby/groupby_helper.hpp>
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
                       groupby_helper& helper,
                       cudf::detail::result_cache& cache,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr)
    : helper(helper), cache(cache), values(values), stream(stream), mr(std::move(mr))
  {
  }

 protected:
  /**
   * @brief Check if the groupby keys are presorted
   */
  [[nodiscard]] bool is_presorted() const { return helper.is_presorted(); }

  /**
   * @brief Get the grouped values
   *
   * Computes the grouped values from @p values on first invocation and returns
   * the stored result on subsequent invocation
   */
  column_view get_grouped_values()
  {
    if (is_presorted()) { return values; }

    // Input order is required by NTH_ELEMENT, COLLECT_LIST and host UDFs even when a prior
    // aggregation on the same values requested a value-sorted view.
    return grouped_values ? grouped_values->view()
                          : (grouped_values = helper.grouped_values(values, stream, mr))->view();
  };

  /**
   * @brief Get grouped values for aggregations that do not depend on row order.
   */
  column_view get_unordered_grouped_values()
  {
    if (is_presorted()) { return values; }
    if (grouped_values) { return grouped_values->view(); }

    // Keep this cache separate: order-sensitive aggregations must always obtain a stable view.
    return unordered_grouped_values
             ? unordered_grouped_values->view()
             : (unordered_grouped_values = helper.unordered_grouped_values(values, stream, mr))
                 ->view();
  }

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
  groupby_helper& helper;             ///< Grouping helper
  cudf::detail::result_cache& cache;  ///< cache of results to store into
  column_view const& values;          ///< Column of values to group and aggregate

  cuda::stream_ref stream;            ///< CUDA stream on which to execute kernels
  rmm::device_async_resource_ref mr;  ///< Memory resource to allocate space for results

  std::unique_ptr<column> sorted_values;   ///< Memoised grouped and sorted values
  std::unique_ptr<column> grouped_values;  ///< Memoised grouped values
  std::unique_ptr<column>
    unordered_grouped_values;  ///< Memoised values with no row-order guarantee
};
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
