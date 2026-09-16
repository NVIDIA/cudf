/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace cudf::detail::rolling {

/**
 * @brief Functor to construct gather-map indices for NTH_ELEMENT rolling aggregation.
 *
 * By definition, the `N`th element is deemed null (i.e. the gather index is set to "nullify")
 * for the following cases:
 *   1. The window has fewer elements than `min_periods`.
 *   2. N falls outside the window, i.e. N ∉ [-window_size, window_size).
 *   3. `null_handling == EXCLUDE`, and the window has fewer than `N` non-null elements.
 *
 * If none of the above holds true, the result is non-null. How the value is determined
 * depends on `null_handling`:
 *   1. `null_handling == INCLUDE`: The required value is the `N`th value from the window's start.
 *       i.e. the gather index is window_start + N (adjusted for negative N).
 *   2. `null_handling == EXCLUDE`: The required value is the `N`th non-null value from the
 *       window's start. i.e. Return index of the `N`th non-null value.
 */
template <null_policy null_handling, typename PrecedingIter, typename FollowingIter>
struct gather_index_calculator {
  size_type n;
  bitmask_type const* input_nullmask;
  bool exclude_nulls;
  PrecedingIter preceding;
  FollowingIter following;
  size_type min_periods;
  cuda::stream_ref stream;

  static size_type constexpr NULL_INDEX =
    cuda::std::numeric_limits<size_type>::min();  // For nullifying with gather.

  gather_index_calculator(size_type n,
                          column_view input,
                          PrecedingIter preceding,
                          FollowingIter following,
                          size_type min_periods,
                          cuda::stream_ref stream)
    : n{n},
      input_nullmask{input.null_mask()},
      exclude_nulls{null_handling == null_policy::EXCLUDE and input.has_nulls()},
      preceding{preceding},
      following{following},
      min_periods{min_periods},
      stream{stream}
  {
  }

  /// For `null_policy::EXCLUDE`, find gather index for `N`th non-null value.
  template <typename Iter>
  size_type __device__ index_of_nth_non_null(Iter begin, size_type window_size) const
  {
    auto reqd_valid_count     = n >= 0 ? n : (-n - 1);
    auto const pred_nth_valid = [&reqd_valid_count, input_nullmask = input_nullmask](size_type j) {
      return cudf::bit_is_set(input_nullmask, j) && reqd_valid_count-- == 0;
    };
    auto const end   = begin + window_size;
    auto const found = thrust::find_if(thrust::seq, begin, end, pred_nth_valid);
    return found == end ? NULL_INDEX : *found;
  }

  size_type __device__ operator()(size_type i) const
  {
    // preceding[i] includes the current row.
    auto const window_size = preceding[i] + following[i];
    if (min_periods > window_size) { return NULL_INDEX; }

    auto const wrapped_n = n >= 0 ? n : (window_size + n);
    if (wrapped_n < 0 || wrapped_n > (window_size - 1)) {
      return NULL_INDEX;  // n lies outside the window.
    }

    // Out of short-circuit exits.
    // If nulls don't need to be excluded, a fixed window offset calculation is sufficient.
    auto const window_start = i - preceding[i] + 1;
    if (not exclude_nulls) { return window_start + wrapped_n; }

    // Must exclude nulls. Must examine each row in the window.
    auto const window_end = window_start + window_size;
    return n >= 0 ? index_of_nth_non_null(cuda::counting_iterator{window_start}, window_size)
                  : index_of_nth_non_null(
                      cuda::std::make_reverse_iterator(cuda::counting_iterator{window_end}),
                      window_size);
  }
};

/**
 * @brief Helper function for NTH_ELEMENT window aggregation
 *
 * The `N`th element is deemed null for the following cases:
 *    1. The window has fewer elements than `min_periods`.
 *    2. N falls outside the window, i.e. N ∉ [-window_size, window_size).
 *    3. `null_handling == EXCLUDE`, and the window has fewer than `N` non-null elements.
 *
 *  If none of the above holds true, the result is non-null. How the value is determined
 *  depends on `null_handling`:
 *    1. `null_handling == INCLUDE`: The required value is the `N`th value from the window's start.
 *    2. `null_handling == EXCLUDE`: The required value is the `N`th *non-null* value from the
 *        window's start. If the window has fewer than `N` non-null values, the result is null.
 *
 * @tparam null_handling Whether to include/exclude null rows in the window
 * @tparam PrecedingIter Type of iterator for preceding window
 * @tparam FollowingIter Type of iterator for following window
 * @param n The index of the element to be returned
 * @param input The input column
 * @param preceding Iterator specifying the preceding window bound
 * @param following Iterator specifying the following window bound
 * @param min_periods The minimum number of rows required in the window
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column the `n`th element of the specified window for each row
 */
template <null_policy null_handling, typename PrecedingIter, typename FollowingIter>
std::unique_ptr<column> nth_element(size_type n,
                                    column_view const& input,
                                    PrecedingIter preceding,
                                    FollowingIter following,
                                    size_type min_periods,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  auto const gather_iter = cudf::detail::make_counting_transform_iterator(
    0,
    gather_index_calculator<null_handling, PrecedingIter, FollowingIter>{
      n, input, preceding, following, min_periods, stream});

  auto gather_map = rmm::device_uvector<size_type>(input.size(), stream);
  thrust::copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               gather_iter,
               gather_iter + input.size(),
               gather_map.begin());

  auto gathered = cudf::detail::gather(table_view{{input}},
                                       gather_map,
                                       cudf::out_of_bounds_policy::NULLIFY,
                                       negative_index_policy::NOT_ALLOWED,
                                       stream,
                                       mr)
                    ->release();
  return std::move(gathered.front());
}

}  // namespace cudf::detail::rolling
