/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "segmented_sort_fast.cuh"
#include "segmented_sort_keys.cuh"
#include "sort.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_segmented_sort.cuh>
#include <cuda/iterator>
#include <thrust/for_each.h>

namespace cudf {
namespace detail {

/**
 * @brief Seeds the segmented sort's paired index buffers with the identity permutation
 *
 * `d_values_out` must start as the identity: the segmented sort leaves rows outside every segment
 * unwritten, and the contract keeps such rows mapped to themselves.
 */
struct identity_seed_pair {
  size_type* d_values_in;
  size_type* d_values_out;
  __device__ void operator()(size_type idx) const
  {
    d_values_in[idx]  = idx;
    d_values_out[idx] = idx;
  }
};

/**
 * @brief Functor performs faster segmented sort on eligible columns
 */
template <sort_method method>
struct column_fast_sort_fn {
  /**
   * @brief Run-time check for faster segmented sort on an eligible column
   *
   * Integral types and the fixed-point reps (via `dispatch_storage_type`) are supported; the
   * `__int128` DECIMAL128 rep is admitted for the unstable sort only.
   */
  static bool is_fast_sort_supported(column_view const& col)
  {
    auto const decimal128_ok =
      col.type().id() == type_id::DECIMAL128 and method == sort_method::UNSTABLE;
    return !col.has_nulls() and (cudf::is_integral(col.type()) ||
                                 (cudf::is_fixed_point(col.type()) and
                                  (col.type().id() != type_id::DECIMAL128 or decimal128_ok)));
  }

  /**
   * @brief Compile-time check for supporting fast segmented sort for a specific type
   *
   * The `dispatch_storage_type` lets the integral check cover the fixed-point reps; `__int128` is
   * admitted for the unstable sort only, matching the run-time gate.
   */
  template <typename T>
  static constexpr bool is_fast_sort_supported()
  {
    return (cudf::is_integral<T>() and !std::is_same_v<__int128, T>) or
           (std::is_same_v<__int128, T> and method == sort_method::UNSTABLE);
  }

  template <typename T>
  void fast_sort(column_view const& input,
                 column_view const& segment_offsets,
                 mutable_column_view& indices,
                 bool ascending,
                 rmm::cuda_stream_view stream)
  {
    // CUB's segmented sort functions cannot accept iterators.
    // We create a temporary column here for it to use.
    auto temp_col                   = cudf::detail::allocate_like(input,
                                                input.size(),
                                                mask_allocation_policy::NEVER,
                                                stream,
                                                cudf::get_current_device_resource_ref());
    mutable_column_view output_view = temp_col->mutable_view();
    // One pass seeds CUB's value-input buffer and identity-prefills the output `indices`.
    rmm::device_uvector<size_type> temp_indices(input.size(), stream);
    thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::counting_iterator<size_type>{0},
                       input.size(),
                       identity_seed_pair{temp_indices.data(), indices.begin<size_type>()});

    // DeviceSegmentedSort is faster than DeviceSegmentedRadixSort at this time
    auto fast_sort_impl = [stream](bool ascending, [[maybe_unused]] auto&&... args) {
      rmm::device_buffer d_temp_storage;
      size_t temp_storage_bytes = 0;
      if (ascending) {
        if constexpr (method == sort_method::STABLE) {
          cub::DeviceSegmentedSort::StableSortPairs(
            d_temp_storage.data(), temp_storage_bytes, std::forward<decltype(args)>(args)...);
          d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
          cub::DeviceSegmentedSort::StableSortPairs(
            d_temp_storage.data(), temp_storage_bytes, std::forward<decltype(args)>(args)...);
        } else {
          cub::DeviceSegmentedSort::SortPairs(
            d_temp_storage.data(), temp_storage_bytes, std::forward<decltype(args)>(args)...);
          d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
          cub::DeviceSegmentedSort::SortPairs(
            d_temp_storage.data(), temp_storage_bytes, std::forward<decltype(args)>(args)...);
        }
      } else {
        if constexpr (method == sort_method::STABLE) {
          cub::DeviceSegmentedSort::StableSortPairsDescending(
            d_temp_storage.data(), temp_storage_bytes, std::forward<decltype(args)>(args)...);
          d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
          cub::DeviceSegmentedSort::StableSortPairsDescending(
            d_temp_storage.data(), temp_storage_bytes, std::forward<decltype(args)>(args)...);
        } else {
          cub::DeviceSegmentedSort::SortPairsDescending(
            d_temp_storage.data(), temp_storage_bytes, std::forward<decltype(args)>(args)...);
          d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
          cub::DeviceSegmentedSort::SortPairsDescending(
            d_temp_storage.data(), temp_storage_bytes, std::forward<decltype(args)>(args)...);
        }
      }
    };

    fast_sort_impl(ascending,
                   input.begin<T>(),
                   output_view.begin<T>(),
                   temp_indices.data(),
                   indices.begin<size_type>(),
                   input.size(),
                   segment_offsets.size() - 1,
                   segment_offsets.begin<size_type>(),
                   segment_offsets.begin<size_type>() + 1,
                   stream.value());
  }

  template <typename T, CUDF_ENABLE_IF(is_fast_sort_supported<T>())>
  void operator()(column_view const& input,
                  column_view const& segment_offsets,
                  mutable_column_view& indices,
                  bool ascending,
                  rmm::cuda_stream_view stream)
  {
    fast_sort<T>(input, segment_offsets, indices, ascending, stream);
  }

  template <typename T, CUDF_ENABLE_IF(!is_fast_sort_supported<T>())>
  void operator()(
    column_view const&, column_view const&, mutable_column_view&, bool, rmm::cuda_stream_view)
  {
    CUDF_FAIL("Column type cannot be used with fast-sort function");
  }
};

/**
 * @brief Performs faster sort on eligible columns
 *
 * Check the `is_fast_sort_supported()==true` on the input column before using this function.
 *
 * @param input Column to sort
 * @param segment_offsets Identifies segments to sort within
 * @param column_order Sort ascending or descending
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <sort_method method>
std::unique_ptr<column> fast_segmented_sorted_order(column_view const& input,
                                                    column_view const& segment_offsets,
                                                    order const& column_order,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  // Allocated unfilled: the sort functor identity-seeds it along with CUB's value-input buffer.
  auto sorted_indices = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, input.size(), mask_state::UNALLOCATED, stream, mr);
  auto indices_view = sorted_indices->mutable_view();

  cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                               column_fast_sort_fn<method>{},
                                               input,
                                               segment_offsets,
                                               indices_view,
                                               column_order == order::ASCENDING,
                                               stream);
  return sorted_indices;
}

/**
 * @brief Builds indices to identify segments to sort
 *
 * The segments are added to the input table-view keys so they
 * are lexicographically sorted within the segmented groups.
 *
 * @code{.pseudo}
 * Example 1:
 * num_rows = 10
 * offsets = {0, 3, 7, 10}
 * segment-indices -> { 3,3,3, 7,7,7,7, 10,10,10 }
 * @endcode
 *
 * @code{.pseudo}
 * Example 2: (offsets do not cover all indices)
 * num_rows = 10
 * offsets = {3, 7}
 * segment-indices -> { 0,1,2, 7,7,7,7, 8,9,10 }
 * @endcode
 *
 * @param num_rows Total number of rows in the input keys to sort
 * @param offsets The offsets identifying the segments
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
rmm::device_uvector<size_type> get_segment_indices(size_type num_rows,
                                                   column_view const& offsets,
                                                   rmm::cuda_stream_view stream);

/**
 * @brief Whether the segment offsets span every row `[0, num_rows]`
 *
 * The packed, tiered, and strings fast paths label elements by dense segment ordinal and index the
 * output by raw offset, so partial-coverage offsets -- allowed by the public contract -- must take
 * the comparison path. Requiring at least two offsets also excludes `num_segments == 0`, whose
 * zero-width segment field would shift a 64-bit packed key by 64 (undefined behavior).
 *
 * @param segment_offsets The segment offsets (segment count plus one)
 * @param num_rows Element count the offsets must span
 * @param stream CUDA stream used for the boundary reads
 * @return true when there are at least two offsets spanning `[0, num_rows]`
 */
inline bool fast_path_offsets_cover_all_rows(column_view const& segment_offsets,
                                             size_type num_rows,
                                             rmm::cuda_stream_view stream)
{
  auto const num_offsets = segment_offsets.size();
  if (num_offsets < 2) { return false; }
  auto const first = cudf::detail::make_pinned_vector_async(
    device_span<size_type const>{segment_offsets.begin<size_type>(), 1}, stream);
  auto const last = cudf::detail::make_pinned_vector_async(
    device_span<size_type const>{segment_offsets.begin<size_type>() + (num_offsets - 1), 1},
    stream);
  stream.synchronize();
  return first[0] == 0 and last[0] == num_rows;
}

/**
 * @brief Segmented sorted-order utility
 *
 * Returns the indices that map the column to a segmented sorted table.
 * Automatically handles calling accelerated code paths as appropriate.
 *
 * @tparam method Specifies sort is stable or not
 * @param keys Table to sort
 * @param segment_offsets Identifies the segments within the keys
 * @param column_order Sort order for each column in the keys
 * @param null_precedence Where to place the null entries for each column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to allocate any returned objects
 */
template <sort_method method>
std::unique_ptr<column> segmented_sorted_order_common(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (keys.num_rows() == 0 || keys.num_columns() == 0) {
    return cudf::make_empty_column(type_to_id<size_type>());
  }

  CUDF_EXPECTS(segment_offsets.type() == data_type(type_to_id<size_type>()),
               "segment offsets should be size_type");

  if (not column_order.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(keys.num_columns()) == column_order.size(),
                 "Mismatch between number of columns and column order.");
  }

  if (not null_precedence.empty()) {
    CUDF_EXPECTS(static_cast<std::size_t>(keys.num_columns()) == null_precedence.size(),
                 "Mismatch between number of columns and null_precedence size.");
  }

  // Unstable-only fast paths for a single fixed-width key column, routed by type class and list
  // shape; any explicit (order, null_order) is folded into the fast-path sort keys.
  if constexpr (method == sort_method::UNSTABLE) {
    if (keys.num_columns() == 1 and segment_offsets.size() > 0 and column_order.size() == 1 and
        null_precedence.size() == 1) {
      // Routing runs first so a column routed to `comparison` skips the probe's stream sync.
      auto const path =
        choose_fixed_width_sort_path(keys.column(0), keys.num_rows(), segment_offsets, stream);
      if (path != fixed_width_sort_path::comparison and
          fast_path_offsets_cover_all_rows(segment_offsets, keys.num_rows(), stream)) {
        auto const polarity = resolve_sort_polarity(
          keys.column(0).has_nulls(), column_order.front(), null_precedence.front());
        switch (path) {
          case fixed_width_sort_path::tiered:  // short lists, or any nulls
            return fast_segmented_sorted_order_tiered(
              keys.column(0), segment_offsets, polarity, stream, mr);
          case fixed_width_sort_path::packed_radix:  // long no-null lists
            return fast_segmented_sorted_order_numeric_packed(
              keys.column(0), segment_offsets, polarity, stream, mr);
          case fixed_width_sort_path::cub_segmented:  // wider-rep no-null mid band
            return fast_segmented_sorted_order<method>(
              keys.column(0), segment_offsets, column_order.front(), stream, mr);
          case fixed_width_sort_path::comparison: break;  // excluded by the guard above
        }
      }
    }
  }

  // fast-path for single column sort:
  // - single-column table
  // - both stable and unstable sort
  // - no nulls and allowable fixed-width type
  // - size and width are limited -- based on benchmark results
  if (keys.num_columns() == 1 and
      column_fast_sort_fn<method>::is_fast_sort_supported(keys.column(0)) and
      (segment_offsets.size() > 0) and
      prefer_cub_segmented_sort(keys.num_rows(), segment_offsets.size()) and
      // DECIMAL128 wins with CUB only in its measured short-list band; above it CUB loses
      // several-fold to the comparison sort.
      (keys.column(0).type().id() != type_id::DECIMAL128 or
       keys.num_rows() / segment_offsets.size() <= DECIMAL128_CUB_MAX_AVG_LIST_SIZE)) {
    auto const col_order = column_order.empty() ? order::ASCENDING : column_order.front();
    return fast_segmented_sorted_order<method>(
      keys.column(0), segment_offsets, col_order, stream, mr);
  }

  // Unstable-only fast paths for a single STRING key column. Segments that all fit the warp tile
  // take the graduated in-warp merge sort, which beats the prefix-radix engine there; otherwise a
  // radix sort over packed leading-byte prefixes with byte-window tie-breaking. Any explicit
  // (order, null_order) is folded into both engines' keys.
  if constexpr (method == sort_method::UNSTABLE) {
    if (keys.num_columns() == 1 and keys.column(0).type().id() == type_id::STRING and
        (segment_offsets.size() > 0) and column_order.size() == 1 and
        null_precedence.size() == 1 and
        fast_path_offsets_cover_all_rows(segment_offsets, keys.num_rows(), stream)) {
      auto const polarity = resolve_sort_polarity(
        keys.column(0).has_nulls(), column_order.front(), null_precedence.front());
      if (strings_grad_all_segments_fit(segment_offsets, stream)) {
        return fast_segmented_sorted_order_strings_grad(
          keys.column(0), segment_offsets, polarity, stream, mr);
      }
      return fast_segmented_sorted_order_strings_prefix(
        keys.column(0), segment_offsets, polarity, stream, mr);
    }
  }

  // Get segment id of each element in all segments.
  auto segment_ids = get_segment_indices(keys.num_rows(), segment_offsets, stream);

  // insert segment id before all columns.
  std::vector<column_view> keys_with_segid;
  keys_with_segid.reserve(keys.num_columns() + 1);
  keys_with_segid.push_back(column_view(
    data_type(type_to_id<size_type>()), segment_ids.size(), segment_ids.data(), nullptr, 0));
  keys_with_segid.insert(keys_with_segid.end(), keys.begin(), keys.end());
  auto segid_keys = table_view(keys_with_segid);

  auto prepend_default = [](auto const& vector, auto default_value) {
    if (vector.empty()) return vector;
    std::remove_cv_t<std::remove_reference_t<decltype(vector)>> pre_vector;
    pre_vector.reserve(pre_vector.size() + 1);
    pre_vector.push_back(default_value);
    pre_vector.insert(pre_vector.end(), vector.begin(), vector.end());
    return pre_vector;
  };
  auto child_column_order    = prepend_default(column_order, order::ASCENDING);
  auto child_null_precedence = prepend_default(null_precedence, null_order::AFTER);

  // return sorted order of child columns
  if constexpr (method == sort_method::STABLE) {
    return detail::stable_sorted_order(
      segid_keys, child_column_order, child_null_precedence, stream, mr);
  } else {
    return detail::sorted_order(segid_keys, child_column_order, child_null_precedence, stream, mr);
  }
}

template <sort_method method>
std::unique_ptr<table> segmented_sort_by_key_common(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(values.num_rows() == keys.num_rows(),
               "Mismatch in number of rows for values and keys");
  auto sorted_order =
    segmented_sorted_order_common<method>(keys,
                                          segment_offsets,
                                          column_order,
                                          null_precedence,
                                          stream,
                                          cudf::get_current_device_resource_ref());
  // Gather segmented sort of child value columns
  auto result = detail::gather(values,
                               sorted_order->view(),
                               out_of_bounds_policy::DONT_CHECK,
                               negative_index_policy::NOT_ALLOWED,
                               stream,
                               mr);
  // A nullable but null-free input still yields a gather-allocated all-valid mask; release it so
  // output columns are nullable only when they contain nulls.
  for (auto i = size_type{0}; i < result->num_columns(); ++i) {
    auto& col = result->get_column(i);
    if (col.null_count() == 0) { col.set_null_mask(rmm::device_buffer{}, 0); }
  }
  return result;
}

}  // namespace detail
}  // namespace cudf
