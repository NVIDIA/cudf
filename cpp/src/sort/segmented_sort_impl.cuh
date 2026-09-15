/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_segmented_sort.cuh>
#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @brief Functor performs faster segmented sort on eligible columns
 */
template <sort_method method>
struct column_fast_sort_fn {
  /**
   * @brief Run-time check for faster segmented sort on an eligible column
   *
   * Fast segmented sort can handle integral types including
   * decimal types if dispatch_storage_type is used but it does not support int128.
   */
  static bool is_fast_sort_supported(column_view const& col)
  {
    return !col.has_nulls() and
           (cudf::is_integral(col.type()) ||
            (cudf::is_fixed_point(col.type()) and (col.type().id() != type_id::DECIMAL128)));
  }

  /**
   * @brief Compile-time check for supporting fast segmented sort for a specific type
   *
   * The dispatch_storage_type means we can check for integral types to
   * include fixed-point types but the CUB limitation means we need to exclude int128.
   */
  template <typename T>
  static constexpr bool is_fast_sort_supported()
  {
    return cudf::is_integral<T>() and !std::is_same_v<__int128, T>;
  }

  template <typename T>
  void fast_sort(column_view const& input,
                 column_view const& segment_offsets,
                 mutable_column_view& indices,
                 bool ascending,
                 cuda::stream_ref stream)
  {
    // CUB's segmented sort functions cannot accept iterators.
    // We create a temporary column here for it to use.
    auto temp_col                   = cudf::detail::allocate_like(input,
                                                input.size(),
                                                mask_allocation_policy::NEVER,
                                                stream,
                                                cudf::get_current_device_resource_ref());
    mutable_column_view output_view = temp_col->mutable_view();
    auto temp_indices =
      cudf::column(cudf::column_view(indices.type(), indices.size(), indices.head(), nullptr, 0),
                   stream,
                   cudf::get_current_device_resource_ref());

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
                   temp_indices.view().begin<size_type>(),
                   indices.begin<size_type>(),
                   input.size(),
                   segment_offsets.size() - 1,
                   segment_offsets.begin<size_type>(),
                   segment_offsets.begin<size_type>() + 1,
                   stream.get());
  }

  template <typename T, CUDF_ENABLE_IF(is_fast_sort_supported<T>())>
  void operator()(column_view const& input,
                  column_view const& segment_offsets,
                  mutable_column_view& indices,
                  bool ascending,
                  cuda::stream_ref stream)
  {
    fast_sort<T>(input, segment_offsets, indices, ascending, stream);
  }

  template <typename T, CUDF_ENABLE_IF(!is_fast_sort_supported<T>())>
  void operator()(
    column_view const&, column_view const&, mutable_column_view&, bool, cuda::stream_ref)
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
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  // Unfortunately, CUB's segmented sort functions cannot accept iterators.
  // We have to build a pre-filled sequence of indices as input.
  auto sorted_indices = cudf::detail::sequence(
    input.size(),
    numeric_scalar<size_type>{0, true, stream, cudf::get_current_device_resource_ref()},
    stream,
    mr);
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
 * ```
 * Example 1:
 * num_rows = 10
 * offsets = {0, 3, 7, 10}
 * segment-indices -> { 3,3,3, 7,7,7,7, 10,10,10 }
 * ```
 *
 * ```
 * Example 2: (offsets do not cover all indices)
 * num_rows = 10
 * offsets = {3, 7}
 * segment-indices -> { 0,1,2, 7,7,7,7, 8,9,10 }
 * ```
 *
 * @param num_rows Total number of rows in the input keys to sort
 * @param offsets The offsets identifying the segments
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
rmm::device_uvector<size_type> get_segment_indices(size_type num_rows,
                                                   column_view const& offsets,
                                                   cuda::stream_ref stream);

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
  cuda::stream_ref stream,
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

  // the average row size for which to prefer fast sort
  constexpr cudf::size_type MAX_AVG_LIST_SIZE_FOR_FAST_SORT{100};
  // the maximum row count for which to prefer fast sort
  constexpr cudf::size_type MAX_LIST_SIZE_FOR_FAST_SORT{1 << 18};

  // fast-path for single column sort:
  // - single-column table
  // - not stable-sort
  // - no nulls and allowable fixed-width type
  // - size and width are limited -- based on benchmark results
  if (keys.num_columns() == 1 and
      column_fast_sort_fn<method>::is_fast_sort_supported(keys.column(0)) and
      (segment_offsets.size() > 0) and
      (((keys.num_rows() / segment_offsets.size()) < MAX_AVG_LIST_SIZE_FOR_FAST_SORT) or
       (keys.num_rows() < MAX_LIST_SIZE_FOR_FAST_SORT))) {
    auto const col_order = column_order.empty() ? order::ASCENDING : column_order.front();
    return fast_segmented_sorted_order<method>(
      keys.column(0), segment_offsets, col_order, stream, mr);
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
                                                    cuda::stream_ref stream,
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
  return detail::gather(values,
                        sorted_order->view(),
                        out_of_bounds_policy::DONT_CHECK,
                        negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}

}  // namespace detail
}  // namespace cudf
