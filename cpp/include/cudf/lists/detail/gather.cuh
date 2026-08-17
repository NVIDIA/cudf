/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {

/**
 * @brief The information needed to create an iterator to gather level N+1
 *
 * @ref make_gather_data
 */
struct gather_data {
  // The offsets column from our parent list (level N)
  std::unique_ptr<column> offsets;
  // For each offset in the above offsets column, the original offset value
  // prior to being gathered.
  // Example:
  // If the offsets[3] == 6  (representing row 3 of the new column)
  // And the original value it was itself gathered from was 15, then
  // base_offsets[3] == 15
  rmm::device_uvector<int32_t> base_offsets;
  // size of the gather map that will be generated from this data
  size_type gather_map_size;
};

/**
 * @copydoc cudf::make_gather_data(cudf::lists_column_view const& source_column,
 *                                 MapItType gather_map,
 *                                 size_type gather_map_size,
 *                                 rmm::cuda_stream_view stream,
 *                                 rmm::device_async_resource_ref mr)
 *
 * @param prev_base_offsets The buffer backing the base offsets used in the gather map. We can
 *                          free this buffer before allocating the new one to keep peak memory
 *                          usage down.
 */
template <bool NullifyOutOfBounds, typename MapItType>
gather_data make_gather_data(cudf::lists_column_view const& source_column,
                             MapItType gather_map,
                             size_type gather_map_size,
                             rmm::device_uvector<int32_t>&& prev_base_offsets,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  // size of the gather map is the # of output rows
  size_type output_count = gather_map_size;

  // offsets of the source column
  int32_t const* src_offsets{source_column.offsets().data<int32_t>() + source_column.offset()};
  size_type const src_size = source_column.size();

  auto const source_column_nullmask = source_column.null_mask();

  auto sizes_itr = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<int32_t>([source_column_nullmask,
                                         source_column_offset = source_column.offset(),
                                         gather_map,
                                         output_count,
                                         src_offsets,
                                         src_size] __device__(int32_t index) -> int32_t {
      int32_t offset_index = index < output_count ? gather_map[index] : 0;

      // if this is an invalid index, this will be a NULL list
      if (NullifyOutOfBounds && ((offset_index < 0) || (offset_index >= src_size))) { return 0; }

      // If the source row is null, the output row size must be 0.
      if (source_column_nullmask != nullptr &&
          not cudf::bit_is_set(source_column_nullmask, source_column_offset + offset_index)) {
        return 0;
      }

      // the length of this list
      return src_offsets[offset_index + 1] - src_offsets[offset_index];
    }));

  auto [dst_offsets_c, map_size] =
    cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + output_count, stream, mr);

  // handle sliced columns
  size_type const shift =
    source_column.offset() > 0
      ? cudf::detail::get_value<int32_t>(source_column.offsets(), source_column.offset(), stream)
      : 0;

  // generate the base offsets
  rmm::device_uvector<int32_t> base_offsets = rmm::device_uvector<int32_t>(output_count, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    gather_map,
    gather_map + output_count,
    base_offsets.data(),
    [source_column_nullmask,
     source_column_offset = source_column.offset(),
     src_offsets,
     src_size,
     shift] __device__(int32_t index) {
      // if this is an invalid index, this will be a NULL list
      if (NullifyOutOfBounds && ((index < 0) || (index >= src_size))) { return 0; }

      // If the source row is null, the output row size must be 0.
      if (source_column_nullmask != nullptr &&
          not cudf::bit_is_set(source_column_nullmask, source_column_offset + index)) {
        return 0;
      }

      return src_offsets[index] - shift;
    });

  // Retrieve size of the resulting gather map for level N+1 (the last offset)
  auto const child_gather_map_size = static_cast<size_type>(map_size);
  return {std::move(dst_offsets_c), std::move(base_offsets), child_gather_map_size};
}

/**
 * @brief Generates the data needed to create a `gather_map` for the next level of
 * recursion in a hierarchy of list columns.
 *
 * Gathering from a single level of a list column is similar to gathering from
 * a string column.  Each row represents a list bounded by offsets.
 *
 * @code{.pseudo}
 * Example:
 * Level 0 : List<List<int>>
 *           Size : 3
 *           Offsets : [0, 2, 5, 10]
 *
 * This represents a column with 3 rows.
 * Row 0 has 2 elements (bounded by offsets 0,2)
 * Row 1 has 3 elements (bounded by offsets 2,5)
 * Row 2 has 5 elements (bounded by offsets 5,10)
 *
 * If we wanted to gather rows 0 and 2 the offsets for our outgoing column
 * would be the compacted ranges (0,2) and (5,10). The level 1 column
 * then looks like
 *
 * Level 1 : List<int>
 *           Size : 2
 *           Offsets : [0, 2, 7]
 *
 * However, we need to then gather one level further, because at the bottom we have
 * a column of integers.  We cannot gather the elements in the ranges (0, 2) and (2, 7).
 * Instead, we have to gather elements in the ranges from the Level 0 column (0, 2) and (5, 10).
 * So our gather_map iterator will need to know these "base" offsets to index properly.
 * Specifically:
 *
 * Offsets        : [0, 2, 7]    The offsets for Level 1
 * Base Offsets   : [0, 5]       The corresponding base offsets from Level 0
 *
 * Using this we can create an iterator that generates the sequence which properly indexes the
 * final integer values we want to gather.
 *
 * [0, 1, 5, 6, 7, 8, 9]
 * @endcode
 *
 * Thinking generally, this means that to produce a gather_map for level N+1, we need to use the
 * offsets from level N and the "base" offsets from level N-1. So we are always carrying along
 * one extra buffer of these "base" offsets which keeps our memory usage well controlled.
 *
 * @code{.pseudo}
 * Example:
 *
 * "column of lists of lists of ints"
 * {
 *   {
 *      {2, 3}, {4, 5}
 *   },
 *   {
 *      {6, 7, 8}, {9, 10, 11}, {12, 13, 14}
 *   },
 *   {
 *      {15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}
 *   }
 * }
 *
 * List<List<int32_t>>:
 * Length : 3
 * Offsets : 0, 2, 5, 10
 * Children :
 *    List<int32_t>:
 *    Length : 10
 *    Offsets : 0, 2, 4, 7, 10, 13, 15, 17, 19, 21, 23
 *       Children :
 *           2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 17, 18, 17, 18, 17, 18
 *
 * Final column, doing gather([0, 2])
 *
 * {
 *   {
 *      {2, 3}, {4, 5}
 *   },
 *   {
 *      {15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}
 *   }
 * }
 *
 * List<List<int32_t>>:
 * Length : 2
 * Offsets : 0, 2, 7
 * Children :
 *    List<int32_t>:
 *    Length : 7
 *    Offsets : 0, 2, 4, 6, 8, 10, 12, 14
 *       Children :
 *          2, 3, 4, 5, 15, 16, 17, 18, 17, 18, 17, 18, 17, 18
 * @endcode
 *
 * @tparam MapItType Iterator type to access the incoming column.
 * @tparam NullifyOutOfBounds Nullify values in `gather_map` that are out of bounds
 * @param source_column View into the column to gather from
 * @param gather_map Iterator access to the gather map for `source_column` map
 * @param gather_map_size Size of the gather map.
 * @param stream CUDA stream on which to execute kernels
 * @param mr Memory resource to use for all allocations
 *
 * @returns The gather_data struct needed to construct the gather map for the
 *          next level of recursion.
 */
template <bool NullifyOutOfBounds, typename MapItType>
gather_data make_gather_data(cudf::lists_column_view const& source_column,
                             MapItType gather_map,
                             size_type gather_map_size,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  return make_gather_data<NullifyOutOfBounds, MapItType>(
    source_column,
    gather_map,
    gather_map_size,
    rmm::device_uvector<int32_t>{0, stream, mr},
    stream,
    mr);
}

/**
 * @brief Gather a list column from a hierarchy of list columns.
 *
 * The recursion continues from here at least 1 level further.
 *
 * @param list View into the list column to gather from
 * @param gd The gather_data needed to construct a gather map iterator for this level
 * @param stream CUDA stream on which to execute kernels
 * @param mr Memory resource to use for all allocations
 *
 * @returns column with elements gathered based on `gather_data`
 */
std::unique_ptr<column> gather_list_nested(lists_column_view const& list,
                                           gather_data& gd,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @brief Gather a leaf column from a hierarchy of list columns.
 *
 * The recursion terminates here.
 *
 * @param column View into the column to gather from
 * @param gd The gather_data needed to construct a gather map iterator for this level
 * @param stream CUDA stream on which to execute kernels
 * @param mr Memory resource to use for all allocations
 *
 * @returns column with elements gathered based on `gather_data`
 */
std::unique_ptr<column> gather_list_leaf(column_view const& column,
                                         gather_data const& gd,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::segmented_gather(lists_column_view const& source_column,
 *                                        lists_column_view const& gather_map_list,
 *                                        out_of_bounds_policy bounds_policy,
 *                                        rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream on which to execute kernels
 */
std::unique_ptr<column> segmented_gather(lists_column_view const& source_column,
                                         lists_column_view const& gather_map_list,
                                         out_of_bounds_policy bounds_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace lists
}  // namespace cudf
