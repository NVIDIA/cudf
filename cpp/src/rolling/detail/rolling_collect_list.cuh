/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/extrema.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {

/**
 * @brief Creates the offsets child of the result of the `COLLECT_LIST` window aggregation
 *
 * Given the input column, the preceding/following window bounds, and `min_periods`,
 * the sizes of each list row may be computed. These values can then be used to
 * calculate the offsets for the result of `COLLECT_LIST`.
 *
 * Note: If `min_periods` exceeds the number of observations for a window, the size
 * is set to `0` (since the result is `null`).
 */
template <typename PrecedingIter, typename FollowingIter>
std::unique_ptr<column> create_collect_offsets(size_type input_size,
                                               PrecedingIter preceding_begin,
                                               FollowingIter following_begin,
                                               size_type min_periods,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  // Materialize offsets column.
  auto static constexpr size_data_type = data_type{type_to_id<size_type>()};
  auto sizes                           = make_fixed_width_column(size_data_type,
                                       input_size,
                                       mask_state::UNALLOCATED,
                                       stream,
                                       cudf::get_current_device_resource_ref());
  auto mutable_sizes                   = sizes->mutable_view();

  // Consider the following preceding/following values:
  //    preceding = [1,2,2,2,2]
  //    following = [1,1,1,1,0]
  // The sum of the vectors should yield the window sizes:
  //  prec + foll = [2,3,3,3,2]
  //
  // If min_periods=2, all rows have at least `min_periods` observations.
  // But if min_periods=3, rows at indices 0 and 4 have too few observations, and must return
  // null. The sizes at these positions must be 0, i.e.
  //  prec + foll = [0,3,3,3,0]
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    preceding_begin,
                    preceding_begin + input_size,
                    following_begin,
                    mutable_sizes.begin<size_type>(),
                    cuda::proclaim_return_type<size_type>(
                      [min_periods] __device__(auto const preceding, auto const following) {
                        return (preceding + following) < min_periods ? 0 : (preceding + following);
                      }));

  // Convert `sizes` to an offsets column, via inclusive_scan():
  auto offsets_column = std::get<0>(cudf::detail::make_offsets_child_column(
    sizes->view().begin<size_type>(), sizes->view().end<size_type>(), stream, mr));
  return offsets_column;
}

/**
 * @brief Generate mapping of each row in the COLLECT_LIST result's child column
 * to the index of the row it belongs to.
 *
 *  If
 *         input col == [A,B,C,D,E]
 *    and  preceding == [1,2,2,2,2],
 *    and  following == [1,1,1,1,0],
 *  then,
 *        collect result       == [ [A,B], [A,B,C], [B,C,D], [C,D,E], [D,E] ]
 *   i.e. result offset column == [0,2,5,8,11,13],
 *    and result child  column == [A,B,A,B,C,B,C,D,C,D,E,D,E].
 *  Mapping back to `input`    == [0,1,0,1,2,1,2,3,2,3,4,3,4]
 */
std::unique_ptr<column> get_list_child_to_list_row_mapping(cudf::column_view const& offsets,
                                                           cuda::stream_ref stream);

/**
 * @brief Create gather map to generate the child column of the result of
 * the `COLLECT_LIST` window aggregation.
 */
template <typename PrecedingIter>
std::unique_ptr<column> create_collect_gather_map(column_view const& child_offsets,
                                                  column_view const& per_row_mapping,
                                                  PrecedingIter preceding_iter,
                                                  cuda::stream_ref stream)
{
  auto gather_map = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                            per_row_mapping.size(),
                                            mask_state::UNALLOCATED,
                                            stream,
                                            cudf::get_current_device_resource_ref());
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{per_row_mapping.size()},
    gather_map->mutable_view().template begin<size_type>(),
    cuda::proclaim_return_type<size_type>(
      [d_offsets =
         child_offsets.template begin<int32_t>(),  // E.g.     [0,   2,     5,     8,     11, 13]
       d_groups =
         per_row_mapping.template begin<size_type>(),  // E.g. [0,0, 1,1,1, 2,2,2, 3,3,3, 4,4]
       d_prev = preceding_iter] __device__(auto i) {
        auto group              = d_groups[i];
        auto group_start_offset = d_offsets[group];
        auto relative_index     = i - group_start_offset;

        return (group - d_prev[group] + 1) + relative_index;
      }));
  return gather_map;
}

/**
 * @brief Count null entries in result of COLLECT_LIST.
 */
size_type count_child_nulls(column_view const& input,
                            std::unique_ptr<column> const& gather_map,
                            cuda::stream_ref stream);

/**
 * @brief Purge entries for null inputs from gather_map, and adjust offsets.
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> purge_null_entries(
  column_view const& input,
  column_view const& gather_map,
  column_view const& offsets,
  size_type num_child_nulls,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

template <typename PrecedingIter, typename FollowingIter>
std::unique_ptr<column> rolling_collect_list(column_view const& input,
                                             column_view const& default_outputs,
                                             PrecedingIter preceding_begin,
                                             FollowingIter following_begin,
                                             size_type min_periods,
                                             null_policy null_handling,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(default_outputs.is_empty(),
               "COLLECT_LIST window function does not support default values.");

  if (input.is_empty()) return empty_like(input);

  // Materialize collect list's offsets.
  auto offsets =
    create_collect_offsets(input.size(), preceding_begin, following_begin, min_periods, stream, mr);

  // Map each element of the collect() result's child column
  // to the index where it appears in the input.
  auto per_row_mapping = get_list_child_to_list_row_mapping(offsets->view(), stream);

  // Generate gather map to produce the collect() result's child column.
  auto gather_map =
    create_collect_gather_map(offsets->view(), per_row_mapping->view(), preceding_begin, stream);

  // If gather_map collects null elements, and null_policy == EXCLUDE,
  // those elements must be filtered out, and offsets recomputed.
  if (null_handling == null_policy::EXCLUDE && input.has_nulls()) {
    auto num_child_nulls = count_child_nulls(input, gather_map, stream);
    if (num_child_nulls != 0) {
      std::tie(gather_map, offsets) =
        purge_null_entries(input, *gather_map, *offsets, num_child_nulls, stream, mr);
    }
  }

  // gather(), to construct child column.
  auto gather_output = cudf::detail::gather(table_view{std::vector<column_view>{input}},
                                            gather_map->view(),
                                            cudf::out_of_bounds_policy::DONT_CHECK,
                                            cudf::negative_index_policy::NOT_ALLOWED,
                                            stream,
                                            mr);

  auto [null_mask, null_count] = valid_if(
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{input.size()},
    [preceding_begin, following_begin, min_periods] __device__(auto i) {
      return (preceding_begin[i] + following_begin[i]) >= min_periods;
    },
    stream,
    mr);

  return make_lists_column(input.size(),
                           std::move(offsets),
                           std::move(gather_output->release()[0]),
                           null_count,
                           std::move(null_mask));
}

}  // namespace detail
}  // namespace cudf
