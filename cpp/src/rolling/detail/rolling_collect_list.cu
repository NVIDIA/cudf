/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rolling_collect_list.cuh"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {

/**
 * @see cudf::detail::get_list_child_to_list_row_mapping
 */
std::unique_ptr<column> get_list_child_to_list_row_mapping(cudf::column_view const& offsets,
                                                           rmm::cuda_stream_view stream)
{
  // First, scatter the count for each repeated offset (except the first and last),
  // into a column of N `0`s, where N == number of child rows.
  // For example:
  //   offsets        == [0, 2, 5, 8, 11, 13]
  //   scatter result == [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  //
  // An example with empty list row at index 2:
  //   offsets        == [0, 2, 5, 5, 8, 11, 13]
  //   scatter result == [0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0]
  //

  auto const num_child_rows{
    cudf::detail::get_value<size_type>(offsets, offsets.size() - 1, stream)};
  auto per_row_mapping       = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                                 num_child_rows,
                                                 mask_state::UNALLOCATED,
                                                 stream,
                                                 cudf::get_current_device_resource_ref());
  auto per_row_mapping_begin = per_row_mapping->mutable_view().template begin<size_type>();
  thrust::fill_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 per_row_mapping_begin,
                 num_child_rows,
                 0);

  auto const begin = cuda::counting_iterator<size_type>{0};
  thrust::scatter_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     begin,
                     begin + offsets.size() - 1,
                     offsets.begin<size_type>(),
                     begin,  // stencil iterator
                     per_row_mapping_begin,
                     [offset = offsets.begin<size_type>()] __device__(auto i) {
                       return offset[i] != offset[i + 1];
                     });  // [0,0,1,0,0,3,...]

  // Next, generate mapping with inclusive_scan(max) on the scatter result.
  // For the example above:
  //   scatter result == [0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0]
  //   inclusive_scan == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
  //
  // For the case with an empty list at index 2:
  //   scatter result == [0, 0, 1, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0]
  //   inclusive_scan == [0, 0, 1, 1, 1, 3, 3, 3, 4, 4, 4, 5, 5]
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         per_row_mapping_begin,
                         per_row_mapping_begin + num_child_rows,
                         per_row_mapping_begin,
                         cuda::maximum{});
  return per_row_mapping;
}

/**
 * @see cudf::detail::count_child_nulls
 */
size_type count_child_nulls(column_view const& input,
                            std::unique_ptr<column> const& gather_map,
                            rmm::cuda_stream_view stream)
{
  auto input_device_view = column_device_view::create(input, stream);

  auto input_row_is_null = [d_input = *input_device_view] __device__(auto i) {
    return d_input.is_null_nocheck(i);
  };

  return cudf::detail::count_if(gather_map->view().begin<size_type>(),
                                gather_map->view().end<size_type>(),
                                input_row_is_null,
                                stream);
}

/**
 * @see cudf::detail::rolling_collect_list
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> purge_null_entries(
  column_view const& input,
  column_view const& gather_map,
  column_view const& offsets,
  size_type num_child_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto input_device_view = column_device_view::create(input, stream);

  auto input_row_not_null = [d_input = *input_device_view] __device__(auto i) -> bool {
    return d_input.is_valid_nocheck(i);
  };

  // Purge entries in gather_map that correspond to null input.
  auto new_gather_map = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                                gather_map.size() - num_child_nulls,
                                                mask_state::UNALLOCATED,
                                                stream,
                                                mr);
  cudf::detail::copy_if_async(gather_map.template begin<size_type>(),
                              gather_map.template end<size_type>(),
                              new_gather_map->mutable_view().template begin<size_type>(),
                              input_row_not_null,
                              stream);

  // Recalculate offsets after null entries are purged.
  auto new_sizes = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                           input.size(),
                                           mask_state::UNALLOCATED,
                                           stream,
                                           cudf::get_current_device_resource_ref());

  thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   new_sizes->mutable_view().template begin<size_type>(),
                   new_sizes->mutable_view().template end<size_type>(),
                   [d_gather_map  = gather_map.template begin<size_type>(),
                    d_old_offsets = offsets.template begin<size_type>(),
                    input_row_not_null] __device__(auto i) {
                     return thrust::count_if(thrust::seq,
                                             d_gather_map + d_old_offsets[i],
                                             d_gather_map + d_old_offsets[i + 1],
                                             input_row_not_null);
                   });

  auto new_offsets = std::get<0>(
    cudf::detail::make_offsets_child_column(new_sizes->view().template begin<size_type>(),
                                            new_sizes->view().template end<size_type>(),
                                            stream,
                                            mr));

  return std::make_pair<std::unique_ptr<column>, std::unique_ptr<column>>(std::move(new_gather_map),
                                                                          std::move(new_offsets));
}

}  // namespace detail
}  // namespace cudf
