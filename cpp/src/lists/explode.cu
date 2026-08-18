/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/explode.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <memory>
#include <type_traits>

namespace cudf {
namespace detail {

// explode column gather map uses cudf::out_of_bounds_policy::NULLIFY to
// fill nulls where there are invalid indices
constexpr size_type InvalidIndex = -1;

namespace {

std::unique_ptr<table> build_table(
  table_view const& input_table,
  size_type const explode_column_idx,
  column_view const& sliced_child,
  cudf::device_span<size_type const> gather_map,
  cuda::std::optional<cudf::device_span<size_type const>> explode_col_gather_map,
  cuda::std::optional<rmm::device_uvector<size_type>> position_array,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto select_iter = thrust::make_transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0},
    [explode_column_idx](size_type i) { return i >= explode_column_idx ? i + 1 : i; });

  auto gathered_table =
    detail::gather(input_table.select(select_iter, select_iter + input_table.num_columns() - 1),
                   gather_map.begin(),
                   gather_map.end(),
                   cudf::out_of_bounds_policy::DONT_CHECK,
                   stream,
                   mr);

  std::vector<std::unique_ptr<column>> columns = gathered_table->release();

  columns.insert(columns.begin() + explode_column_idx,
                 explode_col_gather_map
                   ? std::move(detail::gather(table_view({sliced_child}),
                                              explode_col_gather_map->begin(),
                                              explode_col_gather_map->end(),
                                              cudf::out_of_bounds_policy::NULLIFY,
                                              stream,
                                              mr)
                                 ->release()[0])
                   : std::make_unique<column>(sliced_child, stream, mr));

  if (position_array) {
    size_type position_size = position_array->size();
    // build the null mask for position based on invalid entries in gather map
    auto nullmask = explode_col_gather_map ? valid_if(
                                               explode_col_gather_map->begin(),
                                               explode_col_gather_map->end(),
                                               [] __device__(auto i) { return i != InvalidIndex; },
                                               stream,
                                               mr)
                                           : std::pair<rmm::device_buffer, size_type>{
                                               rmm::device_buffer(0, stream), size_type{0}};

    columns.insert(columns.begin() + explode_column_idx,
                   std::make_unique<column>(data_type(type_to_id<size_type>()),
                                            position_size,
                                            position_array->release(),
                                            std::move(nullmask.first),
                                            nullmask.second));
  }

  return std::make_unique<table>(std::move(columns));
}
}  // namespace

std::unique_ptr<table> explode(table_view const& input_table,
                               size_type const explode_column_idx,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  lists_column_view explode_col{input_table.column(explode_column_idx)};
  auto sliced_child = explode_col.get_sliced_child(stream);
  rmm::device_uvector<size_type> gather_map(sliced_child.size(), stream);

  // Sliced columns may require rebasing of the offsets.
  auto offsets = explode_col.offsets_begin();
  // offsets + 1 here to skip the 0th offset, which removes a - 1 operation later.
  auto offsets_minus_one = thrust::make_transform_iterator(
    cuda::std::next(offsets), cuda::proclaim_return_type<size_type>([offsets] __device__(auto i) {
      return (i - offsets[0]) - 1;
    }));
  auto counting_iter = cuda::counting_iterator<cudf::size_type>{0};

  // This looks like an off-by-one bug, but what is going on here is that we need to reduce each
  // result from `lower_bound` by 1 to build the correct gather map. This can be accomplished by
  // skipping the first entry and using the result of `lower_bound` directly.
  thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      offsets_minus_one,
                      offsets_minus_one + explode_col.size(),
                      counting_iter,
                      counting_iter + gather_map.size(),
                      gather_map.begin());

  return build_table(input_table,
                     explode_column_idx,
                     sliced_child,
                     gather_map,
                     cuda::std::nullopt,
                     cuda::std::nullopt,
                     stream,
                     mr);
}

std::unique_ptr<table> explode_position(table_view const& input_table,
                                        size_type const explode_column_idx,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  lists_column_view explode_col{input_table.column(explode_column_idx)};
  auto sliced_child = explode_col.get_sliced_child(stream);
  rmm::device_uvector<size_type> gather_map(sliced_child.size(), stream);

  // Sliced columns may require rebasing of the offsets.
  auto offsets = explode_col.offsets_begin();
  // offsets + 1 here to skip the 0th offset, which removes a - 1 operation later.
  auto offsets_minus_one = thrust::make_transform_iterator(
    offsets + 1, cuda::proclaim_return_type<size_type>([offsets] __device__(auto i) {
      return (i - offsets[0]) - 1;
    }));
  auto counting_iter = cuda::counting_iterator<cudf::size_type>{0};

  rmm::device_uvector<size_type> pos(sliced_child.size(), stream, mr);

  // This looks like an off-by-one bug, but what is going on here is that we need to reduce each
  // result from `lower_bound` by 1 to build the correct gather map. This can be accomplished by
  // skipping the first entry and using the result of `lower_bound` directly.
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    counting_iter,
    counting_iter + gather_map.size(),
    gather_map.begin(),
    cuda::proclaim_return_type<size_type>([position_array = pos.data(),
                                           offsets_minus_one,
                                           offsets,
                                           offset_size =
                                             explode_col.size()] __device__(auto idx) -> size_type {
      auto lb_idx = cuda::std::distance(
        offsets_minus_one,
        thrust::lower_bound(thrust::seq, offsets_minus_one, offsets_minus_one + offset_size, idx));
      position_array[idx] = idx - (offsets[lb_idx] - offsets[0]);
      return lb_idx;
    }));

  return build_table(input_table,
                     explode_column_idx,
                     sliced_child,
                     gather_map,
                     cuda::std::nullopt,
                     std::move(pos),
                     stream,
                     mr);
}

std::unique_ptr<table> explode_outer(table_view const& input_table,
                                     size_type const explode_column_idx,
                                     bool include_position,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  lists_column_view explode_col{input_table.column(explode_column_idx)};
  auto sliced_child  = explode_col.get_sliced_child(stream);
  auto counting_iter = cuda::counting_iterator<cudf::size_type>{0};
  auto offsets       = explode_col.offsets_begin();

  // number of nulls or empty lists found so far in the explode column
  rmm::device_uvector<size_type> null_or_empty_offset(explode_col.size(), stream);

  auto null_or_empty = thrust::make_transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::proclaim_return_type<size_type>(
      [offsets, offsets_size = explode_col.size() - 1] __device__(int idx) {
        return (idx > offsets_size || (offsets[idx + 1] != offsets[idx])) ? 0 : 1;
      }));
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         null_or_empty,
                         null_or_empty + explode_col.size(),
                         null_or_empty_offset.begin());

  auto null_or_empty_count =
    null_or_empty_offset.size() > 0 ? null_or_empty_offset.back_element(stream) : 0;
  if (null_or_empty_count == 0) {
    // performance penalty to run the below loop if there are no nulls or empty lists.
    // run simple explode instead
    return include_position ? detail::explode_position(input_table, explode_column_idx, stream, mr)
                            : detail::explode(input_table, explode_column_idx, stream, mr);
  }

  auto gather_map_size = sliced_child.size() + null_or_empty_count;

  rmm::device_uvector<size_type> gather_map(gather_map_size, stream);
  rmm::device_uvector<size_type> explode_col_gather_map(gather_map_size, stream);
  rmm::device_uvector<size_type> pos(include_position ? gather_map_size : 0, stream, mr);

  // offsets + 1 here to skip the 0th offset, which removes a - 1 operation later.
  auto offsets_minus_one = thrust::make_transform_iterator(
    cuda::std::next(offsets), cuda::proclaim_return_type<size_type>([offsets] __device__(auto i) {
      return (i - offsets[0]) - 1;
    }));

  auto fill_gather_maps = [offsets_minus_one,
                           gather_map_p             = gather_map.begin(),
                           explode_col_gather_map_p = explode_col_gather_map.begin(),
                           position_array           = pos.begin(),
                           sliced_child_size        = sliced_child.size(),
                           null_or_empty_offset_p   = null_or_empty_offset.begin(),
                           include_position,
                           offsets,
                           null_or_empty,
                           offset_size = explode_col.offsets().size() - 1] __device__(auto idx) {
    if (idx < sliced_child_size) {
      auto lb_idx = cuda::std::distance(
        offsets_minus_one,
        thrust::lower_bound(
          thrust::seq, offsets_minus_one, offsets_minus_one + (offset_size), idx));
      auto index_to_write                      = null_or_empty_offset_p[lb_idx] + idx;
      gather_map_p[index_to_write]             = lb_idx;
      explode_col_gather_map_p[index_to_write] = idx;
      if (include_position) {
        position_array[index_to_write] = idx - (offsets[lb_idx] - offsets[0]);
      }
    }
    if (null_or_empty[idx]) {
      auto invalid_index          = null_or_empty_offset_p[idx] == 0
                                      ? offsets[idx]
                                      : offsets[idx] + null_or_empty_offset_p[idx] - 1;
      gather_map_p[invalid_index] = idx;

      explode_col_gather_map_p[invalid_index] = InvalidIndex;
      if (include_position) { position_array[invalid_index] = 0; }
    }
  };

  // we need to do this loop at least explode_col times or we may not properly fill in null and
  // empty entries.
  auto loop_count = std::max(sliced_child.size(), explode_col.size());

  // Fill in gather map with all the child column's entries
  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   counting_iter,
                   counting_iter + loop_count,
                   fill_gather_maps);

  return build_table(
    input_table,
    explode_column_idx,
    sliced_child,
    gather_map,
    explode_col_gather_map,
    include_position ? std::move(pos) : cuda::std::optional<rmm::device_uvector<size_type>>{},
    stream,
    mr);
}

}  // namespace detail

/**
 * @copydoc cudf::explode(table_view const&, size_type, cuda::stream_ref,
 * rmm::device_async_resource_ref)
 */
std::unique_ptr<table> explode(table_view const& input_table,
                               size_type explode_column_idx,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input_table.column(explode_column_idx).type().id() == type_id::LIST,
               "Unsupported non-list column");
  return detail::explode(input_table, explode_column_idx, stream, mr);
}

/**
 * @copydoc cudf::explode_position(table_view const&, size_type, cuda::stream_ref,
 * rmm::device_async_resource_ref)
 */
std::unique_ptr<table> explode_position(table_view const& input_table,
                                        size_type explode_column_idx,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input_table.column(explode_column_idx).type().id() == type_id::LIST,
               "Unsupported non-list column");
  return detail::explode_position(input_table, explode_column_idx, stream, mr);
}

/**
 * @copydoc cudf::explode_outer(table_view const&, size_type, cuda::stream_ref,
 * rmm::device_async_resource_ref)
 */
std::unique_ptr<table> explode_outer(table_view const& input_table,
                                     size_type explode_column_idx,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input_table.column(explode_column_idx).type().id() == type_id::LIST,
               "Unsupported non-list column");
  return detail::explode_outer(input_table, explode_column_idx, false, stream, mr);
}

/**
 * @copydoc cudf::explode_outer_position(table_view const&, size_type,
 * cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<table> explode_outer_position(table_view const& input_table,
                                              size_type explode_column_idx,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(input_table.column(explode_column_idx).type().id() == type_id::LIST,
               "Unsupported non-list column");
  return detail::explode_outer(input_table, explode_column_idx, true, stream, mr);
}

}  // namespace cudf
