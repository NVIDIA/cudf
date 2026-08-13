/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/lists/detail/scatter_helper.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cinttypes>

namespace cudf {
namespace lists {
namespace detail {

template <typename IndexIterator>
rmm::device_uvector<unbound_list_view> list_vector_from_column(
  unbound_list_view::label_type label,
  cudf::lists_column_device_view const& lists_column,
  IndexIterator index_begin,
  IndexIterator index_end,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto n_rows = cuda::std::distance(index_begin, index_end);

  auto vector = rmm::device_uvector<unbound_list_view>(n_rows, stream, mr);

  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    index_begin,
                    index_end,
                    vector.begin(),
                    cuda::proclaim_return_type<unbound_list_view>(
                      [label, lists_column] __device__(size_type row_index) {
                        return unbound_list_view{label, lists_column, row_index};
                      }));

  return vector;
}

/**
 * @brief General implementation of scattering into list column
 *
 * Scattering `source` into `target` according to `scatter_map`.
 * The view order of `source` and `target` can be specified by
 * `source_vector` and `target_vector` respectively.
 *
 * @tparam MapIterator must produce index values within the target column.
 *
 * @param source_vector A vector of `unbound_list_view` into source column
 * @param target_vector A vector of `unbound_list_view` into target column
 * @param scatter_map_begin Start iterator of scatter map
 * @param scatter_map_end End iterator of scatter map
 * @param source Source column view
 * @param target Target column view
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column.
 */
template <typename MapIterator>
std::unique_ptr<column> scatter_impl(rmm::device_uvector<unbound_list_view> const& source_vector,
                                     rmm::device_uvector<unbound_list_view>& target_vector,
                                     MapIterator scatter_map_begin,
                                     MapIterator scatter_map_end,
                                     column_view const& source,
                                     column_view const& target,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(have_same_types(source, target), "Mismatched column types.");

  auto const child_column_type = lists_column_view(target).child().type();

  // Scatter.
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  source_vector.begin(),
                  source_vector.end(),
                  scatter_map_begin,
                  target_vector.begin());

  auto const source_lists_column_view =
    lists_column_view(source);  // Checks that this is a list column.
  auto const target_lists_column_view =
    lists_column_view(target);  // Checks that target is a list column.

  auto list_size_begin = thrust::make_transform_iterator(
    target_vector.begin(),
    cuda::proclaim_return_type<size_type>([] __device__(unbound_list_view l) { return l.size(); }));
  auto offsets_column = std::get<0>(cudf::detail::make_offsets_child_column(
    list_size_begin, list_size_begin + target.size(), stream, mr));

  auto child_column = build_lists_child_column_recursive(child_column_type,
                                                         target_vector,
                                                         offsets_column->view(),
                                                         source_lists_column_view,
                                                         target_lists_column_view,
                                                         stream,
                                                         mr);

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));
  children.emplace_back(std::move(child_column));
  auto null_mask = target.has_nulls() ? cudf::detail::copy_bitmask(target, stream, mr)
                                      : rmm::device_buffer{0, stream, mr};

  // The output column from this function only has null masks copied from the target columns.
  // That is still not a correct final null mask for the scatter result.
  // In addition, that null mask may overshadow the non-null rows (lists) scattered from the source
  // column. Thus, avoid using `cudf::make_lists_column` since it calls `purge_nonempty_nulls`.
  return std::make_unique<column>(data_type{type_id::LIST},
                                  target.size(),
                                  rmm::device_buffer{},
                                  std::move(null_mask),
                                  target.null_count(),
                                  std::move(children));
}

/**
 * @brief Scatters lists into a copy of the target column
 * according to a scatter map.
 *
 * The scatter is performed according to the scatter iterator such that row
 * `scatter_map[i]` of the output column is replaced by the source list-row.
 * All other rows of the output column equal corresponding rows of the target table.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * The caller must update the null mask in the output column.
 *
 * @tparam MapIterator must produce index values within the target column.
 *
 * @param source Source column view
 * @param scatter_map_begin Start iterator of scatter map
 * @param scatter_map_end End iterator of scatter map
 * @param target Target column view
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column.
 */
template <typename MapIterator>
std::unique_ptr<column> scatter(column_view const& source,
                                MapIterator scatter_map_begin,
                                MapIterator scatter_map_end,
                                column_view const& target,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto const num_rows = target.size();
  if (num_rows == 0) { return cudf::empty_like(target); }

  auto const source_device_view = column_device_view::create(source, stream);
  auto const scatter_map_size   = cuda::std::distance(scatter_map_begin, scatter_map_end);
  auto const source_vector =
    list_vector_from_column(unbound_list_view::label_type::SOURCE,
                            cudf::lists_column_device_view(*source_device_view),
                            cuda::counting_iterator<size_type>{0},
                            cuda::counting_iterator{static_cast<size_type>(scatter_map_size)},
                            stream,
                            mr);

  auto const target_device_view = column_device_view::create(target, stream);
  auto target_vector            = list_vector_from_column(unbound_list_view::label_type::TARGET,
                                               cudf::lists_column_device_view(*target_device_view),
                                               cuda::counting_iterator<size_type>{0},
                                               cuda::counting_iterator<size_type>{num_rows},
                                               stream,
                                               mr);

  return scatter_impl(
    source_vector, target_vector, scatter_map_begin, scatter_map_end, source, target, stream, mr);
}

/**
 * @brief Scatters list scalar (a single row) into a copy of the target column
 * according to a scatter map.
 *
 * Returns a copy of the target column where every row specified in the `scatter_map`
 * is replaced by the row value.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * The caller must update the null mask in the output column.
 *
 * @tparam MapIterator must produce index values within the target column.
 *
 * @param slr Source scalar, specifying row data
 * @param scatter_map_begin Start iterator of scatter map
 * @param scatter_map_end End iterator of scatter map
 * @param target Target column view
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column.
 */
template <typename MapIterator>
std::unique_ptr<column> scatter(scalar const& slr,
                                MapIterator scatter_map_begin,
                                MapIterator scatter_map_end,
                                column_view const& target,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto const num_rows = target.size();
  if (num_rows == 0) { return cudf::empty_like(target); }

  auto lv                      = static_cast<list_scalar const*>(&slr);
  bool slr_valid               = slr.is_valid(stream);
  rmm::device_buffer null_mask = slr_valid
                                   ? cudf::create_null_mask(1, mask_state::UNALLOCATED, stream, mr)
                                   : cudf::create_null_mask(1, mask_state::ALL_NULL, stream, mr);
  auto offset_column =
    make_numeric_column(data_type{type_to_id<size_type>()}, 2, mask_state::UNALLOCATED, stream, mr);
  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   offset_column->mutable_view().begin<size_type>(),
                   offset_column->mutable_view().end<size_type>(),
                   0,
                   lv->view().size());
  auto wrapped = column_view(data_type{type_id::LIST},
                             1,
                             nullptr,
                             static_cast<bitmask_type const*>(null_mask.data()),
                             slr_valid ? 0 : 1,
                             0,
                             {offset_column->view(), lv->view()});

  auto const source_device_view = column_device_view::create(wrapped, stream);
  auto const scatter_map_size   = cuda::std::distance(scatter_map_begin, scatter_map_end);
  auto const source_vector =
    list_vector_from_column(unbound_list_view::label_type::SOURCE,
                            cudf::lists_column_device_view(*source_device_view),
                            cuda::make_constant_iterator<size_type>(0),
                            cuda::make_constant_iterator<size_type>(0) + scatter_map_size,
                            stream,
                            mr);

  auto const target_device_view = column_device_view::create(target, stream);
  auto target_vector            = list_vector_from_column(unbound_list_view::label_type::TARGET,
                                               cudf::lists_column_device_view(*target_device_view),
                                               cuda::counting_iterator<size_type>{0},
                                               cuda::counting_iterator<size_type>{num_rows},
                                               stream,
                                               mr);

  return scatter_impl(
    source_vector, target_vector, scatter_map_begin, scatter_map_end, wrapped, target, stream, mr);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
