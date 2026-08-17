/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/lists/detail/extract.hpp>
#include <cudf/lists/detail/gather.cuh>
#include <cudf/lists/extract.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>

#include <limits>

namespace cudf {
namespace lists {
namespace detail {
namespace {

/**
 * @brief Helper to construct a column of indices, for use with `segmented_gather()`.
 *
 * When indices are specified as a column, e.g. `{5, -4, 3, -2, 1, null}`,
 * the column returned is:                      `{5, -4, 3, -2, 1, MAX_SIZE_TYPE}`.
 * All null indices are replaced with `MAX_SIZE_TYPE = numeric_limits<size_type>::max()`.
 *
 * The returned column can then be used to construct a lists column, for use
 * with `segmented_gather()`.
 */
std::unique_ptr<cudf::column> make_index_child(column_view const& indices,
                                               size_type,
                                               rmm::cuda_stream_view stream)
{
  // New column, near identical to `indices`, except with null values replaced.
  // `segmented_gather()` on a null index should produce a null row.
  if (not indices.nullable()) {
    return std::make_unique<column>(indices, stream, cudf::get_current_device_resource_ref());
  }

  auto const d_indices = column_device_view::create(indices, stream);
  // Replace null indices with MAX_SIZE_TYPE, so that gather() returns null for them.
  auto const null_replaced_iter_begin =
    cudf::detail::make_null_replacement_iterator(*d_indices, std::numeric_limits<size_type>::max());
  auto index_child = make_numeric_column(data_type{type_to_id<size_type>()},
                                         indices.size(),
                                         mask_state::UNALLOCATED,
                                         stream,
                                         cudf::get_current_device_resource_ref());
  thrust::copy_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 null_replaced_iter_begin,
                 indices.size(),
                 index_child->mutable_view().begin<size_type>());
  return index_child;
}

/**
 * @brief Helper to construct a column of indices, for use with `segmented_gather()`.
 *
 * When indices are specified as a size_type, e.g. `7`,
 * the column returned is: `{ 7, 7, 7, 7, 7 }`.
 *
 * The returned column can then be used to construct a lists column, for use
 * with `segmented_gather()`.
 */
std::unique_ptr<cudf::column> make_index_child(size_type index,
                                               size_type num_rows,
                                               rmm::cuda_stream_view stream)
{
  auto index_child =  // [index, index, index, ..., index]
    make_numeric_column(data_type{type_to_id<size_type>()},
                        num_rows,
                        mask_state::UNALLOCATED,
                        stream,
                        cudf::get_current_device_resource_ref());
  thrust::fill_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 index_child->mutable_view().begin<size_type>(),
                 num_rows,
                 index);
  return index_child;
}

/**
 * @brief Helper to construct offsets column for an index vector.
 *
 * Constructs the sequence: `{ 0, 1, 2, 3, ... num_lists + 1}`.
 * This may be used to construct an "index-list" column, where each list row
 * has a single element.
 */
std::unique_ptr<cudf::column> make_index_offsets(size_type num_lists, rmm::cuda_stream_view stream)
{
  return cudf::detail::sequence(
    num_lists + 1,
    cudf::scalar_type_t<int32_t>(0, true, stream, cudf::get_current_device_resource_ref()),
    stream,
    cudf::get_current_device_resource_ref());
}

}  // namespace

/**
 * @copydoc cudf::lists::extract_list_element
 * @tparam index_t The type used to specify the index values (either column_view or size_type)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename index_t>
std::unique_ptr<column> extract_list_element_impl(lists_column_view lists_column,
                                                  index_t const& index,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto const num_lists = lists_column.size();
  if (num_lists == 0) { return empty_like(lists_column.child()); }

  // Given an index (or indices vector), an index lists column may be constructed,
  // with each list row having a single element.
  // E.g.
  // 1. If index = 7, index_lists_column = { {7}, {7}, {7}, {7}, ... }.
  // 2. If indices = {4, 3, 2, 1, null},
  //    index_lists_column = { {4}, {3}, {2}, {1}, {MAX_SIZE_TYPE} }.

  auto const index_lists_column = make_lists_column(num_lists,
                                                    make_index_offsets(num_lists, stream),
                                                    make_index_child(index, num_lists, stream),
                                                    0,
                                                    {});

  // We want the output of `segmented_gather` to be a lists column in which each list has exactly
  // one element, even for the null lists.
  // Thus, the input into `segmented_gather` should not be nullable.
  auto const lists_column_removed_null_mask = lists_column_view{
    column_view{data_type{type_id::LIST},
                lists_column.size(),
                nullptr,  // data
                nullptr,  // null_mask
                0,        // null_count
                lists_column.offset(),
                std::vector<column_view>{lists_column.child_begin(), lists_column.child_end()}}};

  auto extracted_lists = segmented_gather(lists_column_removed_null_mask,
                                          index_lists_column->view(),
                                          out_of_bounds_policy::NULLIFY,
                                          stream,
                                          mr);

  auto output =
    std::move(extracted_lists->release().children[lists_column_view::child_column_index]);
  if (!lists_column.has_nulls()) { return output; }

  // The input lists column may have non-empty nulls if it is nullable, although this is rare.
  // In such cases, the extracted elements corresponding to these non-empty nulls may not be null.
  // Thus, we need to superimpose nulls from the input column into the output to make sure each
  // input null list always results in a null output row.
  return cudf::structs::detail::superimpose_and_sanitize_nulls(
    lists_column.null_mask(), lists_column.null_count(), std::move(output), stream, mr);
}

/**
 * @copydoc cudf::lists::extract_list_element
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             size_type const index,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  return detail::extract_list_element_impl(lists_column, index, stream, mr);
}

std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             column_view const& indices,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  return detail::extract_list_element_impl(lists_column, indices, stream, mr);
}

}  // namespace detail

/**
 * @copydoc cudf::lists::extract_list_element(lists_column_view const&,
 *                                            size_type,
 *                                            rmm::device_async_resource_ref)
 */
std::unique_ptr<column> extract_list_element(lists_column_view const& lists_column,
                                             size_type index,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_list_element(lists_column, index, stream, mr);
}

/**
 * @copydoc cudf::lists::extract_list_element(lists_column_view const&,
 *                                            column_view const&,
 *                                            rmm::device_async_resource_ref)
 */
std::unique_ptr<column> extract_list_element(lists_column_view const& lists_column,
                                             column_view const& indices,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(indices.size() == lists_column.size(),
               "Index column must have as many elements as lists column.");
  return detail::extract_list_element(lists_column, indices, stream, mr);
}

}  // namespace lists
}  // namespace cudf
