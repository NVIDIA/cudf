/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {
namespace {
/**
 * @brief Concatenate lists within the same row into one list, ignoring any null list during
 * concatenation.
 */
std::unique_ptr<column> concatenate_lists_ignore_null(column_view const& input,
                                                      bool build_null_mask,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.size();

  auto out_offsets = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);

  auto const d_out_offsets  = out_offsets->mutable_view().template begin<int32_t>();
  auto const d_row_offsets  = lists_column_view(input).offsets_begin();
  auto const d_list_offsets = lists_column_view(lists_column_view(input).child()).offsets_begin();

  // Concatenating the lists at the same row by converting the entry offsets from the child column
  // into row offsets of the root column. Those entry offsets are subtracted by the first entry
  // offset to output zero-based offsets.
  auto const iter = cuda::counting_iterator<size_type>{0};
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    iter,
                    iter + num_rows + 1,
                    d_out_offsets,
                    [d_row_offsets, d_list_offsets] __device__(auto const idx) {
                      auto const start_offset = d_list_offsets[d_row_offsets[0]];
                      return d_list_offsets[d_row_offsets[idx]] - start_offset;
                    });

  // The child column of the output lists column is just copied from the input column.
  auto out_entries = std::make_unique<column>(
    lists_column_view(lists_column_view(input).get_sliced_child(stream)).get_sliced_child(stream),
    stream,
    mr);

  auto [null_mask, null_count] = [&] {
    if (!build_null_mask)
      return std::pair(cudf::detail::copy_bitmask(input, stream, mr), input.null_count());

    // The output row will be null only if all lists on the input row are null.
    auto const lists_dv_ptr = column_device_view::create(lists_column_view(input).child(), stream);
    return cudf::detail::valid_if(
      iter,
      iter + num_rows,
      [d_row_offsets, lists_dv = *lists_dv_ptr, iter] __device__(auto const idx) {
        return thrust::any_of(
          thrust::seq,
          iter + d_row_offsets[idx],
          iter + d_row_offsets[idx + 1],
          [&] __device__(auto const list_idx) { return lists_dv.is_valid(list_idx); });
      },
      stream,
      mr);
  }();

  return make_lists_column(num_rows,
                           std::move(out_offsets),
                           std::move(out_entries),
                           null_count,
                           null_count > 0 ? std::move(null_mask) : rmm::device_buffer{});
}

/**
 * @brief Generate list offsets and list validities for the output lists column.
 *
 * This function is called only when (has_null_list == true and null_policy == NULLIFY_OUTPUT_ROW).
 */
std::pair<std::unique_ptr<column>, rmm::device_uvector<int8_t>>
generate_list_offsets_and_validities(column_view const& input,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.size();

  auto const lists_of_lists_dv_ptr = column_device_view::create(input, stream);
  auto const lists_dv_ptr   = column_device_view::create(lists_column_view(input).child(), stream);
  auto const d_row_offsets  = lists_column_view(input).offsets_begin();
  auto const d_list_offsets = lists_column_view(lists_column_view(input).child()).offsets_begin();

  // The array of int8_t stores validities for the output list elements.
  auto validities = rmm::device_uvector<int8_t>(num_rows, stream);

  // Compute output list sizes and validities.
  auto sizes_itr = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>([lists_of_lists_dv = *lists_of_lists_dv_ptr,
                                           lists_dv          = *lists_dv_ptr,
                                           d_row_offsets,
                                           d_list_offsets,
                                           d_validities =
                                             validities.begin()] __device__(auto const idx) {
      if (d_row_offsets[idx] == d_row_offsets[idx + 1]) {  // This is a null/empty row.
        d_validities[idx] = static_cast<int8_t>(lists_of_lists_dv.is_valid(idx));
        return size_type{0};
      }
      // The output row will not be null only if all lists on the input row are not null.
      auto const iter = cuda::counting_iterator<size_type>{0};
      auto const is_valid =
        thrust::all_of(thrust::seq,
                       iter + d_row_offsets[idx],
                       iter + d_row_offsets[idx + 1],
                       [&] __device__(auto const list_idx) { return lists_dv.is_valid(list_idx); });
      d_validities[idx] = static_cast<int8_t>(is_valid);
      if (!is_valid) { return size_type{0}; }

      // Compute size of the output list as sum of sizes of all lists in the current input row.
      return d_list_offsets[d_row_offsets[idx + 1]] - d_list_offsets[d_row_offsets[idx]];
    }));
  // Compute offsets from sizes.
  auto out_offsets = std::get<0>(
    cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + num_rows, stream, mr));

  return {std::move(out_offsets), std::move(validities)};
}

/**
 * @brief Gather entries from the input lists column, ignoring rows that have null list elements.
 *
 * This function is called only when (has_null_list == true and null_policy == NULLIFY_OUTPUT_ROW).
 */
std::unique_ptr<column> gather_list_entries(column_view const& input,
                                            column_view const& output_list_offsets,
                                            size_type num_rows,
                                            size_type num_output_entries,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  auto const child_col      = lists_column_view(input).child();
  auto const entry_col      = lists_column_view(child_col).child();
  auto const d_row_offsets  = lists_column_view(input).offsets_begin();
  auto const d_list_offsets = lists_column_view(child_col).offsets_begin();
  auto gather_map           = rmm::device_uvector<size_type>(num_output_entries, stream);

  // Fill the gather map with indices of the lists from the child column of the input column.
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    num_rows,
    [d_row_offsets,
     d_list_offsets,
     d_indices = gather_map.begin(),
     d_out_list_offsets =
       output_list_offsets.template begin<int32_t>()] __device__(size_type const idx) {
      // The output row has been identified as a null/empty list during list size computation.
      if (d_out_list_offsets[idx + 1] == d_out_list_offsets[idx]) { return; }

      // The indices of the list elements on the row `idx` of the input column.
      thrust::sequence(thrust::seq,
                       d_indices + d_out_list_offsets[idx],
                       d_indices + d_out_list_offsets[idx + 1],
                       d_list_offsets[d_row_offsets[idx]]);
    });

  auto result = cudf::detail::gather(table_view{{entry_col}},
                                     gather_map,
                                     out_of_bounds_policy::DONT_CHECK,
                                     cudf::negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

std::unique_ptr<column> concatenate_lists_nullifying_rows(column_view const& input,
                                                          cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  // Generate offsets and validities of the output lists column.
  auto [list_offsets, list_validities] = generate_list_offsets_and_validities(input, stream, mr);
  auto const offsets_view              = list_offsets->view();

  auto const num_rows           = input.size();
  auto const num_output_entries = cudf::detail::get_value<int32_t>(offsets_view, num_rows, stream);

  auto list_entries =
    gather_list_entries(input, offsets_view, num_rows, num_output_entries, stream, mr);
  auto [null_mask, null_count] = cudf::detail::valid_if(
    list_validities.begin(), list_validities.end(), cuda::std::identity{}, stream, mr);

  return make_lists_column(num_rows,
                           std::move(list_offsets),
                           std::move(list_entries),
                           null_count,
                           null_count ? std::move(null_mask) : rmm::device_buffer{});
}

}  // namespace

/**
 * @copydoc cudf::lists::concatenate_list_elements
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate_list_elements(column_view const& input,
                                                  concatenate_null_policy null_policy,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == type_id::LIST,
               "Input column must be a lists column.",
               std::invalid_argument);

  auto const child = lists_column_view(input).child();
  CUDF_EXPECTS(child.type().id() == type_id::LIST,
               "Child of the input lists column must also be a lists column.",
               std::invalid_argument);

  if (input.size() == 0) { return cudf::empty_like(input); }

  // Guard: when the inner list column has 0 rows, every outer row is either null or a
  // valid-but-empty list.  The kernels below read the inner column's offsets buffer via
  // a raw device pointer; for a 0-row list column that buffer may be unallocated (zero
  // bytes), so the pointer is invalid and the read triggers cudaErrorIllegalAddress.
  // Build the result directly: all-zero offsets, a 0-row child, and the outer null mask.
  //
  // null_policy need not be consulted: child.size() == 0 implies child.has_nulls() == false,
  // so both policies would dispatch to concatenate_lists_ignore_null with build_null_mask=false,
  // which simply copies the outer null mask — exactly what we do below.
  if (child.size() == 0) {
    auto const num_rows = input.size();
    auto out_offsets    = cudf::make_column_from_scalar(
      numeric_scalar<int32_t>(0, true, stream, mr), num_rows + 1, stream, mr);
    // Use the grandchild's schema (not child's) so out_entries has type T, not list<T>.
    auto out_entries = cudf::empty_like(lists_column_view(child).child());
    return make_lists_column(num_rows,
                             std::move(out_offsets),
                             std::move(out_entries),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input, stream, mr));
  }

  bool const has_null_list = child.has_nulls();
  return (null_policy == concatenate_null_policy::IGNORE || !has_null_list)
           ? concatenate_lists_ignore_null(input, has_null_list, stream, mr)
           : concatenate_lists_nullifying_rows(input, stream, mr);
}

}  // namespace detail

/**
 * @copydoc cudf::lists::concatenate_list_elements
 */
std::unique_ptr<column> concatenate_list_elements(column_view const& input,
                                                  concatenate_null_policy null_policy,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_list_elements(input, null_policy, stream, mr);
}

}  // namespace lists
}  // namespace cudf
