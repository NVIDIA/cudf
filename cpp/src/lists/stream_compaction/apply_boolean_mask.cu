/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/reduction/detail/segmented_reduction_functions.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

namespace cudf::lists {
namespace detail {

std::unique_ptr<column> apply_mask(lists_column_view const& input,
                                   lists_column_view const& boolean_mask,
                                   cudf::detail::mask_type mask_kind,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(boolean_mask.child().type().id() == type_id::BOOL8, "Mask must be of type BOOL8.");
  CUDF_EXPECTS(input.size() == boolean_mask.size(),
               "Boolean masks column must have same number of rows as input.");
  auto const num_rows = input.size();

  if (num_rows == 0) { return cudf::empty_like(input.parent()); }

  auto constexpr offset_data_type = data_type{type_id::INT32};

  auto const mask_sliced_child = boolean_mask.get_sliced_child(stream);

  auto const make_filtered_child = [&] {
    auto filtered = cudf::detail::apply_mask(cudf::table_view{{input.get_sliced_child(stream)}},
                                             mask_sliced_child,
                                             mask_kind,
                                             stream,
                                             mr)
                      ->release();
    return std::move(filtered.front());
  };

  auto const make_output_offsets = [&] {
    auto mask_sliced_offsets =
      cudf::detail::slice(
        boolean_mask.offsets(), {boolean_mask.offset(), boolean_mask.size() + 1}, stream)
        .front();

    auto const sizes = [&] {
      // For DELETION, invert the `mask_sliced_child` so segmented_sum only counts `non-null` and
      // `false` elements in the original mask.
      auto const inverted_mask_child =
        (mask_kind == cudf::detail::mask_type::RETENTION)
          ? std::unique_ptr<column>{}
          : cudf::detail::unary_operation(mask_sliced_child,
                                          cudf::unary_operator::NOT,
                                          stream,
                                          cudf::get_current_device_resource_ref());
      auto const effective_mask_sliced_child = (mask_kind == cudf::detail::mask_type::RETENTION)
                                                 ? mask_sliced_child
                                                 : inverted_mask_child->view();
      return cudf::reduction::detail::segmented_sum(effective_mask_sliced_child,
                                                    mask_sliced_offsets,
                                                    offset_data_type,
                                                    null_policy::EXCLUDE,
                                                    std::nullopt,
                                                    stream,
                                                    cudf::get_current_device_resource_ref());
    }();
    auto const d_sizes     = column_device_view::create(*sizes, stream);
    auto const sizes_begin = cudf::detail::make_null_replacement_iterator(*d_sizes, size_type{0});
    auto const sizes_end   = sizes_begin + sizes->size();
    auto output_offsets    = cudf::make_numeric_column(
      offset_data_type, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
    auto output_offsets_view = output_offsets->mutable_view();

    // Could have attempted an exclusive_scan(), but it would not compute the last entry.
    // Instead, inclusive_scan(), followed by writing `0` to the head of the offsets column.
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           sizes_begin,
                           sizes_end,
                           output_offsets_view.begin<int32_t>() + 1);
    CUDF_CUDA_TRY(
      cudaMemsetAsync(output_offsets_view.begin<int32_t>(), 0, sizeof(int32_t), stream.value()));
    return output_offsets;
  };

  return cudf::make_lists_column(input.size(),
                                 make_output_offsets(),
                                 make_filtered_child(),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr));
}
}  // namespace detail

std::unique_ptr<column> apply_boolean_mask(lists_column_view const& input,
                                           lists_column_view const& boolean_mask,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(input, boolean_mask, cudf::detail::mask_type::RETENTION, stream, mr);
}

std::unique_ptr<column> apply_deletion_mask(lists_column_view const& input,
                                            lists_column_view const& deletion_mask,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(input, deletion_mask, cudf::detail::mask_type::DELETION, stream, mr);
}

}  // namespace cudf::lists
