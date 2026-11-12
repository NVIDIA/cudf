/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/reduction/detail/segmented_reduction_functions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

namespace cudf::lists {
namespace detail {

std::unique_ptr<column> apply_mask(lists_column_view const& input,
                                   lists_column_view const& boolean_mask,
                                   cudf::detail::mask_type mask_kind,
                                   null_policy null_handling,
                                   cuda::stream_ref stream,
                                   cudf::memory_resources mr)
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
                                             null_handling,
                                             stream,
                                             mr)
                      ->release();
    return std::move(filtered.front());
  };

  auto const make_output_offsets = [&] {
    auto mask_sliced_offsets =
      cudf::detail::slice(boolean_mask.offsets(),
                          {boolean_mask.offset(), boolean_mask.offset() + boolean_mask.size() + 1},
                          stream)
        .front();

    auto const sizes = [&] {
      auto const is_retention = (mask_kind == cudf::detail::mask_type::RETENTION);
      // `mask_sliced_offsets` index the full mask child, not `mask_sliced_child`
      auto const mask_child = boolean_mask.child();
      // For INCLUDE, replace null mask elements with the value that keeps the element
      auto const null_replaced_mask_child =
        (null_handling == null_policy::INCLUDE and mask_child.has_nulls())
          ? cudf::detail::replace_nulls(mask_child,
                                        cudf::numeric_scalar<bool>(is_retention, true, stream),
                                        stream,
                                        cudf::get_current_device_resource_ref())
          : std::unique_ptr<column>{};
      auto const counted_mask_child =
        null_replaced_mask_child ? null_replaced_mask_child->view() : mask_child;
      // For DELETION, invert the mask so segmented_sum only counts the kept (`false`) elements
      auto const inverted_mask_child =
        is_retention ? std::unique_ptr<column>{}
                     : cudf::detail::unary_operation(counted_mask_child,
                                                     cudf::unary_operator::NOT,
                                                     stream,
                                                     cudf::get_current_device_resource_ref());
      return cudf::reduction::detail::segmented_sum(
        is_retention ? counted_mask_child : inverted_mask_child->view(),
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
      cudaMemsetAsync(output_offsets_view.begin<int32_t>(), 0, sizeof(int32_t), stream.get()));
    return output_offsets;
  };

  return cudf::make_lists_column(input.size(),
                                 make_output_offsets(),
                                 make_filtered_child(),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr));
}
}  // namespace detail

std::unique_ptr<column> apply_retention_mask(lists_column_view const& input,
                                             lists_column_view const& retention_mask,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources mr)
{
  return apply_retention_mask(input, retention_mask, null_policy::EXCLUDE, stream, mr);
}

std::unique_ptr<column> apply_retention_mask(lists_column_view const& input,
                                             lists_column_view const& retention_mask,
                                             null_policy null_handling,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(
    input, retention_mask, cudf::detail::mask_type::RETENTION, null_handling, stream, mr);
}

std::unique_ptr<column> apply_boolean_mask(lists_column_view const& input,
                                           lists_column_view const& boolean_mask,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr)
{
  return apply_retention_mask(input, boolean_mask, null_policy::EXCLUDE, stream, mr);
}

std::unique_ptr<column> apply_deletion_mask(lists_column_view const& input,
                                            lists_column_view const& deletion_mask,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  return apply_deletion_mask(input, deletion_mask, null_policy::EXCLUDE, stream, mr);
}

std::unique_ptr<column> apply_deletion_mask(lists_column_view const& input,
                                            lists_column_view const& deletion_mask,
                                            null_policy null_handling,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::apply_mask(
    input, deletion_mask, cudf::detail::mask_type::DELETION, null_handling, stream, mr);
}

}  // namespace cudf::lists
