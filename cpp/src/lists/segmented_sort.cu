/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {

namespace {

/**
 * @brief Create output offsets for segmented sort
 *
 * This creates a normalized set of offsets from the offsets child column of the input.
 */
std::unique_ptr<column> build_output_offsets(lists_column_view const& input,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  auto output_offset = make_numeric_column(
    input.offsets().type(), input.size() + 1, mask_state::UNALLOCATED, stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    input.offsets_begin(),
                    input.offsets_end(),
                    output_offset->mutable_view().begin<size_type>(),
                    [first = input.offsets_begin()] __device__(auto offset_index) {
                      return offset_index - *first;
                    });
  return output_offset;
}

}  // namespace

std::unique_ptr<column> sort_lists(lists_column_view const& input,
                                   order column_order,
                                   null_order null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return empty_like(input.parent());

  auto output_offset = build_output_offsets(input, stream, mr);
  auto const child   = input.get_sliced_child(stream);

  auto const sorted_child_table = cudf::detail::segmented_sort_by_key(table_view{{child}},
                                                                      table_view{{child}},
                                                                      output_offset->view(),
                                                                      {column_order},
                                                                      {null_precedence},
                                                                      stream,
                                                                      mr);

  // The sort cannot add nulls: release a null-free child's all-valid mask and skip the parent
  // bitmask copy when it holds no nulls, so the output is nullable only where nulls are present.
  auto sorted_child = std::move(sorted_child_table->release().front());
  // Defensive: keeps sort_lists' no-spurious-mask contract independent of the by-key internals --
  // currently a no-op, since the shared gather already normalizes.
  if (sorted_child->null_count() == 0) { sorted_child->set_null_mask(rmm::device_buffer{}, 0); }

  return make_lists_column(input.size(),
                           std::move(output_offset),
                           std::move(sorted_child),
                           input.null_count(),
                           input.null_count() == 0
                             ? rmm::device_buffer{}
                             : cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

std::unique_ptr<column> stable_sort_lists(lists_column_view const& input,
                                          order column_order,
                                          null_order null_precedence,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return empty_like(input.parent()); }

  auto output_offset = build_output_offsets(input, stream, mr);
  auto const child   = input.get_sliced_child(stream);

  auto const sorted_child_table = cudf::detail::stable_segmented_sort_by_key(table_view{{child}},
                                                                             table_view{{child}},
                                                                             output_offset->view(),
                                                                             {column_order},
                                                                             {null_precedence},
                                                                             stream,
                                                                             mr);

  // The sort cannot add nulls: release a null-free child's all-valid mask and skip the parent
  // bitmask copy when it holds no nulls, so the output is nullable only where nulls are present.
  auto sorted_child = std::move(sorted_child_table->release().front());
  // Defensive: keeps stable_sort_lists' no-spurious-mask contract independent of the by-key
  // internals -- currently a no-op, since the shared gather already normalizes.
  if (sorted_child->null_count() == 0) { sorted_child->set_null_mask(rmm::device_buffer{}, 0); }

  return make_lists_column(input.size(),
                           std::move(output_offset),
                           std::move(sorted_child),
                           input.null_count(),
                           input.null_count() == 0
                             ? rmm::device_buffer{}
                             : cudf::detail::copy_bitmask(input.parent(), stream, mr));
}
}  // namespace detail

std::unique_ptr<column> sort_lists(lists_column_view const& input,
                                   order column_order,
                                   null_order null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sort_lists(input, column_order, null_precedence, stream, mr);
}

std::unique_ptr<column> stable_sort_lists(lists_column_view const& input,
                                          order column_order,
                                          null_order null_precedence,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_sort_lists(input, column_order, null_precedence, stream, mr);
}

}  // namespace lists
}  // namespace cudf
