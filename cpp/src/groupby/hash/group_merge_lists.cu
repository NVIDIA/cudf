/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/gather.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_merge_lists(column_view const& values,
                                          cudf::device_span<size_type const> group_offsets,
                                          size_type num_groups,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(values.type().id() == type_id::LIST,
               "Input to `group_merge_lists` must be a lists column.");
  CUDF_EXPECTS(!values.nullable(),
               "Input to `group_merge_lists` must be a non-nullable lists column.");

  auto offsets_column = make_numeric_column(
    data_type(type_id::INT32), num_groups + 1, mask_state::UNALLOCATED, stream, mr);

  // Generate offsets of the output lists column by gathering from the provided group offsets and
  // the input list offsets.
  //
  // For example:
  //   values        = [[2, 1], [], [4, -1, -2], [], [<NA>, 4, <NA>]]
  //   list_offsets  =  [0,     2,   2,           5,   5              8]
  //   group_offsets = [0,                        3,                  5]
  //
  //   then, the output offsets_column is [0, 5, 8].
  //
  thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 group_offsets.begin(),
                 group_offsets.end(),
                 lists_column_view(values).offsets_begin(),
                 offsets_column->mutable_view().template begin<int32_t>());

  // The child column of the output lists column is just copied from the input column.
  auto child_column =
    std::make_unique<column>(lists_column_view(values).get_sliced_child(stream), stream, mr);

  return make_lists_column(
    num_groups, std::move(offsets_column), std::move(child_column), 0, rmm::device_buffer{});
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
