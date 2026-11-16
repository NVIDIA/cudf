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
#include <thrust/transform.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_merge_lists(column_view const& values,
                                          cudf::device_span<size_type const> group_offsets,
                                          size_type num_groups,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  CUDF_EXPECTS(values.type().id() == type_id::LIST,
               "Input to `group_merge_lists` must be a lists column.");
  CUDF_EXPECTS(!values.nullable(),
               "Input to `group_merge_lists` must be a non-nullable lists column.");

  auto offsets_column = make_numeric_column(
    data_type(type_id::INT32), num_groups + 1, mask_state::UNALLOCATED, stream, mr);

  // Generate offsets of the output lists column by gathering from the provided group offsets and
  // the input list offsets. The output child starts at the first list of the input, so the offsets
  // are taken relative to it.
  //
  // For example:
  //   values        = [[2, 1], [], [4, -1, -2], [], [<NA>, 4, <NA>]]
  //   list_offsets  =  [0,     2,   2,           5,   5              8]
  //   group_offsets = [0,                        3,                  5]
  //
  //   then, the output offsets_column is [0, 5, 8].
  //
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    group_offsets.begin(),
                    group_offsets.end(),
                    offsets_column->mutable_view().template begin<int32_t>(),
                    [list_offsets = lists_column_view(values).offsets_begin()] __device__(
                      size_type row) -> size_type { return list_offsets[row] - list_offsets[0]; });

  // The child column of the output lists column is just copied from the input column.
  auto child_column =
    std::make_unique<column>(lists_column_view(values).get_sliced_child(stream), stream, mr);

  return make_lists_column(num_groups,
                           std::move(offsets_column),
                           std::move(child_column),
                           0,
                           cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
