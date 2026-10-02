/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_single_pass_reduction_util.cuh"

namespace cudf::groupby::detail {

std::unique_ptr<column> group_argminmax(column_view const& values,
                                        size_type num_groups,
                                        cudf::device_span<size_type const> group_labels,
                                        bool is_argmin,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  auto dispatch_type = cudf::is_dictionary(values.type())
                         ? dictionary_column_view(values).keys().type()
                         : values.type();
  return type_dispatcher(dispatch_type,
                         group_argminmax_dispatcher{},
                         values,
                         num_groups,
                         group_labels,
                         is_argmin,
                         stream,
                         mr);
}

}  // namespace cudf::groupby::detail
