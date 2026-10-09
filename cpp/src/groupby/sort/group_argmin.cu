/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_argminmax.hpp"

namespace cudf::groupby::detail {

std::unique_ptr<column> group_argmin(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& key_sort_order,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  return group_argminmax(values, num_groups, group_labels, key_sort_order, true, stream, mr);
}

}  // namespace cudf::groupby::detail
