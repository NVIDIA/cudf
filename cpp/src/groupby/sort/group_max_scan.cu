/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_scan_util.cuh"

#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> max_scan(column_view const& values,
                                 size_type num_groups,
                                 cudf::device_span<size_type const> group_labels,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  return type_dispatcher(values.type(),
                         group_scan_dispatcher<aggregation::MAX>{},
                         values,
                         num_groups,
                         group_labels,
                         stream,
                         mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
