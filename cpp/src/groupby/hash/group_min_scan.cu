/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/hash/group_scan_util.cuh"

#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> min_scan(column_view const& values,
                                 size_type num_groups,
                                 cudf::device_span<size_type const> group_labels,
                                 nan_semantics semantics,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
{
  // TOTALLY_ORDERED only changes floating-point results, so only float and double are instantiated
  // with DeviceMinTotallyOrdered.
  if (semantics == nan_semantics::TOTALLY_ORDERED) {
    if (values.type().id() == type_id::FLOAT32) {
      return group_scan_functor<aggregation::MIN, float>::invoke_with<DeviceMinTotallyOrdered>(
        values, num_groups, group_labels, stream, mr);
    }
    if (values.type().id() == type_id::FLOAT64) {
      return group_scan_functor<aggregation::MIN, double>::invoke_with<DeviceMinTotallyOrdered>(
        values, num_groups, group_labels, stream, mr);
    }
  }
  return type_dispatcher(values.type(),
                         group_scan_dispatcher<aggregation::MIN>{},
                         values,
                         num_groups,
                         group_labels,
                         stream,
                         mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
