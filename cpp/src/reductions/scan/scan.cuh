/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/reduction.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/stream>

#include <utility>

namespace cudf {
namespace detail {

// logical-and scan of the null mask of the input view
std::pair<rmm::device_buffer, size_type> mask_scan(column_view const& input_view,
                                                   scan_type inclusive,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr);

// exponentially weighted moving average of the input
std::unique_ptr<column> exponentially_weighted_moving_average(column_view const& input,
                                                              scan_aggregation const& agg,
                                                              cuda::stream_ref stream,
                                                              rmm::device_async_resource_ref mr);

template <template <typename> typename DispatchFn>
std::unique_ptr<column> scan_agg_dispatch(column_view const& input,
                                          scan_aggregation const& agg,
                                          bitmask_type const* output_mask,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  switch (agg.kind) {
    case aggregation::SUM:
      return type_dispatcher<dispatch_storage_type>(
        input.type(), DispatchFn<DeviceSum>(), input, output_mask, stream, mr);
    case aggregation::MIN:
      return type_dispatcher<dispatch_storage_type>(
        input.type(), DispatchFn<DeviceMin>(), input, output_mask, stream, mr);
    case aggregation::MAX:
      return type_dispatcher<dispatch_storage_type>(
        input.type(), DispatchFn<DeviceMax>(), input, output_mask, stream, mr);
    case aggregation::PRODUCT:
      // a product scan on a decimal type with non-zero scale would result in each element having
      // a different scale, and because scale is stored once per column, this is not possible
      CUDF_EXPECTS(!is_fixed_point(input.type()), "decimal32/64/128 cannot support product scan");
      return type_dispatcher<dispatch_storage_type>(
        input.type(), DispatchFn<DeviceProduct>(), input, output_mask, stream, mr);
    case aggregation::EWMA: return exponentially_weighted_moving_average(input, agg, stream, mr);
    case aggregation::COUNT_VALID:
      return type_dispatcher<dispatch_storage_type>(
        input.type(), DispatchFn<DeviceCount>(), input, output_mask, stream, mr);
    case aggregation::COUNT_ALL:
      return type_dispatcher<dispatch_storage_type>(
        input.type(), DispatchFn<DeviceCount>(), input, nullptr, stream, mr);
    default: CUDF_FAIL("Unsupported aggregation operator for scan");
  }
}

}  // namespace detail
}  // namespace cudf
