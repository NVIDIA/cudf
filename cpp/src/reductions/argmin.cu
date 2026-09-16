/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "extrema_utils.cuh"

namespace cudf::reduction::detail {

std::unique_ptr<scalar> argmin(column_view const& input,
                               data_type dispatch_type,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  return type_dispatcher(
    dispatch_type, simple::detail::arg_minmax_dispatcher<aggregation::ARGMIN>{}, input, stream, mr);
}

}  // namespace cudf::reduction::detail
