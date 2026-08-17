/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "simple.cuh"

#include <cudf/reduction/detail/segmented_reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace reduction {
namespace detail {

std::unique_ptr<cudf::column> segmented_sum_of_squares(column_view const& col,
                                                       device_span<size_type const> offsets,
                                                       cudf::data_type const output_dtype,
                                                       null_policy null_handling,
                                                       cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr)
{
  using reducer = simple::detail::column_type_dispatcher<op::sum_of_squares>;
  return cudf::type_dispatcher(
    col.type(), reducer{}, col, offsets, output_dtype, null_handling, std::nullopt, stream, mr);
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
