/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compound.cuh"

#include <cudf/reduction/detail/segmented_reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace reduction {
namespace detail {

// segmented_variance is intentionally co-located with segmented_standard_deviation in this
// translation unit. Both reductions use the same var_std intermediate and segmented CUB reduction
// shape; keeping them together avoids emitting duplicate device kernel instantiations.
std::unique_ptr<cudf::column> segmented_standard_deviation(column_view const& col,
                                                           device_span<size_type const> offsets,
                                                           cudf::data_type const output_dtype,
                                                           null_policy null_handling,
                                                           size_type ddof,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  using reducer = compound::detail::compound_segmented_dispatcher<op::standard_deviation>;
  return cudf::type_dispatcher(
    col.type(), reducer(), col, offsets, output_dtype, null_handling, ddof, stream, mr);
}

std::unique_ptr<cudf::column> segmented_variance(column_view const& col,
                                                 device_span<size_type const> offsets,
                                                 cudf::data_type const output_dtype,
                                                 null_policy null_handling,
                                                 size_type ddof,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  using reducer = compound::detail::compound_segmented_dispatcher<op::variance>;
  return cudf::type_dispatcher(
    col.type(), reducer(), col, offsets, output_dtype, null_handling, ddof, stream, mr);
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
