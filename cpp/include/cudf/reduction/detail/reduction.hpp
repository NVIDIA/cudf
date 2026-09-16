/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <optional>

namespace cudf {
namespace reduction::detail {

/**
 * @copydoc cudf::reduce(column_view const&, reduce_aggregation const&, data_type,
 * std::optional<std::reference_wrapper<scalar const>>, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<scalar> reduce(column_view const& col,
                               reduce_aggregation const& agg,
                               data_type output_dtype,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

}  // namespace reduction::detail
}  // namespace cudf
