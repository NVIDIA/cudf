/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::unary_operation
 */
std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::unary_operator op,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::is_valid
 */
std::unique_ptr<cudf::column> is_valid(cudf::column_view const& input,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::cast
 */
std::unique_ptr<column> cast(column_view const& input,
                             data_type type,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::is_nan
 */
std::unique_ptr<column> is_nan(cudf::column_view const& input,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::is_not_nan
 */
std::unique_ptr<column> is_not_nan(cudf::column_view const& input,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
