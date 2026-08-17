/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/binaryop.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
//! Inner interfaces and implementations
namespace detail {

/**
 * @copydoc cudf::binary_operation(column_view const&, column_view const&,
 * std::string const&, data_type, cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         std::string const& ptx,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::binary_operation(scalar const&, column_view const&, binary_operator,
 * data_type, cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::binary_operation(column_view const&, scalar const&, binary_operator,
 * data_type, cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::binary_operation(column_view const&, column_view const&,
 * binary_operator, data_type, cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);
}  // namespace detail
}  // namespace cudf
