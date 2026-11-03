/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/filling.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::sequence(size_type, scalar const&, scalar const&,cuda::stream_ref
 * stream,rmm::device_async_resource_ref)
 */
std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 scalar const& step,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::sequence(size_type, scalar const&, cuda::stream_ref,
 * rmm::device_async_resource_ref)
 */
std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::calendrical_month_sequence(size_type size,
 *                                           scalar const& init,
 *                                           size_type months,
 *                                           rmm::device_async_resource_ref mr)
 */
std::unique_ptr<cudf::column> calendrical_month_sequence(size_type size,
                                                         scalar const& init,
                                                         size_type months,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
