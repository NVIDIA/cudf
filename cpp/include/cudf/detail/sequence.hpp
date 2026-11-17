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
 * stream,cudf::memory_resources)
 */
std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 scalar const& step,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr);

/**
 * @copydoc cudf::sequence(size_type, scalar const&, cuda::stream_ref,
 * cudf::memory_resources)
 */
std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr);

/**
 * @copydoc cudf::calendrical_month_sequence(size_type size,
 *                                           scalar const& init,
 *                                           size_type months,
 *                                           cudf::memory_resources mr)
 */
std::unique_ptr<cudf::column> calendrical_month_sequence(size_type size,
                                                         scalar const& init,
                                                         size_type months,
                                                         cuda::stream_ref stream,
                                                         cudf::memory_resources mr);

}  // namespace detail
}  // namespace cudf
