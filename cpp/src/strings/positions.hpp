/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf::strings::detail {

/**
 * @brief Create offsets for position values within a strings column
 *
 * The positions usually identify target sub-strings in the input column.
 * The offsets identify the set of positions for each string row.
 *
 * @param input Strings column corresponding to the input positions
 * @param positions Indices of target bytes within the input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned objects' device memory
 * @return Offsets of the position values for each string in input
 */
std::unique_ptr<column> create_offsets_from_positions(strings_column_view const& input,
                                                      device_span<int64_t const> const& positions,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr);

}  // namespace cudf::strings::detail
