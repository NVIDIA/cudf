/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "join/join_common_utils.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf::detail {

/**
 * @brief Removes unmatched-right entries from FULL join maps to produce LEFT join maps.
 *
 * A FULL join map already contains `(JoinNoMatch, right_index)` entries. These must be removed
 * before applying LEFT filtering because `finalize_full_join` reconstructs the unmatched-right
 * complement after the predicate has been applied.
 */
VectorPair full_to_left_join_indices(device_span<size_type const> left_indices,
                                     device_span<size_type const> right_indices,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
