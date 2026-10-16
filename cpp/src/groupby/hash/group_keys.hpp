/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/buffer>
#include <cuda/stream>

namespace cudf::groupby::detail::hash {

/// The keys grouped by the HashCSR build.
struct grouped_keys {
  cuda::device_buffer<size_type> key_rows;       ///< First `num_groups` entries identify group keys
  cuda::device_buffer<size_type> group_offsets;  ///< `num_groups + 1` offsets into `grouped_rows`
  cuda::device_buffer<size_type> grouped_rows;   ///< Input rows reordered so groups are contiguous
  size_type num_groups;                          ///< Number of representative rows in `key_rows`
};

/**
 * @brief Groups input keys with HashCSR, optionally materializing group offsets and grouped rows.
 *
 * @param keys Table whose rows define groups
 * @param include_null_keys Whether rows containing null keys participate in grouping
 * @param need_group_offsets Whether to return offsets for each group
 * @param need_grouped_rows Whether to return grouped input row indices; requires group offsets
 * @param stream CUDA stream used for device operations
 * @param mr Memory resources for grouping outputs and temporary allocations
 * @throws std::invalid_argument If grouped rows are requested without group offsets
 * @return Representative rows, group offsets, and grouped row indices as requested
 */
grouped_keys group_keys(table_view const& keys,
                        null_policy include_null_keys,
                        bool need_group_offsets,
                        bool need_grouped_rows,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash
