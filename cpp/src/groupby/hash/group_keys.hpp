/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/buffer>
#include <cuda/stream>

#include <cstddef>
#include <optional>
#include <utility>

namespace cudf::groupby::detail::hash {

/// The keys grouped by the HashCSR build.
class grouped_keys {
  cuda::device_buffer<size_type> _key_storage;

 public:
  grouped_keys(cuda::device_buffer<size_type>&& indices,
               std::optional<cuda::device_buffer<size_type>>&& offsets,
               std::optional<cuda::device_buffer<size_type>>&& rows,
               size_type num_groups)
    : _key_storage{std::move(indices)},
      key_indices{_key_storage.data(), static_cast<std::size_t>(num_groups)},
      group_offsets{std::move(offsets)},
      grouped_rows{std::move(rows)}
  {
  }

  grouped_keys(grouped_keys const&)            = delete;
  grouped_keys& operator=(grouped_keys const&) = delete;
  grouped_keys(grouped_keys&&)                 = default;
  grouped_keys& operator=(grouped_keys&&)      = default;

  /// Input row indices for unique keys; size equals the number of groups.
  device_span<size_type const> key_indices;
  /// `key_indices.size() + 1` group offsets, present when requested.
  std::optional<cuda::device_buffer<size_type>> group_offsets;
  /// Input row indices in contiguous groups, present when requested.
  std::optional<cuda::device_buffer<size_type>> grouped_rows;
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
