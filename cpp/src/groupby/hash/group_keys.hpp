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
               size_type num_groups,
               std::optional<cuda::device_buffer<size_type>>&& labels = std::nullopt)
    : _key_storage{std::move(indices)},
      key_indices{_key_storage.data(), static_cast<std::size_t>(num_groups)},
      group_offsets{std::move(offsets)},
      grouped_rows{std::move(rows)},
      group_labels{std::move(labels)}
  {
  }

  grouped_keys(grouped_keys const&)            = delete;
  grouped_keys& operator=(grouped_keys const&) = delete;
  grouped_keys(grouped_keys&&)                 = default;
  grouped_keys& operator=(grouped_keys&&)      = default;

  /// Input row indices for distinct keys; size equals the number of groups.
  device_span<size_type const> key_indices;
  /// `key_indices.size() + 1` group offsets, present when requested.
  std::optional<cuda::device_buffer<size_type>> group_offsets;
  /// Input row indices in contiguous groups, present when requested.
  std::optional<cuda::device_buffer<size_type>> grouped_rows;
  /// Group label of each grouped row, present when kept from sorting stable grouped rows or
  /// cached by a later request.
  std::optional<cuda::device_buffer<size_type>> group_labels;
};

/**
 * @brief Groups input keys with HashCSR, optionally materializing group offsets and grouped rows.
 *
 * @param keys Table whose rows define groups
 * @param include_null_keys Whether rows containing null keys participate in grouping
 * @param need_group_offsets Whether to return offsets for each group
 * @param need_grouped_rows Whether to return grouped input row indices; requires group offsets
 * @param stable_rows Whether grouped rows retain their input order within each group
 * @param keep_labels Whether to also return the group label of each stable grouped row when they
 * are sorted together
 * @param stream CUDA stream used for device operations
 * @param mr Memory resources for grouping outputs and temporary allocations
 * @throws std::invalid_argument If grouped rows are requested without group offsets
 * @return Representative rows, group offsets, grouped row indices, and group labels as requested
 */
grouped_keys group_keys(table_view const& keys,
                        null_policy include_null_keys,
                        bool need_group_offsets,
                        bool need_grouped_rows,
                        bool stable_rows,
                        bool keep_labels,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr);

/// Non-owning key and value buffers sorted together by `sort_pairs`.
struct pair_buffers {
  size_type* keys;
  size_type* values;
};

/**
 * @brief Stably sorts key-value pairs by the key bits in `[begin_bit, end_bit)`.
 *
 * Both buffer pairs remain owned by the caller. Either pair may hold the sorted keys or values.
 *
 * @param input Keys and values to sort
 * @param alternate Storage for `num_items` keys and values used while sorting
 * @param num_items Number of pairs to sort
 * @param begin_bit Least significant key bit to sort by
 * @param end_bit One past the most significant key bit to sort by
 * @param stream CUDA stream used for device operations
 * @param mr Memory resources whose temporary resource holds the sort scratch
 * @return The buffers holding the sorted keys and values
 */
pair_buffers sort_pairs(pair_buffers input,
                        pair_buffers alternate,
                        size_type num_items,
                        int begin_bit,
                        int end_bit,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash
