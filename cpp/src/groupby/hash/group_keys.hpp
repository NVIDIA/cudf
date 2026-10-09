/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

namespace cudf::groupby::detail::hash {

/// Non-owning label and row buffers used by the stable group-order sort.
struct group_row_buffers {
  size_type* labels;
  size_type* rows;
};

/**
 * @brief Stably sorts rows by the low `end_bit` bits of their group labels.
 *
 * Both buffers remain owned by the caller. The returned pointers independently identify the
 * selected label and row outputs, which may reside in either input or alternate storage.
 */
group_row_buffers stable_sort_group_rows(group_row_buffers input,
                                         group_row_buffers alternate,
                                         size_type num_rows,
                                         int end_bit,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr);

/// The keys grouped by the HashCSR build.
struct grouped_keys {
  size_type num_groups;
  size_type num_grouped_rows;
  rmm::device_uvector<size_type> key_rows;  ///< One representative input row per group; empty when
                                            ///< the group-offset prefix already holds these rows
  rmm::device_uvector<size_type> group_offsets;  ///< `num_groups + 1` offsets into `grouped_rows`
  rmm::device_uvector<size_type> grouped_rows;   ///< Input rows reordered so groups are contiguous
  rmm::device_uvector<size_type> group_labels;   ///< Group of each grouped row when the stable
                                                 ///< build kept its sort keys, otherwise empty
};

/**
 * @brief Groups input keys with HashCSR, optionally materializing group offsets and grouped rows.
 *
 * When grouped rows are requested, `stable_rows` retains their original order within each group,
 * and `keep_labels` additionally returns the group of every grouped row when available.
 */
grouped_keys group_keys(table_view const& keys,
                        null_policy include_null_keys,
                        bool need_group_offsets,
                        bool need_grouped_rows,
                        bool stable_rows,
                        bool keep_labels,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash
