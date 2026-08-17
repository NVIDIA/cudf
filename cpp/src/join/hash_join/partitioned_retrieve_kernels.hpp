/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels_common.cuh"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <memory>
#include <utility>

namespace cudf::detail {

/**
 * @brief Probes the hash table for each key and writes matching index pairs.
 *
 * Reduces match_counts to derive the total output size, allocates output buffers,
 * and launches the retrieve kernel. `left_offset` is added to each stored probe-row index when
 * writing to `left_indices`, so callers can produce indices in the full probe
 * table's coordinate space directly from a slice-local `keys` array.
 *
 * @return A pair of device vectors [left_indices, right_indices].
 */
template <bool IsOuter, typename Ref>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
launch_partitioned_retrieve(probe_key_type const* keys,
                            thread_index_type n,
                            size_type const* match_counts,
                            Ref ref,
                            size_type left_offset,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
