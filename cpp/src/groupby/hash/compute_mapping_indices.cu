/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_mapping_indices.cuh"
#include "compute_mapping_indices.hpp"

namespace cudf::groupby::detail::hash {
template int32_t max_active_blocks_mapping_kernel<hash_set_ref_t<cuco::insert_and_find_tag>>();

template void compute_mapping_indices<hash_set_ref_t<cuco::insert_and_find_tag>>(
  size_type grid_size,
  size_type num_rows,
  hash_set_ref_t<cuco::insert_and_find_tag> global_set,
  bitmask_type const* row_bitmask,
  size_type* local_mapping_index,
  size_type* global_mapping_index,
  size_type* block_cardinality,
  cuda::std::atomic_flag* needs_global_memory_fallback,
  cuda::stream_ref stream);
}  // namespace cudf::groupby::detail::hash
