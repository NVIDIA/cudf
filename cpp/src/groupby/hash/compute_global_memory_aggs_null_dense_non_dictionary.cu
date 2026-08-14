/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_global_memory_aggs_null_kernels.cuh"

namespace cudf::groupby::detail::hash {

void launch_null_dense_non_dictionary(size_type const* target_indices,
                                      aggregation::Kind const* aggs,
                                      table_device_view const& input_values,
                                      mutable_table_device_view const& output_values,
                                      int64_t num_items,
                                      rmm::cuda_stream_view stream)
{
  launch_null_dense_filtered<false>(
    target_indices, aggs, input_values, output_values, num_items, stream);
}

}  // namespace cudf::groupby::detail::hash
