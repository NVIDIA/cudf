/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_global_memory_aggs_null_kernels.cuh"

namespace cudf::groupby::detail::hash {

void launch_null_sparse_non_dictionary(nullable_insert_and_find_ref set_ref,
                                       bitmask_type const* row_bitmask,
                                       aggregation::Kind const* aggs,
                                       table_device_view const& input_values,
                                       mutable_table_device_view const& output_values,
                                       size_type num_rows,
                                       rmm::cuda_stream_view stream)
{
  launch_null_sparse_filtered<false>(
    set_ref, row_bitmask, aggs, input_values, output_values, num_rows, stream);
}

}  // namespace cudf::groupby::detail::hash
