/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partitioned_count_kernels.cuh"
#include "ref_types.cuh"

namespace cudf::detail {

template void launch_partitioned_count<true, primitive_count_ref_t>(
  probe_key_type const*, thread_index_type, size_type*, primitive_count_ref_t, cuda::stream_ref);

template void launch_partitioned_count<true, nested_count_ref_t>(
  probe_key_type const*, thread_index_type, size_type*, nested_count_ref_t, cuda::stream_ref);

template void launch_partitioned_count<true, flat_count_ref_t>(
  probe_key_type const*, thread_index_type, size_type*, flat_count_ref_t, cuda::stream_ref);

}  // namespace cudf::detail
