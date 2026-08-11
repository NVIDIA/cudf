/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if 0  // Replaced by the shared HashCSR outer count kernel.
#include "partitioned_count_kernels.cuh"
#include "ref_types.cuh"

namespace cudf::detail {

template void launch_partitioned_count<true, primitive_count_ref_t>(probe_key_type const*,
                                                                    thread_index_type,
                                                                    size_type*,
                                                                    primitive_count_ref_t,
                                                                    rmm::cuda_stream_view);

template void launch_partitioned_count<true, nested_count_ref_t>(
  probe_key_type const*, thread_index_type, size_type*, nested_count_ref_t, rmm::cuda_stream_view);

template void launch_partitioned_count<true, flat_count_ref_t>(
  probe_key_type const*, thread_index_type, size_type*, flat_count_ref_t, rmm::cuda_stream_view);

}  // namespace cudf::detail
#endif
