/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels_common.cuh"

#include <cuda/stream>

namespace cudf::detail {

/// Launch the partitioned_count kernel.
template <bool IsOuter, typename Ref>
void launch_partitioned_count(probe_key_type const* keys,
                              thread_index_type n,
                              size_type* output,
                              Ref ref,
                              cuda::stream_ref stream);

}  // namespace cudf::detail
