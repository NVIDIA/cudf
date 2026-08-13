/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_single_pass_aggs.cuh"
#include "compute_single_pass_aggs.hpp"

namespace cudf::groupby::detail::hash {
template std::pair<rmm::device_uvector<size_type>, bool>
compute_single_pass_aggs<nullable_global_set_t>(nullable_global_set_t& global_set,
                                                bitmask_type const* row_bitmask,
                                                std::span<aggregation_request const> requests,
                                                cudf::detail::result_cache* cache,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash
