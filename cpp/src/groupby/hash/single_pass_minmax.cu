/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "single_pass_reductions.cuh"

namespace cudf::groupby::detail::hash {

template std::unique_ptr<column> compute_reduction<aggregation::MIN>(reduction_context const& ctx,
                                                                     cuda::stream_ref stream,
                                                                     cudf::memory_resources mr);
template std::unique_ptr<column> compute_reduction<aggregation::MAX>(reduction_context const& ctx,
                                                                     cuda::stream_ref stream,
                                                                     cudf::memory_resources mr);

template std::vector<std::unique_ptr<column>> compute_reductions<aggregation::MIN>(
  host_span<reduction_context const>,
  std::span<int8_t const>,
  cuda::stream_ref,
  cudf::memory_resources);

template std::vector<std::unique_ptr<column>> compute_reductions<aggregation::MAX>(
  host_span<reduction_context const>,
  std::span<int8_t const>,
  cuda::stream_ref,
  cudf::memory_resources);

}  // namespace cudf::groupby::detail::hash
