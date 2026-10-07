/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "single_pass_reductions.cuh"

namespace cudf::groupby::detail::hash {

template std::unique_ptr<column> compute_reduction<aggregation::SUM_OVERFLOW>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash
