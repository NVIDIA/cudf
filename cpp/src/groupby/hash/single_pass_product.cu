/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "single_pass_reductions.cuh"

namespace cudf::groupby::detail::hash::single_pass {

template std::unique_ptr<column> compute_reduction<aggregation::PRODUCT>(
  reduction_context const& ctx, cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash::single_pass
