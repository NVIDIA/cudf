/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf {
class column;
}

namespace cudf::groupby::detail::hash {

struct reduction_context;

// Primitive ARGMIN/ARGMAX share one type-dispatch and device-launch owner.
std::unique_ptr<column> compute_argminmax(reduction_context const& ctx,
                                          bool is_argmin,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr);

// The expensive nested comparator has a separate owner for parallel compilation.
std::unique_ptr<column> compute_nested_argminmax(reduction_context const& ctx,
                                                 bool is_argmin,
                                                 cuda::stream_ref stream,
                                                 cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash
