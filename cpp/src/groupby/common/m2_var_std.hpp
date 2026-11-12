/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

namespace cudf::groupby::detail {

/// Whether M2, VARIANCE and STD support values of type `type`.
bool is_m2_supported(data_type type);

std::unique_ptr<column> compute_variance(column_view const& m2,
                                         column_view const& count,
                                         size_type ddof,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr);

std::unique_ptr<column> compute_std(column_view const& m2,
                                    column_view const& count,
                                    size_type ddof,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr);

}  // namespace cudf::groupby::detail
