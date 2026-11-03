/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

namespace cudf::groupby::detail {

/// Whether M2, VARIANCE and STD support values of type `type`.
bool is_m2_supported(data_type type);

std::unique_ptr<column> compute_m2(data_type source_type,
                                   column_view const& sum_sqr,
                                   column_view const& sum,
                                   column_view const& count,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

std::unique_ptr<column> compute_variance(column_view const& m2,
                                         column_view const& count,
                                         size_type ddof,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

std::unique_ptr<column> compute_std(column_view const& m2,
                                    column_view const& count,
                                    size_type ddof,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail
