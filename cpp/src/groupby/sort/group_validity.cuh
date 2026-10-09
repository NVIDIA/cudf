/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf::groupby::detail {

/**
 * @brief Writes whether each sorted group contains a valid value.
 *
 * The declaration-only interface lets reductions share the validity kernel without
 * depending on the value type or aggregation. Labels must match the values' row count,
 * and the output must contain one entry per contiguous group.
 */
void reduce_group_validity(device_span<size_type const> group_labels,
                           column_device_view const& values,
                           device_span<bool> validity,
                           cuda::stream_ref stream);

}  // namespace cudf::groupby::detail
