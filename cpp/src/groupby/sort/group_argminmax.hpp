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

// Dispatch in the owner TU so supported types come from cudf's central type mapping.
void launch_argminmax_reduction(cudf::device_span<cudf::size_type const> group_labels,
                                data_type value_type,
                                column_device_view const& values,
                                bool has_nulls,
                                bool arg_min,
                                cudf::size_type* output,
                                cuda::stream_ref stream);

// Keep the definition in private owner TUs so ARGMIN and ARGMAX share device instantiations.
template <typename BinOp>
void launch_argminmax_reduction(cudf::device_span<cudf::size_type const> group_labels,
                                BinOp const& binop,
                                cudf::size_type* output,
                                cuda::stream_ref stream);

}  // namespace cudf::groupby::detail
