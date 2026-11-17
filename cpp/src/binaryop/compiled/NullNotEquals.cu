/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "binary_ops.cuh"

namespace cudf::binops::compiled {
template void apply_binary_op<ops::NullNotEquals>(mutable_column_view&,
                                                  column_view const&,
                                                  column_view const&,
                                                  bool is_lhs_scalar,
                                                  bool is_rhs_scalar,
                                                  cuda::stream_ref);
}  // namespace cudf::binops::compiled
