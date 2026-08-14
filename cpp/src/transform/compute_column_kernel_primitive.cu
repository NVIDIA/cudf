/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_column_kernel.cuh"
#include "compute_column_kernel.hpp"

namespace cudf::detail {
template void launch_compute_column_kernel<false, false>(
  table_device_view const& table_device,
  ast::detail::expression_device_view device_expression_data,
  mutable_column_device_view& mutable_output_device,
  cudf::detail::grid_1d const& config,
  size_t shmem_per_block,
  cuda::stream_ref stream);
}  // namespace cudf::detail
