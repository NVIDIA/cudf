/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/filter_join_indices/filter_join_indices_output_size_kernel.cuh"
#include "join/filter_join_indices/filter_join_indices_output_size_kernel.hpp"

namespace cudf::detail {
template void launch_filter_output_size_kernel<false, false>(
  cudf::table_device_view const& left_table,
  cudf::table_device_view const& right_table,
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::ast::detail::expression_device_view device_expression_data,
  cudf::detail::grid_1d const& config,
  std::size_t shmem_per_block,
  cudf::join_kind join_kind,
  cudf::size_type* output_counts,
  rmm::cuda_stream_view stream);
}  // namespace cudf::detail
