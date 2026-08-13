/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_helper_group_offsets.cuh"

namespace cudf::groupby::detail::sort {

size_type compute_nested_group_offsets(table_view const& keys,
                                       size_type const* sorted_order,
                                       size_type size,
                                       rmm::device_uvector<size_type>& group_offsets,
                                       rmm::cuda_stream_view stream)
{
  return compute_group_offsets<true>(keys, sorted_order, size, group_offsets, stream);
}

}  // namespace cudf::groupby::detail::sort
