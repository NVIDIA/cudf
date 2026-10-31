/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby_helper_group_offsets.cuh"

namespace cudf::groupby::detail {

size_type compute_nested_group_offsets(table_view const& keys,
                                       device_span<size_type> group_offsets,
                                       cuda::stream_ref stream)
{
  return compute_group_offsets<true>(keys, group_offsets, stream);
}

}  // namespace cudf::groupby::detail
