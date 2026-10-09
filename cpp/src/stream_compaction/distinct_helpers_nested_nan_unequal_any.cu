/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "distinct_helpers.cuh"
#include "distinct_helpers.hpp"

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

namespace cudf::detail {

template rmm::device_uvector<size_type> reduce_by_row_keep_any(
  distinct_set_t<cudf::detail::row::equality::device_row_comparator<
                   true,
                   cudf::nullate::DYNAMIC,
                   cudf::detail::row::equality::physical_equality_comparator>,
                 distinct_precomputed_hash>& set,
  size_type num_rows,
  cuda::stream_ref stream,
  cuda::mr::device_resource_ref mr);

}  // namespace cudf::detail
