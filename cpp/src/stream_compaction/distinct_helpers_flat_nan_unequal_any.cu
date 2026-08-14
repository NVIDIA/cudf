/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "distinct_helpers.cuh"
#include "distinct_helpers.hpp"

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf::detail {

template rmm::device_uvector<size_type> reduce_by_row_keep_any(
  distinct_set_t<cudf::detail::row::equality::device_row_comparator<
    false,
    cudf::nullate::DYNAMIC,
    cudf::detail::row::equality::physical_equality_comparator>>& set,
  size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
