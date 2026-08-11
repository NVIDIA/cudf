/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/range_rolling.hpp"
#include "detail/range_utils.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>

namespace cudf::detail {

std::unique_ptr<column> dispatch_range_window(
  bounded_open,
  column_view const& orderby,
  rolling::direction direction,
  order order,
  std::optional<rolling::preprocessed_group_info> const& grouping,
  bool nulls_at_start,
  scalar const* row_delta,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return type_dispatcher(orderby.type(),
                         rolling::range_window_clamper<bounded_open>{},
                         orderby,
                         direction,
                         order,
                         grouping,
                         nulls_at_start,
                         row_delta,
                         stream,
                         mr);
}

}  // namespace cudf::detail
