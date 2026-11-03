/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/rolling.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/rolling.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

namespace cudf {

// Applies a fixed-size rolling window function to the values in a column, with default output
// specified
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rolling_window(
    input, default_outputs, preceding_window, following_window, min_periods, agg, stream, mr);
}

// Applies a fixed-size rolling window function to the values in a column, without default specified
std::unique_ptr<column> rolling_window(column_view const& input,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto defaults =
    cudf::is_dictionary(input.type()) ? dictionary_column_view(input).indices() : input;
  return detail::rolling_window(input,
                                empty_like(defaults)->view(),
                                preceding_window,
                                following_window,
                                min_periods,
                                agg,
                                stream,
                                mr);
}

// Applies a variable-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& preceding_window,
                                       column_view const& following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rolling_window(
    input, preceding_window, following_window, min_periods, agg, stream, mr);
}

bool is_valid_rolling_aggregation(data_type source, aggregation::Kind kind)
{
  return detail::is_valid_rolling_aggregation(source, kind);
}
}  // namespace cudf
