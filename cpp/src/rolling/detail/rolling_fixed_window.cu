/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rolling.cuh"
#include "rolling_udf.cuh"
#include "rolling_utils.cuh"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf::detail {

// Applies a fixed-size rolling window function to the values in a column.
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       size_type preceding_window,
                                       size_type following_window,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (input.is_empty()) { return cudf::detail::empty_output_for_rolling_aggregation(input, agg); }

  CUDF_EXPECTS((min_periods >= 0), "min_periods must be non-negative");

  CUDF_EXPECTS((default_outputs.is_empty() || default_outputs.size() == input.size()),
               "Defaults column must be either empty or have as many rows as the input column.");

  CUDF_EXPECTS(-(preceding_window - 1) <= following_window,
               "Preceding window bounds must precede the following window bounds.");

  if (agg.kind == aggregation::CUDA || agg.kind == aggregation::PTX) {
    // TODO: In future, might need to clamp preceding/following to column boundaries.
    return cudf::detail::rolling_window_udf(input,
                                            cudf::detail::fixed_window_wrapper(preceding_window),
                                            cudf::detail::fixed_window_wrapper(following_window),
                                            min_periods,
                                            agg,
                                            stream,
                                            mr);
  } else {
    namespace utils = cudf::detail::rolling;
    auto groups     = utils::ungrouped{input.size()};
    auto preceding =
      utils::make_clamped_window_iterator<utils::direction::PRECEDING>(preceding_window, groups);
    auto following =
      utils::make_clamped_window_iterator<utils::direction::FOLLOWING>(following_window, groups);
    return cudf::detail::rolling_window(
      input, default_outputs, preceding, following, min_periods, agg, stream, mr);
  }
}
}  // namespace cudf::detail
