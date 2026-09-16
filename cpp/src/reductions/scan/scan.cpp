/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scan.hpp>
#include <cudf/reduction.hpp>

namespace cudf {
namespace detail {
namespace {
std::unique_ptr<column> scan(column_view const& input,
                             scan_aggregation const& agg,
                             scan_type inclusive,
                             null_policy null_handling,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  if (agg.kind == aggregation::RANK) {
    CUDF_EXPECTS(inclusive == scan_type::INCLUSIVE,
                 "Rank aggregation operator requires an inclusive scan");
    auto const& rank_agg = static_cast<cudf::detail::rank_aggregation const&>(agg);
    if (rank_agg._method == rank_method::MIN) {
      if (rank_agg._percentage == rank_percentage::NONE) {
        return inclusive_rank_scan(input, stream, mr);
      } else if (rank_agg._percentage == rank_percentage::ONE_NORMALIZED) {
        return inclusive_one_normalized_percent_rank_scan(input, stream, mr);
      }
    } else if (rank_agg._method == rank_method::DENSE) {
      return inclusive_dense_rank_scan(input, stream, mr);
    }
    CUDF_FAIL("Unsupported rank aggregation method for inclusive scan");
  }

  return inclusive == scan_type::EXCLUSIVE
           ? detail::scan_exclusive(input, agg, null_handling, stream, mr)
           : detail::scan_inclusive(input, agg, null_handling, stream, mr);
}

}  // namespace
}  // namespace detail

std::unique_ptr<column> scan(column_view const& input,
                             scan_aggregation const& agg,
                             scan_type inclusive,
                             null_policy null_handling,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::scan(input, agg, inclusive, null_handling, stream, mr);
}

}  // namespace cudf
