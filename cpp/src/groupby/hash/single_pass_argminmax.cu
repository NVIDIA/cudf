/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "single_pass_argminmax_impl.cuh"
#include "single_pass_reductions.cuh"

#include <cudf/detail/utilities/element_argminmax.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf::groupby::detail::hash {

namespace {

struct argminmax_dispatcher {
  template <typename T>
  std::unique_ptr<column> operator()(reduction_context const& ctx,
                                     bool is_argmin,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    if constexpr (!is_reduction_supported<T>(aggregation::ARGMIN)) {
      CUDF_FAIL("Unsupported type for hash groupby aggregation");
    } else if constexpr (cudf::is_nested<T>()) {
      return compute_nested_argminmax(ctx, is_argmin, stream, mr);
    } else {
      // Both operations retain one comparator type, including the representation mapping used
      // by decimal and chrono columns. The selected row is always an original input index.
      return select_group_rows(
        ctx,
        [&](auto output) {
          reduce_groups(ctx.grouped,
                        ctx.grouped.rows.begin(),
                        output,
                        cudf::detail::element_argminmax_fn<rep_type_t<T>>{
                          ctx.d_values, ctx.values.has_nulls(), is_argmin},
                        is_argmin ? cudf::detail::ARGMIN_SENTINEL : cudf::detail::ARGMAX_SENTINEL,
                        stream,
                        mr);
        },
        stream,
        mr);
    }
  }
};

}  // namespace

std::unique_ptr<column> compute_argminmax(reduction_context const& ctx,
                                          bool is_argmin,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  return type_dispatcher(ctx.values_type, argminmax_dispatcher{}, ctx, is_argmin, stream, mr);
}

template std::unique_ptr<column> compute_reduction<aggregation::ARGMIN>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);
template std::unique_ptr<column> compute_reduction<aggregation::ARGMAX>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash
