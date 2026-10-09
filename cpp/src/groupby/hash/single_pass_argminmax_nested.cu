/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reductions/nested_types_extrema_utils.cuh"
#include "single_pass_argminmax_impl.cuh"

#include <cuda/std/algorithm>

namespace cudf::groupby::detail::hash {

namespace {

template <typename BinOp>
struct first_row_argminmax_fn {
  BinOp binop;
  bool is_argmin;

  __device__ size_type operator()(size_type lhs, size_type rhs) const
  {
    // Equal rows select the smaller row index whatever order the CSR rows and partial reductions
    // arrive in: the row comparator keeps its second argument on ties for ARGMIN and its first
    // for ARGMAX, so the smaller index takes that position.
    auto const first  = cuda::std::min(lhs, rhs);
    auto const second = cuda::std::max(lhs, rhs);
    return is_argmin ? binop(second, first) : binop(first, second);
  }
};

struct valid_nested_index_fn {
  column_device_view values;
  size_type sentinel;
  bool has_nulls;

  __device__ size_type operator()(size_type row) const
  {
    return has_nulls && values.is_null_nocheck(row) ? sentinel : row;
  }
};

}  // namespace

std::unique_ptr<column> compute_nested_argminmax(reduction_context const& ctx,
                                                 bool is_argmin,
                                                 cuda::stream_ref stream,
                                                 cudf::memory_resources mr)
{
  return select_group_rows(
    ctx,
    [&](auto output) {
      // Top-level nulls become the sentinel before comparison. Keep only child nulls in the
      // comparator view, retaining the input offset for sliced lists and structs.
      auto const values     = column_view{ctx.values.type(),
                                      ctx.values.size(),
                                      ctx.values.head(),
                                      nullptr,
                                      0,
                                      ctx.values.offset(),
                                          {ctx.values.child_begin(), ctx.values.child_end()}};
      using generator       = cudf::reduction::detail::arg_minmax_binop_generator;
      auto const comparator = is_argmin
                                ? generator::create<aggregation::ARGMIN>(values, stream, mr)
                                : generator::create<aggregation::ARGMAX>(values, stream, mr);
      auto const sentinel =
        is_argmin ? cudf::detail::ARGMIN_SENTINEL : cudf::detail::ARGMAX_SENTINEL;
      auto const indices = cuda::transform_iterator{
        ctx.grouped.rows.begin(),
        valid_nested_index_fn{ctx.d_values, sentinel, ctx.values.has_nulls()}};
      reduce_groups(ctx.grouped,
                    indices,
                    output,
                    first_row_argminmax_fn{comparator.binop(), is_argmin},
                    sentinel,
                    stream,
                    mr);
    },
    stream,
    mr);
}

}  // namespace cudf::groupby::detail::hash
