/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reductions/nested_types_extrema_utils.cuh"
#include "single_pass_reductions.cuh"

namespace cudf::groupby::detail::hash {

namespace {

template <typename BinOp>
struct input_order_argminmax_fn {
  BinOp binop;

  __device__ size_type operator()(size_type lhs, size_type rhs) const
  {
    // Match a stable input-order fold on ties: ARGMIN retains the last row and ARGMAX the first.
    // CSR rows and partial reductions can arrive in either order.
    return lhs < rhs ? binop(lhs, rhs) : binop(rhs, lhs);
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
  auto result = make_size_type_column(ctx, stream, mr);
  if (ctx.num_groups == 0) { return result; }

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
  auto const comparator = is_argmin ? generator::create<aggregation::ARGMIN>(values, stream)
                                    : generator::create<aggregation::ARGMAX>(values, stream);
  auto const sentinel   = is_argmin ? cudf::detail::ARGMIN_SENTINEL : cudf::detail::ARGMAX_SENTINEL;
  auto const indices =
    cuda::transform_iterator{ctx.grouped.rows.begin(),
                             valid_nested_index_fn{ctx.d_values, sentinel, ctx.values.has_nulls()}};
  reduce_groups(ctx.grouped,
                indices,
                result->mutable_view().begin<size_type>(),
                input_order_argminmax_fn{comparator.binop()},
                sentinel,
                stream,
                mr);
  set_group_null_mask(*result, ctx, stream, mr);
  return result;
}

template std::unique_ptr<column> compute_reduction<aggregation::ARGMIN>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);
template std::unique_ptr<column> compute_reduction<aggregation::ARGMAX>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash
