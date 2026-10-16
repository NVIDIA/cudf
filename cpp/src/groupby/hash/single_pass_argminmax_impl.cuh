/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "grouped_reductions.cuh"
#include "single_pass_argminmax.hpp"
#include "single_pass_reductions.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <cuda/iterator>
#include <cuda/std/cstddef>

#include <utility>

namespace cudf::groupby::detail::hash {

/// Stores each group's selected row and clears the validity of groups without a valid row.
struct select_group_row_fn {
  size_type* out;
  bitmask_type* mask;
  column_device_view values;

  __device__ void operator()(cuda::std::ptrdiff_t group, size_type row) const
  {
    out[group] = row;
    if (mask != nullptr && (row < 0 || row >= values.size() || values.is_null(row))) {
      cudf::clear_bit(mask, static_cast<size_type>(group));
    }
  }
};

/**
 * @brief Selects one input row per group and derives the result validity from that row.
 *
 * `reduce(output)` must write the selected row of every group through `output`. The selected row
 * is valid iff the group contains any valid row. An all-null group may select a null row or the
 * sentinel, so both are checked before reading its validity bit.
 */
template <typename Reduce>
std::unique_ptr<column> select_group_rows(reduction_context const& ctx,
                                          Reduce&& reduce,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  auto result = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                        ctx.num_groups,
                                        mask_state::UNALLOCATED,
                                        stream,
                                        mr.get_output_mr());
  if (ctx.num_groups == 0) { return result; }

  auto const out = result->mutable_view().begin<size_type>();
  auto null_mask =
    cudf::create_null_mask(ctx.num_groups,
                           ctx.nullable ? mask_state::ALL_VALID : mask_state::UNALLOCATED,
                           stream,
                           mr.get_output_mr());
  auto const mask = reinterpret_cast<bitmask_type*>(null_mask.data());
  reduce(cuda::tabulate_output_iterator{select_group_row_fn{out, mask, ctx.d_values}});
  if (ctx.nullable) {
    auto const null_count = count_group_nulls(mask, ctx.num_groups, stream, mr);
    result->set_null_mask(std::move(null_mask), null_count);
  }
  return result;
}

}  // namespace cudf::groupby::detail::hash
