/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "grouped_reductions.cuh"
#include "merge_m2.cuh"
#include "single_pass_reductions.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

namespace cudf::groupby::detail::hash {
namespace {

template <typename T>
struct grouped_m2 {
  size_type const* rows;
  value_accessor<T> value;
  bool has_nulls;

  __device__ m2_value operator()(size_type position) const
  {
    auto const row = rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return {0.0, false}; }
    auto const x = static_cast<double>(value(row));
    return {x, true};
  }
};

template <count_type T>
struct split_m2 {
  __device__ cuda::std::tuple<double, T> operator()(m2_state<T> const& state) const
  {
    return {state.m2, state.count};
  }
};

template <count_type CountT>
struct m2_dispatch {
  template <typename T>
  void operator()(reduction_context const& ctx,
                  double* m2,
                  CountT* count,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr) const
  {
    if constexpr (cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_m2<T>{
          ctx.grouped.rows.data(), value_accessor<T>{ctx.d_values}, ctx.values.has_nulls()});
      auto const output =
        cuda::transform_output_iterator{cuda::make_zip_iterator(m2, count), split_m2<CountT>{}};
      reduce_groups(
        ctx.grouped, values, output, merge_m2<CountT>{}, m2_state<CountT>{0, 0.0, 0.0}, stream, mr);
    } else {
      CUDF_FAIL("Invalid source type for M2 aggregation.");
    }
  }
};

}  // namespace

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> compute_m2_and_count(
  reduction_context const& ctx,
  bool m2_intermediate,
  bool count_intermediate,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  using count_type = size_type;
  // Both results are non-nullable, including zero M2/count for all-null groups.
  auto m2    = make_numeric_column(data_type{type_id::FLOAT64},
                                ctx.num_groups,
                                mask_state::UNALLOCATED,
                                stream,
                                m2_intermediate ? mr.get_temporary_mr() : mr.get_output_mr());
  auto count = make_numeric_column(data_type{cudf::type_to_id<count_type>()},
                                   ctx.num_groups,
                                   mask_state::UNALLOCATED,
                                   stream,
                                   count_intermediate ? mr.get_temporary_mr() : mr.get_output_mr());
  if (ctx.num_groups > 0) {
    type_dispatcher(ctx.values_type,
                    m2_dispatch<count_type>{},
                    ctx,
                    m2->mutable_view().begin<double>(),
                    count->mutable_view().begin<count_type>(),
                    stream,
                    mr);
  }
  return {std::move(m2), std::move(count)};
}

}  // namespace cudf::groupby::detail::hash
