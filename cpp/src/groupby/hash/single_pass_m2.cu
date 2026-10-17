/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "grouped_reductions.cuh"
#include "single_pass_reductions.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/tuple>

namespace cudf::groupby::detail::hash {
namespace {

// Counts within a chunk are bounded. Cache rounded double reciprocals so the hot
// per-value and chunk-tree merges need a multiply rather than a full FP64 division.
// Cross-chunk merges with larger counts retain the ordinary division.
__constant__ cuda::std::array<double, rows_per_chunk + 1> const inverse_counts = [] {
  cuda::std::array<double, rows_per_chunk + 1> result{};
  for (size_type i = 1; i <= rows_per_chunk; ++i) {
    result[i] = 1.0 / i;
  }
  return result;
}();

template <typename T>
concept count_type = cudf::is_integral_not_bool<T>() || cudf::is_floating_point<T>();

template <count_type T>
struct m2_state {
  T count;
  double mean;
  double m2;
};

struct m2_value {
  double value;
  bool valid;

  template <count_type T>
  __device__ operator m2_state<T>() const
  {
    if (!valid) { return {T{0}, 0.0, 0.0}; }
    return {T{1},
            value,
            cuda::std::isfinite(value) ? 0.0 : cuda::std::numeric_limits<double>::quiet_NaN()};
  }
};

template <count_type T>
struct merge_m2 {
  // Merging the intermediate states is expensive: raking avoids the overlapping partial reductions
  // performed by the default warp-based block algorithm, and benchmarks as more performant.
  // Although this operator is commutative, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
  // benchmarked as slower on B200.
  static constexpr auto block_reduce_algorithm = cub::BLOCK_REDUCE_RAKING;

  __device__ m2_state<T> operator()(m2_state<T> const& a, m2_state<T> const& b) const
  {
    // Avoid inf * 0 for empty partials and retain non-finite singleton M2.
    if (a.count == 0) { return b; }
    if (b.count == 0) { return a; }
    // Make non-finite results independent of the order in which states are merged.
    if (!cuda::std::isfinite(a.mean) || !cuda::std::isfinite(b.mean)) {
      auto constexpr nan = cuda::std::numeric_limits<double>::quiet_NaN();
      return {a.count + b.count, nan, nan};
    }
    auto const count   = a.count + b.count;
    auto const delta   = b.mean - a.mean;
    auto const delta_n = count <= T{rows_per_chunk} ? delta * inverse_counts[count]
                                                    : delta / static_cast<double>(count);
    return {count, a.mean + delta_n * b.count, a.m2 + b.m2 + delta * delta_n * a.count * b.count};
  }

  // Thread-local folds see individual values; collective reductions see full states. This
  // is a simplification of the above merge for the case where the additional state
  // b.count == 1.
  __device__ m2_state<T> operator()(m2_state<T> const& a, m2_value const& b) const
  {
    if (!b.valid) { return a; }
    if (a.count == 0) { return static_cast<m2_state<T>>(b); }
    auto const delta = b.value - a.mean;
    // Preserve the general merge's non-finite/overflow behavior.
    if (!cuda::std::isfinite(delta)) { return (*this)(a, static_cast<m2_state<T>>(b)); }
    auto const count   = a.count + T{1};
    auto const delta_n = count <= T{rows_per_chunk} ? delta * inverse_counts[count]
                                                    : delta / static_cast<double>(count);
    auto const mean    = a.mean + delta_n;
    return {count, mean, a.m2 + delta * (b.value - mean)};
  }
};

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
