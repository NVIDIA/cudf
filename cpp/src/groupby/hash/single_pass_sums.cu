/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "single_pass_reductions.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/array>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace cudf::groupby::detail::hash {

/// Sums accumulated together when several additive aggregations are requested on one column.
template <typename Result, bool HasPadding = (alignof(Result) > alignof(size_type))>
struct fused_sums {
  Result sum;
  Result sum_of_squares;
  size_type count;
  // Materialize padding so vectorized loads of stored partials read initialized bytes.
  cuda::std::array<size_type, alignof(Result) / sizeof(size_type) - 1> padding{};
};

template <typename Result>
struct fused_sums<Result, false> {
  Result sum;
  Result sum_of_squares;
  size_type count;
};

template <typename Result>
struct fused_sums_plus {
  __device__ fused_sums<Result> operator()(fused_sums<Result> const& lhs,
                                           fused_sums<Result> const& rhs) const
  {
    return {lhs.sum + rhs.sum, lhs.sum_of_squares + rhs.sum_of_squares, lhs.count + rhs.count};
  }
};

/// Maps a grouped position to the sums contributed by the input row at that position.
template <typename Source, typename Result>
struct grouped_fused_sums_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  bool has_nulls;

  __device__ fused_sums<Result> operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return {Result{0}, Result{0}, 0}; }
    auto const result = static_cast<Result>(value(row));
    return {result, result * result, 1};
  }
};

/// Stores only the requested sums and any counts needed to derive their validity.
template <typename Result>
struct split_fused_sums_fn {
  Result* sum;
  Result* sum_of_squares;
  size_type* count;

  __device__ void operator()(cuda::std::ptrdiff_t group, fused_sums<Result> const& sums) const
  {
    if (sum != nullptr) { sum[group] = sums.sum; }
    if (sum_of_squares != nullptr) { sum_of_squares[group] = sums.sum_of_squares; }
    if (count != nullptr) { count[group] = sums.count; }
  }
};

/// Computes additive aggregations on one column, including the SUM and COUNT_VALID dependencies
/// of MEAN, M2, VARIANCE and STD, with a single grouped reduction.
struct fused_sums_fn {
  template <typename T>
    requires(cudf::detail::is_product_supported<T>())
  std::vector<std::unique_ptr<column>> operator()(reduction_context const& ctx,
                                                  host_span<aggregation::Kind const> kinds,
                                                  std::span<int8_t const> is_intermediate,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr) const
  {
    using Source = rep_type_t<T>;
    using Result = rep_type_t<cudf::detail::target_type_t<T, aggregation::SUM>>;
    static_assert(
      cuda::std::
        is_same_v<Result, rep_type_t<cudf::detail::target_type_t<T, aggregation::SUM_OF_SQUARES>>>);

    auto const needs_validity = [&] {
      if (!ctx.values.has_nulls()) { return false; }
      for (std::size_t i = 0; i < kinds.size(); ++i) {
        if (!is_intermediate[i] && kinds[i] != aggregation::COUNT_VALID) { return true; }
      }
      return false;
    }();
    // Keep the common accumulator while omitting outputs that no caller or mask consumes.
    auto const make_output = [&](aggregation::Kind kind,
                                 bool needed_for_mask = false) -> std::unique_ptr<column> {
      auto const it = std::find(kinds.begin(), kinds.end(), kind);
      if (it == kinds.end() && !needed_for_mask) { return nullptr; }
      auto const requested = it != kinds.end() && !is_intermediate[it - kinds.begin()];
      return make_fixed_width_column(cudf::detail::target_type(ctx.values_type, kind),
                                     ctx.num_groups,
                                     mask_state::UNALLOCATED,
                                     stream,
                                     requested ? mr.get_output_mr() : mr.get_temporary_mr());
    };
    auto sum            = make_output(aggregation::SUM);
    auto sum_of_squares = make_output(aggregation::SUM_OF_SQUARES);
    auto count          = make_output(aggregation::COUNT_VALID, needs_validity);
    // Moving a requested count column into results preserves the allocation used by later masks.
    auto const counts = count ? count->mutable_view().template begin<size_type>() : nullptr;
    if (ctx.num_groups > 0) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_fused_sums_fn<Source, Result>{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()});
      auto const outputs = cuda::tabulate_output_iterator{split_fused_sums_fn<Result>{
        sum ? sum->mutable_view().template begin<Result>() : nullptr,
        sum_of_squares ? sum_of_squares->mutable_view().template begin<Result>() : nullptr,
        counts}};
      reduce_groups(ctx.grouped,
                    values,
                    outputs,
                    fused_sums_plus<Result>{},
                    fused_sums<Result>{Result{0}, Result{0}, 0},
                    stream,
                    mr);
    }

    std::vector<std::unique_ptr<column>> results;
    column const* masked = nullptr;
    for (std::size_t i = 0; i < kinds.size(); ++i) {
      auto result = kinds[i] == aggregation::SUM              ? std::move(sum)
                    : kinds[i] == aggregation::SUM_OF_SQUARES ? std::move(sum_of_squares)
                                                              : std::move(count);
      // A sum is null when its group has no valid row, which the valid count already tells.
      auto const nullable =
        !is_intermediate[i] && kinds[i] != aggregation::COUNT_VALID && ctx.values.has_nulls();
      if (nullable && ctx.num_groups > 0) {
        if (masked == nullptr) {
          auto [null_mask, null_count] =
            make_mask_from_counts(counts, counts + ctx.num_groups, stream, mr);
          result->set_null_mask(std::move(null_mask), null_count);
          masked = result.get();
        } else {
          // Every sum of the column shares the validity of the first one.
          result->set_null_mask(
            cudf::detail::copy_bitmask(
              masked->view().null_mask(), 0, ctx.num_groups, stream, mr.get_output_mr()),
            masked->null_count());
        }
      }
      results.push_back(std::move(result));
    }
    return results;
  }

  template <typename T>
    requires(!cudf::detail::is_product_supported<T>())
  std::vector<std::unique_ptr<column>> operator()(reduction_context const&,
                                                  host_span<aggregation::Kind const>,
                                                  std::span<int8_t const>,
                                                  cuda::stream_ref,
                                                  cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for fused hash groupby sums");
  }
};

template std::unique_ptr<column> compute_reduction<aggregation::SUM>(reduction_context const& ctx,
                                                                     cuda::stream_ref stream,
                                                                     cudf::memory_resources mr);
template std::unique_ptr<column> compute_reduction<aggregation::SUM_OF_SQUARES>(
  reduction_context const& ctx, cuda::stream_ref stream, cudf::memory_resources mr);

std::vector<std::unique_ptr<column>> compute_fused_sums(reduction_context const& ctx,
                                                        host_span<aggregation::Kind const> kinds,
                                                        std::span<int8_t const> is_intermediate,
                                                        cuda::stream_ref stream,
                                                        cudf::memory_resources mr)
{
  return type_dispatcher(ctx.values_type, fused_sums_fn{}, ctx, kinds, is_intermediate, stream, mr);
}

template std::vector<std::unique_ptr<column>> compute_reductions<aggregation::SUM>(
  host_span<reduction_context const>,
  std::span<int8_t const>,
  cuda::stream_ref,
  cudf::memory_resources);

template std::vector<std::unique_ptr<column>> compute_reductions<aggregation::SUM_OF_SQUARES>(
  host_span<reduction_context const>,
  std::span<int8_t const>,
  cuda::stream_ref,
  cudf::memory_resources);

}  // namespace cudf::groupby::detail::hash
