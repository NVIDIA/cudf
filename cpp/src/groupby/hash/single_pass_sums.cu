/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "single_pass_reductions.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace cudf::groupby::detail::hash {

/// Sums accumulated together when several additive aggregations are requested on one column.
template <typename Result>
struct fused_sums {
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

/// Splits the reduced sums into the SUM, SUM_OF_SQUARES and COUNT_VALID outputs.
template <typename Result>
struct split_fused_sums_fn {
  __device__ cuda::std::tuple<Result, Result, size_type> operator()(
    fused_sums<Result> const& sums) const
  {
    return {sums.sum, sums.sum_of_squares, sums.count};
  }
};

/// Extrema retain the input representation while SUM uses its promoted result type.
template <typename Source, typename Result>
struct fused_minmax_sum {
  Source minimum;
  Source maximum;
  Result sum;
  bool valid;
};

template <typename Source, typename Result>
struct fused_minmax_sum_op {
  bool compute_sum;

  __device__ fused_minmax_sum<Source, Result> operator()(
    fused_minmax_sum<Source, Result> const& lhs, fused_minmax_sum<Source, Result> const& rhs) const
  {
    return {cudf::detail::corresponding_operator_t<aggregation::MIN>{}(lhs.minimum, rhs.minimum),
            cudf::detail::corresponding_operator_t<aggregation::MAX>{}(lhs.maximum, rhs.maximum),
            compute_sum
              ? cudf::detail::corresponding_operator_t<aggregation::SUM>{}(lhs.sum, rhs.sum)
              : Result{0},
            lhs.valid || rhs.valid};
  }
};

template <typename Source, typename Result>
struct grouped_fused_minmax_sum_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  bool has_nulls;
  bool compute_sum;
  fused_minmax_sum<Source, Result> identity;

  __device__ fused_minmax_sum<Source, Result> operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return identity; }
    auto const result = value(row);
    return {result, result, compute_sum ? static_cast<Result>(result) : Result{0}, true};
  }
};

template <typename Source, typename Result>
struct split_fused_minmax_sum_fn {
  Source* minimum;
  Source* maximum;
  Result* sum;
  bool* valid;

  __device__ void operator()(cuda::std::ptrdiff_t group,
                             fused_minmax_sum<Source, Result> const& value) const
  {
    minimum[group] = value.minimum;
    maximum[group] = value.maximum;
    if (sum != nullptr) { sum[group] = value.sum; }
    if (valid != nullptr) { valid[group] = value.valid; }
  }
};

/// Computes the SUM, SUM_OF_SQUARES and COUNT_VALID aggregations requested on one column, as
/// extracted for MEAN, M2, VARIANCE and STD, with a single grouped reduction.
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

    // Every sum is reduced, but only explicitly requested results use the output resource.
    auto const make_output = [&](aggregation::Kind kind) {
      auto const it        = std::find(kinds.begin(), kinds.end(), kind);
      auto const requested = it != kinds.end() && !is_intermediate[it - kinds.begin()];
      return make_fixed_width_column(cudf::detail::target_type(ctx.values_type, kind),
                                     ctx.num_groups,
                                     mask_state::UNALLOCATED,
                                     stream,
                                     requested ? mr.get_output_mr() : mr.get_temporary_mr());
    };
    auto sum            = make_output(aggregation::SUM);
    auto sum_of_squares = make_output(aggregation::SUM_OF_SQUARES);
    auto count          = make_output(aggregation::COUNT_VALID);
    auto const counts   = count->view().template begin<size_type>();
    if (ctx.num_groups > 0) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_fused_sums_fn<Source, Result>{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()});
      auto const outputs = cuda::transform_output_iterator{
        cuda::make_zip_iterator(sum->mutable_view().template begin<Result>(),
                                sum_of_squares->mutable_view().template begin<Result>(),
                                count->mutable_view().template begin<size_type>()),
        split_fused_sums_fn<Result>{}};
      reduce_groups(ctx.grouped,
                    values,
                    outputs,
                    fused_sums_plus<Result>{},
                    fused_sums<Result>{Result{0}, Result{0}, 0},
                    stream,
                    mr);
    }

    std::vector<std::unique_ptr<column>> results;
    for (std::size_t i = 0; i < kinds.size(); ++i) {
      auto result = kinds[i] == aggregation::SUM              ? std::move(sum)
                    : kinds[i] == aggregation::SUM_OF_SQUARES ? std::move(sum_of_squares)
                                                              : std::move(count);
      // A sum is null when its group has no valid row, which the valid count already tells.
      auto const nullable =
        !is_intermediate[i] && kinds[i] != aggregation::COUNT_VALID && ctx.values.has_nulls();
      if (nullable && ctx.num_groups > 0) {
        auto [null_mask, null_count] =
          make_mask_from_counts(counts, counts + ctx.num_groups, stream, mr);
        result->set_null_mask(std::move(null_mask), null_count);
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

/// Reduces consecutive MIN/MAX/SUM requests on one input with one value load per row.
/// MIN/MAX pairs share this reducer without allocating or computing an unrequested SUM.
struct fused_minmax_sum_fn {
  template <typename T>
    requires(is_reduction_supported<T>(aggregation::SUM) &&
             is_reduction_supported<T>(aggregation::MIN) &&
             is_reduction_supported<T>(aggregation::MAX))
  std::vector<std::unique_ptr<column>> operator()(reduction_context const& ctx,
                                                  host_span<aggregation::Kind const> kinds,
                                                  std::span<int8_t const> is_intermediate,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr) const
  {
    using Source           = rep_type_t<T>;
    using Result           = rep_type_t<cudf::detail::target_type_t<T, aggregation::SUM>>;
    using Min              = cudf::detail::corresponding_operator_t<aggregation::MIN>;
    using Max              = cudf::detail::corresponding_operator_t<aggregation::MAX>;
    using Sum              = cudf::detail::corresponding_operator_t<aggregation::SUM>;
    auto const make_output = [&](aggregation::Kind kind) {
      auto const it        = std::find(kinds.begin(), kinds.end(), kind);
      auto const requested = it != kinds.end() && !is_intermediate[it - kinds.begin()];
      return make_fixed_width_column(cudf::detail::target_type(ctx.values_type, kind),
                                     ctx.num_groups,
                                     mask_state::UNALLOCATED,
                                     stream,
                                     requested ? mr.get_output_mr() : mr.get_temporary_mr());
    };
    auto minimum = make_output(aggregation::MIN);
    auto maximum = make_output(aggregation::MAX);
    auto sum     = std::find(kinds.begin(), kinds.end(), aggregation::SUM) == kinds.end()
                     ? nullptr
                     : make_output(aggregation::SUM);
    auto const needs_validity =
      ctx.values.has_nulls() &&
      std::ranges::any_of(is_intermediate, [](auto intermediate) { return !intermediate; });
    cuda::device_buffer<bool> group_valid(
      stream, mr.get_temporary_mr(), needs_validity ? ctx.num_groups : 0, cuda::no_init);
    if (ctx.num_groups > 0) {
      auto const identity = fused_minmax_sum<Source, Result>{Min::template identity<Source>(),
                                                             Max::template identity<Source>(),
                                                             Sum::template identity<Result>(),
                                                             false};
      auto const values   = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_fused_minmax_sum_fn<Source, Result>{ctx.grouped.rows.data(),
                                                      ctx.accessor<Source>(),
                                                      ctx.values.has_nulls(),
                                                      sum != nullptr,
                                                      identity});
      auto const outputs = cuda::tabulate_output_iterator{split_fused_minmax_sum_fn<Source, Result>{
        minimum->mutable_view().template begin<Source>(),
        maximum->mutable_view().template begin<Source>(),
        sum ? sum->mutable_view().template begin<Result>() : nullptr,
        group_valid.data()}};
      reduce_groups(ctx.grouped,
                    values,
                    outputs,
                    fused_minmax_sum_op<Source, Result>{sum != nullptr},
                    identity,
                    stream,
                    mr);
    }
    std::vector<std::unique_ptr<column>> results;
    for (std::size_t i = 0; i < kinds.size(); ++i) {
      auto result = kinds[i] == aggregation::MIN   ? std::move(minimum)
                    : kinds[i] == aggregation::MAX ? std::move(maximum)
                                                   : std::move(sum);
      if (!is_intermediate[i] && ctx.values.has_nulls() && ctx.num_groups > 0) {
        auto [null_mask, null_count] = make_mask_from_validity(
          group_valid.data(), group_valid.data() + group_valid.size(), stream, mr);
        result->set_null_mask(std::move(null_mask), null_count);
      }
      results.push_back(std::move(result));
    }
    return results;
  }

  template <typename T>
    requires(!(is_reduction_supported<T>(aggregation::SUM) &&
               is_reduction_supported<T>(aggregation::MIN) &&
               is_reduction_supported<T>(aggregation::MAX)))
  std::vector<std::unique_ptr<column>> operator()(reduction_context const&,
                                                  host_span<aggregation::Kind const>,
                                                  std::span<int8_t const>,
                                                  cuda::stream_ref,
                                                  cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for fused hash groupby extrema and sum");
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

std::vector<std::unique_ptr<column>> compute_fused_minmax_sum(
  reduction_context const& ctx,
  host_span<aggregation::Kind const> kinds,
  std::span<int8_t const> is_intermediate,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  return type_dispatcher(
    ctx.values_type, fused_minmax_sum_fn{}, ctx, kinds, is_intermediate, stream, mr);
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
