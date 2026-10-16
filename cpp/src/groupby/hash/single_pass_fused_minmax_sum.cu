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
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/array>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace cudf::groupby::detail::hash {

/// Extrema retain the input representation while SUM uses its promoted result type.
template <typename Source, typename Result>
struct fused_minmax_sum {
  static_assert(alignof(Result) >= alignof(Source));
  // Put SUM first to avoid interior padding for narrow Source types. Keep the prior state
  // size while naming every trailing byte, since CUB can vector-load stored partials.
  static constexpr auto alignment = alignof(Source) > alignof(Result) ? alignof(Source)
                                                                      : alignof(Result);
  static constexpr auto old_sum_offset =
    (2 * sizeof(Source) + alignof(Result) - 1) / alignof(Result) * alignof(Result);
  Result sum;
  Source minimum;
  Source maximum;
  bool valid;
  cuda::std::array<unsigned char, old_sum_offset + alignment - 2 * sizeof(Source) - 1> padding{};
};

template <typename Source, typename Result>
struct fused_minmax_sum_op {
  bool compute_sum;

  __device__ fused_minmax_sum<Source, Result> operator()(
    fused_minmax_sum<Source, Result> const& lhs, fused_minmax_sum<Source, Result> const& rhs) const
  {
    return {compute_sum
              ? cudf::detail::corresponding_operator_t<aggregation::SUM>{}(lhs.sum, rhs.sum)
              : Result{0},
            cudf::detail::corresponding_operator_t<aggregation::MIN>{}(lhs.minimum, rhs.minimum),
            cudf::detail::corresponding_operator_t<aggregation::MAX>{}(lhs.maximum, rhs.maximum),
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
    return {compute_sum ? static_cast<Result>(result) : Result{0}, result, result, true};
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
    if (minimum != nullptr) { minimum[group] = value.minimum; }
    if (maximum != nullptr) { maximum[group] = value.maximum; }
    if (sum != nullptr) { sum[group] = value.sum; }
    if (valid != nullptr) { valid[group] = value.valid; }
  }
};

/// Reduces consecutive MIN/MAX/SUM requests on one input with one value load per row.
/// MIN/MAX pairs share this reducer without allocating or computing an unrequested SUM.
/// The common accumulator is retained when an extremum output is not requested.
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
    auto const make_output = [&](aggregation::Kind kind) -> std::unique_ptr<column> {
      auto const it = std::find(kinds.begin(), kinds.end(), kind);
      if (it == kinds.end()) { return nullptr; }
      auto const requested = !is_intermediate[it - kinds.begin()];
      return make_fixed_width_column(cudf::detail::target_type(ctx.values_type, kind),
                                     ctx.num_groups,
                                     mask_state::UNALLOCATED,
                                     stream,
                                     requested ? mr.get_output_mr() : mr.get_temporary_mr());
    };
    auto minimum = make_output(aggregation::MIN);
    auto maximum = make_output(aggregation::MAX);
    auto sum     = make_output(aggregation::SUM);
    auto const needs_validity =
      ctx.values.has_nulls() &&
      std::ranges::any_of(is_intermediate, [](auto intermediate) { return !intermediate; });
    cuda::device_buffer<bool> group_valid(
      stream, mr.get_temporary_mr(), needs_validity ? ctx.num_groups : 0, cuda::no_init);
    if (ctx.num_groups > 0) {
      auto const identity = fused_minmax_sum<Source, Result>{Sum::template identity<Result>(),
                                                             Min::template identity<Source>(),
                                                             Max::template identity<Source>(),
                                                             false};
      auto const values   = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_fused_minmax_sum_fn<Source, Result>{ctx.grouped.rows.data(),
                                                      ctx.accessor<Source>(),
                                                      ctx.values.has_nulls(),
                                                      sum != nullptr,
                                                      identity});
      auto const outputs = cuda::tabulate_output_iterator{split_fused_minmax_sum_fn<Source, Result>{
        minimum ? minimum->mutable_view().template begin<Source>() : nullptr,
        maximum ? maximum->mutable_view().template begin<Source>() : nullptr,
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
    column const* masked = nullptr;
    for (std::size_t i = 0; i < kinds.size(); ++i) {
      auto result = kinds[i] == aggregation::MIN   ? std::move(minimum)
                    : kinds[i] == aggregation::MAX ? std::move(maximum)
                                                   : std::move(sum);
      if (!is_intermediate[i] && ctx.values.has_nulls() && ctx.num_groups > 0) {
        if (masked == nullptr) {
          auto [null_mask, null_count] = make_mask_from_validity(
            group_valid.data(), group_valid.data() + group_valid.size(), stream, mr);
          result->set_null_mask(std::move(null_mask), null_count);
          masked = result.get();
        } else {
          // Every extremum and sum of the column shares the validity of the first one.
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

}  // namespace cudf::groupby::detail::hash
