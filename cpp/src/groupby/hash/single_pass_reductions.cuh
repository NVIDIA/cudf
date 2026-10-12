/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_single_pass_aggs.hpp"
#include "grouped_reductions.cuh"
#include "single_pass_reductions.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/element_argminmax.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction/detail/sum_overflow.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {

template <typename T>
value_accessor<T> reduction_context::accessor() const
{
  return value_accessor<T>{d_values};
}

/// Maps a grouped position to the value of the input row at that position, substituting
/// `null_value` for null rows and optionally squaring the value.
template <typename Source, typename Target, bool Square>
struct grouped_value_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  Target null_value;
  bool has_nulls;

  __device__ bool col_is_null(size_type row) const
  {
    return has_nulls && value.col.is_null_nocheck(row);
  }

  __device__ Target compute(size_type row) const
  {
    auto const result = static_cast<Target>(value(row));
    if constexpr (Square) { return result * result; }
    return result;
  }

  __device__ Target operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    return col_is_null(row) ? null_value : compute(row);
  }
};

/// A reduced value together with whether any of the reduced rows was valid, so that a nullable
/// aggregation and its null mask come out of one pass.
template <typename Result>
struct valid_value {
  Result value;
  bool valid;
};

template <typename Op, typename Result>
struct valid_value_op {
  __device__ valid_value<Result> operator()(valid_value<Result> const& lhs,
                                            valid_value<Result> const& rhs) const
  {
    return {Op{}(lhs.value, rhs.value), lhs.valid || rhs.valid};
  }
};

/// Maps a grouped position to the value of the input row at that position and its validity; null
/// rows contribute the identity.
template <typename Source, typename Target, bool Square>
struct grouped_valid_value_fn {
  grouped_value_fn<Source, Target, Square> value;

  __device__ valid_value<Target> operator()(size_type position) const
  {
    auto const row = value.grouped_rows[position];
    if (value.col_is_null(row)) { return {value.null_value, false}; }
    return {value.compute(row), true};
  }
};

template <typename Result>
struct split_valid_value_fn {
  __device__ cuda::std::tuple<Result, bool> operator()(valid_value<Result> const& v) const
  {
    return {v.value, v.valid};
  }
};

/// Maps a grouped position to a SUM_OVERFLOW accumulator, treating nulls as a zero contribution.
template <typename DeviceType, bool Nullable = false>
struct grouped_sum_overflow_fn {
  size_type const* grouped_rows;
  value_accessor<DeviceType> value;
  bool has_nulls;

  using accumulator = cudf::reduction::detail::sum_overflow_result<DeviceType>;
  using result_type = cuda::std::conditional_t<Nullable, valid_value<accumulator>, accumulator>;

  __device__ result_type operator()(size_type position) const
  {
    auto const row    = grouped_rows[position];
    bool const valid  = !has_nulls || !value.col.is_null_nocheck(row);
    auto const result = accumulator{valid ? value(row) : DeviceType{0}, 0};
    if constexpr (Nullable) {
      return valid_value<accumulator>{result, valid};
    } else {
      return result;
    }
  }
};

/// Splits a reduced accumulator into the sum and overflow-flag children of the output struct.
template <typename DeviceType>
struct split_sum_overflow_fn {
  __device__ cuda::std::tuple<DeviceType, bool> operator()(
    cudf::reduction::detail::sum_overflow_result<DeviceType> const& accumulator) const
  {
    return {accumulator.sum, accumulator.wraps != 0};
  }
};

/// Representation used to share reduction instantiations across column types.
/// `device_storage_type_t` unwraps decimals but leaves chrono wrappers intact, so chrono columns
/// additionally use `T::rep` to reduce as their underlying integers.
template <typename T>
struct rep_type {
  using type = device_storage_type_t<T>;
};

template <typename T>
  requires(cudf::is_chrono<T>())
struct rep_type<T> {
  using type = typename T::rep;
};

template <typename T>
using rep_type_t = typename rep_type<T>::type;

template <aggregation::Kind K>
struct grouped_reduction_fn {
  template <typename T>
    requires(is_reduction_supported<T>(K) &&
             (K == aggregation::SUM || K == aggregation::PRODUCT ||
              K == aggregation::SUM_OF_SQUARES || K == aggregation::MIN || K == aggregation::MAX))
  std::unique_ptr<column> operator()(reduction_context const& ctx,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    using Source = rep_type_t<T>;
    using Result = rep_type_t<cudf::detail::target_type_t<T, K>>;
    using Op     = cudf::detail::corresponding_operator_t<K>;

    auto result = make_fixed_width_column(cudf::detail::target_type(ctx.values_type, K),
                                          ctx.num_groups,
                                          mask_state::UNALLOCATED,
                                          stream,
                                          mr.get_output_mr());
    if (ctx.num_groups == 0) { return result; }

    using value_fn      = grouped_value_fn<Source, Result, K == aggregation::SUM_OF_SQUARES>;
    auto const identity = Op::template identity<Result>();
    auto const value =
      value_fn{ctx.grouped.rows.data(), ctx.accessor<Source>(), identity, ctx.values.has_nulls()};
    auto const output = result->mutable_view().begin<Result>();
    if (!ctx.nullable) {
      reduce_groups(ctx.grouped,
                    cudf::detail::make_counting_transform_iterator(0, value),
                    output,
                    Op{},
                    identity,
                    stream,
                    mr);
      return result;
    }

    // The validity of a group (any valid row) rides along with its value in one pass.
    cuda::device_buffer<bool> group_valid(
      stream, mr.get_temporary_mr(), ctx.num_groups, cuda::no_init);
    auto const values = cudf::detail::make_counting_transform_iterator(
      0, grouped_valid_value_fn < Source, Result, K == aggregation::SUM_OF_SQUARES > {value});
    auto const outputs = cuda::transform_output_iterator{
      cuda::make_zip_iterator(output, group_valid.data()), split_valid_value_fn<Result>{}};
    reduce_groups(ctx.grouped,
                  values,
                  outputs,
                  valid_value_op<Op, Result>{},
                  valid_value<Result>{identity, false},
                  stream,
                  mr);
    auto [null_mask, null_count] = make_mask_from_validity(
      group_valid.data(), group_valid.data() + group_valid.size(), stream, mr);
    result->set_null_mask(std::move(null_mask), null_count);
    return result;
  }

  template <typename T>
    requires(is_reduction_supported<T>(K) && (K == aggregation::ARGMIN || K == aggregation::ARGMAX))
  std::unique_ptr<column> operator()(reduction_context const& ctx,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    auto result = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                          ctx.num_groups,
                                          mask_state::UNALLOCATED,
                                          stream,
                                          mr.get_output_mr());
    if (ctx.num_groups == 0) { return result; }

    // The selected row is valid iff the group contains any valid row. An all-null group
    // may select a null row or the sentinel, so check both before reading its validity bit.
    constexpr auto is_argmin = K == aggregation::ARGMIN;
    auto const reduce        = [&](auto output) {
      reduce_groups(ctx.grouped,
                    ctx.grouped.rows.begin(),
                    output,
                    cudf::detail::element_argminmax_fn<rep_type_t<T>>{
                      ctx.d_values, ctx.values.has_nulls(), is_argmin},
                    is_argmin ? cudf::detail::ARGMIN_SENTINEL : cudf::detail::ARGMAX_SENTINEL,
                    stream,
                    mr);
    };
    auto const out = result->mutable_view().begin<size_type>();
    auto null_mask =
      cudf::create_null_mask(ctx.num_groups,
                             ctx.nullable ? mask_state::ALL_VALID : mask_state::UNALLOCATED,
                             stream,
                             mr.get_output_mr());
    auto const mask = reinterpret_cast<bitmask_type*>(null_mask.data());
    reduce(cuda::tabulate_output_iterator{
      [out, mask, values = ctx.d_values] __device__(cuda::std::ptrdiff_t group, size_type row) {
        out[group] = row;
        if (mask != nullptr && (row < 0 || row >= values.size() || values.is_null(row))) {
          cudf::clear_bit(mask, static_cast<size_type>(group));
        }
      }});
    if (ctx.nullable) {
      auto const null_count = count_group_nulls(mask, ctx.num_groups, stream, mr);
      result->set_null_mask(std::move(null_mask), null_count);
    }
    return result;
  }

  template <typename T>
    requires(is_reduction_supported<T>(K) && K == aggregation::SUM_OVERFLOW)
  std::unique_ptr<column> operator()(reduction_context const& ctx,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    using Source      = rep_type_t<T>;
    using accumulator = cudf::reduction::detail::sum_overflow_result<Source>;

    auto sum_child = make_fixed_width_column(
      ctx.values_type, ctx.num_groups, mask_state::UNALLOCATED, stream, mr.get_output_mr());
    auto overflow_child = make_fixed_width_column(data_type{type_id::BOOL8},
                                                  ctx.num_groups,
                                                  mask_state::UNALLOCATED,
                                                  stream,
                                                  mr.get_output_mr());
    auto null_mask = cudf::create_null_mask(0, mask_state::UNALLOCATED, stream, mr.get_output_mr());
    size_type null_count{0};
    if (ctx.num_groups > 0) {
      if (ctx.nullable) {
        null_mask =
          cudf::create_null_mask(ctx.num_groups, mask_state::ALL_VALID, stream, mr.get_output_mr());
        auto const mask   = reinterpret_cast<bitmask_type*>(null_mask.data());
        auto const values = cudf::detail::make_counting_transform_iterator(
          0,
          grouped_sum_overflow_fn<Source, true>{
            ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()});
        auto const output = cuda::tabulate_output_iterator{
          [sums      = sum_child->mutable_view().begin<Source>(),
           overflows = overflow_child->mutable_view().begin<bool>(),
           mask] __device__(cuda::std::ptrdiff_t group, valid_value<accumulator> const& result) {
            sums[group]      = result.value.sum;
            overflows[group] = result.value.wraps != 0;
            if (!result.valid) { cudf::clear_bit(mask, static_cast<size_type>(group)); }
          }};
        reduce_groups(
          ctx.grouped,
          values,
          output,
          valid_value_op<cudf::reduction::detail::overflow_sum_op<Source>, accumulator>{},
          valid_value<accumulator>{accumulator{}, false},
          stream,
          mr);
        null_count = count_group_nulls(mask, ctx.num_groups, stream, mr);
      } else {
        auto const values = cudf::detail::make_counting_transform_iterator(
          0,
          grouped_sum_overflow_fn<Source>{
            ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()});
        auto const children = cuda::transform_output_iterator{
          cuda::make_zip_iterator(sum_child->mutable_view().begin<Source>(),
                                  overflow_child->mutable_view().begin<bool>()),
          split_sum_overflow_fn<Source>{}};
        reduce_groups(ctx.grouped,
                      values,
                      children,
                      cudf::reduction::detail::overflow_sum_op<Source>{},
                      accumulator{},
                      stream,
                      mr);
      }
    }

    std::vector<std::unique_ptr<column>> children;
    children.push_back(std::move(sum_child));
    children.push_back(std::move(overflow_child));
    return create_structs_hierarchy(ctx.num_groups,
                                    std::move(children),
                                    null_count,
                                    std::move(null_mask),
                                    stream,
                                    mr.get_output_mr());
  }

  template <typename T>
    requires(!is_reduction_supported<T>(K))
  std::unique_ptr<column> operator()(reduction_context const&,
                                     cuda::stream_ref,
                                     cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for hash groupby aggregation");
  }
};

/// Prepares compatible columns together, retaining each output's allocation resource.
template <aggregation::Kind K>
struct grouped_reductions_fn {
  template <typename T>
    requires(is_reduction_supported<T>(K))
  std::vector<std::unique_ptr<column>> operator()(host_span<reduction_context const> contexts,
                                                  std::span<int8_t const> is_intermediate,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr) const
  {
    using Source           = rep_type_t<T>;
    using Result           = rep_type_t<cudf::detail::target_type_t<T, K>>;
    using Op               = cudf::detail::corresponding_operator_t<K>;
    using value_fn         = grouped_value_fn<Source, Result, K == aggregation::SUM_OF_SQUARES>;
    auto const& first      = contexts.front();
    auto const num_columns = static_cast<size_type>(contexts.size());
    auto const temp_mr     = mr.get_temporary_mr();
    auto const identity    = Op::template identity<Result>();
    std::vector<std::unique_ptr<column>> results;
    results.reserve(num_columns);
    for (size_type i = 0; i < num_columns; ++i) {
      results.push_back(
        make_fixed_width_column(cudf::detail::target_type(contexts[i].values_type, K),
                                first.num_groups,
                                mask_state::UNALLOCATED,
                                stream,
                                is_intermediate[i] ? temp_mr : mr.get_output_mr()));
    }
    if (first.num_groups == 0) { return results; }

    auto const reduce = [&]<bool Nullable>() {
      using Value   = cuda::std::conditional_t<Nullable, valid_value<Result>, Result>;
      using ValueFn = cuda::std::conditional_t<
        Nullable,
        grouped_valid_value_fn<Source, Result, K == aggregation::SUM_OF_SQUARES>,
        value_fn>;
      using ValueIterator =
        decltype(cudf::detail::make_counting_transform_iterator(0, std::declval<ValueFn>()));
      auto const make_output = [](Result* data, bool* validity) {
        if constexpr (Nullable) {
          return cuda::transform_output_iterator{cuda::make_zip_iterator(data, validity),
                                                 split_valid_value_fn<Result>{}};
        } else {
          return data;
        }
      };
      using OutputIterator = decltype(make_output(nullptr, nullptr));
      using Descriptor     = column_reduction<ValueIterator, OutputIterator>;
      static_assert(cuda::std::is_trivially_copyable_v<Descriptor>);
      auto columns = cudf::detail::make_empty_host_vector<Descriptor>(num_columns, stream);
      cuda::device_buffer<bool> group_valid(
        stream,
        temp_mr,
        Nullable ? static_cast<std::size_t>(num_columns) * first.num_groups : 0,
        cuda::no_init);
      for (size_type i = 0; i < num_columns; ++i) {
        auto const& ctx  = contexts[i];
        auto const value = value_fn{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), identity, ctx.values.has_nulls()};
        auto const values = [&] {
          if constexpr (Nullable) {
            return cudf::detail::make_counting_transform_iterator(0, ValueFn{value});
          } else {
            return cudf::detail::make_counting_transform_iterator(0, value);
          }
        }();
        auto* validity =
          Nullable ? group_valid.data() + static_cast<std::size_t>(i) * first.num_groups : nullptr;
        columns.push_back(
          {values, make_output(results[i]->mutable_view().template begin<Result>(), validity)});
      }
      cuda::device_buffer<Descriptor> device_columns(
        stream, temp_mr, columns.size(), cuda::no_init);
      cudf::detail::cuda_memcpy(
        device_span<Descriptor>{device_columns.data(), device_columns.size()},
        host_span<Descriptor const>{columns},
        stream);
      auto const operation = [] {
        if constexpr (Nullable) {
          return valid_value_op<Op, Result>{};
        } else {
          return Op{};
        }
      }();
      Value const init = [&] {
        if constexpr (Nullable) {
          return Value{identity, false};
        } else {
          return identity;
        }
      }();
      reduce_group_columns(first.grouped,
                           reduction_columns{device_columns.data(), num_columns},
                           operation,
                           init,
                           stream,
                           mr);
      if constexpr (Nullable) {
        for (size_type i = 0; i < num_columns; ++i) {
          auto const begin = group_valid.data() + static_cast<std::size_t>(i) * first.num_groups;
          auto const resources = is_intermediate[i] ? cudf::memory_resources{temp_mr, temp_mr} : mr;
          auto [mask, null_count] =
            make_mask_from_validity(begin, begin + first.num_groups, stream, resources);
          results[i]->set_null_mask(std::move(mask), null_count);
        }
      }
    };
    if (first.nullable) {
      reduce.template operator()<true>();
    } else {
      reduce.template operator()<false>();
    }
    return results;
  }

  template <typename T>
    requires(!is_reduction_supported<T>(K))
  std::vector<std::unique_ptr<column>> operator()(host_span<reduction_context const>,
                                                  std::span<int8_t const>,
                                                  cuda::stream_ref,
                                                  cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for batched hash groupby aggregation");
  }
};

template <aggregation::Kind K>
std::vector<std::unique_ptr<column>> compute_reductions(host_span<reduction_context const> contexts,
                                                        std::span<int8_t const> is_intermediate,
                                                        cuda::stream_ref stream,
                                                        cudf::memory_resources mr)
{
  return type_dispatcher(contexts.front().values_type,
                         grouped_reductions_fn<K>{},
                         contexts,
                         is_intermediate,
                         stream,
                         mr);
}

template <aggregation::Kind K>
std::unique_ptr<column> compute_reduction(reduction_context const& ctx,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  return type_dispatcher(ctx.values_type, grouped_reduction_fn<K>{}, ctx, stream, mr);
}

}  // namespace cudf::groupby::detail::hash
