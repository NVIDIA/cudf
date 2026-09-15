/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "single_pass_reductions.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/element_argminmax.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction/detail/sum_overflow.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/detail/env_dispatch.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/dispatch/dispatch_segmented_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_reduce.cuh>
#include <cub/util_device.cuh>
#include <cuda/devices>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <cuda/version>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash::single_pass {

/// Reads a fixed-width element, going through the keys when the column is a dictionary.
template <typename T>
struct value_accessor {
  column_device_view col;
  bool is_dictionary;

  __device__ T operator()(size_type row) const
  {
    if (is_dictionary) {
      auto const keys = col.child(dictionary_column_view::keys_column_index);
      return keys.element<T>(static_cast<size_type>(col.element<dictionary32>(row)));
    }
    return col.element<T>(row);
  }
};

template <typename T>
value_accessor<T> reduction_context::accessor() const
{
  return {d_values, is_dictionary(values.type())};
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
template <typename DeviceType>
struct grouped_sum_overflow_fn {
  size_type const* grouped_rows;
  value_accessor<DeviceType> value;
  bool has_nulls;

  __device__ cudf::reduction::detail::sum_overflow_result<DeviceType> operator()(
    size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return {DeviceType{0}, 0}; }
    return {value(row), 0};
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

constexpr bool is_fusable_sum(aggregation::Kind kind)
{
  return kind == aggregation::SUM || kind == aggregation::SUM_OF_SQUARES ||
         kind == aggregation::COUNT_VALID;
}

/// Groups this small on average are reduced as segments packed several per block: one block per
/// segment would leave most of the device idle.
constexpr size_type min_avg_rows_per_segment = 128;

/// Groups longer than this are reduced chunk by chunk so that every block has a bounded range.
constexpr size_type rows_per_chunk = 1 << 14;

/// Chunk length when the segments are packed several per block, so that a thread or a sub-warp
/// never walks a long group alone.
constexpr size_type packed_rows_per_chunk = 1 << 10;

/// Threads that share one segment on the packed path.
constexpr int packed_threads_per_group = 8;

// The packed path calls `cub::detail::segmented_reduce::dispatch` directly, an internal entry point
// whose namespace, signature and hint semantics were verified in CCCL 3.5 only.
static_assert(CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 5,
              "re-verify cub::detail::segmented_reduce::dispatch and the max_segment_size hint "
              "(dispatch_segmented_reduce.cuh, kernels/kernel_segmented_reduce.cuh) for this CCCL");

/**
 * @brief Policy selector for cub::DeviceSegmentedReduce whose medium path hands each segment to
 * `packed_threads_per_group` threads instead of a full warp.
 *
 * The large (one block per segment) and small (one thread per segment) policies are those of
 * CUB's own selector, so the large path reduces in exactly the order it does today. Only the
 * medium tile changes: `packed_threads_per_group * items_per_thread` rows instead of
 * `32 * items_per_thread`, and `threads_per_block / packed_threads_per_group` segments per block.
 *
 * Verified against cub/device/dispatch/tuning/tuning_segmented_reduce.cuh:23-63 (the policy
 * structs), :101-140 (the default selector) and cub/util_device.cuh:865 (the selector concept:
 * stateless, `operator()(cuda::compute_capability) -> cub::SegmentedReducePolicy`). The kernel
 * evaluates it as a constant expression for `__launch_bounds__` (kernel_segmented_reduce.cuh:113),
 * hence `constexpr` and host/device.
 */
template <typename AccumT, typename OffsetT, typename Op>
struct subwarp_segmented_reduce_policy_selector {
  [[nodiscard]] CUDF_HOST_DEVICE constexpr cub::SegmentedReducePolicy operator()(
    cuda::compute_capability cc) const
  {
    auto const base =
      cub::detail::segmented_reduce::policy_selector_from_types<AccumT, OffsetT, Op>{}(cc);
    auto const& large = base.large_reduce;
    return cub::SegmentedReducePolicy{large,
                                      cub::SegmentedReduceWarpReducePolicy{large.threads_per_block,
                                                                           packed_threads_per_group,
                                                                           large.items_per_thread,
                                                                           large.vec_size,
                                                                           large.load_modifier},
                                      base.small_reduce};
  }
};

/// The stream and temporary-storage resource handed to the CUB algorithms. Spelled out without
/// class template argument deduction, which the host compiler rejects for these types outside of
/// a template.
using cub_stream_prop_t = cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>;
using cub_mr_prop_t =
  cuda::std::execution::prop<cuda::mr::get_memory_resource_t, rmm::device_async_resource_ref>;
using cub_env_t = cuda::std::execution::env<cub_stream_prop_t, cub_mr_prop_t>;

inline cub_env_t make_cub_env(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  return cub_env_t{cub_stream_prop_t{cuda::get_stream_t{}, stream},
                   cub_mr_prop_t{cuda::mr::get_memory_resource_t{}, mr}};
}

/**
 * @brief Reduces segments with CUB, packing several per block when `avg_rows` is positive.
 *
 * `cub::DeviceSegmentedReduce::Reduce` always launches one block per segment. Its dispatch also
 * accepts a segment size hint, which is what makes it hand every segment to one thread (hint
 * within the small tile) or to one sub-warp (hint within the medium tile) instead; the kernel's
 * agents loop over as many tiles as a segment has, so a segment longer than the hint is still
 * reduced correctly, only by that thread or sub-warp alone. Segments averaging at most the small
 * tile of the accumulator get a thread each, longer ones a sub-warp each.
 *
 * @param avg_rows Average number of rows per segment; zero selects one block per segment
 */
template <typename ValueIterator, typename OutputIterator, typename Op, typename T>
void reduce_segments(device_span<size_type const> offsets,
                     size_type avg_rows,
                     ValueIterator values,
                     OutputIterator output,
                     Op op,
                     T init,
                     cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr)
{
  using offset_type   = cub::detail::common_iterator_value_t<size_type const*, size_type const*>;
  using accum_type    = cuda::std::__accumulator_t<Op, cub::detail::it_value_t<ValueIterator>, T>;
  using selector_type = subwarp_segmented_reduce_policy_selector<accum_type, offset_type, Op>;

  std::size_t hint = 0;
  if (avg_rows > 0) {
    cuda::compute_capability cc{};
    CUDF_CUDA_TRY(cub::detail::ptx_compute_cap(cc));
    auto const policy      = selector_type{}(cc);
    auto const small_tile  = policy.small_reduce.items_per_tile();
    auto const medium_tile = policy.medium_reduce.items_per_tile();
    CUDF_EXPECTS(small_tile + 1 <= medium_tile, "Unexpected segmented reduce tuning");
    hint = avg_rows <= small_tile ? std::size_t{1} : static_cast<std::size_t>(small_tile) + 1;
  }
  CUDF_CUDA_TRY(cub::detail::dispatch_with_env(
    make_cub_env(stream, mr),
    [&](auto, void* d_temp_storage, std::size_t& temp_storage_bytes, cudaStream_t launch_stream) {
      return cub::detail::segmented_reduce::dispatch<accum_type, offset_type>(
        d_temp_storage,
        temp_storage_bytes,
        values,
        output,
        static_cast<cuda::std::int64_t>(offsets.size() - 1),
        offsets.begin(),
        offsets.begin() + 1,
        op,
        init,
        hint,
        launch_stream,
        selector_type{});
    }));
}

/// Reduces the grouped values of every group into one output element per group. `packed_rows`
/// estimates the rows requiring value loads per segment; zero selects one block per segment.
template <typename ValueIterator, typename OutputIterator, typename Op, typename T>
void reduce_groups(grouped_rows const& grouped,
                   size_type packed_rows,
                   ValueIterator values,
                   OutputIterator output,
                   Op op,
                   T init,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr)
{
  if (grouped.group_chunks.is_empty()) {
    reduce_segments(grouped.offsets, packed_rows, values, output, op, init, stream, mr);
    return;
  }
  rmm::device_uvector<T> partials(grouped.chunk_offsets.size() - 1, stream, mr);
  reduce_segments(
    grouped.chunk_offsets, packed_rows, values, partials.begin(), op, init, stream, mr);
  reduce_segments(grouped.group_chunks,
                  packed_rows > 0 ? packed_rows_per_chunk : 0,
                  partials.begin(),
                  output,
                  op,
                  init,
                  stream,
                  mr);
}

/// The device representation of a column element. Chrono and fixed-point columns reduce as their
/// integer reps, so the reduction kernels are only instantiated once per representation.
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

template <aggregation::Kind K, typename T>
constexpr bool is_reduction_supported()
{
  switch (K) {
    case aggregation::SUM:
      return cudf::is_numeric<T>() || cudf::is_duration<T>() || cudf::is_fixed_point<T>();
    case aggregation::PRODUCT:
    case aggregation::SUM_OF_SQUARES: return cudf::detail::is_product_supported<T>();
    case aggregation::MIN:
    case aggregation::MAX: return cudf::is_fixed_width<T>() && is_relationally_comparable<T, T>();
    case aggregation::ARGMIN:
    case aggregation::ARGMAX: return is_relationally_comparable<T, T>();
    case aggregation::SUM_OVERFLOW: return cudf::detail::sum_overflow_supported<T>;
    default: return false;
  }
}

template <aggregation::Kind K>
struct reduce_fn {
  template <typename T>
    requires(is_reduction_supported<K, T>() &&
             (K == aggregation::SUM || K == aggregation::PRODUCT ||
              K == aggregation::SUM_OF_SQUARES || K == aggregation::MIN || K == aggregation::MAX))
  std::unique_ptr<column> operator()(reduction_context const& ctx, cudf::memory_resources mr) const
  {
    using Source = rep_type_t<T>;
    using Result = rep_type_t<cudf::detail::target_type_t<T, K>>;
    using Op     = cudf::detail::corresponding_operator_t<K>;

    auto result = make_fixed_width_column(cudf::detail::target_type(ctx.values_type, K),
                                          ctx.num_groups,
                                          mask_state::UNALLOCATED,
                                          ctx.stream,
                                          mr.get_output_mr());
    if (ctx.num_groups == 0) { return result; }

    using value_fn      = grouped_value_fn<Source, Result, K == aggregation::SUM_OF_SQUARES>;
    auto const identity = Op::template identity<Result>();
    auto const value =
      value_fn{ctx.grouped.rows.data(), ctx.accessor<Source>(), identity, ctx.values.has_nulls()};
    auto const output = result->mutable_view().begin<Result>();
    if (!ctx.nullable) {
      reduce_groups(ctx.grouped,
                    ctx.grouped.packed_rows,
                    cudf::detail::make_counting_transform_iterator(0, value),
                    output,
                    Op{},
                    identity,
                    ctx.stream,
                    mr.get_temporary_mr());
      return result;
    }

    // The validity of a group (any valid row) rides along with its value in one pass.
    rmm::device_uvector<bool> group_valid(ctx.num_groups, ctx.stream, mr.get_temporary_mr());
    auto const values = cudf::detail::make_counting_transform_iterator(
      0, grouped_valid_value_fn < Source, Result, K == aggregation::SUM_OF_SQUARES > {value});
    auto const outputs = cuda::transform_output_iterator{
      cuda::make_zip_iterator(output, group_valid.begin()), split_valid_value_fn<Result>{}};
    auto packed_rows = ctx.grouped.packed_rows;
    if (packed_rows > 0 && ctx.grouped.group_chunks.is_empty()) {
      // Null rows still read their validity, but skip the value load. Use the expected number
      // of valid rows per group when choosing how many threads share a segment.
      auto const valid_rows =
        static_cast<cuda::std::int64_t>(ctx.values.size() - ctx.values.null_count());
      auto const avg_valid_rows = cudf::util::div_rounding_up_safe(
        packed_rows * valid_rows, static_cast<cuda::std::int64_t>(ctx.values.size()));
      packed_rows = static_cast<size_type>(std::max<cuda::std::int64_t>(avg_valid_rows, 1));
    }
    reduce_groups(ctx.grouped,
                  packed_rows,
                  values,
                  outputs,
                  valid_value_op<Op, Result>{},
                  valid_value<Result>{identity, false},
                  ctx.stream,
                  mr.get_temporary_mr());
    auto [null_mask, null_count] = cudf::detail::valid_if(
      group_valid.begin(), group_valid.end(), cuda::std::identity{}, ctx.stream, mr);
    result->set_null_mask(std::move(null_mask), null_count);
    return result;
  }

  template <typename T>
    requires(is_reduction_supported<K, T>() &&
             (K == aggregation::ARGMIN || K == aggregation::ARGMAX))
  std::unique_ptr<column> operator()(reduction_context const& ctx, cudf::memory_resources mr) const
  {
    auto result = make_size_type_column(ctx, mr);
    if (ctx.num_groups == 0) { return result; }

    // The grouped rows are the input row indices themselves, so reducing them with the
    // element comparator yields the input index of each group's extremum. The sentinel identity
    // loses against every valid row and is left in place for all-null groups.
    constexpr auto is_argmin = K == aggregation::ARGMIN;
    reduce_groups(ctx.grouped,
                  ctx.grouped.packed_rows,
                  ctx.grouped.rows.begin(),
                  result->mutable_view().begin<size_type>(),
                  cudf::detail::element_argminmax_fn<rep_type_t<T>>{
                    ctx.d_values, ctx.values.has_nulls(), is_argmin},
                  is_argmin ? cudf::detail::ARGMIN_SENTINEL : cudf::detail::ARGMAX_SENTINEL,
                  ctx.stream,
                  mr.get_temporary_mr());
    set_group_null_mask(*result, ctx, mr);
    return result;
  }

  template <typename T>
    requires(is_reduction_supported<K, T>() && K == aggregation::SUM_OVERFLOW)
  std::unique_ptr<column> operator()(reduction_context const& ctx, cudf::memory_resources mr) const
  {
    using Source      = rep_type_t<T>;
    using accumulator = cudf::reduction::detail::sum_overflow_result<Source>;

    auto sum_child = make_fixed_width_column(
      ctx.values_type, ctx.num_groups, mask_state::UNALLOCATED, ctx.stream, mr.get_output_mr());
    auto overflow_child = make_fixed_width_column(data_type{type_id::BOOL8},
                                                  ctx.num_groups,
                                                  mask_state::UNALLOCATED,
                                                  ctx.stream,
                                                  mr.get_output_mr());
    if (ctx.num_groups > 0) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_sum_overflow_fn<Source>{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()});
      auto const children = cuda::transform_output_iterator{
        cuda::make_zip_iterator(sum_child->mutable_view().begin<Source>(),
                                overflow_child->mutable_view().begin<bool>()),
        split_sum_overflow_fn<Source>{}};
      reduce_groups(ctx.grouped,
                    ctx.grouped.packed_rows,
                    values,
                    children,
                    cudf::reduction::detail::overflow_sum_op<Source>{},
                    accumulator{},
                    ctx.stream,
                    mr.get_temporary_mr());
    }

    auto [null_mask, null_count] = ctx.nullable && ctx.num_groups > 0
                                     ? reduce_group_validity(ctx, mr)
                                     : std::pair{rmm::device_buffer{}, size_type{0}};
    std::vector<std::unique_ptr<column>> children;
    children.push_back(std::move(sum_child));
    children.push_back(std::move(overflow_child));
    return create_structs_hierarchy(ctx.num_groups,
                                    std::move(children),
                                    null_count,
                                    std::move(null_mask),
                                    ctx.stream,
                                    mr.get_output_mr());
  }

  template <typename T>
    requires(!is_reduction_supported<K, T>())
  std::unique_ptr<column> operator()(reduction_context const&, cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for hash groupby aggregation");
  }
};

/// Computes the SUM, SUM_OF_SQUARES and COUNT_VALID aggregations requested on one column, as
/// extracted for MEAN, M2, VARIANCE and STD, with a single segmented reduction.
struct fused_sums_fn {
  template <typename T>
    requires(cudf::detail::is_product_supported<T>())
  std::vector<std::unique_ptr<column>> operator()(reduction_context const& ctx,
                                                  host_span<aggregation::Kind const> kinds,
                                                  std::span<int8_t const> is_intermediate,
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
                                     ctx.stream,
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
                    ctx.grouped.packed_rows,
                    values,
                    outputs,
                    fused_sums_plus<Result>{},
                    fused_sums<Result>{Result{0}, Result{0}, 0},
                    ctx.stream,
                    mr.get_temporary_mr());
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
        auto [null_mask, null_count] = cudf::detail::valid_if(
          counts,
          counts + ctx.num_groups,
          [] __device__(size_type count) { return count > 0; },
          ctx.stream,
          mr);
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
                                                  cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for fused hash groupby sums");
  }
};

template <aggregation::Kind K>
std::unique_ptr<column> compute_reduction(reduction_context const& ctx, cudf::memory_resources mr)
{
  return type_dispatcher(ctx.values_type, reduce_fn<K>{}, ctx, mr);
}

}  // namespace cudf::groupby::detail::hash::single_pass
