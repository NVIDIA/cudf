/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/hash/single_pass_reductions.cuh"
#include "warp_aggregate.hpp"

#include <cudf/detail/copy.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include <cstdint>

namespace cudf::groupby {
namespace {

namespace hash           = detail::hash;
namespace cg             = cooperative_groups;
constexpr int block_size = 256;

template <typename T, aggregation::Kind K>
constexpr bool is_warp_aggregation_supported = [] {
  if constexpr (cuda::std::is_same_v<T, numeric::decimal128> && K == aggregation::SUM) {
    return true;  // This kind has a dedicated existing 128-bit atomic addition.
  }
  if constexpr (hash::is_reduction_supported<T>(K) &&
                (K == aggregation::SUM || K == aggregation::SUM_OF_SQUARES ||
                 K == aggregation::PRODUCT || K == aggregation::MIN || K == aggregation::MAX)) {
    using Result = hash::rep_type_t<cudf::detail::target_type_t<T, K>>;
    return sizeof(hash::rep_type_t<T>) <= sizeof(int64_t) && sizeof(Result) <= sizeof(int64_t) &&
           cudf::has_atomic_support<Result>();
  }
  return false;
}();

struct aggregation_column {
  void* output{};
  bitmask_type* validity{};
};

struct aggregation_columns {
  aggregation_column sum, min, max, product, sum_of_squares, count_valid, count_all;

  aggregation_column& operator[](aggregation::Kind kind)
  {
    switch (kind) {
      case aggregation::SUM: return sum;
      case aggregation::MIN: return min;
      case aggregation::MAX: return max;
      case aggregation::PRODUCT: return product;
      case aggregation::SUM_OF_SQUARES: return sum_of_squares;
      case aggregation::COUNT_VALID: return count_valid;
      case aggregation::COUNT_ALL: return count_all;
      default: CUDF_FAIL("Unsupported kind for warp aggregation");
    }
  }
};

// Public CG collectives shuffle this trivially-copyable representation without narrowing it.
// Unsigned word arithmetic keeps every carry and avoids signed overflow in the local reduction.
struct int128_words {
  uint64_t low;
  uint64_t high;
};
static_assert(sizeof(int128_words) == sizeof(__int128_t));
static_assert(cuda::std::is_trivially_copyable_v<int128_words>);

struct int128_sum {
  __device__ int128_words operator()(int128_words lhs, int128_words rhs) const
  {
    auto const low = lhs.low + rhs.low;
    return {low, lhs.high + rhs.high + static_cast<uint64_t>(low < lhs.low)};
  }
};

template <typename T, typename Op>
__device__ T reduce_group_value(cg::coalesced_group const& thread_group, T value, Op operation)
{
  if constexpr (cuda::std::is_same_v<T, __int128_t>) {
    static_assert(cuda::std::is_same_v<Op, cudf::DeviceSum>);
    auto const bits = static_cast<__uint128_t>(value);
    auto const packed =
      int128_words{static_cast<uint64_t>(bits), static_cast<uint64_t>(bits >> 64)};
    auto const result = cg::reduce(thread_group, packed, int128_sum{});
    auto const output = (static_cast<__uint128_t>(result.high) << 64) | result.low;
    return cuda::std::bit_cast<__int128_t>(output);
  } else {
    return cg::reduce(thread_group, value, operation);
  }
}

template <typename T, aggregation::Kind K>
__device__ void aggregate_value(hash::rep_type_t<T> input,
                                cg::coalesced_group const& thread_group,
                                size_type target,
                                aggregation_column column)
{
  if constexpr (is_warp_aggregation_supported<T, K>) {
    if (column.output == nullptr) { return; }
    using Result = hash::rep_type_t<cudf::detail::target_type_t<T, K>>;
    using Op     = cudf::detail::corresponding_operator_t<K>;
    auto value   = static_cast<Result>(input);
    if constexpr (K == aggregation::SUM_OF_SQUARES) { value *= value; }
    auto combine = thread_group.size() > 1;
    if constexpr (cuda::std::is_floating_point_v<Result> &&
                  (cuda::std::is_same_v<Op, cudf::DeviceMin> ||
                   cuda::std::is_same_v<Op, cudf::DeviceMax>)) {
      if (combine) {
        // Existing MIN/MAX are order-sensitive for NaNs and signed-zero ties. Keep
        // every original atomic in those thread groups.
        auto const exceptional = cuda::std::isnan(value) || value == 0;
        combine = cg::reduce(thread_group, static_cast<int>(exceptional), cg::plus<int>{}) == 0;
      }
    }
    auto const combined = combine ? reduce_group_value(thread_group, value, Op{}) : value;
    if (!combine || thread_group.thread_rank() == 0) {
      if (column.validity != nullptr && !cudf::bit_is_set(column.validity, target)) {
        cudf::set_bit(column.validity, target);
      }
      auto* const output = static_cast<Result*>(column.output);
      if constexpr (cuda::std::is_same_v<Result, __int128_t>) {
        static_assert(cuda::std::is_same_v<Op, cudf::DeviceSum>);
        cudf::detail::atomic_add(output + target, combined);
      } else {
        cudf::detail::genericAtomicOperation(output + target, combined, Op{});
      }
    }
  }
}

__device__ void aggregate_count(cg::coalesced_group const& thread_group,
                                size_type target,
                                aggregation_column column)
{
  if (column.output != nullptr && thread_group.thread_rank() == 0) {
    cudf::detail::atomic_add(static_cast<size_type*>(column.output) + target,
                             static_cast<size_type>(thread_group.size()));
  }
}

// Equal targets can be noncontiguous within a warp; labeled_partition groups lanes with the same
// target. All requested aggregations share the input load and valid-row partition.
template <typename T>
CUDF_KERNEL void __launch_bounds__(block_size)
  aggregate_warp_kernel(size_type num_rows,
                        size_type const* targets,
                        detail::value_accessor<hash::rep_type_t<T>> values,
                        bool has_nulls,
                        aggregation_columns outputs)
{
  auto const row = cudf::detail::grid_1d::global_thread_id();
  if (row >= num_rows) { return; }
  auto const target = targets[row];
  if (target == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) { return; }

  // COUNT_ALL also includes rows with null values.
  if (outputs.count_all.output != nullptr) {
    auto const all_rows = cg::labeled_partition(cg::coalesced_threads(), target);
    aggregate_count(all_rows, target, outputs.count_all);
  }
  if (has_nulls && values.col.is_null_nocheck(row)) { return; }

  auto const thread_group = cg::labeled_partition(cg::coalesced_threads(), target);
  aggregate_count(thread_group, target, outputs.count_valid);
  if constexpr (is_warp_aggregation_supported<T, aggregation::SUM> ||
                is_warp_aggregation_supported<T, aggregation::MIN> ||
                is_warp_aggregation_supported<T, aggregation::MAX> ||
                is_warp_aggregation_supported<T, aggregation::PRODUCT> ||
                is_warp_aggregation_supported<T, aggregation::SUM_OF_SQUARES>) {
    // Count-only requests do not need to read or decode the input value.
    if (outputs.sum.output || outputs.min.output || outputs.max.output || outputs.product.output ||
        outputs.sum_of_squares.output) {
      auto const value = values(row);
      aggregate_value<T, aggregation::SUM>(value, thread_group, target, outputs.sum);
      aggregate_value<T, aggregation::MIN>(value, thread_group, target, outputs.min);
      aggregate_value<T, aggregation::MAX>(value, thread_group, target, outputs.max);
      aggregate_value<T, aggregation::PRODUCT>(value, thread_group, target, outputs.product);
      aggregate_value<T, aggregation::SUM_OF_SQUARES>(
        value, thread_group, target, outputs.sum_of_squares);
    }
  }
}

struct aggregation_dispatch {
  template <typename T>
    requires(!cudf::is_nested<T>() && !cuda::std::is_same_v<T, dictionary32>)
  void operator()(column_view const& input,
                  aggregation_columns outputs,
                  device_span<size_type const> targets,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr) const
  {
    auto const values   = column_device_view::create(input, stream, mr.get_temporary_mr());
    auto const num_rows = static_cast<size_type>(targets.size());
    cudf::detail::grid_1d const grid{num_rows, block_size};
    aggregate_warp_kernel<T><<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      num_rows,
      targets.data(),
      detail::value_accessor<hash::rep_type_t<T>>{*values},
      input.has_nulls(),
      outputs);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  template <typename T>
    requires(cudf::is_nested<T>() || cuda::std::is_same_v<T, dictionary32>)
  void operator()(column_view const&,
                  aggregation_columns,
                  device_span<size_type const>,
                  cuda::stream_ref,
                  cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for warp aggregation");
  }
};

}  // namespace

void compute_warp_aggregates(table_view const& values,
                             host_span<aggregation::Kind const> kinds,
                             device_span<size_type const> targets,
                             mutable_table_view const& output,
                             cuda::stream_ref stream,
                             cudf::memory_resources mr)
{
  for (size_type begin = 0; begin < values.num_columns();) {
    auto const& input = values.column(begin);
    aggregation_columns columns{};
    auto end = begin;
    while (end < values.num_columns() &&
           cudf::detail::is_shallow_equivalent(input, values.column(end))) {
      auto& column = columns[kinds[end]];
      // Separate requests can alias the same input. Keep both outputs for repeated kinds.
      if (column.output != nullptr) { break; }
      column = {output.column(end).head(), output.column(end).null_mask()};
      ++end;
    }
    auto const type = cudf::is_dictionary(input.type())
                        ? dictionary_column_view(input).keys().type()
                        : input.type();
    type_dispatcher(type, aggregation_dispatch{}, input, columns, targets, stream, mr);
    begin = end;
  }
}

}  // namespace cudf::groupby
