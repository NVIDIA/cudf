/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "groupby/hash/single_pass_functors.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/warp/warp_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace cudf::groupby {

namespace {

constexpr int aggs_block_size      = 256;
constexpr int aggs_warps_per_block = aggs_block_size / cudf::detail::warp_size;
/// Bytes of `cub::WarpReduce` temporary storage reserved per warp.
constexpr int warp_reduce_storage_size = 64;

/*
 * Whether aggregation `k` of `Source` values reduces runs of equal keys within a warp.  The rest
 * update per element through `compute_single_pass_aggs_dense_output_fn`.
 */
template <typename Source, aggregation::Kind k>
CUDF_HOST_DEVICE constexpr bool has_warp_reduce_path()
{
  constexpr bool is_decimal = cuda::std::is_same_v<Source, numeric::decimal32> ||
                              cuda::std::is_same_v<Source, numeric::decimal64> ||
                              cuda::std::is_same_v<Source, numeric::decimal128>;
  constexpr bool is_numeric =
    (cuda::std::is_integral_v<Source> && !cuda::std::is_same_v<Source, bool>) ||
    cuda::std::is_floating_point_v<Source> || is_decimal;
  constexpr bool is_count  = k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL;
  constexpr bool is_sum    = k == aggregation::SUM || k == aggregation::SUM_OF_SQUARES;
  constexpr bool is_minmax = k == aggregation::MIN || k == aggregation::MAX;
  return cudf::detail::is_valid_aggregation<Source, k>() &&
         (is_count || (is_sum && is_numeric) || (is_minmax && is_numeric && !is_decimal));
}

struct has_warp_reduce_path_fn {
  template <typename Source, aggregation::Kind k>
  constexpr bool operator()() const noexcept
  {
    return has_warp_reduce_path<Source, k>();
  }
};

/*
 * Element aggregator that first reduces runs of consecutive lanes updating the same target.
 *
 * Every lane of the warp holds a row of the same column, so the whole warp takes the same
 * dispatch branch and the warp collectives see converged lanes.  Each run of equal `key`s is
 * reduced with `cub::WarpReduce::HeadSegmentedReduce` and only the head of the run updates the
 * target, which removes most of the atomic contention for sorted keys or long runs of equal keys.
 */
struct warp_reduce_aggregator {
  bool active;
  size_type key;
  size_type target_row;
  void* storage;

  template <typename Source, aggregation::Kind k>
  __device__ void operator()(mutable_column_device_view target,
                             size_type,
                             column_device_view source,
                             size_type source_index) const noexcept
  {
    constexpr bool is_count  = k == aggregation::COUNT_VALID || k == aggregation::COUNT_ALL;
    constexpr bool is_minmax = k == aggregation::MIN || k == aggregation::MAX;

    // Only columns with a warp path are launched through this aggregator.
    if constexpr (!has_warp_reduce_path<Source, k>()) {
      return;
    } else {
      using Target      = cudf::detail::target_type_t<Source, k>;
      using T           = cudf::device_storage_type_t<Target>;
      using S           = cudf::device_storage_type_t<Source>;
      using WarpReduce  = cub::WarpReduce<T>;
      using ValidReduce = cub::WarpReduce<int>;
      static_assert(sizeof(typename WarpReduce::TempStorage) <= warp_reduce_storage_size);
      static_assert(sizeof(typename ValidReduce::TempStorage) <= warp_reduce_storage_size);

      // Lanes without a valid input contribute the identity of the aggregation.
      auto const valid = active && (k == aggregation::COUNT_ALL || source.is_valid(source_index));
      auto const value = [&]() -> T {
        if constexpr (is_count) {
          return static_cast<T>(valid ? 1 : 0);
        } else if constexpr (is_minmax) {
          constexpr bool is_float = cuda::std::is_floating_point_v<T>;
          constexpr T identity    = k == aggregation::MIN
                                      ? (is_float ? cuda::std::numeric_limits<T>::infinity()
                                                  : cuda::std::numeric_limits<T>::max())
                                      : (is_float ? -cuda::std::numeric_limits<T>::infinity()
                                                  : cuda::std::numeric_limits<T>::lowest());
          return valid ? static_cast<T>(source.element<S>(source_index)) : identity;
        } else {
          auto const v = valid ? static_cast<T>(source.element<S>(source_index)) : T{0};
          if constexpr (k == aggregation::SUM_OF_SQUARES) { return v * v; }
          return v;
        }
      }();

      auto const lane     = threadIdx.x % cudf::detail::warp_size;
      auto const prev_key = __shfl_up_sync(0xffff'ffffu, key, 1);
      int const is_head   = lane == 0 || prev_key != key;
      auto& warp_storage  = *static_cast<typename WarpReduce::TempStorage*>(storage);
      auto& valid_storage = *static_cast<typename ValidReduce::TempStorage*>(storage);
      auto const reduced  = [&] {
        if constexpr (k == aggregation::MIN) {
          return WarpReduce(warp_storage).HeadSegmentedReduce(value, is_head, cudf::DeviceMin{});
        } else if constexpr (k == aggregation::MAX) {
          return WarpReduce(warp_storage).HeadSegmentedReduce(value, is_head, cudf::DeviceMax{});
        } else {
          return WarpReduce(warp_storage).HeadSegmentedSum(value, is_head);
        }
      }();
      // The two reductions share the temporary storage.
      __syncwarp();
      auto const num_valid = ValidReduce(valid_storage).HeadSegmentedSum(valid ? 1 : 0, is_head);

      if (!active || !is_head || num_valid == 0) { return; }
      auto* const element = &target.element<T>(target_row);
      if constexpr (k == aggregation::MIN) {
        cudf::detail::atomic_min(element, reduced);
      } else if constexpr (k == aggregation::MAX) {
        cudf::detail::atomic_max(element, reduced);
      } else {
        cudf::detail::atomic_add(element, reduced);
      }
      if constexpr (!is_count) {
        if (target.is_null(target_row)) { target.set_valid(target_row); }
      }
    }
  }
};

struct aggs_fn {
  size_type const* target_indices;
  aggregation::Kind const* aggs;
  table_device_view input_values;
  mutable_table_device_view output_values;

  __device__ void operator()(size_type col_idx, size_type row, void* storage) const
  {
    auto const lane     = static_cast<size_type>(threadIdx.x % cudf::detail::warp_size);
    auto const in_range = row < input_values.num_rows();
    auto const target   = in_range ? target_indices[row] : cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    auto const active   = target != cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
    // Inactive lanes get keys no group has, so they form runs of their own.
    auto const key         = active ? target : -1 - lane;
    auto const target_row  = active ? target : 0;
    auto const& source_col = input_values.column(col_idx);
    cudf::detail::dispatch_type_and_aggregation(
      source_col.type(),
      aggs[col_idx],
      warp_reduce_aggregator{active, key, target_row, storage},
      output_values.column(col_idx),
      target_row,
      source_col,
      in_range ? row : 0);
  }
};

/// Each warp processes 32 consecutive rows of one column per step.
template <typename Fn>
CUDF_KERNEL void __launch_bounds__(aggs_block_size)
  aggs_kernel(size_type num_rows, size_type num_cols, Fn fn)
{
  __shared__ alignas(16) char storage[aggs_warps_per_block][warp_reduce_storage_size];
  auto const warp          = threadIdx.x / cudf::detail::warp_size;
  auto const lane          = static_cast<size_type>(threadIdx.x % cudf::detail::warp_size);
  auto const warps_per_col = static_cast<int64_t>(
    cudf::util::div_rounding_up_safe(num_rows, static_cast<size_type>(cudf::detail::warp_size)));
  auto const num_steps = warps_per_col * num_cols;
  auto const stride    = static_cast<int64_t>(gridDim.x) * aggs_warps_per_block;
  for (auto step = static_cast<int64_t>(blockIdx.x) * aggs_warps_per_block + warp; step < num_steps;
       step += stride) {
    auto const col = static_cast<size_type>(step / warps_per_col);
    auto const row =
      static_cast<size_type>((step % warps_per_col) * cudf::detail::warp_size) + lane;
    fn(col, row, storage[warp]);
    __syncwarp();
  }
}

int max_active_blocks_aggs_kernel()
{
  int max_active_blocks{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, aggs_kernel<aggs_fn>, aggs_block_size, 0));
  return max_active_blocks;
}

}  // namespace

void streaming_groupby::impl::split_agg_columns(table_view const& values, cuda::stream_ref stream)
{
  auto const mr = cudf::get_current_device_resource_ref();
  for (size_type i = 0; i < values.num_columns(); ++i) {
    auto const warp_reduced = cudf::detail::dispatch_type_and_aggregation(
      values.column(i).type(), _agg_kinds[i], has_warp_reduce_path_fn{});
    (warp_reduced ? _warp_reduced_aggs : _elementwise_aggs).columns.push_back(i);
  }
  for (auto* subset : {&_warp_reduced_aggs, &_elementwise_aggs}) {
    if (subset->columns.empty()) { continue; }
    std::vector<aggregation::Kind> kinds;
    std::vector<mutable_column_view> results;
    for (auto const i : subset->columns) {
      kinds.push_back(_agg_kinds[i]);
      results.push_back(_agg_results->get_column(i).mutable_view());
    }
    subset->d_agg_kinds = std::make_unique<rmm::device_uvector<aggregation::Kind>>(
      cudf::detail::make_device_uvector_async(kinds, stream, mr));
    subset->d_results = mutable_table_device_view::create(mutable_table_view{results}, stream);
  }
}

void streaming_groupby::impl::do_aggregate(table_view const& data, cuda::stream_ref stream)
{
  ensure_not_invalidated();

  auto const batch_size = data.num_rows();
  if (batch_size == 0) { return; }

  CUDF_EXPECTS(batch_size <= _max_distinct_keys,
               "Batch size (" + std::to_string(batch_size) + ") exceeds max_distinct_keys (" +
                 std::to_string(_max_distinct_keys) + ").",
               std::invalid_argument);

  CUDF_EXPECTS(static_cast<int64_t>(_max_distinct_keys) + static_cast<int64_t>(batch_size) <=
                 static_cast<int64_t>(std::numeric_limits<size_type>::max()),
               "Transient key encoding (max_distinct_keys + batch_size) would overflow size_type.",
               std::invalid_argument);

  // The transient key encoding is only valid while a single insertion is in flight, so
  // insertion is serialized across concurrent callers on the host and, via the event, on the
  // device.  The aggregation below is per-group atomic and runs unserialized.
  auto const result = [&] {
    std::lock_guard const lock{_insert_mutex};

    // Re-check under the lock: another caller may have invalidated the object since the
    // fail-fast check above.
    ensure_not_invalidated();

    if (!_initialized) { initialize(data, stream); }

    auto const batch_keys = data.select(_key_indices);

    update_nullable_state(batch_keys);

    if (!_key_set) { create_key_set(stream); }

    _insert_done.wait(stream);
    auto inserted = probe_and_insert(batch_keys, stream);
    _insert_done.record(stream);
    return inserted;
  }();

  auto const values_view = data.select(_value_col_indices);

  if (!_warp_reduced_aggs.columns.empty()) {
    auto const d_values =
      table_device_view::create(values_view.select(_warp_reduced_aggs.columns), stream);
    auto const num_cols  = static_cast<size_type>(_warp_reduced_aggs.columns.size());
    auto const num_steps = static_cast<int64_t>(cudf::util::div_rounding_up_safe(
                             batch_size, static_cast<size_type>(cudf::detail::warp_size))) *
                           num_cols;
    auto const num_blocks = static_cast<int>(std::min<int64_t>(
      cudf::util::div_rounding_up_safe<int64_t>(num_steps, aggs_warps_per_block),
      static_cast<int64_t>(max_active_blocks_aggs_kernel()) * cudf::detail::num_multiprocessors()));
    aggs_kernel<<<num_blocks, aggs_block_size, 0, stream.get()>>>(
      batch_size,
      num_cols,
      aggs_fn{result.target_indices.begin(),
              _warp_reduced_aggs.d_agg_kinds->data(),
              *d_values,
              *_warp_reduced_aggs.d_results});
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  if (!_elementwise_aggs.columns.empty()) {
    auto const d_values =
      table_device_view::create(values_view.select(_elementwise_aggs.columns), stream);
    auto const num_items =
      static_cast<int64_t>(batch_size) * static_cast<int64_t>(_elementwise_aggs.columns.size());
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<int64_t>(0),
      num_items,
      detail::hash::compute_single_pass_aggs_dense_output_fn{result.target_indices.begin(),
                                                             _elementwise_aggs.d_agg_kinds->data(),
                                                             *d_values,
                                                             *_elementwise_aggs.d_results});
  }
}

}  // namespace cudf::groupby
