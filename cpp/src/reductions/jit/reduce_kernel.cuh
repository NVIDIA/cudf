/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <jit/type_list.cuh>

namespace cudf::jit {

namespace reduce {

template <typename... T>
struct state {
  using types = type_list<T...>;

  cuda::std::tuple<T...> values{};
  errc error{};
};

enum class phase : int8_t {
  REDUCE   = 0,  ///< Update input rows and reduce them into block partials.
  MERGE    = 1,  ///< Merge block partials into the final reduction state.
  FINALIZE = 2   ///< Finalize an already reduced state into output columns.
};

/** @brief Evaluate the filter before loading values for the update callback. */
template <bool NullAware,
          typename Inputs,
          typename FilterInputs,
          typename State,
          typename Filter,
          typename Update,
          typename... Context>
__device__ void update_row(State& state,
                           column_device_view_core const* inputs,
                           column_device_view_core const* filter_inputs,
                           size_type row,
                           Filter&& filter,
                           Update&& update,
                           Context... context)
{
  if constexpr (!NullAware) {
    auto valid = [&]<typename... A> { return (A::is_valid(inputs, row) && ... && true); };
    if (!Inputs::map(valid)) { return; }
  }

  auto load = [&]<typename Accessor>(column_device_view_core const* columns) {
    if constexpr (NullAware) {
      return Accessor::nullable_element(columns, row);
    } else {
      return Accessor::element(columns, row);
    }
  };

  bool selected     = false;
  auto filter_error = FilterInputs::map([&]<typename... A> {
    if constexpr (!NullAware) {
      if (!(A::is_valid(filter_inputs, row) && ... && true)) { return errc::SUCCESS; }
    }
    return filter(context..., row, &selected, load.template operator()<A>(filter_inputs)...);
  });
  state.error       = cuda::std::max(state.error, filter_error);
  if (filter_error != errc::SUCCESS || !selected) { return; }

  auto update_error = Inputs::map([&]<typename... A> {
    return [&]<int... I>(cuda::std::integer_sequence<int, I...>) {
      return update(context...,
                    row,
                    &cuda::std::get<I>(state.values)...,
                    load.template operator()<A>(inputs)...);
    }(State::types::indexed);
  });
  state.error       = cuda::std::max(state.error, update_error);
}

template <typename State, typename Merge, typename... Context>
__device__ State merge(State a, State const& b, Merge&& fn, Context&&... context)
{
  auto merge_error = [&]<int... I>(cuda::std::integer_sequence<int, I...>) {
    return fn(context..., &cuda::std::get<I>(a.values)..., cuda::std::get<I>(b.values)...);
  }(State::types::indexed);

  a.error = cuda::std::max(a.error, b.error);
  a.error = cuda::std::max(a.error, merge_error);

  return a;
}

template <typename T>
__device__ T shuffle_down(T value, int offset)
{
  constexpr auto num_words = cudf::detail::div_rounding_up_safe(sizeof(T), sizeof(uint32_t));
  uint32_t words[num_words]{};
  memcpy(words, &value, sizeof(T));
#pragma unroll
  for (int i = 0; i < num_words; ++i) {
    words[i] = __shfl_down_sync(0xFFFF'FFFFU, words[i], offset);
  }
  T result{};
  memcpy(&result, words, sizeof(T));
  return result;
}

template <int GroupSize, typename State, typename Merge>
  requires(GroupSize > 0 && GroupSize <= detail::warp_size && (GroupSize & (GroupSize - 1)) == 0)
__device__ State warp_combine(State local, Merge&& merge, bool active)
{
  auto rank = threadIdx.x % GroupSize;
#pragma unroll
  for (int offset = GroupSize / 2; offset != 0; offset /= 2) {
    auto other = shuffle_down(local, offset);
    if (active && rank < offset) { local = merge(local, other); }
  }
  return local;
}

template <int BlockSize, typename State, typename Merge>
  requires(BlockSize >= detail::warp_size && BlockSize <= 1024 &&
           BlockSize % detail::warp_size == 0 && (BlockSize & (BlockSize - 1)) == 0)
__device__ State block_combine(State local, Merge&& merge)
{
  constexpr int warps = BlockSize / detail::warp_size;
  auto lane           = threadIdx.x % detail::warp_size;
  auto warp           = threadIdx.x / detail::warp_size;
  local               = warp_combine<detail::warp_size>(local, merge, true);

  if constexpr (warps > 1) {
    // Raw storage avoids constructing shared states; only initialized warp leaders are read.
    __shared__ __align__(alignof(State)) unsigned char storage[warps * sizeof(State)];
    auto* leaders = reinterpret_cast<State*>(storage);

    if (lane == 0) { leaders[warp] = local; }

    __syncthreads();

    if (warp == 0) {
      if (lane < warps) { local = leaders[lane]; }

      local = warp_combine<warps>(local, merge, lane < warps);
    }
  }
  return local;
}

template <typename T>
CUDF_HOST_DEVICE constexpr T& deref_value(T& value)
{
  return value;
}

template <typename T>
CUDF_HOST_DEVICE constexpr T deref_value(cuda::std::optional<T> const& value)
{
  return value.value_or(T{});
}

template <typename T>
CUDF_HOST_DEVICE constexpr bool is_valid(cuda::std::optional<T> const& value)
{
  return value.has_value();
}

template <typename T>
CUDF_HOST_DEVICE constexpr bool is_valid(T const&)
{
  return true;
}

template <bool NullAware, typename Outputs, typename State, typename Finalize, typename... Context>
__device__ void finalize(State const& state,
                         mutable_column_device_view_core const* __restrict__ outputs,
                         errc* __restrict__ error,
                         Finalize&& finalize,
                         Context&&... context)
{
  Outputs::map([&]<typename... A> {
    auto values = [&] {
      if constexpr (NullAware) {
        return cuda::std::tuple{A::nullable_output_arg(outputs, 0)...};
      } else {
        return cuda::std::tuple{A::output_arg(outputs, 0)...};
      }
    }();

    auto err = [&]<int... I>(cuda::std::integer_sequence<int, I...>) {
      return finalize(
        context..., &cuda::std::get<A::index>(values)..., cuda::std::get<I>(state.values)...);
    }(State::types::indexed);

    auto assign = [&]<typename Out> {
      auto& value = cuda::std::get<Out::index>(values);
      Out::assign(outputs, 0, deref_value(value));
      Out::set_null_mask_word(outputs, 0, is_valid(value));
    };
    (assign.template operator()<A>(), ...);

    *error = cuda::std::max(state.error, err);
  });
}

}  // namespace reduce

/**
 * @brief Executes one phase of a user-defined scalar reduction.
 *
 * All threads participate in a grid-stride traversal and a block reduction:
 * - `REDUCE` initializes a state per thread, updates selected input rows, and writes one
 *   partial state per block to `results`.
 * - `MERGE` initializes an identity per thread, merges `num_rows` states from `partials`,
 *   and writes one reduced state per block to `results`.
 * - `FINALIZE` invokes the finalizer on `partials[0]` from global thread zero. It does not
 *   initialize or merge states, or use `results` and `scratch`.
 *
 * With `finalize_inline`, global thread zero also finalizes `results[0]` after `REDUCE` or
 * `MERGE`. This requires a single-block launch; the function has no grid-wide synchronization.
 * Dependent phases must execute in stream order. Empty inputs still reduce the identity state.
 *
 * @tparam NullAware Whether callbacks receive nullable input/output values
 * @tparam Inputs Type list of input accessors in update-parameter order
 * @tparam FilterInputs Type list of input accessors in filter-parameter order
 * @tparam Outputs Type list of output accessors in finalization-parameter order
 * @tparam BlockSize Number of threads per block
 * @tparam UseWarpReduce Use warp shuffles and shared warp leaders; otherwise use global scratch
 * @tparam Filter Callable selecting rows before the update callback
 * @tparam State State containing values and error
 * @tparam Init Callable `errc(state pointers...)` that initializes every field to its identity
 * @tparam Update Callable `errc(row, state pointers..., input values...)`
 * @tparam Merge Callable `errc(destination pointers..., source values...)`
 * @tparam Finalize Callable `errc(output pointers..., state values...)`
 *
 * @param phase Reduction phase to execute
 * @param num_rows Count of input rows
 * @param inputs Device input views consumed by the update accessors in `REDUCE`
 * @param filter_inputs Device input views consumed by the filter accessors in `REDUCE`
 * @param outputs Device array of `Outputs::size` single-row views, used only during finalization
 * @param partials Source states for `MERGE`, or one final state for `FINALIZE`
 * @param results Writable array of `gridDim.x` states
 * @param scratch Writable array of `gridDim.x * BlockSize` states when `UseWarpReduce` is false and
 * the phase is not `FINALIZE`
 * @param error Device error slot written only when finalization runs
 * @param finalize_inline Whether to finalize `results[0]` in a single-block reduction/merge launch
 */
template <bool NullAware,
          typename Inputs,
          typename FilterInputs,
          typename Outputs,
          int BlockSize,
          bool UseWarpReduce,
          typename State,
          typename Init,
          typename Filter,
          typename Update,
          typename Merge,
          typename Finalize>
  requires((BlockSize & (BlockSize - 1)) == 0)
__device__ void reduce_kernel(reduce::phase phase,
                              size_type num_rows,
                              column_device_view_core const* __restrict__ inputs,
                              column_device_view_core const* __restrict__ filter_inputs,
                              mutable_column_device_view_core const* __restrict__ outputs,
                              State const* __restrict__ partials,
                              State* __restrict__ results,
                              State* __restrict__ scratch,
                              errc* __restrict__ error,
                              bool finalize_inline,
                              Init&& init,
                              Filter&& filter,
                              Update&& update,
                              Merge&& merge,
                              Finalize&& finalize)
{
  if (phase == reduce::phase::FINALIZE) {
    if (detail::grid_1d::global_thread_id() == 0) {
      reduce::finalize<NullAware, Outputs>(*partials, outputs, error, finalize);
    }
    return;
  }

  State local;

  local.error = [&]<int... I>(cuda::std::integer_sequence<int, I...>) {
    return init(&cuda::std::get<I>(local.values)...);
  }(State::types::indexed);

  auto start  = detail::grid_1d::global_thread_id();
  auto stride = detail::grid_1d::grid_stride();

  auto do_merge = [&](auto const& a, auto const& b) { return reduce::merge(a, b, merge); };

  for (auto row = start; row < num_rows; row += stride) {
    if (phase == reduce::phase::MERGE) {
      local = do_merge(local, partials[row]);
      continue;
    }

    reduce::update_row<NullAware, Inputs, FilterInputs>(
      local, inputs, filter_inputs, row, filter, update);
  }

  if constexpr (UseWarpReduce) {
    auto result = reduce::block_combine<BlockSize>(local, do_merge);

    if (threadIdx.x == 0) { results[blockIdx.x] = result; }
  } else {
    // Very wide states create excessive register pressure with shuffles. This tree keeps the same
    // callback contract using disjoint, aligned global scratch slots for each block.
    auto* block        = scratch + static_cast<size_t>(blockIdx.x) * BlockSize;
    block[threadIdx.x] = local;

    __syncthreads();

    for (int offset = BlockSize / 2; offset != 0; offset /= 2) {
      if (threadIdx.x < offset) {
        block[threadIdx.x] = do_merge(block[threadIdx.x], block[threadIdx.x + offset]);
      }

      __syncthreads();
    }

    if (threadIdx.x == 0) { results[blockIdx.x] = block[0]; }
  }

  __syncthreads();

  if (finalize_inline) {
    if (detail::grid_1d::global_thread_id() == 0) {
      reduce::finalize<NullAware, Outputs>(results[0], outputs, error, finalize);
    }
  }
}

}  // namespace cudf::jit
