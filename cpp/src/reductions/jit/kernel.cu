/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/errc.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/atomic>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <jit/column_accessor.cuh>
#include <jit/sync.cuh>
#include <jit/type_list.cuh>

#pragma nv_hdrstop

#include <cudf/detail/kernel_instance.cuh>
namespace cudf::jit {
// Shuffle fields independently: state padding is never read, and wrappers and sub-word
// fields retain their bit representation. Constant-sized copies lower to register moves.
template <typename T>
__device__ T reduction_shuffle_down(T value, int offset)
{
  static_assert(cuda::std::is_trivially_copyable_v<T>);
  cuda::std::array<uint32_t, (sizeof(T) + sizeof(uint32_t) - 1) / sizeof(uint32_t)> words{};
  cuda::std::memcpy(words.data(), &value, sizeof(T));
#pragma unroll
  for (int i = 0; i < words.size(); ++i) {
    words[i] = __shfl_down_sync(0xffff'ffffu, words[i], offset);
  }
  T result{};
  cuda::std::memcpy(&result, words.data(), sizeof(T));
  return result;
}

// Fixed-point wrappers have internal padding. Shuffle their representation and scale
// explicitly so no indeterminate padding bits are used as shuffle operands.
template <typename Rep, numeric::Radix Rad>
__device__ numeric::fixed_point<Rep, Rad> reduction_shuffle_down(
  numeric::fixed_point<Rep, Rad> value, int offset)
{
  auto rep   = reduction_shuffle_down(value.value(), offset);
  auto scale = reduction_shuffle_down(static_cast<int32_t>(value.scale()), offset);
  return numeric::fixed_point<Rep, Rad>{
    numeric::scaled_integer<Rep>{rep, numeric::scale_type{scale}}};
}
}  // namespace cudf::jit

#include <cudf/detail/operation_udf.cuh>

namespace cudf::jit {

// Normalize transform's CUDA callback conventions, while LTO always returns int.
template <typename F, typename... Args>
__device__ int32_t reduction_invoke(F f, Args... args)
{
  if constexpr (cuda::std::is_void_v<decltype(f(args...))>) {
    f(args...);
    return 0;
  } else {
    return static_cast<int32_t>(f(args...));
  }
}

template <bool HasUserData, typename F, typename Args>
__device__ int32_t reduction_apply(F f, void* user_data, Args args)
{
  return cuda::std::apply(
    [&](auto... a) {
      if constexpr (HasUserData) {
        return reduction_invoke(f, user_data, a...);
      } else {
        return reduction_invoke(f, a...);
      }
    },
    args);
}

// Dictionary accessors decode to a key value. Form the optional from that value type rather
// than from the dictionary type tag used to select the device-view accessor.
template <typename Accessor>
__device__ auto reduction_input(column_device_view_core const* input_cols, size_type row)
{
  using value_type = cuda::std::remove_cvref_t<decltype(Accessor::element(input_cols, row))>;
  if (Accessor::is_valid(input_cols, row)) {
    return cuda::std::optional<value_type>{Accessor::element(input_cols, row)};
  }
  return cuda::std::optional<value_type>{};
}

template <bool HasUserData>
struct reduction_merge {
  void* user_data;

  __device__ reduce_state operator()(reduce_state a, reduce_state const& b) const
  {
    a.error = cuda::std::max(a.error, b.error);
    // Even a callback that failed without writing its state cannot expose uninitialized values
    // to a subsequent callback. The value-initialized fields may still be copied by the block
    // combine.
    if (a.error == 0) {
      a.error =
        reduction_apply<HasUserData>([](auto... args) { return cudf_reduce_merge(args...); },
                                     user_data,
                                     cuda::std::tuple_cat(a.pointers(), b.values()));
    }
    return a;
  }
};

// All lanes execute each shuffle, including lanes with no input rows. Only the
// participating tree nodes invoke merge; inactive lanes cannot contribute errors/state.
template <int BlockSize, typename Merge>
__device__ reduce_state reduction_block_combine(reduce_state local, Merge merge)
{
  static_assert(BlockSize >= detail::warp_size && BlockSize % detail::warp_size == 0);
  constexpr int warps = BlockSize / detail::warp_size;
  auto const lane     = threadIdx.x % detail::warp_size;
  auto const warp     = threadIdx.x / detail::warp_size;
#pragma unroll
  for (int offset = detail::warp_size / 2; offset != 0; offset /= 2) {
    auto other = local.shuffle_down(offset);
    if (lane < offset) { local = merge(local, other); }
  }
  if constexpr (warps > 1) {
    // Raw storage avoids constructing shared states; only initialized warp leaders are read.
    __shared__ __align__(alignof(reduce_state)) unsigned char storage[warps * sizeof(reduce_state)];
    auto* leaders = reinterpret_cast<reduce_state*>(storage);
    if (lane == 0) { leaders[warp] = local; }
    __syncthreads();
    if (warp == 0) {
      if (lane < warps) { local = leaders[lane]; }
#pragma unroll
      for (int offset = warps / 2; offset != 0; offset /= 2) {
        auto other = local.shuffle_down(offset);
        if (lane < offset) { local = merge(local, other); }
      }
    }
  }
  return local;
}

template <bool HasUserData, typename Outputs>
__device__ void reduction_finalize(reduce_state const& state,
                                   void* user_data,
                                   mutable_column_device_view_core const* output_cols,
                                   size_type const* output_sizes,
                                   size_type max_output_size,
                                   int32_t* status)
{
  if (state.error != 0) { return; }
  auto const start  = detail::grid_1d::global_thread_id();
  auto const stride = detail::grid_1d::grid_stride();
  auto const limit  = util::round_up_safe<thread_index_type>(max_output_size, detail::warp_size);
  int32_t error     = 0;

  // All lanes participate in the per-column ballots, including the last partially filled warp.
  for (auto index = start; index < limit; index += stride) {
    auto values   = Outputs::map([]<typename... A>() {
      return cuda::std::tuple<cuda::std::optional<typename A::element_type>...>{};
    });
    auto pointers = Outputs::map([&]<typename... A>() {
      return cuda::std::tuple{
        (index < output_sizes[A::index] ? &cuda::std::get<A::index>(values) : nullptr)...};
    });

    if (index < max_output_size) {
      auto e = reduction_apply<HasUserData>(
        [](auto... args) { return cudf_reduce_finalize(args...); },
        user_data,
        cuda::std::tuple_cat(
          cuda::std::tuple{static_cast<size_type>(index),
                           cuda::std::span<size_type const>{output_sizes, Outputs::size}},
          pointers,
          state.values()));
      error = cuda::std::max(error, e);
    }

    Outputs::map([&]<typename... A>() {
      auto assign = [&]<typename Out>() {
        auto const active     = index < output_sizes[Out::index];
        auto const mask       = __ballot_sync(0xffff'ffffu, active);
        auto const valid      = cuda::std::get<Out::index>(values).has_value();
        auto const valid_mask = __ballot_sync(0xffff'ffffu, active && valid);
        if (active) {
          if (valid) {
            Out::assign(
              output_cols, static_cast<size_type>(index), *cuda::std::get<Out::index>(values));
          }
          if (warp_elect(mask)) {
            Out::set_null_mask_word(output_cols, index / detail::warp_size, valid_mask);
            auto const nulls = __popc(mask) - __popc(valid_mask);
            if (nulls != 0) { atomicAdd(status + 1 + Out::index, nulls); }
          }
        }
      };
      (assign.template operator()<A>(), ...);
    });
  }

  if (error != 0) {
    cuda::atomic_ref ref(*status);
    ref.fetch_max(error, cuda::std::memory_order_relaxed);
  }
}

template <bool NullAware,
          bool HasUserData,
          typename Inputs,
          typename Outputs,
          int BlockSize,
          bool UseWarpReduce>
__device__ void reduce_kernel(int phase,
                              size_type num_rows,
                              void* user_data,
                              column_device_view_core const* input_cols,
                              mutable_column_device_view_core const* output_cols,
                              size_type const* output_sizes,
                              size_type max_output_size,
                              reduce_state const* partials,
                              reduce_state* results,
                              reduce_state* scratch,
                              int32_t* status,
                              bool finalize_inline)
{
  if (phase == 2) {
    reduction_finalize<HasUserData, Outputs>(
      *partials, user_data, output_cols, output_sizes, max_output_size, status);
    return;
  }

  reduce_state local{};
  local.error = reduction_apply<HasUserData>(
    [](auto... args) { return cudf_reduce_init(args...); }, user_data, local.pointers());
  reduction_merge<HasUserData> merge{user_data};
  auto const start  = detail::grid_1d::global_thread_id();
  auto const stride = detail::grid_1d::grid_stride();

  for (auto row = start; row < num_rows; row += stride) {
    if (local.error != 0) { break; }
    if (phase == 1) {
      local = merge(local, partials[row]);
      continue;
    }

    auto update = [&](auto values) {
      if constexpr (!NullAware) {
        bool valid = cuda::std::apply([](auto... a) { return (a.has_value() && ...); }, values);
        if (!valid) { return; }
        auto unwrapped =
          cuda::std::apply([](auto... a) { return cuda::std::tuple{*a...}; }, values);
        local.error = reduction_apply<HasUserData>(
          [](auto... args) { return cudf_reduce_update(args...); },
          user_data,
          cuda::std::tuple_cat(
            cuda::std::tuple{static_cast<size_type>(row)}, local.pointers(), unwrapped));
      } else {
        local.error = reduction_apply<HasUserData>(
          [](auto... args) { return cudf_reduce_update(args...); },
          user_data,
          cuda::std::tuple_cat(
            cuda::std::tuple{static_cast<size_type>(row)}, local.pointers(), values));
      }
    };

#if CUDF_REDUCE_HAS_MAP
    auto inputs = Inputs::map([&]<typename... A>() {
      return cuda::std::tuple{reduction_input<A>(input_cols, static_cast<size_type>(row))...};
    });
    auto mapped = reduce_mapped_types::map(
      []<typename... T>() { return cuda::std::tuple<cuda::std::optional<T>...>{}; });
    auto mapped_ptrs = cuda::std::apply([](auto&... a) { return cuda::std::tuple{&a...}; }, mapped);
    local.error =
      cuda::std::apply([](auto... a) { return static_cast<int32_t>(cudf_reduce_map(a...)); },
                       cuda::std::tuple_cat(mapped_ptrs, inputs));
    if (local.error == 0) { update(mapped); }
#else
    if constexpr (NullAware) {
      update(Inputs::map([&]<typename... A>() {
        return cuda::std::tuple{reduction_input<A>(input_cols, static_cast<size_type>(row))...};
      }));
    } else {
      auto const valid = Inputs::map([&]<typename... A>() {
        return (A::is_valid(input_cols, static_cast<size_type>(row)) && ...);
      });
      if (!valid) { continue; }
      auto inputs = Inputs::map([&]<typename... A>() {
        return cuda::std::tuple{A::element(input_cols, static_cast<size_type>(row))...};
      });
      local.error = reduction_apply<HasUserData>(
        [](auto... args) { return cudf_reduce_update(args...); },
        user_data,
        cuda::std::tuple_cat(
          cuda::std::tuple{static_cast<size_type>(row)}, local.pointers(), inputs));
    }
#endif
  }

  if constexpr (UseWarpReduce) {
    auto result = reduction_block_combine<BlockSize>(local, merge);
    if (threadIdx.x == 0) { results[blockIdx.x] = result; }
  } else {
    // Very wide states create excessive register pressure with shuffles. This tree keeps the same
    // callback contract using disjoint, aligned global scratch slots for each block.
    auto* block        = scratch + static_cast<size_t>(blockIdx.x) * BlockSize;
    block[threadIdx.x] = local;
    __syncthreads();
    for (int offset = BlockSize / 2; offset != 0; offset /= 2) {
      if (threadIdx.x < offset) {
        block[threadIdx.x] = merge(block[threadIdx.x], block[threadIdx.x + offset]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) { results[blockIdx.x] = block[0]; }
  }

  __syncthreads();
  if (threadIdx.x == 0 && results[blockIdx.x].error != 0) {
    cuda::atomic_ref ref(*status);
    ref.fetch_max(results[blockIdx.x].error, cuda::std::memory_order_relaxed);
  }
  if (finalize_inline) {
    reduction_finalize<HasUserData, Outputs>(
      results[0], user_data, output_cols, output_sizes, max_output_size, status);
  }
}

}  // namespace cudf::jit

extern "C" __global__ void cudf_kernel_entry(
  int phase,
  cudf::size_type num_rows,
  void* user_data,
  cudf::column_device_view_core const* input_cols,
  cudf::mutable_column_device_view_core const* output_cols,
  cudf::size_type const* output_sizes,
  cudf::size_type max_output_size,
  void const* partials,
  void* results,
  void* scratch,
  int32_t* status,
  bool finalize_inline)
{
  CUDF_KERNEL_INSTANCE(phase,
                       num_rows,
                       user_data,
                       input_cols,
                       output_cols,
                       output_sizes,
                       max_output_size,
                       static_cast<cudf::jit::reduce_state const*>(partials),
                       static_cast<cudf::jit::reduce_state*>(results),
                       static_cast<cudf::jit::reduce_state*>(scratch),
                       status,
                       finalize_inline);
}
