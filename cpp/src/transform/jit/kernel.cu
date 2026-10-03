/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/errc.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <jit/column_accessor.cuh>
#include <jit/column_device_view_wrappers.cuh>
#include <jit/sync.cuh>
#include <jit/type_list.cuh>

#pragma nv_hdrstop  // The above headers are used by the kernel below and need to be included before
                    // it. Each UDF will have a different operation_udf.cuh generated for it, so we
                    // need to put this pragma before including it to avoid PCH mismatch.

// clang-format off
// This header is an inlined header that defines the GENERIC_TRANSFORM_OP function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/operation_udf.cuh>
// clang-format on

#ifndef CUDF_LTO_MODE
#define CUDF_UDF_TYPE int()
#endif

// Use LTO-dispatch for transform operators if we're in LTO mode. This allows the operator to be
// defined in a separate translation unit and compiled with LTO, which can result in better
// performance due to more optimization opportunities
#ifdef CUDF_LTO_MODE
#define GENERIC_TRANSFORM_OP(...) ::cudf::jit::lto::transform(__VA_ARGS__)
#endif

namespace cudf {
namespace jit {
namespace lto {

using transform_type = CUDF_UDF_TYPE;

extern "C" __device__ transform_type transform;

}  // namespace lto

// Expanding the argument packs directly avoids instantiating concatenated tuple types and
// cuda::std::apply machinery, which is expensive in NVRTC's C++ frontend.
template <typename... Args>
  requires requires(Args... args) { GENERIC_TRANSFORM_OP(args...); }
__device__ errc invoke_transform(Args... args)
{
  if constexpr (!cuda::std::is_void_v<decltype(GENERIC_TRANSFORM_OP(args...))>) {
    return static_cast<errc>(GENERIC_TRANSFORM_OP(args...));
  } else {
    (void)GENERIC_TRANSFORM_OP(args...);
    return errc::SUCCESS;
  }
}

template <bool has_user_data, typename... Args>
__device__ errc invoke_transform_op(void* user_data, size_type row, Args... args)
{
  if constexpr (has_user_data) {
    return invoke_transform(user_data, row, args...);
  } else {
    return invoke_transform(args...);
  }
}

/// @brief The generic transform kernel. Supports all types and nullability combinations.
template <bool is_null_aware, bool has_user_data, typename InputAccessors, typename OutputAccessors>
__device__ void transform_kernel(size_type row_size,
                                 bitmask_type const* __restrict__ stencil,
                                 void* __restrict__ user_data,
                                 column_device_view_core const* __restrict__ input_cols,
                                 mutable_column_device_view_core const* __restrict__ output_cols,
                                 int32_t* __restrict__ max_error)
{
  auto start        = detail::grid_1d::global_thread_id();
  auto stride       = detail::grid_1d::grid_stride();
  auto thread_error = errc::SUCCESS;

  // Keep row_index wide: the final stride increment and warp padding can exceed size_type's range.
  // Only narrow to row after checking bounds, so column accessors and UDFs receive a safe
  // size_type.
  if constexpr (!is_null_aware) {
    for (auto row_index = start; row_index < row_size; row_index += stride) {
      auto const row = static_cast<size_type>(row_index);
      if (stencil != nullptr && !bit_is_set(stencil, row)) { continue; }

      auto outs = OutputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::output_arg(output_cols, row)...}; });

      auto row_error = OutputAccessors::map([&]<typename... Out>() {
        return InputAccessors::map([&]<typename... In>() {
          return invoke_transform_op<has_user_data>(
            user_data, row, &cuda::std::get<Out::index>(outs)..., In::element(input_cols, row)...);
        });
      });

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, row, cuda::std::get<A::index>(outs)), ...);
      });

      thread_error = cuda::std::max(thread_error, row_error);
    }
  } else {
    // Keep every lane in a warp on the same loop iteration when writing validity.
    auto warp_padded_size = util::round_up_safe<thread_index_type>(row_size, detail::warp_size);

    for (auto row_index = start; row_index < warp_padded_size; row_index += stride) {
      auto active_mask = __ballot_sync(0xffff'ffffu, row_index < row_size);
      if (row_index >= row_size) { continue; }
      auto const row = static_cast<size_type>(row_index);

      auto outs = OutputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::null_output_arg(output_cols, row)...}; });

      auto row_error = OutputAccessors::map([&]<typename... Out>() {
        return InputAccessors::map([&]<typename... In>() {
          return invoke_transform_op<has_user_data>(user_data,
                                                    row,
                                                    &cuda::std::get<Out::index>(outs)...,
                                                    In::nullable_element(input_cols, row)...);
        });
      });

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, row, *cuda::std::get<A::index>(outs)), ...);
        (warp_compact_validity<A>(
           active_mask, output_cols, row, cuda::std::get<A::index>(outs).has_value()),
         ...);
      });

      thread_error = cuda::std::max(thread_error, row_error);
    }
  }

  // early exit if no error occurred
  if (thread_error == errc::SUCCESS) { return; }

  cuda::atomic_ref ref(*max_error);
  ref.fetch_max(static_cast<int32_t>(thread_error), cuda::std::memory_order_relaxed);
}

}  // namespace jit
}  // namespace cudf

// The entry point for the JIT compiled kernel. This is the C-ABI function that will be used to
// retrieve the `CuFunction` for the kernel from the compiled module. This is because we don't want
// to track the scope-dependent C++ mangled name of the kernel, and can just use a fixed name to
// retrieve the `CuFunction` of the kernel.
// A C++-mangled symbol has ambiguous and complex resolution rules, and can change based on the
// scope of the function, the types of the arguments, and other factors that will not be known until
// after compilation. By using a fixed C-ABI symbol name for the kernel entry point, we can avoid
// these issues and ensure that we can always retrieve the correct `CuFunction` for the kernel
// regardless of the context in which it was compiled or used.
extern "C" __global__ void cudf_kernel_entry(
  cudf::size_type row_size,
  cudf::bitmask_type const* __restrict__ stencil,
  void* __restrict__ user_data,
  cudf::column_device_view_core const* __restrict__ input_cols,
  cudf::mutable_column_device_view_core const* __restrict__ output_cols,
  int32_t* __restrict__ max_error)
{
  CUDF_KERNEL_INSTANCE(row_size, stencil, user_data, input_cols, output_cols, max_error);
}
