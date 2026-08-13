/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/row_ir/opcode.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
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

  for (auto row = start; row < row_size; row += stride) {
    auto operation = [&]<typename Args>(Args args) {
      // TODO: static assert invocable
      auto func = [&](auto... a) {
        if constexpr (!cuda::std::is_void_v<decltype(GENERIC_TRANSFORM_OP(a...))>) {
          return static_cast<cudf::errc>(GENERIC_TRANSFORM_OP(a...));
        } else {
          (void)GENERIC_TRANSFORM_OP(a...);
          return errc::SUCCESS;
        }
      };

      if constexpr (has_user_data) {
        return cuda::std::apply(func, cuda::std::tuple_cat(cuda::std::tuple{user_data, row}, args));
      } else {
        return cuda::std::apply(func, args);
      }
    };

    if constexpr (!is_null_aware) {
      if (stencil != nullptr && !bit_is_set(stencil, row)) { continue; }

      auto ins = InputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::element(input_cols, row)...}; });

      auto outs = OutputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::output_arg(output_cols, row)...}; });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto row_error = operation(cuda::std::tuple_cat(out_ptrs, ins));

      OutputAccessors::map([&]<typename... A>() {
        (A::assign(output_cols, row, cuda::std::get<A::index>(outs)), ...);
      });

      thread_error = cuda::std::max(thread_error, row_error);

    } else {
      auto active_mask = __ballot_sync(__activemask(), row < row_size);

      auto ins = InputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::nullable_element(input_cols, row)...}; });

      auto outs = OutputAccessors::map(
        [&]<typename... A>() { return cuda::std::tuple{A::null_output_arg(output_cols, row)...}; });

      auto out_ptrs =
        cuda::std::apply([&](auto&... args) { return cuda::std::tuple{&args...}; }, outs);

      auto row_error = operation(cuda::std::tuple_cat(out_ptrs, ins));

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
