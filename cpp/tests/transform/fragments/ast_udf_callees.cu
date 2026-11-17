/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Device functions that AST JIT expressions call (cudf::ast::jit::call with an ast::jit::udf),
// in tests/ast/jit_expressions_tests.cpp and tests/join/mixed_join_tests.cu.

#include <cudf/errc.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/optional>

extern "C" __device__ cudf::errc lto_add_one(int32_t* out, int32_t a)
{
  *out = a + 1;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lto_twice(int32_t* out, int32_t a)
{
  *out = a * 2;
  return cudf::errc::SUCCESS;
}

// Fails a row with a negative argument.
extern "C" __device__ cudf::errc lto_checked_halve(int32_t* out, int32_t a)
{
  if (a < 0) { return cudf::errc::ARITHMETIC_OVERFLOW; }
  *out = a / 2;
  return cudf::errc::SUCCESS;
}

// Divides 100 by its argument, failing a zero with DIVISION_BY_ZERO and a negative argument with
// the lower code ARITHMETIC_OVERFLOW.
extern "C" __device__ cudf::errc lto_checked_hundred_over(int32_t* out, int32_t a)
{
  if (a == 0) { return cudf::errc::DIVISION_BY_ZERO; }
  if (a < 0) { return cudf::errc::ARITHMETIC_OVERFLOW; }
  *out = 100 / a;
  return cudf::errc::SUCCESS;
}

// Fails a row with the error code it is given; zero succeeds.
extern "C" __device__ cudf::errc lto_fail_with(int32_t* out, int32_t code)
{
  *out = code;
  return static_cast<cudf::errc>(code);
}

// Named like a parameter and a local of the generated function, which must not hide them.
extern "C" __device__ cudf::errc in_0(int32_t* out, int32_t a)
{
  *out = a + 1;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc tmp_0(int32_t* out, int32_t a)
{
  *out = a * 2;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lto_xor_popcount(int32_t* out, int32_t a, int32_t b)
{
  *out = __popc(a ^ b);
  return cudf::errc::SUCCESS;
}

__device__ unsigned long long issued_ids = 0;

// Returns a new number on every call and ignores its argument, so it is not pure.
extern "C" __device__ cudf::errc lto_next_id(int64_t* out, int32_t)
{
  *out = static_cast<int64_t>(atomicAdd(&issued_ids, 1ULL));
  return cudf::errc::SUCCESS;
}

// The Monday on or before a day; 1970-01-01 was a Thursday.
extern "C" __device__ cudf::errc lto_week_start_ns(cudf::timestamp_ns* out, cudf::timestamp_ns in)
{
  constexpr long long ticks_per_day = 86'400'000'000'000LL;
  auto const ticks                  = in.time_since_epoch().count();
  auto days                         = ticks / ticks_per_day;
  if (ticks % ticks_per_day != 0 && ticks < 0) { --days; }
  auto const shifted = days + 3;
  auto weeks         = shifted / 7;
  if (shifted % 7 != 0 && shifted < 0) { --weeks; }
  *out = cudf::timestamp_ns{cudf::duration_ns{(weeks * 7 - 3) * ticks_per_day}};
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lto_decimal_add(numeric::decimal64* out,
                                                 numeric::decimal32 a,
                                                 numeric::decimal64 b,
                                                 int32_t output_scale)
{
  auto const widened_a = numeric::decimal64{numeric::scaled_integer<int64_t>{a.value(), a.scale()}};
  auto const sum       = (widened_a + b).rescaled(numeric::scale_type{output_scale});
  // The caller uses only the representation, so this scale deliberately differs from output_scale.
  *out = numeric::decimal64{numeric::scaled_integer<int64_t>{sum.value(), numeric::scale_type{0}}};
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lto_string_tail(cudf::string_view* out, cudf::string_view input)
{
  *out = input.empty() ? input : input.substr(1, input.length() - 1);
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lto_nullable_string_tail(
  cuda::std::optional<cudf::string_view>* out, cuda::std::optional<cudf::string_view> input)
{
  auto const value = input.value_or(cudf::string_view{});
  *out             = value.empty() ? value : value.substr(1, value.length() - 1);
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lto_constant(int32_t* out)
{
  *out = 42;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lto_nullable_constant(cuda::std::optional<int32_t>* out)
{
  *out = cuda::std::nullopt;
  return cudf::errc::SUCCESS;
}
