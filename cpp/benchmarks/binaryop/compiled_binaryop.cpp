/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/transform.hpp>

#include <cudf_benchmark_fragments.hpp>
#include <nvbench/nvbench.cuh>

template <typename TypeLhs, typename TypeRhs, typename TypeOut>
void BM_compiled_binaryop(nvbench::state& state, cudf::binary_operator binop)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  auto const source_table = create_random_table(
    {cudf::type_to_id<TypeLhs>(), cudf::type_to_id<TypeRhs>()}, row_count{num_rows});

  auto lhs = cudf::column_view(source_table->get_column(0));
  auto rhs = cudf::column_view(source_table->get_column(1));

  auto output_dtype = cudf::data_type(cudf::type_to_id<TypeOut>());

  // Call once for hot cache.
  cudf::binary_operation(lhs, rhs, binop, output_dtype);

  // use number of bytes read and written to global memory
  state.add_global_memory_reads<TypeLhs>(num_rows);
  state.add_global_memory_reads<TypeRhs>(num_rows);
  state.add_global_memory_writes<TypeOut>(num_rows);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::binary_operation(lhs, rhs, binop, output_dtype); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

#define BM_STRINGIFY(a) #a

// TODO tparam boolean for null.
#define BM_BINARYOP_BENCHMARK_DEFINE(name, lhs, rhs, bop, tout)               \
  static void name(::nvbench::state& st)                                      \
  {                                                                           \
    ::BM_compiled_binaryop<lhs, rhs, tout>(st, ::cudf::binary_operator::bop); \
  }                                                                           \
  NVBENCH_BENCH(name)                                                         \
    .set_name("compiled_binary_op_" BM_STRINGIFY(name))                       \
    .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000})

#define build_name(a, b, c, d) a##_##b##_##c##_##d

#define BINARYOP_BENCHMARK_DEFINE(lhs, rhs, bop, tout) \
  BM_BINARYOP_BENCHMARK_DEFINE(build_name(bop, lhs, rhs, tout), lhs, rhs, bop, tout)

using cudf::duration_D;
using cudf::duration_ms;
using cudf::duration_ns;
using cudf::duration_s;
using cudf::timestamp_D;
using cudf::timestamp_ms;
using cudf::timestamp_s;
using numeric::decimal32;

// clang-format off
BINARYOP_BENCHMARK_DEFINE(float,        int64_t,      ADD,                  int32_t);
BINARYOP_BENCHMARK_DEFINE(float,        float,        ADD,                  float);
BINARYOP_BENCHMARK_DEFINE(timestamp_s,  duration_s,   ADD,                  timestamp_s);
BINARYOP_BENCHMARK_DEFINE(duration_s,   duration_D,   SUB,                  duration_ms);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int64_t,      SUB,                  int64_t);
BINARYOP_BENCHMARK_DEFINE(float,        float,        MUL,                  int64_t);
BINARYOP_BENCHMARK_DEFINE(float,        float,        MUL,                  float);
BINARYOP_BENCHMARK_DEFINE(duration_s,   int64_t,      MUL,                  duration_s);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int64_t,      DIV,                  int64_t);
BINARYOP_BENCHMARK_DEFINE(duration_ms,  int32_t,      DIV,                  duration_ms);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int64_t,      TRUE_DIV,             int64_t);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int64_t,      FLOOR_DIV,            int64_t);
BINARYOP_BENCHMARK_DEFINE(double,       double,       MOD,                  double);
BINARYOP_BENCHMARK_DEFINE(duration_ms,  int64_t,      MOD,                  duration_ms);
BINARYOP_BENCHMARK_DEFINE(int32_t,      int64_t,      PMOD,                 double);
BINARYOP_BENCHMARK_DEFINE(int32_t,      uint8_t,      PYMOD,                int64_t);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int64_t,      POW,                  double);
BINARYOP_BENCHMARK_DEFINE(float,        double,       LOG_BASE,             double);
BINARYOP_BENCHMARK_DEFINE(float,        double,       ATAN2,                double);
BINARYOP_BENCHMARK_DEFINE(int,          int,          SHIFT_LEFT,           int);
BINARYOP_BENCHMARK_DEFINE(int16_t,      int64_t,      SHIFT_RIGHT,          int);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int32_t,      SHIFT_RIGHT_UNSIGNED, int64_t);
BINARYOP_BENCHMARK_DEFINE(int64_t,      int32_t,      BITWISE_AND,          int16_t);
BINARYOP_BENCHMARK_DEFINE(int16_t,      int32_t,      BITWISE_OR,           int64_t);
BINARYOP_BENCHMARK_DEFINE(int16_t,      int64_t,      BITWISE_XOR,          int32_t);
BINARYOP_BENCHMARK_DEFINE(double,       int8_t,       LOGICAL_AND,          bool);
BINARYOP_BENCHMARK_DEFINE(int16_t,      int64_t,      LOGICAL_OR,           bool);
BINARYOP_BENCHMARK_DEFINE(int32_t,      int64_t,      EQUAL,                bool);
BINARYOP_BENCHMARK_DEFINE(duration_ms,  duration_ns,  EQUAL,                bool);
BINARYOP_BENCHMARK_DEFINE(decimal32,    decimal32,    NOT_EQUAL,            bool);
BINARYOP_BENCHMARK_DEFINE(timestamp_s,  timestamp_s,  LESS,                 bool);
BINARYOP_BENCHMARK_DEFINE(timestamp_ms, timestamp_s,  GREATER,              bool);
BINARYOP_BENCHMARK_DEFINE(duration_ms,  duration_ns,  NULL_EQUALS,          bool);
BINARYOP_BENCHMARK_DEFINE(duration_ms,  duration_ns,  NULL_NOT_EQUALS,      bool);
BINARYOP_BENCHMARK_DEFINE(decimal32,    decimal32,    NULL_MAX,             decimal32);
BINARYOP_BENCHMARK_DEFINE(timestamp_D,  timestamp_s,  NULL_MIN,             timestamp_s);
// clang-format on

template <typename TypeLhs, typename TypeRhs, typename TypeOut>
void BM_jit_binaryop(nvbench::state& state, cudf::binary_operator binop)
{
  constexpr auto const jit_mul_cuda = R"***(
__device__ void transform(float* out, float a, float b) {
  *out = a * b;
}
)***";

  constexpr auto const jit_add_cuda = R"***(
__device__ void transform(float* out, float a, float b) {
  *out = a + b;
}
)***";

  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const use_lto  = state.get_string("use_lto") == "true";
  static_assert(std::is_same_v<TypeLhs, TypeRhs> && std::is_same_v<TypeRhs, TypeOut>);
  static_assert(std::is_same_v<TypeLhs, float>);

  auto const source_table = create_random_table(
    {cudf::type_to_id<TypeLhs>(), cudf::type_to_id<TypeRhs>()}, row_count{num_rows});

  auto lhs = cudf::column_view(source_table->get_column(0));
  auto rhs = cudf::column_view(source_table->get_column(1));

  size_t fragment_id = 0;
  char const* cuda   = nullptr;

  switch (binop) {
    case cudf::binary_operator::ADD: {
      fragment_id = cudf_benchmark_fragments::add_f32;
      cuda        = jit_add_cuda;
    } break;
    case cudf::binary_operator::MUL: {
      fragment_id = cudf_benchmark_fragments::mul_f32;
      cuda        = jit_mul_cuda;
    } break;
    default: throw std::runtime_error("Unsupported binary operator for JIT benchmark");
  }

  // Call once for hot cache.
  cudf::transform_input inputs[]   = {lhs, rhs};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_to_id<TypeOut>()}, cudf::output_nullability::ALL_VALID}};

  auto const range = cudf_benchmark_fragments::file_ranges[fragment_id];
  std::span<uint8_t const> udf{cudf_benchmark_fragments::files.subspan(range[0], range[1])};

  auto result = use_lto ? cudf::transform_lto(udf,
                                              cudf::lto_binary_type::FATBIN,
                                              cudf::null_aware::NO,
                                              std::nullopt,
                                              inputs,
                                              outputs,
                                              {},
                                              std::nullopt)
                        : cudf::transform(cuda,
                                          cudf::udf_source_type::CUDA,
                                          cudf::null_aware::NO,
                                          std::nullopt,
                                          inputs,
                                          outputs,
                                          {},
                                          std::nullopt);

  // use number of bytes read and written to global memory
  state.add_global_memory_reads<TypeLhs>(num_rows);
  state.add_global_memory_reads<TypeRhs>(num_rows);
  state.add_global_memory_writes<TypeOut>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto result = use_lto ? cudf::transform_lto(udf,
                                                                 cudf::lto_binary_type::FATBIN,
                                                                 cudf::null_aware::NO,
                                                                 std::nullopt,
                                                                 inputs,
                                                                 outputs,
                                                                 {},
                                                                 std::nullopt)
                                           : cudf::transform(cuda,
                                                             cudf::udf_source_type::CUDA,
                                                             cudf::null_aware::NO,
                                                             std::nullopt,
                                                             inputs,
                                                             outputs,
                                                             {},
                                                             std::nullopt);
  });
}

#define BM_JIT_BINARYOP_BENCHMARK_DEFINE(name, lhs, rhs, bop, tout)                    \
  static void name(::nvbench::state& st)                                               \
  {                                                                                    \
    ::BM_jit_binaryop<lhs, rhs, tout>(st, ::cudf::binary_operator::bop);               \
  }                                                                                    \
  NVBENCH_BENCH(name)                                                                  \
    .set_name("jit_binary_op_" BM_STRINGIFY(name))                                     \
    .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000}) \
    .add_string_axis("use_lto", {"true", "false"})

#define build_name_jit(a, b, c, d) a##_##b##_##c##_##d##_jit

#define JIT_BINARYOP_BENCHMARK_DEFINE(lhs, rhs, bop, tout) \
  BM_JIT_BINARYOP_BENCHMARK_DEFINE(build_name_jit(bop, lhs, rhs, tout), lhs, rhs, bop, tout)

// clang-format off
JIT_BINARYOP_BENCHMARK_DEFINE(float,        float,      ADD,                  float);
JIT_BINARYOP_BENCHMARK_DEFINE(float,        float,      MUL,                  float);
// clang-format on
