/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>
#include <nvbench/types.cuh>

#include <random>

namespace {

enum class engine_type : uint8_t { AST = 0, JIT = 1 };

engine_type engine_from_string(std::string_view str)
{
  if (str == "ast") {
    return engine_type::AST;
  } else if (str == "jit") {
    return engine_type::JIT;
  } else {
    CUDF_FAIL("unrecognized engine enum: " + std::string(str));
  }
}

template <typename key_type>
void BM_ast_polynomials(nvbench::state& state)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const order            = static_cast<cudf::size_type>(state.get_int64("order"));
  auto const null_probability = state.get_float64("null_probability");
  auto const engine           = engine_from_string(state.get_string("engine"));

  CUDF_EXPECTS(order > 0, "Polynomial order must be greater than 0");

  data_profile profile;
  profile.set_null_probability(null_probability);
  profile.set_distribution_params(cudf::type_to_id<key_type>(),
                                  distribution_id::NORMAL,
                                  static_cast<key_type>(0),
                                  static_cast<key_type>(1));
  auto table = create_random_table({cudf::type_to_id<key_type>()}, row_count{num_rows}, profile);
  auto column_view = table->get_column(0);

  std::vector<cudf::numeric_scalar<key_type>> constants;
  {
    std::random_device random_device;
    std::mt19937 generator;
    std::uniform_real_distribution<key_type> distribution{0, 1};

    std::transform(cuda::counting_iterator<cudf::size_type>{0},
                   cuda::counting_iterator{order + 1},
                   std::back_inserter(constants),
                   [&](int) { return distribution(generator); });
  }

  cudf::ast::tree tree{};

  auto& column_ref = tree.push(cudf::ast::column_reference{0});

  // computes polynomials: (((ax + b)x + c)x + d)x + e... = ax**4 + bx**3 + cx**2 + dx + e....
  tree.push(cudf::ast::literal{constants[0]});

  for (cudf::size_type i = 0; i < order; i++) {
    auto& product =
      tree.push(cudf::ast::operation{cudf::ast::ast_operator::MUL, tree.back(), column_ref});
    auto& constant = tree.push(cudf::ast::literal{constants[i + 1]});
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::ADD, product, constant});
  }

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(num_rows);
  state.add_global_memory_writes<key_type>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::benchmark::scoped_range range{"benchmark_iteration"};

    switch (engine) {
      case engine_type::AST: {
        cudf::compute_column(*table, tree.back(), launch.get_stream().get_stream());
        break;
      }
      case engine_type::JIT: {
        cudf::compute_column_jit(*table, tree.back(), launch.get_stream().get_stream());
        break;
      }
      default: CUDF_FAIL("Invalid engine type");
    }
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

#define AST_POLYNOMIAL_BENCHMARK_DEFINE(name, key_type)                          \
  static void name(::nvbench::state& st) { ::BM_ast_polynomials<key_type>(st); } \
  NVBENCH_BENCH(name)                                                            \
    .set_name(#name)                                                             \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})   \
    .add_int64_axis("order", {1, 2, 4, 8, 16, 32})                               \
    .add_float64_axis("null_probability", {0.01})                                \
    .add_string_axis("engine", {"ast", "jit"})

}  // namespace

AST_POLYNOMIAL_BENCHMARK_DEFINE(ast_polynomials_float32, float);

AST_POLYNOMIAL_BENCHMARK_DEFINE(ast_polynomials_float64, double);
