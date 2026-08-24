/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>
#include <nvbench/types.cuh>

#include <concepts>
#include <vector>

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

bool boolean_from_string(std::string_view str)
{
  if (str == "true") {
    return true;
  } else if (str == "false") {
    return false;
  } else {
    CUDF_FAIL("unrecognized boolean value: " + std::string(str));
  }
}

template <typename key_type>
void BM_filter_min_max(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_filter_columns =
    static_cast<cudf::size_type>(state.get_int64("num_filter_columns"));
  auto const engine_name = state.get_string("engine");
  auto const nullable    = boolean_from_string(state.get_string("nullable"));
  auto const engine      = engine_from_string(engine_name);

  auto const input_min   = static_cast<key_type>(0);
  auto const input_max   = static_cast<key_type>(100);
  auto const filter_min  = input_min;
  auto const selectivity = state.get_float64("selectivity");
  auto const filter_max  = static_cast<key_type>(input_min + (input_max - input_min) * selectivity);

  auto profile = data_profile{};
  profile.set_distribution_params(
    cudf::type_to_id<key_type>(), distribution_id::UNIFORM, input_min, input_max);
  profile.set_null_probability(nullable ? std::optional{0.3} : std::nullopt);

  std::vector<std::unique_ptr<cudf::column>> filter_columns;
  std::transform(cuda::counting_iterator<cudf::size_type>{0},
                 cuda::counting_iterator{num_filter_columns},
                 std::back_inserter(filter_columns),
                 [&](auto) {
                   return create_random_column(
                     cudf::type_to_id<key_type>(), row_count{num_rows}, profile);
                 });

  std::vector<cudf::column_view> filter_column_views;
  std::transform(filter_columns.begin(),
                 filter_columns.end(),
                 std::back_inserter(filter_column_views),
                 [](auto const& col) { return col->view(); });

  std::string type_name = cudf::type_to_name(cudf::data_type{cudf::type_to_id<key_type>()});

  auto udf = std::format(
    R"***(
    __device__ void transform(bool * out, {0} c0, {0} min, {0} max) {{
    *out = (c0 >= min && c0 <= max);
    }}
    )***",
    type_name);

  auto tree              = cudf::ast::tree{};
  auto min_scalar        = cudf::numeric_scalar<key_type>{filter_min};
  auto max_scalar        = cudf::numeric_scalar<key_type>{filter_max};
  auto min_scalar_column = cudf::make_column_from_scalar(min_scalar, 1);
  auto max_scalar_column = cudf::make_column_from_scalar(max_scalar, 1);

  {
    auto& column_ref  = tree.push(cudf::ast::column_reference{0});
    auto& min_literal = tree.push(cudf::ast::literal{min_scalar});
    auto& max_literal = tree.push(cudf::ast::literal{max_scalar});
    auto& filter_min  = tree.push(
      cudf::ast::operation{cudf::ast::ast_operator::GREATER_EQUAL, column_ref, min_literal});
    auto& filter_max =
      tree.push(cudf::ast::operation{cudf::ast::ast_operator::LESS_EQUAL, column_ref, max_literal});
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::LOGICAL_AND, filter_min, filter_max});
  }

  auto const predicate_column =
    create_random_column(cudf::type_to_id<key_type>(), row_count{num_rows}, profile);

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(static_cast<std::size_t>(num_rows));
  state.add_global_memory_writes<key_type>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto stream = launch.get_stream().get_stream();
    auto mr     = cudf::get_current_device_resource_ref();

    switch (engine) {
      case engine_type::AST: {
        auto predicate_table      = cudf::table_view{{predicate_column->view()}};
        auto filter_table         = cudf::table_view{filter_column_views};
        auto const filter_boolean = cudf::compute_column(predicate_table, tree.back(), stream, mr);
        auto const result =
          cudf::apply_boolean_mask(filter_table, filter_boolean->view(), stream, mr);
      } break;
      case engine_type::JIT: {
        cudf::filter_input predicate_inputs[] = {
          predicate_column->view(),
          cudf::scalar_column_view(min_scalar_column->view()),
          cudf::scalar_column_view(max_scalar_column->view())};
        auto result = cudf::filter_extended(predicate_inputs,
                                            udf,
                                            filter_column_views,
                                            cudf::udf_source_type::CUDA,
                                            std::nullopt,
                                            cudf::null_aware::NO,
                                            cudf::output_nullability::PRESERVE,
                                            stream,
                                            mr);
      } break;
      default: CUDF_UNREACHABLE("Unrecognised engine type requested");
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

#define FILTER_BENCHMARK_DEFINE(name, key_type)                                 \
  static void name(::nvbench::state& st) { ::BM_filter_min_max<key_type>(st); } \
  NVBENCH_BENCH(name)                                                           \
    .set_name(#name)                                                            \
    .add_string_axis("engine", {"ast", "jit"})                                  \
    .add_int64_axis("num_rows", {1'000'000, 10'000'000, 100'000'000})           \
    .add_int64_axis("num_filter_columns", {1, 32})                              \
    .add_string_axis("nullable", {"true", "false"})
}  // namespace

FILTER_BENCHMARK_DEFINE(filter_min_max_int32, int32_t)
  .add_float64_axis("selectivity", {0.001, 0.01, 0.1, 0.5, 0.8});

FILTER_BENCHMARK_DEFINE(filter_min_max_int64, int64_t)
  .add_float64_axis("selectivity", {0.001, 0.01, 0.1, 0.5, 0.8});

FILTER_BENCHMARK_DEFINE(filter_min_max_float32, float)
  .add_float64_axis("selectivity", {0.001, 0.01, 0.1, 0.5, 0.8});

FILTER_BENCHMARK_DEFINE(filter_min_max_float64, double)
  .add_float64_axis("selectivity", {0.001, 0.01, 0.1, 0.5, 0.8});
