/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

enum class executor_type : uint8_t { AST, JIT, JIT_OPT };

executor_type executor_from_string(std::string_view executor)
{
  if (executor == "ast") { return executor_type::AST; }
  if (executor == "jit") { return executor_type::JIT; }
  if (executor == "jit-opt") { return executor_type::JIT_OPT; }
  CUDF_FAIL("unrecognized executor: " + std::string{executor});
}

template <typename LiteralFactory>
std::vector<cudf::ast::tree> make_expression_trees(cudf::size_type table_width,
                                                   cudf::size_type expression_depth,
                                                   LiteralFactory make_literal)
{
  std::vector<cudf::ast::tree> trees;
  trees.reserve(table_width);

  for (cudf::size_type column_index = 0; column_index < table_width; ++column_index) {
    cudf::ast::tree tree;
    cudf::ast::expression const* expression = &tree.push(cudf::ast::column_reference{column_index});

    for (cudf::size_type level = 0; level < expression_depth; ++level) {
      auto& literal = tree.push(make_literal(level));
      expression =
        &tree.push(cudf::ast::operation{cudf::ast::ast_operator::ADD, *expression, literal});
    }
    trees.push_back(std::move(tree));
  }

  return trees;
}

void BM_ast_jit_wide_table(nvbench::state& state)
{
  auto table_width      = static_cast<cudf::size_type>(state.get_int64("table_width"));
  auto rows_per_batch   = static_cast<cudf::size_type>(state.get_int64("rows_per_batch"));
  auto total_rows       = state.get_int64("total_rows");
  auto expression_depth = static_cast<cudf::size_type>(state.get_int64("expression_depth"));
  auto executor         = executor_from_string(state.get_string("executor"));

  if (rows_per_batch > total_rows || total_rows % rows_per_batch != 0) {
    state.skip("rows_per_batch must evenly divide total_rows");
    return;
  }

  auto input = create_sequence_table(cycle_dtypes({cudf::type_id::INT32}, table_width),
                                     row_count{rows_per_batch});

  auto input_view = input->view();

  auto num_batches = total_rows / rows_per_batch;

  std::vector<cudf::numeric_scalar<int32_t>> scalars;
  scalars.reserve(expression_depth);
  for (cudf::size_type level = 0; level < expression_depth; ++level) {
    scalars.emplace_back(level + 1);
  }

  auto scalar_trees =
    make_expression_trees(table_width, expression_depth, [&scalars](cudf::size_type level) {
      return cudf::ast::literal{scalars[level]};
    });

  std::vector<std::unique_ptr<cudf::column>> scalar_columns;
  scalar_columns.reserve(expression_depth);
  for (auto& scalar : scalars) {
    scalar_columns.push_back(cudf::make_column_from_scalar(scalar, 1));
  }

  auto scalar_column_view_trees =
    make_expression_trees(table_width, expression_depth, [&scalar_columns](cudf::size_type level) {
      return cudf::ast::literal{cudf::scalar_column_view{scalar_columns[level]->view()}};
    });

  auto elements = static_cast<std::size_t>(total_rows) * static_cast<std::size_t>(table_width);
  state.add_element_count(elements);
  state.add_global_memory_reads<int32_t>(elements * (expression_depth + 1));
  state.add_global_memory_writes<int32_t>(elements);

  auto mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::benchmark::scoped_range range{"benchmark_iteration"};
    auto stream = launch.get_stream().get_stream();
    std::vector<std::unique_ptr<cudf::column>> outputs;
    outputs.reserve(table_width);

    for (int64_t batch = 0; batch < num_batches; ++batch) {
      outputs.clear();

      auto& trees = (executor == executor_type::JIT_OPT) ? scalar_column_view_trees : scalar_trees;

      for (auto& tree : trees) {
        switch (executor) {
          case executor_type::AST: {
            outputs.push_back(cudf::compute_column(input_view, tree.back(), stream));
            break;
          }
          case executor_type::JIT: {
            outputs.push_back(cudf::compute_column_jit(input_view, tree.back(), stream));
            break;
          }
          case executor_type::JIT_OPT: {
            outputs.push_back(cudf::compute_column_jit(input_view, tree.back(), stream));
            break;
          }
        }
      }
    }
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

NVBENCH_BENCH(BM_ast_jit_wide_table)
  .set_name("ast_jit_wide_table")
  .add_int64_axis("table_width", {1, 16, 64})
  .add_int64_axis("rows_per_batch", {1'024, 16'384, 262'144})
  .add_int64_axis("total_rows", {262'144, 1'048'576})
  .add_int64_axis("expression_depth", {1, 4, 16})
  .add_string_axis("executor", {"ast", "jit", "jit-opt"});
