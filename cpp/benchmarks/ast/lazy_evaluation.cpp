/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace {

enum class strategy : uint8_t { LAZY_AST, LAZY_JIT, EAGER, NULL_MASK, FILTER_SCATTER };

strategy strategy_from_string(std::string_view str)
{
  if (str == "lazy_ast") { return strategy::LAZY_AST; }
  if (str == "lazy_jit") { return strategy::LAZY_JIT; }
  if (str == "eager") { return strategy::EAGER; }
  if (str == "null_mask") { return strategy::NULL_MASK; }
  if (str == "filter_scatter") { return strategy::FILTER_SCATTER; }
  CUDF_FAIL("unrecognized strategy: " + std::string(str));
}

using ast_op = cudf::ast::ast_operator;

// Chain of `heavy_ops` multiply and add operations on columns 0 and 1
cudf::ast::expression const& make_heavy_branch(cudf::ast::tree& tree, int heavy_ops)
{
  auto const& x = tree.push(cudf::ast::column_reference(0));
  auto const& y = tree.push(cudf::ast::column_reference(1));
  auto const* e = &tree.push(cudf::ast::operation(ast_op::MUL, x, y));
  for (int i = 1; i < heavy_ops; ++i) {
    e = &tree.push(cudf::ast::operation(i % 2 ? ast_op::ADD : ast_op::MUL, *e, i % 2 ? x : y));
  }
  return *e;
}

}  // namespace

static void BM_lazy_if_else(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const selectivity = static_cast<double>(state.get_int64("selectivity_percent")) / 100.0;
  auto const heavy_ops   = static_cast<int>(state.get_int64("heavy_ops"));
  auto const kind        = strategy_from_string(state.get_string("strategy"));

  data_profile const value_profile =
    data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_id::INT64, distribution_id::UNIFORM, 1, 3);
  data_profile const cond_profile =
    data_profile_builder().cardinality(0).no_validity().bool_probability_true(selectivity);
  auto x    = create_random_column(cudf::type_id::INT64, row_count{num_rows}, value_profile);
  auto y    = create_random_column(cudf::type_id::INT64, row_count{num_rows}, value_profile);
  auto cond = [&] {
    if (state.get_string("layout") == "random") {
      return create_random_column(cudf::type_id::BOOL8, row_count{num_rows}, cond_profile);
    }
    auto const row_ids = cudf::sequence(num_rows, cudf::numeric_scalar<int32_t>(0));
    return cudf::binary_operation(
      row_ids->view(),
      cudf::numeric_scalar<int32_t>(static_cast<int32_t>(selectivity * num_rows)),
      cudf::binary_operator::LESS,
      cudf::data_type{cudf::type_id::BOOL8});
  }();
  auto const table    = cudf::table_view{{x->view(), y->view(), cond->view()}};
  auto const xy_table = cudf::table_view{{x->view(), y->view()}};

  cudf::ast::tree tree;
  auto const& then_branch = make_heavy_branch(tree, heavy_ops);
  auto const& else_branch = tree.push(cudf::ast::operation(ast_op::SUB, tree.at(0), tree.at(1)));
  auto const& cond_ref    = tree.push(cudf::ast::column_reference(2));
  auto const& lazy =
    tree.push(cudf::ast::operation(ast_op::IF_ELSE, cond_ref, then_branch, else_branch));

  auto const run = [&](strategy s) -> std::unique_ptr<cudf::column> {
    switch (s) {
      case strategy::LAZY_AST: return cudf::compute_column(table, lazy);
      case strategy::LAZY_JIT: return cudf::compute_column_jit(table, lazy);
      case strategy::EAGER: {
        auto then_column = cudf::compute_column(xy_table, then_branch);
        auto else_column = cudf::compute_column(xy_table, else_branch);
        return cudf::copy_if_else(then_column->view(), else_column->view(), cond->view());
      }
      case strategy::NULL_MASK: {
        auto const null_value = cudf::numeric_scalar<int64_t>(0, false);
        auto not_cond         = cudf::unary_operation(cond->view(), cudf::unary_operator::NOT);
        auto then_x           = cudf::copy_if_else(x->view(), null_value, cond->view());
        auto then_y           = cudf::copy_if_else(y->view(), null_value, cond->view());
        auto else_x           = cudf::copy_if_else(x->view(), null_value, not_cond->view());
        auto else_y           = cudf::copy_if_else(y->view(), null_value, not_cond->view());
        auto then_column =
          cudf::compute_column(cudf::table_view{{then_x->view(), then_y->view()}}, then_branch);
        auto else_column =
          cudf::compute_column(cudf::table_view{{else_x->view(), else_y->view()}}, else_branch);
        return cudf::copy_if_else(then_column->view(), else_column->view(), cond->view());
      }
      case strategy::FILTER_SCATTER: {
        auto const zero = cudf::numeric_scalar<int32_t>(0);
        auto row_ids    = cudf::sequence(num_rows, zero);
        auto not_cond   = cudf::unary_operation(cond->view(), cudf::unary_operator::NOT);
        auto then_rows  = cudf::apply_retention_mask(xy_table, cond->view());
        auto else_rows  = cudf::apply_retention_mask(xy_table, not_cond->view());
        auto then_ids =
          cudf::apply_retention_mask(cudf::table_view{{row_ids->view()}}, cond->view());
        auto else_ids =
          cudf::apply_retention_mask(cudf::table_view{{row_ids->view()}}, not_cond->view());
        auto then_column    = cudf::compute_column(then_rows->view(), then_branch);
        auto else_column    = cudf::compute_column(else_rows->view(), else_branch);
        auto scattered_then = cudf::scatter(cudf::table_view{{then_column->view()}},
                                            then_ids->view().column(0),
                                            cudf::table_view{{x->view()}});
        auto scattered      = cudf::scatter(cudf::table_view{{else_column->view()}},
                                       else_ids->view().column(0),
                                       scattered_then->view());
        return std::move(scattered->release().front());
      }
    }
    CUDF_FAIL("Invalid strategy");
  };

  auto const matches = [&] {
    auto const equal = cudf::binary_operation(run(strategy::LAZY_JIT)->view(),
                                              run(kind)->view(),
                                              cudf::binary_operator::NULL_EQUALS,
                                              cudf::data_type{cudf::type_id::BOOL8});
    auto const all   = cudf::reduce(equal->view(),
                                  *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
                                  cudf::data_type{cudf::type_id::BOOL8});
    return static_cast<cudf::numeric_scalar<bool> const&>(*all).value();
  }();
  CUDF_EXPECTS(matches, "strategy does not match the lazy JIT result");

  state.add_global_memory_reads<int64_t>(static_cast<std::size_t>(num_rows) * 2);
  state.add_global_memory_reads<bool>(num_rows);
  state.add_global_memory_writes<int64_t>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) { run(kind); });
}

NVBENCH_BENCH(BM_lazy_if_else)
  .set_name("lazy_if_else")
  .add_string_axis("strategy", {"lazy_ast", "lazy_jit", "eager", "null_mask", "filter_scatter"})
  .add_int64_axis("selectivity_percent", {1, 10, 50, 90, 99})
  .add_string_axis("layout", {"random", "clustered"})
  .add_int64_axis("heavy_ops", {16, 64})
  .add_int64_axis("num_rows", {20'000'000});
