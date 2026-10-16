/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <cstddef>
#include <string>

namespace {

// Cover the FP64 column/scalar arithmetic and register-pressure regression in #23514.
void BM_tpch_q1_compute(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const expression = state.get_string("expression");
  CUDF_EXPECTS(expression == "disc_price" || expression == "charge", "Unknown Q1 expression");
  auto const include_tax = expression == "charge";

  cudf::numeric_scalar<double> ep_value{100.0}, discount_value{0.05}, tax_value{0.08}, one{1.0};
  auto ep          = cudf::sequence(num_rows, ep_value);
  auto discount    = cudf::sequence(num_rows, discount_value);
  auto tax         = include_tax ? cudf::sequence(num_rows, tax_value) : nullptr;
  auto const table = include_tax ? cudf::table_view{{ep->view(), discount->view(), tax->view()}}
                                 : cudf::table_view{{ep->view(), discount->view()}};

  namespace ast = cudf::ast;
  ast::column_reference ep_ref{0}, discount_ref{1}, tax_ref{2};
  ast::literal one_literal{one};
  ast::operation one_sub_discount{ast::ast_operator::SUB, one_literal, discount_ref};
  ast::operation one_add_tax{ast::ast_operator::ADD, one_literal, tax_ref};
  ast::operation disc_price{ast::ast_operator::MUL, ep_ref, one_sub_discount};
  ast::operation charge{ast::ast_operator::MUL, disc_price, one_add_tax};
  auto const& root = include_tax ? charge : disc_price;

  // Compile the UDF before measuring steady-state execution.
  cudf::compute_column_jit(table, root);

  state.add_global_memory_reads<double>(static_cast<std::size_t>(num_rows) * (include_tax ? 3 : 2));
  state.add_global_memory_writes<double>(num_rows);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::compute_column_jit(table, root, launch.get_stream().get_stream());
  });
}

}  // namespace

NVBENCH_BENCH(BM_tpch_q1_compute)
  .set_name("tpch_q1_compute")
  .add_string_axis("expression", {"disc_price", "charge"})
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000, 600'037'902});
