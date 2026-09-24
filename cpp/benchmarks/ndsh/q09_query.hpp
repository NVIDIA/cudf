/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file q09_query.hpp
 * @brief Shared Q9 execution for benchmarks and integration tests, computing profit by
 * nation and year with selectable binary-operation, AST, or transform evaluation.
 */

#pragma once

#include "utilities.hpp"

#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ndsh::q9 {

enum class engine_type : int32_t { BINARYOP = 0, AST = 1, TRANSFORM = 2 };

struct q9_data {
  std::unique_ptr<table_with_names> lineitem;
  std::unique_ptr<table_with_names> nation;
  std::unique_ptr<table_with_names> orders;
  std::unique_ptr<table_with_names> part;
  std::unique_ptr<table_with_names> partsupp;
  std::unique_ptr<table_with_names> supplier;
};

inline std::map<std::string, std::vector<std::string>> const q9_projections{
  {"lineitem",
   {"l_suppkey", "l_partkey", "l_orderkey", "l_extendedprice", "l_discount", "l_quantity"}},
  {"nation", {"n_nationkey", "n_name"}},
  {"orders", {"o_orderkey", "o_orderdate"}},
  {"part", {"p_partkey", "p_name"}},
  {"partsupp", {"ps_suppkey", "ps_partkey", "ps_supplycost"}},
  {"supplier", {"s_suppkey", "s_nationkey"}}};

/**
 * @brief Calculate the amount column
 *
 * @param discount The discount column
 * @param extendedprice The extended price column
 * @param supplycost The supply cost column
 * @param quantity The quantity column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] inline std::unique_ptr<cudf::column> compute_amount_binaryop(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  cudf::column_view const& supplycost,
  cudf::column_view const& quantity,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_BENCHMARK_RANGE();

  auto const one = cudf::numeric_scalar<double>(1);
  auto const one_minus_discount =
    cudf::binary_operation(one, discount, cudf::binary_operator::SUB, discount.type(), stream, mr);
  auto const extendedprice_discounted_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto const extendedprice_discounted      = cudf::binary_operation(extendedprice,
                                                               one_minus_discount->view(),
                                                               cudf::binary_operator::MUL,
                                                               extendedprice_discounted_type,
                                                               stream,
                                                               mr);
  auto const supplycost_quantity_type      = cudf::data_type{cudf::type_id::FLOAT64};
  auto const supplycost_quantity           = cudf::binary_operation(
    supplycost, quantity, cudf::binary_operator::MUL, supplycost_quantity_type, stream, mr);
  auto amount = cudf::binary_operation(extendedprice_discounted->view(),
                                       supplycost_quantity->view(),
                                       cudf::binary_operator::SUB,
                                       extendedprice_discounted->type(),
                                       stream,
                                       mr);
  return amount;
}

[[nodiscard]] inline std::unique_ptr<cudf::column> compute_amount_transform(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  cudf::column_view const& supplycost,
  cudf::column_view const& quantity,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_BENCHMARK_RANGE();

  std::string udf =
    R"***(
  void calculate_price(double * amount, double discount, double extended_price, double supply_cost, double quantity){
    *amount = extended_price * (1 - discount) - supply_cost * quantity;
  }
  )***";

  cudf::transform_input transform_inputs[] = {discount, extendedprice, supplycost, quantity};

  return std::move(
    cudf::transform(udf,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    transform_inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT64},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt,
                    stream,
                    mr)
      ->release()
      .front());
}

[[nodiscard]] inline std::unique_ptr<cudf::column> compute_amount_ast(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  cudf::column_view const& supplycost,
  cudf::column_view const& quantity,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_BENCHMARK_RANGE();

  cudf::ast::tree tree;
  cudf::table_view table{std::vector{discount, extendedprice, supplycost, quantity}};

  auto& discount_ref       = tree.push(cudf::ast::column_reference{0});
  auto& extended_price_ref = tree.push(cudf::ast::column_reference{1});
  auto& supplycost_ref     = tree.push(cudf::ast::column_reference{2});
  auto& quantity_ref       = tree.push(cudf::ast::column_reference{3});

  auto& extended_price_mul_discount =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::MUL, extended_price_ref, discount_ref});

  // AST presently doesn't support literals on LHS, so we expand extended_price * (1 - discount)
  auto& extended_price_discounted = tree.push(cudf::ast::operation{
    cudf::ast::ast_operator::SUB, extended_price_ref, extended_price_mul_discount});

  auto& quantity_float64 =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::CAST_TO_FLOAT64, quantity_ref});

  auto& supply_cost_mul_quantity =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::MUL, supplycost_ref, quantity_float64});
  auto& result = tree.push(cudf::ast::operation{
    cudf::ast::ast_operator::SUB, extended_price_discounted, supply_cost_mul_quantity});

  return cudf::compute_column(table, result, stream, mr);
}

[[nodiscard]] inline std::unique_ptr<cudf::column> compute_amount(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  cudf::column_view const& supplycost,
  cudf::column_view const& quantity,
  engine_type engine,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  switch (engine) {
    case engine_type::BINARYOP:
      return compute_amount_binaryop(discount, extendedprice, supplycost, quantity, stream, mr);
    case engine_type::AST:
      return compute_amount_ast(discount, extendedprice, supplycost, quantity, stream, mr);
    case engine_type::TRANSFORM:
      return compute_amount_transform(discount, extendedprice, supplycost, quantity, stream, mr);
    default: CUDF_UNREACHABLE("invalid engine_type enum");
  }
}

template <typename Read>
q9_data load_data(Read&& read)
{
  return q9_data{read("lineitem", q9_projections.at("lineitem")),
                 read("nation", q9_projections.at("nation")),
                 read("orders", q9_projections.at("orders")),
                 read("part", q9_projections.at("part")),
                 read("partsupp", q9_projections.at("partsupp")),
                 read("supplier", q9_projections.at("supplier"))};
}

inline std::unique_ptr<table_with_names> join_data(q9_data const& data)
{
  CUDF_BENCHMARK_RANGE();

  // Generating the `profit` table
  // Filter the part table using `p_name like '%green%'`
  auto const p_name        = data.part->table().column(1);
  auto const mask          = cudf::strings::like(cudf::strings_column_view(p_name), "%green%");
  auto const part_filtered = apply_mask(data.part, mask);

  // Perform the joins
  auto const join_a =
    apply_inner_join(data.supplier, data.nation, {"s_nationkey"}, {"n_nationkey"});
  auto const join_b = apply_inner_join(data.partsupp, join_a, {"ps_suppkey"}, {"s_suppkey"});
  auto const join_c = apply_inner_join(data.lineitem, part_filtered, {"l_partkey"}, {"p_partkey"});
  auto const join_d = apply_inner_join(data.orders, join_c, {"o_orderkey"}, {"l_orderkey"});
  return apply_inner_join(join_d, join_b, {"l_suppkey", "l_partkey"}, {"s_suppkey", "ps_partkey"});
}

inline std::unique_ptr<table_with_names> compute_profit(
  engine_type engine,
  q9_data const& data,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto joined_table = join_data(data);
  // Calculate the `nation`, `o_year`, and `amount` columns
  auto n_name = std::make_unique<cudf::column>(joined_table->column("n_name"));
  auto o_year = cudf::datetime::extract_datetime_component(
    joined_table->column("o_orderdate"), cudf::datetime::datetime_component::YEAR);

  auto amount = compute_amount(joined_table->column("l_discount"),
                               joined_table->column("l_extendedprice"),
                               joined_table->column("ps_supplycost"),
                               joined_table->column("l_quantity"),
                               engine,
                               stream,
                               mr);

  // Put together the `profit` table
  std::vector<std::unique_ptr<cudf::column>> profit_columns;
  profit_columns.push_back(std::move(n_name));
  profit_columns.push_back(std::move(o_year));
  profit_columns.push_back(std::move(amount));

  auto profit_table = std::make_unique<cudf::table>(std::move(profit_columns));
  auto const profit = std::make_unique<table_with_names>(
    std::move(profit_table), std::vector<std::string>{"nation", "o_year", "amount"});

  // Perform the groupby operation
  auto const groupedby_table = apply_groupby(
    profit,
    groupby_context_t{{"nation", "o_year"},
                      {{"amount", {{cudf::groupby_aggregation::SUM, "sum_profit"}}}}});

  // Perform the orderby operation
  return apply_orderby(
    groupedby_table, {"nation", "o_year"}, {cudf::order::ASCENDING, cudf::order::DESCENDING});
}

template <typename Read>
std::unique_ptr<table_with_names> execute_q9(
  engine_type engine,
  Read&& read,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const data = load_data(std::forward<Read>(read));
  return compute_profit(engine, data, stream, mr);
}

}  // namespace ndsh::q9
