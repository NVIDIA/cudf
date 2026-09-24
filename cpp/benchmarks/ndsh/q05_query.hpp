/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "utilities.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ndsh::q5 {

inline std::map<std::string, std::vector<std::string>> const q5_projections{
  {"orders", {"o_custkey", "o_orderkey", "o_orderdate"}},
  {"customer", {"c_custkey", "c_nationkey"}},
  {"lineitem", {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"}},
  {"supplier", {"s_suppkey", "s_nationkey"}},
  {"nation", {"n_nationkey", "n_regionkey", "n_name"}},
  {"region", {"r_regionkey", "r_name"}}};

/**
 * @brief Calculate the revenue column
 *
 * @param extendedprice The extended price column
 * @param discount The discount column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] inline std::unique_ptr<cudf::column> calculate_revenue(
  cudf::column_view const& extendedprice,
  cudf::column_view const& discount,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const one = cudf::numeric_scalar<double>(1);
  auto const one_minus_discount =
    cudf::binary_operation(one, discount, cudf::binary_operator::SUB, discount.type(), stream, mr);
  auto const revenue_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto revenue            = cudf::binary_operation(extendedprice,
                                        one_minus_discount->view(),
                                        cudf::binary_operator::MUL,
                                        revenue_type,
                                        stream,
                                        mr);
  return revenue;
}

/**
 * read returns owning projected tables, applying supplied predicates if filter_predicates is false.
 * Otherwise filtering happens here. consume receives the result owner by reference and may
 * move it out; its return value is forwarded. This helper adds no final stream synchronization.
 */
template <typename Read, typename Consume>
auto execute_q5(Read&& read, bool filter_predicates, Consume&& consume)
{
  // Define the column projection and filter predicate for the `orders` table
  auto const& orders_cols    = q5_projections.at("orders");
  auto const o_orderdate_ref = cudf::ast::column_reference(std::distance(
    orders_cols.begin(), std::find(orders_cols.begin(), orders_cols.end(), "o_orderdate")));
  auto o_orderdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
  auto const o_orderdate_lower_limit = cudf::ast::literal(o_orderdate_lower);
  auto const o_orderdate_pred_lower  = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, o_orderdate_ref, o_orderdate_lower_limit);
  auto o_orderdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
  auto const o_orderdate_upper_limit = cudf::ast::literal(o_orderdate_upper);
  auto const o_orderdate_pred_upper =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, o_orderdate_ref, o_orderdate_upper_limit);
  auto const orders_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, o_orderdate_pred_lower, o_orderdate_pred_upper);

  // Define the column projection and filter predicate for the `region` table
  auto const& region_cols   = q5_projections.at("region");
  auto const r_name_ref     = cudf::ast::column_reference(std::distance(
    region_cols.begin(), std::find(region_cols.begin(), region_cols.end(), "r_name")));
  auto r_name_value         = cudf::string_scalar("ASIA");
  auto const r_name_literal = cudf::ast::literal(r_name_value);
  auto const region_pred    = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, r_name_ref, r_name_literal);

  std::unique_ptr<cudf::ast::operation> const no_predicate;
  auto const customer = read("customer", q5_projections.at("customer"), no_predicate);
  auto orders         = read("orders", orders_cols, orders_pred);
  auto const lineitem = read("lineitem", q5_projections.at("lineitem"), no_predicate);
  auto const supplier = read("supplier", q5_projections.at("supplier"), no_predicate);
  auto const nation   = read("nation", q5_projections.at("nation"), no_predicate);
  auto region         = read("region", region_cols, region_pred);
  if (filter_predicates) {
    orders = apply_filter(orders, *orders_pred);
    region = apply_filter(region, *region_pred);
  }

  // Perform the joins
  auto const join_a = apply_inner_join(region, nation, {"r_regionkey"}, {"n_regionkey"});
  auto const join_b = apply_inner_join(join_a, customer, {"n_nationkey"}, {"c_nationkey"});
  auto const join_c = apply_inner_join(join_b, orders, {"c_custkey"}, {"o_custkey"});
  auto const join_d = apply_inner_join(join_c, lineitem, {"o_orderkey"}, {"l_orderkey"});
  auto joined_table =
    apply_inner_join(supplier, join_d, {"s_suppkey", "s_nationkey"}, {"l_suppkey", "n_nationkey"});

  // Calculate and append the `revenue` column
  auto revenue =
    calculate_revenue(joined_table->column("l_extendedprice"), joined_table->column("l_discount"));
  (*joined_table).append(revenue, "revenue");

  // Perform the groupby operation
  auto const groupedby_table =
    apply_groupby(joined_table,
                  groupby_context_t{{"n_name"},
                                    {
                                      {"revenue", {{cudf::aggregation::Kind::SUM, "revenue"}}},
                                    }});

  // Perform the order by operation
  auto orderedby_table = apply_orderby(groupedby_table, {"revenue"}, {cudf::order::DESCENDING});
  return consume(orderedby_table);
}

}  // namespace ndsh::q5
