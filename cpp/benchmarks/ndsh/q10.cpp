/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q10.cpp
 * @brief Implement query 10 of the NDS-H benchmark.
 *
 * create view customer as select * from '/tables/scale-1/customer.parquet';
 * create view orders as select * from '/tables/scale-1/orders.parquet';
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 * create view nation as select * from '/tables/scale-1/nation.parquet';
 *
 * select
 *    c_custkey,
 *    c_name,
 *    sum(l_extendedprice * (1 - l_discount)) as revenue,
 *    c_acctbal,
 *    n_name,
 *    c_address,
 *    c_phone,
 *    c_comment
 * from
 *    customer,
 *    orders,
 *    lineitem,
 *    nation
 * where
 *     c_custkey = o_custkey
 *     and l_orderkey = o_orderkey
 *     and o_orderdate >= date '1993-10-01'
 *     and o_orderdate < date '1994-01-01'
 *     and l_returnflag = 'R'
 *     and c_nationkey = n_nationkey
 * group by
 *     c_custkey,
 *     c_name,
 *     c_acctbal,
 *     c_phone,
 *     n_name,
 *     c_address,
 *     c_comment
 * order by
 *     revenue desc;
 */

/**
 * @brief Calculate the revenue column
 *
 * @param extendedprice The extended price column
 * @param discount The discount column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> calculate_revenue(
  cudf::column_view const& extendedprice,
  cudf::column_view const& discount,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
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

void run_ndsh_q10(nvbench::state& state,
                  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  // Define the column projection and filter predicate for the `orders` table
  std::vector<std::string> const orders_cols = {"o_custkey", "o_orderkey", "o_orderdate"};
  auto const o_orderdate_ref                 = cudf::ast::column_reference(std::distance(
    orders_cols.begin(), std::find(orders_cols.begin(), orders_cols.end(), "o_orderdate")));
  auto o_orderdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1993, 10, 1), true);
  auto const o_orderdate_lower_limit = cudf::ast::literal(o_orderdate_lower);
  auto const o_orderdate_pred_lower  = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, o_orderdate_ref, o_orderdate_lower_limit);
  auto o_orderdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
  auto const o_orderdate_upper_limit = cudf::ast::literal(o_orderdate_upper);
  auto const o_orderdate_pred_upper =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, o_orderdate_ref, o_orderdate_upper_limit);
  auto const orders_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, o_orderdate_pred_lower, o_orderdate_pred_upper);

  auto const l_returnflag_ref = cudf::ast::column_reference(3);
  auto r_scalar               = cudf::string_scalar("R");
  auto const r_literal        = cudf::ast::literal(r_scalar);
  auto const lineitem_pred    = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, l_returnflag_ref, r_literal);

  // Read out the tables from parquet files
  // while pushing down the column projections and filter predicates
  auto const customer = read_parquet(
    sources.at("customer").make_source_info(),
    {"c_custkey", "c_name", "c_nationkey", "c_acctbal", "c_address", "c_phone", "c_comment"});
  auto const orders =
    read_parquet(sources.at("orders").make_source_info(), orders_cols, std::move(orders_pred));
  auto const lineitem =
    read_parquet(sources.at("lineitem").make_source_info(),
                 {"l_extendedprice", "l_discount", "l_orderkey", "l_returnflag"},
                 std::move(lineitem_pred));
  auto const nation =
    read_parquet(sources.at("nation").make_source_info(), {"n_name", "n_nationkey"});

  // Perform the joins
  auto const join_a       = apply_inner_join(customer, nation, {"c_nationkey"}, {"n_nationkey"});
  auto const join_b       = apply_inner_join(lineitem, orders, {"l_orderkey"}, {"o_orderkey"});
  auto const joined_table = apply_inner_join(join_a, join_b, {"c_custkey"}, {"o_custkey"});

  // Calculate and append the `revenue` column
  auto revenue =
    calculate_revenue(joined_table->column("l_extendedprice"), joined_table->column("l_discount"));
  (*joined_table).append(revenue, "revenue");

  // Perform the groupby operation
  auto const groupedby_table = apply_groupby(
    joined_table,
    groupby_context_t{
      {"c_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address", "c_comment"},
      {
        {"revenue", {{cudf::aggregation::Kind::SUM, "revenue"}}},
      }});

  // Perform the order by operation
  auto const orderedby_table =
    apply_orderby(groupedby_table, {"revenue"}, {cudf::order::DESCENDING});

  // Write query result to a parquet file
  orderedby_table->to_parquet("q10.parquet");
}

void ndsh_q10(nvbench::state& state)
{
  // Generate the required parquet files in device buffers
  double const scale_factor = state.get_float64("scale_factor");
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"customer", "orders", "lineitem", "nation"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { run_ndsh_q10(state, sources); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q10).set_name("ndsh_q10").add_float64_axis("scale_factor", {0.01, 0.1, 1});
