/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file q03.cpp
 * @brief Implement query 3 of the NDS-H benchmark.
 *
 * select
 *     l_orderkey,
 *     sum(l_extendedprice * (1 - l_discount)) as revenue,
 *     o_orderdate,
 *     o_shippriority
 * from
 *     customer,
 *     orders,
 *     lineitem
 * where
 *     c_mktsegment = 'BUILDING'
 *     and c_custkey = o_custkey
 *     and l_orderkey = o_orderkey
 *     and o_orderdate < '1995-03-15'
 *     and l_shipdate > '1995-03-15'
 * group by
 *     l_orderkey,
 *     o_orderdate,
 *     o_shippriority
 * order by
 *     revenue desc,
 *     o_orderdate
 * limit 10;
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
  return cudf::binary_operation(extendedprice,
                                one_minus_discount->view(),
                                cudf::binary_operator::MUL,
                                cudf::data_type{cudf::type_id::FLOAT64},
                                stream,
                                mr);
}

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q3(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::vector<std::string> const customer_cols = {"c_custkey", "c_mktsegment"};
  auto const c_mktsegment_ref                  = cudf::ast::column_reference(1);
  auto building                                = cudf::string_scalar("BUILDING");
  auto const building_literal                  = cudf::ast::literal(building);
  auto const customer_pred                     = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, c_mktsegment_ref, building_literal);

  std::vector<std::string> const orders_cols = {
    "o_custkey", "o_orderkey", "o_orderdate", "o_shippriority"};
  auto const o_orderdate_ref = cudf::ast::column_reference(2);
  auto orderdate_limit =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 3, 15), true);
  auto const orderdate_literal = cudf::ast::literal(orderdate_limit);
  auto const orders_pred       = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LESS, o_orderdate_ref, orderdate_literal);

  std::vector<std::string> const lineitem_cols = {
    "l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"};
  auto const l_shipdate_ref = cudf::ast::column_reference(3);
  auto shipdate_limit =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 3, 15), true);
  auto const shipdate_literal = cudf::ast::literal(shipdate_limit);
  auto const lineitem_pred    = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::GREATER, l_shipdate_ref, shipdate_literal);

  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace(
    "customer",
    read_parquet(
      sources.at("customer").make_source_info(), customer_cols, std::move(customer_pred)));
  tables.emplace(
    "orders",
    read_parquet(sources.at("orders").make_source_info(), orders_cols, std::move(orders_pred)));
  tables.emplace(
    "lineitem",
    read_parquet(
      sources.at("lineitem").make_source_info(), lineitem_cols, std::move(lineitem_pred)));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q3(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const customer_orders =
    apply_inner_join(tables.at("customer"), tables.at("orders"), {"c_custkey"}, {"o_custkey"});
  auto joined =
    apply_inner_join(customer_orders, tables.at("lineitem"), {"o_orderkey"}, {"l_orderkey"});

  auto revenue = calculate_revenue(joined->column("l_extendedprice"), joined->column("l_discount"));
  joined->append(revenue, "revenue");

  auto const grouped =
    apply_groupby(joined,
                  groupby_context_t{{"l_orderkey", "o_orderdate", "o_shippriority"},
                                    {{"revenue", {{cudf::aggregation::Kind::SUM, "revenue"}}}}});
  auto const ordered = apply_orderby(
    grouped, {"revenue", "o_orderdate"}, {cudf::order::DESCENDING, cudf::order::ASCENDING});
  auto const top_ten = apply_slice(ordered, 0, 10);
  return apply_projection(top_ten, {"l_orderkey", "revenue", "o_orderdate", "o_shippriority"});
}

void ndsh_q3(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"customer", "orders", "lineitem"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q3(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q3(sources);
      result     = execute_ndsh_q3(input);
    } else {
      result = execute_ndsh_q3(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  if (not write_ndsh_results()) { return; }
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q3(sources);
    result     = execute_ndsh_q3(input);
  } else {
    result = execute_ndsh_q3(tables);
  }
  write_ndsh_result(*result, "q03");
}

NVBENCH_BENCH(ndsh_q3)
  .set_name("ndsh_q3")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
