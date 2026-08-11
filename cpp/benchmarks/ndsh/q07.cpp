/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#include <array>

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

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q7(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::vector<std::string> const lineitem_cols = {
    "l_orderkey", "l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"};
  auto const l_shipdate_ref = cudf::ast::column_reference(4);
  auto l_shipdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
  auto const l_shipdate_lower_limit = cudf::ast::literal(l_shipdate_lower);
  auto const l_shipdate_pred_lower  = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, l_shipdate_ref, l_shipdate_lower_limit);
  auto l_shipdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1996, 12, 31), true);
  auto const l_shipdate_upper_limit = cudf::ast::literal(l_shipdate_upper);
  auto const l_shipdate_pred_upper  = cudf::ast::operation(
    cudf::ast::ast_operator::LESS_EQUAL, l_shipdate_ref, l_shipdate_upper_limit);
  auto const lineitem_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, l_shipdate_pred_lower, l_shipdate_pred_upper);

  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace(
    "customer",
    read_parquet(sources.at("customer").make_source_info(), {"c_custkey", "c_nationkey"}));
  tables.emplace(
    "lineitem",
    read_parquet(
      sources.at("lineitem").make_source_info(), lineitem_cols, std::move(lineitem_pred)));
  tables.emplace("nation",
                 read_parquet(sources.at("nation").make_source_info(), {"n_nationkey", "n_name"}));
  tables.emplace(
    "orders", read_parquet(sources.at("orders").make_source_info(), {"o_custkey", "o_orderkey"}));
  tables.emplace(
    "supplier",
    read_parquet(sources.at("supplier").make_source_info(), {"s_suppkey", "s_nationkey"}));
  return tables;
}

std::unique_ptr<table_with_names> filter_nation(std::unique_ptr<table_with_names> const& nation,
                                                std::string const& name,
                                                std::string const& name_column)
{
  auto name_value         = cudf::string_scalar(name);
  auto const name_ref     = cudf::ast::column_reference(nation->column_id("n_name"));
  auto const name_literal = cudf::ast::literal(name_value);
  auto const predicate =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, name_ref, name_literal);
  auto const boolean_mask = cudf::compute_column(nation->table(), predicate);
  auto filtered           = cudf::apply_boolean_mask(nation->table(), boolean_mask->view());
  return std::make_unique<table_with_names>(std::move(filtered),
                                            std::vector<std::string>{"n_nationkey", name_column});
}

std::unique_ptr<table_with_names> make_shipping(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables,
  std::string const& customer_nation_name,
  std::string const& supplier_nation_name)
{
  auto const& customer = tables.at("customer");
  auto const& lineitem = tables.at("lineitem");
  auto const& nation   = tables.at("nation");
  auto const& orders   = tables.at("orders");
  auto const& supplier = tables.at("supplier");

  auto const customer_nation = filter_nation(nation, customer_nation_name, "cust_nation");
  auto const supplier_nation = filter_nation(nation, supplier_nation_name, "supp_nation");
  auto const customers =
    apply_inner_join(customer, customer_nation, {"c_nationkey"}, {"n_nationkey"});
  auto const suppliers =
    apply_inner_join(supplier, supplier_nation, {"s_nationkey"}, {"n_nationkey"});
  auto const items          = apply_inner_join(orders, lineitem, {"o_orderkey"}, {"l_orderkey"});
  auto const customer_items = apply_inner_join(customers, items, {"c_custkey"}, {"o_custkey"});
  auto const joined = apply_inner_join(customer_items, suppliers, {"l_suppkey"}, {"s_suppkey"});

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::make_unique<cudf::column>(joined->column("supp_nation")));
  columns.push_back(std::make_unique<cudf::column>(joined->column("cust_nation")));
  columns.push_back(cudf::datetime::extract_datetime_component(
    joined->column("l_shipdate"), cudf::datetime::datetime_component::YEAR));
  columns.push_back(
    calculate_revenue(joined->column("l_extendedprice"), joined->column("l_discount")));
  return std::make_unique<table_with_names>(
    std::make_unique<cudf::table>(std::move(columns)),
    std::vector<std::string>{"supp_nation", "cust_nation", "l_year", "volume"});
}

std::unique_ptr<table_with_names> execute_ndsh_q7(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const france_to_germany                         = make_shipping(tables, "GERMANY", "FRANCE");
  auto const germany_to_france                         = make_shipping(tables, "FRANCE", "GERMANY");
  std::array<cudf::table_view, 2> const shipping_views = {france_to_germany->table(),
                                                          germany_to_france->table()};
  auto shipping = std::make_unique<table_with_names>(cudf::concatenate(shipping_views),
                                                     france_to_germany->column_names());
  auto const grouped =
    apply_groupby(shipping,
                  groupby_context_t{{"supp_nation", "cust_nation", "l_year"},
                                    {{"volume", {{cudf::aggregation::Kind::SUM, "revenue"}}}}});
  return apply_orderby(grouped,
                       {"supp_nation", "cust_nation", "l_year"},
                       {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING});
}

void ndsh_q7(nvbench::state& state)
{
  double const scale_factor = state.get_float64("scale_factor");
  auto const mode           = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"customer", "lineitem", "nation", "orders", "supplier"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q7(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q7(sources);
      result     = execute_ndsh_q7(input);
    } else {
      result = execute_ndsh_q7(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q7(sources);
    result     = execute_ndsh_q7(input);
  } else {
    result = execute_ndsh_q7(tables);
  }
  result->to_parquet("q7.parquet");
}

NVBENCH_BENCH(ndsh_q7)
  .set_name("ndsh_q7")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
