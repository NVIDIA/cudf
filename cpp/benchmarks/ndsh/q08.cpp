/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q08.cpp
 * @brief Implement query 8 of the NDS-H benchmark.
 *
 * select
 *     o_year,
 *     round(
 *         sum(case
 *             when nation = 'BRAZIL' then volume
 *             else 0
 *         end) / sum(volume)
 *     , 2) as mkt_share
 * from
 *     (
 *         select
 *             extract(year from o_orderdate) as o_year,
 *             l_extendedprice * (1 - l_discount) as volume,
 *             n2.n_name as nation
 *         from
 *             part,
 *             supplier,
 *             lineitem,
 *             orders,
 *             customer,
 *             nation n1,
 *             nation n2,
 *             region
 *         where
 *             p_partkey = l_partkey
 *             and s_suppkey = l_suppkey
 *             and l_orderkey = o_orderkey
 *             and o_custkey = c_custkey
 *             and c_nationkey = n1.n_nationkey
 *             and n1.n_regionkey = r_regionkey
 *             and r_name = 'AMERICA'
 *             and s_nationkey = n2.n_nationkey
 *             and o_orderdate between timestamp '1995-01-01' and timestamp '1996-12-31'
 *             and p_type = 'ECONOMY ANODIZED STEEL'
 *     ) as all_nations
 * group by
 *     o_year
 * order by
 *     o_year;
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q8(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::vector<std::string> const orders_cols = {"o_orderkey", "o_custkey", "o_orderdate"};
  auto const orderdate_ref                   = cudf::ast::column_reference(2);
  auto orderdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
  auto const orderdate_lower_literal = cudf::ast::literal(orderdate_lower);
  auto orderdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1996, 12, 31), true);
  auto const orderdate_upper_literal = cudf::ast::literal(orderdate_upper);
  auto const orderdate_pred_lower    = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, orderdate_ref, orderdate_lower_literal);
  auto const orderdate_pred_upper = cudf::ast::operation(
    cudf::ast::ast_operator::LESS_EQUAL, orderdate_ref, orderdate_upper_literal);
  auto const orders_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, orderdate_pred_lower, orderdate_pred_upper);

  std::vector<std::string> const part_cols = {"p_partkey", "p_type"};
  auto const part_type_ref                 = cudf::ast::column_reference(1);
  auto part_type                           = cudf::string_scalar("ECONOMY ANODIZED STEEL");
  auto const part_type_literal             = cudf::ast::literal(part_type);
  auto const part_pred                     = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, part_type_ref, part_type_literal);

  std::vector<std::string> const region_cols = {"r_regionkey", "r_name"};
  auto const region_name_ref                 = cudf::ast::column_reference(1);
  auto region_name                           = cudf::string_scalar("AMERICA");
  auto const region_name_literal             = cudf::ast::literal(region_name);
  auto const region_pred                     = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, region_name_ref, region_name_literal);

  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace(
    "customer",
    read_parquet(sources.at("customer").make_source_info(), {"c_custkey", "c_nationkey"}));
  tables.emplace(
    "lineitem",
    read_parquet(sources.at("lineitem").make_source_info(),
                 {"l_partkey", "l_suppkey", "l_orderkey", "l_extendedprice", "l_discount"}));
  tables.emplace("nation",
                 read_parquet(sources.at("nation").make_source_info(),
                              {"n_nationkey", "n_regionkey", "n_name"}));
  tables.emplace(
    "orders",
    read_parquet(sources.at("orders").make_source_info(), orders_cols, std::move(orders_pred)));
  tables.emplace(
    "part", read_parquet(sources.at("part").make_source_info(), part_cols, std::move(part_pred)));
  tables.emplace(
    "region",
    read_parquet(sources.at("region").make_source_info(), region_cols, std::move(region_pred)));
  tables.emplace(
    "supplier",
    read_parquet(sources.at("supplier").make_source_info(), {"s_suppkey", "s_nationkey"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q8(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const customer_nation =
    apply_projection(tables.at("nation"), {"n_nationkey", "n_regionkey"});
  auto const supplier_nation = apply_projection(tables.at("nation"), {"n_nationkey", "n_name"});
  auto const america_nations =
    apply_inner_join(customer_nation, tables.at("region"), {"n_regionkey"}, {"r_regionkey"});

  auto const join_a =
    apply_inner_join(tables.at("part"), tables.at("lineitem"), {"p_partkey"}, {"l_partkey"});
  auto const join_b = apply_inner_join(join_a, tables.at("supplier"), {"l_suppkey"}, {"s_suppkey"});
  auto const join_c = apply_inner_join(join_b, tables.at("orders"), {"l_orderkey"}, {"o_orderkey"});
  auto const join_d = apply_inner_join(join_c, tables.at("customer"), {"o_custkey"}, {"c_custkey"});
  auto const join_e = apply_inner_join(join_d, america_nations, {"c_nationkey"}, {"n_nationkey"});
  auto joined       = apply_inner_join(join_e, supplier_nation, {"s_nationkey"}, {"n_nationkey"});

  auto o_year = cudf::datetime::extract_datetime_component(
    joined->column("o_orderdate"), cudf::datetime::datetime_component::YEAR);
  auto const one                = cudf::numeric_scalar<double>(1);
  auto const one_minus_discount = cudf::binary_operation(one,
                                                         joined->column("l_discount"),
                                                         cudf::binary_operator::SUB,
                                                         cudf::data_type{cudf::type_id::FLOAT64});
  auto volume                   = cudf::binary_operation(joined->column("l_extendedprice"),
                                       one_minus_discount->view(),
                                       cudf::binary_operator::MUL,
                                       cudf::data_type{cudf::type_id::FLOAT64});

  auto const nation_ref     = cudf::ast::column_reference(joined->column_id("n_name"));
  auto brazil               = cudf::string_scalar("BRAZIL");
  auto const brazil_literal = cudf::ast::literal(brazil);
  auto const brazil_pred =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, nation_ref, brazil_literal);
  auto const brazil_mask = cudf::compute_column(joined->table(), brazil_pred);
  auto const zero        = cudf::numeric_scalar<double>(0);
  auto brazil_volume     = cudf::copy_if_else(volume->view(), zero, brazil_mask->view());

  joined->append(o_year, "o_year").append(volume, "volume").append(brazil_volume, "brazil_volume");
  auto const grouped = apply_groupby(
    joined,
    groupby_context_t{{"o_year"},
                      {{"volume", {{cudf::aggregation::Kind::SUM, "volume"}}},
                       {"brazil_volume", {{cudf::aggregation::Kind::SUM, "brazil_volume"}}}}});
  auto market_share = cudf::binary_operation(grouped->column("brazil_volume"),
                                             grouped->column("volume"),
                                             cudf::binary_operator::DIV,
                                             cudf::data_type{cudf::type_id::FLOAT64});
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  auto rounded_market_share = cudf::round(market_share->view(), 2, cudf::rounding_method::HALF_UP);
#pragma GCC diagnostic pop

  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.push_back(std::make_unique<cudf::column>(grouped->column("o_year")));
  result_columns.push_back(std::move(rounded_market_share));
  auto result =
    std::make_unique<table_with_names>(std::make_unique<cudf::table>(std::move(result_columns)),
                                       std::vector<std::string>{"o_year", "mkt_share"});
  return apply_orderby(result, {"o_year"}, {cudf::order::ASCENDING});
}

void ndsh_q8(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor,
    {"customer", "lineitem", "nation", "orders", "part", "region", "supplier"},
    sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q8(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q8(sources);
      result     = execute_ndsh_q8(input);
    } else {
      result = execute_ndsh_q8(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q8(sources);
    result     = execute_ndsh_q8(input);
  } else {
    result = execute_ndsh_q8(tables);
  }
  result->to_parquet("q8.parquet");
}

NVBENCH_BENCH(ndsh_q8)
  .set_name("ndsh_q8")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
