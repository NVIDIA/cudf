/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q20.cpp
 * @brief Implement query 20 of the NDS-H benchmark.
 *
 * select
 *     s_name,
 *     s_address
 * from
 *     supplier,
 *     nation
 * where
 *     s_suppkey in (
 *         select
 *             ps_suppkey
 *         from
 *             partsupp
 *         where
 *             ps_partkey in (
 *                 select
 *                     p_partkey
 *                 from
 *                     part
 *                 where
 *                     p_name like 'forest%'
 *             )
 *             and ps_availqty > (
 *                 select
 *                     0.5 * sum(l_quantity)
 *                 from
 *                     lineitem
 *                 where
 *                     l_partkey = ps_partkey
 *                     and l_suppkey = ps_suppkey
 *                     and l_shipdate >= date '1994-01-01'
 *                     and l_shipdate < date '1994-01-01' + interval '1' year
 *             )
 *     )
 *     and s_nationkey = n_nationkey
 *     and n_name = 'CANADA'
 * order by
 *     s_name;
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q20(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::vector<std::string> const lineitem_columns = {
    "l_partkey", "l_suppkey", "l_quantity", "l_shipdate"};
  auto const shipdate = cudf::ast::column_reference{3};
  auto start = cudf::timestamp_scalar<cudf::timestamp_D>{days_since_epoch(1994, 1, 1), true};
  auto end   = cudf::timestamp_scalar<cudf::timestamp_D>{days_since_epoch(1995, 1, 1), true};
  auto const start_literal = cudf::ast::literal{start};
  auto const end_literal   = cudf::ast::literal{end};
  auto const after_start =
    cudf::ast::operation{cudf::ast::ast_operator::GREATER_EQUAL, shipdate, start_literal};
  auto const before_end =
    cudf::ast::operation{cudf::ast::ast_operator::LESS, shipdate, end_literal};
  auto lineitem_filter = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, after_start, before_end);

  std::vector<std::string> const nation_columns = {"n_nationkey", "n_name"};
  auto const nation_name                        = cudf::ast::column_reference{1};
  auto canada                                   = cudf::string_scalar{"CANADA"};
  auto const canada_literal                     = cudf::ast::literal{canada};
  auto nation_filter                            = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, nation_name, canada_literal);

  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace(
    "lineitem",
    read_parquet(
      sources.at("lineitem").make_source_info(), lineitem_columns, std::move(lineitem_filter)));
  tables.emplace(
    "nation",
    read_parquet(
      sources.at("nation").make_source_info(), nation_columns, std::move(nation_filter)));
  tables.emplace("part",
                 read_parquet(sources.at("part").make_source_info(), {"p_partkey", "p_name"}));
  tables.emplace("partsupp",
                 read_parquet(sources.at("partsupp").make_source_info(),
                              {"ps_partkey", "ps_suppkey", "ps_availqty"}));
  tables.emplace("supplier",
                 read_parquet(sources.at("supplier").make_source_info(),
                              {"s_suppkey", "s_nationkey", "s_name", "s_address"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q20(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto quantities = apply_groupby(
    tables.at("lineitem"),
    groupby_context_t{{"l_partkey", "l_suppkey"},
                      {{"l_quantity", {{cudf::aggregation::Kind::SUM, "sum_quantity"}}}}});
  auto const half = cudf::numeric_scalar<double>{0.5};
  auto threshold  = cudf::binary_operation(quantities->column("sum_quantity"),
                                          half,
                                          cudf::binary_operator::MUL,
                                          cudf::data_type{cudf::type_id::FLOAT64});
  quantities->append(threshold, "threshold");

  auto const forest = cudf::string_scalar{"forest"};
  auto forest_mask  = cudf::strings::starts_with(
    cudf::strings_column_view{tables.at("part")->column("p_name")}, forest);
  auto const forest_parts =
    apply_distinct(apply_projection(apply_mask(tables.at("part"), forest_mask), {"p_partkey"}));

  auto const inventory =
    apply_inner_join(forest_parts, tables.at("partsupp"), {"p_partkey"}, {"ps_partkey"});
  auto const inventory_quantity = apply_inner_join(
    inventory, quantities, {"ps_suppkey", "p_partkey"}, {"l_suppkey", "l_partkey"});
  auto available = cudf::binary_operation(inventory_quantity->column("ps_availqty"),
                                          inventory_quantity->column("threshold"),
                                          cudf::binary_operator::GREATER,
                                          cudf::data_type{cudf::type_id::BOOL8});
  auto const eligible_suppliers =
    apply_distinct(apply_projection(apply_mask(inventory_quantity, available), {"ps_suppkey"}));

  auto const canadian_suppliers =
    apply_inner_join(tables.at("supplier"), tables.at("nation"), {"s_nationkey"}, {"n_nationkey"});
  auto const result =
    apply_inner_join(eligible_suppliers, canadian_suppliers, {"ps_suppkey"}, {"s_suppkey"});
  return apply_orderby(
    apply_projection(result, {"s_name", "s_address"}), {"s_name"}, {cudf::order::ASCENDING});
}

void ndsh_q20(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"lineitem", "nation", "part", "partsupp", "supplier"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q20(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q20(sources);
      result     = execute_ndsh_q20(input);
    } else {
      result = execute_ndsh_q20(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q20(sources);
    result     = execute_ndsh_q20(input);
  } else {
    result = execute_ndsh_q20(tables);
  }
  result->to_parquet("q20.parquet");
}

NVBENCH_BENCH(ndsh_q20)
  .set_name("ndsh_q20")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
