/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q21.cpp
 * @brief Implement query 21 of the NDS-H benchmark.
 *
 * select
 *     s_name,
 *     count(*) as numwait
 * from
 *     supplier,
 *     lineitem l1,
 *     orders,
 *     nation
 * where
 *     s_suppkey = l1.l_suppkey
 *     and o_orderkey = l1.l_orderkey
 *     and o_orderstatus = 'F'
 *     and l1.l_receiptdate > l1.l_commitdate
 *     and exists (
 *         select
 *             *
 *         from
 *             lineitem l2
 *         where
 *             l2.l_orderkey = l1.l_orderkey
 *             and l2.l_suppkey <> l1.l_suppkey
 *     )
 *     and not exists (
 *         select
 *             *
 *         from
 *             lineitem l3
 *         where
 *             l3.l_orderkey = l1.l_orderkey
 *             and l3.l_suppkey <> l1.l_suppkey
 *             and l3.l_receiptdate > l3.l_commitdate
 *     )
 *     and s_nationkey = n_nationkey
 *     and n_name = 'SAUDI ARABIA'
 * group by
 *     s_name
 * order by
 *     numwait desc,
 *     s_name
 * limit 100;
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q21(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::vector<std::string> const nation_columns = {"n_nationkey", "n_name"};
  auto const n_name_ref                         = cudf::ast::column_reference{1};
  auto saudi_arabia                             = cudf::string_scalar{"SAUDI ARABIA"};
  auto const saudi_arabia_literal               = cudf::ast::literal{saudi_arabia};
  auto nation_predicate                         = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, n_name_ref, saudi_arabia_literal);

  std::vector<std::string> const orders_columns = {"o_orderkey", "o_orderstatus"};
  auto const o_orderstatus_ref                  = cudf::ast::column_reference{1};
  auto final_status                             = cudf::string_scalar{"F"};
  auto const final_status_literal               = cudf::ast::literal{final_status};
  auto orders_predicate                         = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, o_orderstatus_ref, final_status_literal);

  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace("lineitem",
                 read_parquet(sources.at("lineitem").make_source_info(),
                              {"l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"}));
  tables.emplace(
    "nation",
    read_parquet(
      sources.at("nation").make_source_info(), nation_columns, std::move(nation_predicate)));
  tables.emplace(
    "orders",
    read_parquet(
      sources.at("orders").make_source_info(), orders_columns, std::move(orders_predicate)));
  tables.emplace("supplier",
                 read_parquet(sources.at("supplier").make_source_info(),
                              {"s_suppkey", "s_nationkey", "s_name"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q21(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const& lineitem = tables.at("lineitem");

  auto const late_mask     = cudf::binary_operation(lineitem->column("l_receiptdate"),
                                                lineitem->column("l_commitdate"),
                                                cudf::binary_operator::GREATER,
                                                cudf::data_type{cudf::type_id::BOOL8});
  auto const late_lineitem = apply_mask(lineitem, late_mask);

  auto const supplier_pairs =
    apply_distinct(apply_projection(lineitem, {"l_orderkey", "l_suppkey"}));
  auto const suppliers_per_order = apply_groupby(
    supplier_pairs,
    groupby_context_t{{"l_orderkey"},
                      {{"l_suppkey", {{cudf::aggregation::Kind::COUNT_ALL, "supplier_count"}}}}});
  auto const one           = cudf::numeric_scalar<cudf::size_type>{1};
  auto const multiple_mask = cudf::binary_operation(suppliers_per_order->column("supplier_count"),
                                                    one,
                                                    cudf::binary_operator::GREATER,
                                                    cudf::data_type{cudf::type_id::BOOL8});
  auto const multiple_orders =
    apply_projection(apply_mask(suppliers_per_order, multiple_mask), {"l_orderkey"});

  auto const late_supplier_pairs =
    apply_distinct(apply_projection(late_lineitem, {"l_orderkey", "l_suppkey"}));
  auto const late_suppliers_per_order = apply_groupby(
    late_supplier_pairs,
    groupby_context_t{{"l_orderkey"},
                      {{"l_suppkey", {{cudf::aggregation::Kind::COUNT_ALL, "supplier_count"}}}}});
  auto const single_late_mask =
    cudf::binary_operation(late_suppliers_per_order->column("supplier_count"),
                           one,
                           cudf::binary_operator::EQUAL,
                           cudf::data_type{cudf::type_id::BOOL8});
  auto const single_late_orders =
    apply_projection(apply_mask(late_suppliers_per_order, single_late_mask), {"l_orderkey"});

  auto const qualifying_late =
    apply_left_semi_join(late_lineitem, multiple_orders, {"l_orderkey"}, {"l_orderkey"});
  auto const qualifying_lineitem =
    apply_left_semi_join(qualifying_late, single_late_orders, {"l_orderkey"}, {"l_orderkey"});

  auto const saudi_suppliers =
    apply_inner_join(tables.at("supplier"), tables.at("nation"), {"s_nationkey"}, {"n_nationkey"});
  auto const supplier_lineitem =
    apply_inner_join(qualifying_lineitem, saudi_suppliers, {"l_suppkey"}, {"s_suppkey"});
  auto const joined =
    apply_inner_join(supplier_lineitem, tables.at("orders"), {"l_orderkey"}, {"o_orderkey"});

  auto const grouped = apply_groupby(
    joined,
    groupby_context_t{{"s_name"},
                      {{"l_orderkey", {{cudf::aggregation::Kind::COUNT_ALL, "numwait"}}}}});
  auto const ordered = apply_orderby(
    grouped, {"numwait", "s_name"}, {cudf::order::DESCENDING, cudf::order::ASCENDING});
  return apply_slice(ordered, 0, 100);
}

void ndsh_q21(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"lineitem", "nation", "orders", "supplier"}, sources);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q21(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q21(sources);
      result     = execute_ndsh_q21(input);
    } else {
      result = execute_ndsh_q21(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  if (not write_ndsh_results()) { return; }
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q21(sources);
    result     = execute_ndsh_q21(input);
  } else {
    result = execute_ndsh_q21(tables);
  }
  write_ndsh_result(*result, "q21");
}

NVBENCH_BENCH(ndsh_q21)
  .set_name("ndsh_q21")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
