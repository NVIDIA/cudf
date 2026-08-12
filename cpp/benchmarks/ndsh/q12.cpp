/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q12.cpp
 * @brief Implement query 12 of the NDS-H benchmark.
 *
 * select
 *     l_shipmode,
 *     sum(case
 *         when o_orderpriority = '1-URGENT'
 *             or o_orderpriority = '2-HIGH'
 *             then 1
 *         else 0
 *     end) as high_line_count,
 *     sum(case
 *         when o_orderpriority <> '1-URGENT'
 *             and o_orderpriority <> '2-HIGH'
 *             then 1
 *         else 0
 *     end) as low_line_count
 * from
 *     orders,
 *     lineitem
 * where
 *     o_orderkey = l_orderkey
 *     and l_shipmode in ('MAIL', 'SHIP')
 *     and l_commitdate < l_receiptdate
 *     and l_shipdate < l_commitdate
 *     and l_receiptdate >= date '1994-01-01'
 *     and l_receiptdate < date '1994-01-01' + interval '1' year
 * group by
 *     l_shipmode
 * order by
 *     l_shipmode;
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q12(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace(
    "lineitem",
    read_parquet(sources.at("lineitem").make_source_info(),
                 {"l_orderkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"}));
  tables.emplace(
    "orders",
    read_parquet(sources.at("orders").make_source_info(), {"o_orderkey", "o_orderpriority"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q12(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const& lineitem = tables.at("lineitem");
  auto const& orders   = tables.at("orders");

  auto const shipmode_ref = cudf::ast::column_reference(lineitem->column_id("l_shipmode"));
  auto const commit_ref   = cudf::ast::column_reference(lineitem->column_id("l_commitdate"));
  auto const receipt_ref  = cudf::ast::column_reference(lineitem->column_id("l_receiptdate"));
  auto const ship_ref     = cudf::ast::column_reference(lineitem->column_id("l_shipdate"));

  auto mail               = cudf::string_scalar("MAIL");
  auto ship               = cudf::string_scalar("SHIP");
  auto const mail_literal = cudf::ast::literal(mail);
  auto const ship_literal = cudf::ast::literal(ship);
  auto const mail_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, shipmode_ref, mail_literal);
  auto const ship_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, shipmode_ref, ship_literal);
  auto const shipmode_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, mail_predicate, ship_predicate);

  auto const commit_before_receipt =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, commit_ref, receipt_ref);
  auto const ship_before_commit =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, ship_ref, commit_ref);

  auto receipt_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
  auto receipt_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
  auto const receipt_lower_literal = cudf::ast::literal(receipt_lower);
  auto const receipt_upper_literal = cudf::ast::literal(receipt_upper);
  auto const receipt_after_lower   = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, receipt_ref, receipt_lower_literal);
  auto const receipt_before_upper =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, receipt_ref, receipt_upper_literal);

  auto const date_order_predicate = cudf::ast::operation(
    cudf::ast::ast_operator::LOGICAL_AND, commit_before_receipt, ship_before_commit);
  auto const receipt_predicate = cudf::ast::operation(
    cudf::ast::ast_operator::LOGICAL_AND, receipt_after_lower, receipt_before_upper);
  auto const date_predicate = cudf::ast::operation(
    cudf::ast::ast_operator::LOGICAL_AND, date_order_predicate, receipt_predicate);
  auto const lineitem_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, shipmode_predicate, date_predicate);
  auto const filtered_lineitem = apply_filter(lineitem, lineitem_predicate);

  auto joined = apply_inner_join(orders, filtered_lineitem, {"o_orderkey"}, {"l_orderkey"});
  auto const priority_ref   = cudf::ast::column_reference(joined->column_id("o_orderpriority"));
  auto urgent               = cudf::string_scalar("1-URGENT");
  auto high                 = cudf::string_scalar("2-HIGH");
  auto const urgent_literal = cudf::ast::literal(urgent);
  auto const high_literal   = cudf::ast::literal(high);
  auto const urgent_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, priority_ref, urgent_literal);
  auto const high_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, priority_ref, high_literal);
  auto const high_priority_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, urgent_predicate, high_predicate);
  auto const high_priority = cudf::compute_column(joined->table(), high_priority_predicate);

  auto const false_scalar  = cudf::numeric_scalar<bool>{false};
  auto normalized_priority = cudf::replace_nulls(high_priority->view(), false_scalar);
  auto not_high_priority =
    cudf::unary_operation(normalized_priority->view(), cudf::unary_operator::NOT);
  auto valid_priority  = cudf::is_valid(joined->column("o_orderpriority"));
  auto low_priority    = cudf::binary_operation(not_high_priority->view(),
                                             valid_priority->view(),
                                             cudf::binary_operator::LOGICAL_AND,
                                             cudf::data_type{cudf::type_id::BOOL8});
  auto const one       = cudf::numeric_scalar<int64_t>(1);
  auto const zero      = cudf::numeric_scalar<int64_t>(0);
  auto high_line_count = cudf::copy_if_else(one, zero, normalized_priority->view());
  auto low_line_count  = cudf::copy_if_else(one, zero, low_priority->view());
  joined->append(high_line_count, "high_line_count").append(low_line_count, "low_line_count");

  auto grouped = apply_groupby(
    joined,
    groupby_context_t{{"l_shipmode"},
                      {{"high_line_count", {{cudf::aggregation::Kind::SUM, "high_line_count"}}},
                       {"low_line_count", {{cudf::aggregation::Kind::SUM, "low_line_count"}}}}});
  auto projected = apply_projection(grouped, {"l_shipmode", "high_line_count", "low_line_count"});
  return apply_orderby(projected, {"l_shipmode"}, {cudf::order::ASCENDING});
}

void ndsh_q12(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"lineitem", "orders"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q12(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q12(sources);
      result     = execute_ndsh_q12(input);
    } else {
      result = execute_ndsh_q12(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q12(sources);
    result     = execute_ndsh_q12(input);
  } else {
    result = execute_ndsh_q12(tables);
  }
  result->to_parquet("q12.parquet");
}

NVBENCH_BENCH(ndsh_q12)
  .set_name("ndsh_q12")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
