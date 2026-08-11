/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>

#include <nvbench/nvbench.cuh>

struct q4_data {
  std::unique_ptr<table_with_names> lineitem;
  std::unique_ptr<table_with_names> orders;
};

q4_data load_ndsh_q4(std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::vector<std::string> const lineitem_cols = {"l_orderkey", "l_commitdate", "l_receiptdate"};

  std::vector<std::string> const orders_cols = {"o_orderkey", "o_orderdate", "o_orderpriority"};
  auto const orderdate_ref                   = cudf::ast::column_reference(1);
  auto orderdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1993, 7, 1), true);
  auto orderdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1993, 10, 1), true);
  auto const orderdate_lower_literal = cudf::ast::literal(orderdate_lower);
  auto const orderdate_upper_literal = cudf::ast::literal(orderdate_upper);
  auto const orderdate_lower_pred    = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, orderdate_ref, orderdate_lower_literal);
  auto const orderdate_upper_pred =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, orderdate_ref, orderdate_upper_literal);
  auto const orders_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, orderdate_lower_pred, orderdate_upper_pred);

  return q4_data{
    read_parquet(sources.at("lineitem").make_source_info(), lineitem_cols),
    read_parquet(sources.at("orders").make_source_info(), orders_cols, std::move(orders_pred))};
}

std::unique_ptr<table_with_names> execute_ndsh_q4(q4_data const& data)
{
  auto const commitdate_ref = cudf::ast::column_reference(data.lineitem->column_id("l_commitdate"));
  auto const receiptdate_ref =
    cudf::ast::column_reference(data.lineitem->column_id("l_receiptdate"));
  auto const lineitem_pred =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, commitdate_ref, receiptdate_ref);
  auto const lineitem = apply_filter(data.lineitem, lineitem_pred);
  auto const qualifying_orders =
    apply_left_semi_join(data.orders, lineitem, {"o_orderkey"}, {"l_orderkey"});
  auto const grouped_orders = apply_groupby(
    qualifying_orders,
    groupby_context_t{{"o_orderpriority"},
                      {{"o_orderkey", {{cudf::aggregation::Kind::COUNT_ALL, "order_count"}}}}});
  return apply_orderby(grouped_orders, {"o_orderpriority"}, {cudf::order::ASCENDING});
}

void ndsh_q4(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"lineitem", "orders"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  q4_data data;
  if (mode == query_mode::COMPUTE_ONLY) { data = load_ndsh_q4(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q4(sources);
      result     = execute_ndsh_q4(input);
    } else {
      result = execute_ndsh_q4(data);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q4(sources);
    result     = execute_ndsh_q4(input);
  } else {
    result = execute_ndsh_q4(data);
  }
  result->to_parquet("q4.parquet");
}

NVBENCH_BENCH(ndsh_q4)
  .set_name("ndsh_q4")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
