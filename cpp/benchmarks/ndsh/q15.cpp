/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q15.cpp
 * @brief Implement query 15 of the NDS-H benchmark.
 *
 * with revenue (supplier_no, total_revenue) as (
 *     select
 *         l_suppkey,
 *         sum(l_extendedprice * (1 - l_discount))
 *     from
 *         lineitem
 *     where
 *         l_shipdate >= date '1996-01-01'
 *         and l_shipdate < date '1996-01-01' + interval '3' month
 *     group by
 *         l_suppkey
 * )
 * select
 *     s_suppkey,
 *     s_name,
 *     s_address,
 *     s_phone,
 *     total_revenue
 * from
 *     supplier,
 *     revenue
 * where
 *     s_suppkey = supplier_no
 *     and total_revenue = (
 *         select
 *             max(total_revenue)
 *         from
 *             revenue
 *     )
 * order by
 *     s_suppkey;
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q15(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace("lineitem",
                 read_parquet(sources.at("lineitem").make_source_info(),
                              {"l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"}));
  tables.emplace("supplier",
                 read_parquet(sources.at("supplier").make_source_info(),
                              {"s_suppkey", "s_name", "s_address", "s_phone"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q15(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const& lineitem = tables.at("lineitem");

  auto const shipdate_ref = cudf::ast::column_reference(lineitem->column_id("l_shipdate"));
  auto lower = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1996, 1, 1), true);
  auto upper = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1996, 4, 1), true);
  auto const lower_literal = cudf::ast::literal(lower);
  auto const upper_literal = cudf::ast::literal(upper);
  auto const after_lower =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, shipdate_ref, lower_literal);
  auto const before_upper =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, shipdate_ref, upper_literal);
  auto const date_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, after_lower, before_upper);
  auto filtered_lineitem = apply_filter(lineitem, date_predicate);

  auto const one          = cudf::numeric_scalar<double>(1);
  auto one_minus_discount = cudf::binary_operation(one,
                                                   filtered_lineitem->column("l_discount"),
                                                   cudf::binary_operator::SUB,
                                                   cudf::data_type{cudf::type_id::FLOAT64});
  auto revenue            = cudf::binary_operation(filtered_lineitem->column("l_extendedprice"),
                                        one_minus_discount->view(),
                                        cudf::binary_operator::MUL,
                                        cudf::data_type{cudf::type_id::FLOAT64});
  filtered_lineitem->append(revenue, "revenue");
  auto const grouped = apply_groupby(
    filtered_lineitem,
    groupby_context_t{{"l_suppkey"},
                      {{"revenue", {{cudf::aggregation::Kind::SUM, "total_revenue"}}}}});

  auto const max_aggregation = cudf::make_max_aggregation<cudf::reduce_aggregation>();
  auto max_revenue           = cudf::reduce(
    grouped->column("total_revenue"), *max_aggregation, cudf::data_type{cudf::type_id::FLOAT64});
  auto max_mask                = cudf::binary_operation(grouped->column("total_revenue"),
                                         *max_revenue,
                                         cudf::binary_operator::EQUAL,
                                         cudf::data_type{cudf::type_id::BOOL8});
  auto const maximum_suppliers = apply_mask(grouped, max_mask);
  auto const joined =
    apply_inner_join(tables.at("supplier"), maximum_suppliers, {"s_suppkey"}, {"l_suppkey"});
  auto const selected =
    apply_projection(joined, {"s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"});
  return apply_orderby(selected, {"s_suppkey"}, {cudf::order::ASCENDING});
}

void ndsh_q15(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"lineitem", "supplier"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q15(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q15(sources);
      result     = execute_ndsh_q15(input);
    } else {
      result = execute_ndsh_q15(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q15(sources);
    result     = execute_ndsh_q15(input);
  } else {
    result = execute_ndsh_q15(tables);
  }
  result->to_parquet("q15.parquet");
}

NVBENCH_BENCH(ndsh_q15)
  .set_name("ndsh_q15")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
