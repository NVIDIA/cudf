/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q13.cpp
 * @brief Implement query 13 of the NDS-H benchmark.
 *
 * select
 *   c_count,
 *   count(*) as custdist
 * from (
 *   select
 *     c_custkey,
 *     count(o_orderkey) as c_count
 *   from customer left outer join orders on
 *     c_custkey = o_custkey
 *     and o_comment not like '%special%requests%'
 *   group by c_custkey
 * ) as c_orders
 * group by c_count
 * order by custdist desc, c_count desc;
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q13(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace("customer",
                 read_parquet(sources.at("customer").make_source_info(), {"c_custkey"}));
  tables.emplace("orders",
                 read_parquet(sources.at("orders").make_source_info(),
                              {"o_custkey", "o_orderkey", "o_comment"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q13(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const& customer = tables.at("customer");
  auto const& orders   = tables.at("orders");

  auto const program = cudf::strings::regex_program::create("special.*requests");
  auto matches =
    cudf::strings::contains_re(cudf::strings_column_view{orders->column("o_comment")}, *program);
  auto keep_orders           = cudf::unary_operation(matches->view(), cudf::unary_operator::NOT);
  auto const filtered_orders = apply_mask(orders, keep_orders);

  auto const joined = apply_left_join(customer, filtered_orders, {"c_custkey"}, {"o_custkey"});

  cudf::groupby::groupby customer_groups(joined->select({"c_custkey"}));
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = joined->column("o_orderkey");
  requests[0].aggregations.push_back(
    cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
  auto [customer_keys, customer_counts] = customer_groups.aggregate(requests);
  auto count_columns                    = customer_keys->release();
  count_columns.push_back(std::move(customer_counts[0].results[0]));
  auto counts =
    std::make_unique<table_with_names>(std::make_unique<cudf::table>(std::move(count_columns)),
                                       std::vector<std::string>{"c_custkey", "c_count"});

  auto const distribution = apply_groupby(
    counts,
    groupby_context_t{{"c_count"},
                      {{"c_custkey", {{cudf::aggregation::Kind::COUNT_ALL, "custdist"}}}}});
  return apply_orderby(
    distribution, {"custdist", "c_count"}, {cudf::order::DESCENDING, cudf::order::DESCENDING});
}

void ndsh_q13(nvbench::state& state)
{
  double const scale_factor = state.get_float64("scale_factor");
  auto const mode           = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"customer", "orders"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q13(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q13(sources);
      result     = execute_ndsh_q13(input);
    } else {
      result = execute_ndsh_q13(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  if (not write_ndsh_results()) { return; }
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q13(sources);
    result     = execute_ndsh_q13(input);
  } else {
    result = execute_ndsh_q13(tables);
  }
  write_ndsh_result(*result, "q13");
}

NVBENCH_BENCH(ndsh_q13)
  .set_name("ndsh_q13")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
