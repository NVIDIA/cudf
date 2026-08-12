/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q18.cpp
 * @brief Implement query 18 of the NDS-H benchmark.
 *
 * select
 *     c_name,
 *     c_custkey,
 *     o_orderkey,
 *     o_orderdate,
 *     o_totalprice,
 *     sum(l_quantity)
 * from
 *     customer,
 *     orders,
 *     lineitem
 * where
 *     o_orderkey in (
 *         select
 *             l_orderkey
 *         from
 *             lineitem
 *         group by
 *             l_orderkey having
 *                 sum(l_quantity) > 300
 *     )
 *     and c_custkey = o_custkey
 *     and o_orderkey = l_orderkey
 * group by
 *     c_name,
 *     c_custkey,
 *     o_orderkey,
 *     o_orderdate,
 *     o_totalprice
 * order by
 *     o_totalprice desc,
 *     o_orderdate
 * limit 100;
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q18(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace("customer",
                 read_parquet(sources.at("customer").make_source_info(), {"c_custkey", "c_name"}));
  tables.emplace(
    "lineitem",
    read_parquet(sources.at("lineitem").make_source_info(), {"l_orderkey", "l_quantity"}));
  tables.emplace("orders",
                 read_parquet(sources.at("orders").make_source_info(),
                              {"o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q18(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const quantity_by_order = apply_groupby(
    tables.at("lineitem"),
    groupby_context_t{{"l_orderkey"},
                      {{"l_quantity", {{cudf::aggregation::Kind::SUM, "sum_quantity"}}}}});
  auto const threshold    = cudf::numeric_scalar<int64_t>{300};
  auto mask               = cudf::binary_operation(quantity_by_order->column("sum_quantity"),
                                     threshold,
                                     cudf::binary_operator::GREATER,
                                     cudf::data_type{cudf::type_id::BOOL8});
  auto const large_orders = apply_mask(quantity_by_order, mask);

  auto const qualifying_orders =
    apply_left_semi_join(tables.at("orders"), large_orders, {"o_orderkey"}, {"l_orderkey"});
  auto const orders_lineitem =
    apply_inner_join(qualifying_orders, tables.at("lineitem"), {"o_orderkey"}, {"l_orderkey"});
  auto const joined =
    apply_inner_join(tables.at("customer"), orders_lineitem, {"c_custkey"}, {"o_custkey"});
  auto const grouped = apply_groupby(
    joined,
    groupby_context_t{{"c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"},
                      {{"l_quantity", {{cudf::aggregation::Kind::SUM, "sum(l_quantity)"}}}}});
  auto const ordered = apply_orderby(
    grouped, {"o_totalprice", "o_orderdate"}, {cudf::order::DESCENDING, cudf::order::ASCENDING});
  return apply_slice(ordered, 0, 100);
}

void ndsh_q18(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"customer", "orders", "lineitem"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q18(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q18(sources);
      result     = execute_ndsh_q18(input);
    } else {
      result = execute_ndsh_q18(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  if (not write_ndsh_results()) { return; }
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q18(sources);
    result     = execute_ndsh_q18(input);
  } else {
    result = execute_ndsh_q18(tables);
  }
  write_ndsh_result(*result, "q18");
}

NVBENCH_BENCH(ndsh_q18)
  .set_name("ndsh_q18")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
