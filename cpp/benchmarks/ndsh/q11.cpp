/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q11.cpp
 * @brief Implement query 11 of the NDS-H benchmark.
 *
 * select
 *     ps_partkey,
 *     round(sum(ps_supplycost * ps_availqty), 2) as value
 * from
 *     partsupp, supplier, nation
 * where
 *     ps_suppkey = s_suppkey
 *     and s_nationkey = n_nationkey
 *     and n_name = 'GERMANY'
 * group by
 *     ps_partkey
 * having
 *     sum(ps_supplycost * ps_availqty) > (
 *         select
 *             sum(ps_supplycost * ps_availqty) * (0.0001 / scale_factor)
 *         from
 *             partsupp, supplier, nation
 *         where
 *             ps_suppkey = s_suppkey
 *             and s_nationkey = n_nationkey
 *             and n_name = 'GERMANY'
 *     )
 * order by
 *     value desc;
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q11(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::vector<std::string> const nation_columns = {"n_nationkey", "n_name"};
  auto const nation_name                        = cudf::ast::column_reference{1};
  auto germany                                  = cudf::string_scalar{"GERMANY"};
  auto const germany_literal                    = cudf::ast::literal{germany};
  auto nation_filter                            = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, nation_name, germany_literal);

  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace(
    "nation",
    read_parquet(
      sources.at("nation").make_source_info(), nation_columns, std::move(nation_filter)));
  tables.emplace("partsupp",
                 read_parquet(sources.at("partsupp").make_source_info(),
                              {"ps_suppkey", "ps_supplycost", "ps_availqty", "ps_partkey"}));
  tables.emplace(
    "supplier",
    read_parquet(sources.at("supplier").make_source_info(), {"s_suppkey", "s_nationkey"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q11(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables,
  double scale_factor)
{
  auto const partsupp_supplier =
    apply_inner_join(tables.at("partsupp"), tables.at("supplier"), {"ps_suppkey"}, {"s_suppkey"});
  auto joined =
    apply_inner_join(partsupp_supplier, tables.at("nation"), {"s_nationkey"}, {"n_nationkey"});

  auto value = cudf::binary_operation(joined->column("ps_supplycost"),
                                      joined->column("ps_availqty"),
                                      cudf::binary_operator::MUL,
                                      cudf::data_type{cudf::type_id::FLOAT64});
  joined->append(value, "value");

  auto const sum_aggregation = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto total                 = cudf::reduce(
    joined->column("value"), *sum_aggregation, cudf::data_type{cudf::type_id::FLOAT64});
  auto total_column           = cudf::make_column_from_scalar(*total, 1);
  auto const threshold_factor = cudf::numeric_scalar<double>{0.0001 / scale_factor};
  auto threshold_column       = cudf::binary_operation(total_column->view(),
                                                 threshold_factor,
                                                 cudf::binary_operator::MUL,
                                                 cudf::data_type{cudf::type_id::FLOAT64});
  auto threshold              = cudf::get_element(threshold_column->view(), 0);

  auto grouped = apply_groupby(
    joined,
    groupby_context_t{{"ps_partkey"}, {{"value", {{cudf::aggregation::Kind::SUM, "value"}}}}});
  auto mask     = cudf::binary_operation(grouped->column("value"),
                                     *threshold,
                                     cudf::binary_operator::GREATER,
                                     cudf::data_type{cudf::type_id::BOOL8});
  auto filtered = apply_mask(grouped, mask);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  auto rounded = cudf::round(filtered->column("value"), 2, cudf::rounding_method::HALF_UP);
#pragma GCC diagnostic pop
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::make_unique<cudf::column>(filtered->column("ps_partkey")));
  columns.push_back(std::move(rounded));
  auto result =
    std::make_unique<table_with_names>(std::make_unique<cudf::table>(std::move(columns)),
                                       std::vector<std::string>{"ps_partkey", "value"});
  return apply_orderby(result, {"value"}, {cudf::order::DESCENDING});
}

void ndsh_q11(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"nation", "partsupp", "supplier"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q11(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q11(sources);
      result     = execute_ndsh_q11(input, scale_factor);
    } else {
      result = execute_ndsh_q11(tables, scale_factor);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q11(sources);
    result     = execute_ndsh_q11(input, scale_factor);
  } else {
    result = execute_ndsh_q11(tables, scale_factor);
  }
  result->to_parquet("q11.parquet");
}

NVBENCH_BENCH(ndsh_q11)
  .set_name("ndsh_q11")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
