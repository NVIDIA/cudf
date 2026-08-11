/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvbench/nvbench.cuh>

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q14(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::vector<std::string> const lineitem_columns = {
    "l_partkey", "l_shipdate", "l_extendedprice", "l_discount"};
  auto const shipdate_ref = cudf::ast::column_reference{1};
  auto lower = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 9, 1), true);
  auto upper = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 10, 1), true);
  auto const lower_literal = cudf::ast::literal{lower};
  auto const upper_literal = cudf::ast::literal{upper};
  auto const after_lower =
    cudf::ast::operation{cudf::ast::ast_operator::GREATER_EQUAL, shipdate_ref, lower_literal};
  auto const before_upper =
    cudf::ast::operation{cudf::ast::ast_operator::LESS, shipdate_ref, upper_literal};
  auto shipdate_filter = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, after_lower, before_upper);

  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace(
    "lineitem",
    read_parquet(
      sources.at("lineitem").make_source_info(), lineitem_columns, std::move(shipdate_filter)));
  tables.emplace("part",
                 read_parquet(sources.at("part").make_source_info(), {"p_partkey", "p_type"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q14(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const joined =
    apply_inner_join(tables.at("lineitem"), tables.at("part"), {"l_partkey"}, {"p_partkey"});

  auto const one          = cudf::numeric_scalar<double>{1};
  auto one_minus_discount = cudf::binary_operation(one,
                                                   joined->column("l_discount"),
                                                   cudf::binary_operator::SUB,
                                                   cudf::data_type{cudf::type_id::FLOAT64});
  auto revenue            = cudf::binary_operation(joined->column("l_extendedprice"),
                                        one_minus_discount->view(),
                                        cudf::binary_operator::MUL,
                                        cudf::data_type{cudf::type_id::FLOAT64});

  auto promo      = cudf::strings::starts_with(cudf::strings_column_view{joined->column("p_type")},
                                          cudf::string_scalar{"PROMO"});
  auto const zero = cudf::numeric_scalar<double>{0};
  auto promo_revenue = cudf::copy_if_else(revenue->view(), zero, promo->view());
  auto const promo_total =
    apply_reduction(promo_revenue->view(), cudf::aggregation::Kind::SUM, "promo_total");
  auto const revenue_total =
    apply_reduction(revenue->view(), cudf::aggregation::Kind::SUM, "revenue_total");

  auto const hundred = cudf::numeric_scalar<double>{100};
  auto scaled_promo  = cudf::binary_operation(promo_total->column("promo_total"),
                                             hundred,
                                             cudf::binary_operator::MUL,
                                             cudf::data_type{cudf::type_id::FLOAT64});
  auto percentage    = cudf::binary_operation(scaled_promo->view(),
                                           revenue_total->column("revenue_total"),
                                           cudf::binary_operator::DIV,
                                           cudf::data_type{cudf::type_id::FLOAT64});
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  auto rounded = cudf::round(percentage->view(), 2, cudf::rounding_method::HALF_UP);
#pragma GCC diagnostic pop

  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.push_back(std::move(rounded));
  return std::make_unique<table_with_names>(
    std::make_unique<cudf::table>(std::move(result_columns)),
    std::vector<std::string>{"promo_revenue"});
}

void ndsh_q14(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"lineitem", "part"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q14(sources); }
  std::unique_ptr<table_with_names> result;
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q14(sources);
      result     = execute_ndsh_q14(input);
    } else {
      result = execute_ndsh_q14(tables);
    }
  });
  result->to_parquet("q14.parquet");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q14)
  .set_name("ndsh_q14")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
