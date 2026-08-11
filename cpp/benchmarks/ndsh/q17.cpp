/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>

#include <nvbench/nvbench.cuh>

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q17(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::vector<std::string> const part_columns = {"p_partkey", "p_brand", "p_container"};
  auto const brand_ref                        = cudf::ast::column_reference{1};
  auto const container_ref                    = cudf::ast::column_reference{2};
  auto brand                                  = cudf::string_scalar{"Brand#23"};
  auto container                              = cudf::string_scalar{"MED BOX"};
  auto const brand_literal                    = cudf::ast::literal{brand};
  auto const container_literal                = cudf::ast::literal{container};
  auto const brand_predicate =
    cudf::ast::operation{cudf::ast::ast_operator::EQUAL, brand_ref, brand_literal};
  auto const container_predicate =
    cudf::ast::operation{cudf::ast::ast_operator::EQUAL, container_ref, container_literal};
  auto part_predicate = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, brand_predicate, container_predicate);

  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace("lineitem",
                 read_parquet(sources.at("lineitem").make_source_info(),
                              {"l_partkey", "l_quantity", "l_extendedprice"}));
  tables.emplace(
    "part",
    read_parquet(sources.at("part").make_source_info(), part_columns, std::move(part_predicate)));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q17(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const joined =
    apply_inner_join(tables.at("part"), tables.at("lineitem"), {"p_partkey"}, {"l_partkey"});
  auto grouped = apply_groupby(
    joined,
    groupby_context_t{{"p_partkey"},
                      {{"l_quantity", {{cudf::aggregation::Kind::MEAN, "avg_quantity"}}}}});

  auto const threshold_factor = cudf::numeric_scalar<double>{0.2};
  auto threshold              = cudf::binary_operation(grouped->column("avg_quantity"),
                                          threshold_factor,
                                          cudf::binary_operator::MUL,
                                          cudf::data_type{cudf::type_id::FLOAT64});
  grouped->append(threshold, "quantity_threshold");
  auto const with_threshold = apply_inner_join(joined, grouped, {"p_partkey"}, {"p_partkey"});

  auto mask           = cudf::binary_operation(with_threshold->column("l_quantity"),
                                     with_threshold->column("quantity_threshold"),
                                     cudf::binary_operator::LESS,
                                     cudf::data_type{cudf::type_id::BOOL8});
  auto const filtered = apply_mask(with_threshold, mask);
  auto const total =
    apply_reduction(filtered->column("l_extendedprice"), cudf::aggregation::Kind::SUM, "total");
  auto const seven = cudf::numeric_scalar<double>{7.0};
  auto avg_yearly  = cudf::binary_operation(total->column("total"),
                                           seven,
                                           cudf::binary_operator::DIV,
                                           cudf::data_type{cudf::type_id::FLOAT64});
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  auto rounded = cudf::round(avg_yearly->view(), 2, cudf::rounding_method::HALF_UP);
#pragma GCC diagnostic pop

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(rounded));
  return std::make_unique<table_with_names>(std::make_unique<cudf::table>(std::move(columns)),
                                            std::vector<std::string>{"avg_yearly"});
}

void ndsh_q17(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"lineitem", "part"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q17(sources); }
  std::unique_ptr<table_with_names> result;
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q17(sources);
      result     = execute_ndsh_q17(input);
    } else {
      result = execute_ndsh_q17(tables);
    }
  });
  result->to_parquet("q17.parquet");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q17)
  .set_name("ndsh_q17")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
