/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvbench/nvbench.cuh>

#include <optional>

struct q2_data {
  std::unique_ptr<table_with_names> nation;
  std::unique_ptr<table_with_names> part;
  std::unique_ptr<table_with_names> partsupp;
  std::unique_ptr<table_with_names> region;
  std::unique_ptr<table_with_names> supplier;
};

q2_data load_ndsh_q2(std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::vector<std::string> const part_columns = {"p_partkey", "p_size", "p_type", "p_mfgr"};
  auto const p_size_ref                       = cudf::ast::column_reference{1};
  auto p_size                                 = cudf::numeric_scalar<int8_t>{15};
  auto const p_size_literal                   = cudf::ast::literal{p_size};
  auto part_predicate                         = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, p_size_ref, p_size_literal);

  std::vector<std::string> const region_columns = {"r_regionkey", "r_name"};
  auto const r_name_ref                         = cudf::ast::column_reference{1};
  auto r_name                                   = cudf::string_scalar{"EUROPE"};
  auto const r_name_literal                     = cudf::ast::literal{r_name};
  auto region_predicate                         = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, r_name_ref, r_name_literal);

  return q2_data{
    read_parquet(sources.at("nation").make_source_info(), {"n_nationkey", "n_regionkey", "n_name"}),
    read_parquet(sources.at("part").make_source_info(), part_columns, std::move(part_predicate)),
    read_parquet(sources.at("partsupp").make_source_info(),
                 {"ps_partkey", "ps_suppkey", "ps_supplycost"}),
    read_parquet(
      sources.at("region").make_source_info(), region_columns, std::move(region_predicate)),
    read_parquet(
      sources.at("supplier").make_source_info(),
      {"s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"})};
}

std::unique_ptr<table_with_names> execute_ndsh_q2(q2_data const& data)
{
  auto const brass = cudf::string_scalar{"BRASS"};
  auto const mask =
    cudf::strings::ends_with(cudf::strings_column_view{data.part->column("p_type")}, brass);
  auto const part = apply_mask(data.part, mask);

  auto const join_a = apply_inner_join(part, data.partsupp, {"p_partkey"}, {"ps_partkey"});
  auto const join_b = apply_inner_join(join_a, data.supplier, {"ps_suppkey"}, {"s_suppkey"});
  auto const join_c = apply_inner_join(join_b, data.nation, {"s_nationkey"}, {"n_nationkey"});
  auto const joined = apply_inner_join(join_c, data.region, {"n_regionkey"}, {"r_regionkey"});

  auto const minimum_cost = apply_groupby(
    joined,
    groupby_context_t{{"p_partkey"},
                      {{"ps_supplycost", {{cudf::aggregation::Kind::MIN, "ps_supplycost"}}}}});
  auto const minimum_suppliers = apply_inner_join(
    minimum_cost, joined, {"p_partkey", "ps_supplycost"}, {"p_partkey", "ps_supplycost"});
  auto const selected = apply_projection(
    minimum_suppliers,
    {"s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr", "s_address", "s_phone", "s_comment"});
  auto const ordered = apply_orderby(selected,
                                     {"s_acctbal", "n_name", "s_name", "p_partkey"},
                                     {cudf::order::DESCENDING,
                                      cudf::order::ASCENDING,
                                      cudf::order::ASCENDING,
                                      cudf::order::ASCENDING});
  return apply_slice(ordered, 0, 100);
}

void ndsh_q2(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"nation", "part", "partsupp", "region", "supplier"}, sources);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto data =
    mode == query_mode::COMPUTE_ONLY ? std::optional{load_ndsh_q2(sources)} : std::nullopt;
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q2(sources);
      result     = execute_ndsh_q2(input);
    } else {
      result = execute_ndsh_q2(*data);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q2(sources);
    result     = execute_ndsh_q2(input);
  } else {
    result = execute_ndsh_q2(*data);
  }
  result->to_parquet("q2.parquet");
}

NVBENCH_BENCH(ndsh_q2)
  .set_name("ndsh_q2")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
