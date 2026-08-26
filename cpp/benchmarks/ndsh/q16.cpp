/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q16.cpp
 * @brief Implement query 16 of the NDS-H benchmark.
 *
 * select
 *     p_brand,
 *     p_type,
 *     p_size,
 *     count(distinct ps_suppkey) as supplier_cnt
 * from
 *     partsupp,
 *     part
 * where
 *     p_partkey = ps_partkey
 *     and p_brand <> 'Brand#45'
 *     and p_type not like 'MEDIUM POLISHED%'
 *     and p_size in (49, 14, 23, 45, 19, 3, 36, 9)
 *     and ps_suppkey not in (
 *         select
 *             s_suppkey
 *         from
 *             supplier
 *         where
 *             s_comment like '%Customer%Complaints%'
 *     )
 * group by
 *     p_brand,
 *     p_type,
 *     p_size
 * order by
 *     supplier_cnt desc,
 *     p_brand,
 *     p_type,
 *     p_size;
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q16(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace("part",
                 read_parquet(sources.at("part").make_source_info(),
                              {"p_partkey", "p_brand", "p_type", "p_size"}));
  tables.emplace(
    "partsupp",
    read_parquet(sources.at("partsupp").make_source_info(), {"ps_partkey", "ps_suppkey"}));
  tables.emplace(
    "supplier",
    read_parquet(sources.at("supplier").make_source_info(), {"s_suppkey", "s_comment"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q16(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const& part = tables.at("part");

  auto const excluded_brand = cudf::string_scalar{"Brand#45"};
  auto brand_mask           = cudf::binary_operation(part->column("p_brand"),
                                           excluded_brand,
                                           cudf::binary_operator::NOT_EQUAL,
                                           cudf::data_type{cudf::type_id::BOOL8});

  auto const excluded_type = cudf::string_scalar{"MEDIUM POLISHED"};
  auto excluded_type_mask =
    cudf::strings::starts_with(cudf::strings_column_view{part->column("p_type")}, excluded_type);
  auto type_mask = cudf::unary_operation(excluded_type_mask->view(), cudf::unary_operator::NOT);

  std::unique_ptr<cudf::column> size_mask;
  for (auto const size : {int8_t{49},
                          int8_t{14},
                          int8_t{23},
                          int8_t{45},
                          int8_t{19},
                          int8_t{3},
                          int8_t{36},
                          int8_t{9}}) {
    auto const value = cudf::numeric_scalar<int8_t>{size};
    auto match       = cudf::binary_operation(part->column("p_size"),
                                        value,
                                        cudf::binary_operator::EQUAL,
                                        cudf::data_type{cudf::type_id::BOOL8});
    if (size_mask) {
      size_mask = cudf::binary_operation(size_mask->view(),
                                         match->view(),
                                         cudf::binary_operator::LOGICAL_OR,
                                         cudf::data_type{cudf::type_id::BOOL8});
    } else {
      size_mask = std::move(match);
    }
  }

  auto part_mask           = cudf::binary_operation(brand_mask->view(),
                                          type_mask->view(),
                                          cudf::binary_operator::LOGICAL_AND,
                                          cudf::data_type{cudf::type_id::BOOL8});
  part_mask                = cudf::binary_operation(part_mask->view(),
                                     size_mask->view(),
                                     cudf::binary_operator::LOGICAL_AND,
                                     cudf::data_type{cudf::type_id::BOOL8});
  auto const filtered_part = apply_mask(part, part_mask);

  auto const program  = cudf::strings::regex_program::create("Customer.*Complaints");
  auto complaint_mask = cudf::strings::contains_re(
    cudf::strings_column_view{tables.at("supplier")->column("s_comment")}, *program);
  auto const complaint_suppliers = apply_mask(tables.at("supplier"), complaint_mask);

  auto const joined =
    apply_inner_join(filtered_part, tables.at("partsupp"), {"p_partkey"}, {"ps_partkey"});
  auto const eligible =
    apply_left_anti_join(joined, complaint_suppliers, {"ps_suppkey"}, {"s_suppkey"});
  auto const grouped = apply_groupby(
    eligible,
    groupby_context_t{{"p_brand", "p_type", "p_size"},
                      {{"ps_suppkey", {{cudf::aggregation::Kind::NUNIQUE, "supplier_cnt"}}}}});
  return apply_orderby(grouped,
                       {"supplier_cnt", "p_brand", "p_type", "p_size"},
                       {cudf::order::DESCENDING,
                        cudf::order::ASCENDING,
                        cudf::order::ASCENDING,
                        cudf::order::ASCENDING});
}

void ndsh_q16(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"part", "partsupp", "supplier"}, sources);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q16(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q16(sources);
      result     = execute_ndsh_q16(input);
    } else {
      result = execute_ndsh_q16(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  if (not write_ndsh_results()) { return; }
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q16(sources);
    result     = execute_ndsh_q16(input);
  } else {
    result = execute_ndsh_q16(tables);
  }
  write_ndsh_result(*result, "q16");
}

NVBENCH_BENCH(ndsh_q16)
  .set_name("ndsh_q16")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
