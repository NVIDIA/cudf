/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvbench/nvbench.cuh>

#include <optional>

/**
 * @file q22.cpp
 * @brief Implement query 22 of the NDS-H benchmark.
 *
 * select
 *     cntrycode,
 *     count(*) as numcust,
 *     sum(c_acctbal) as totacctbal
 * from (
 *     select
 *         substring(c_phone from 1 for 2) as cntrycode,
 *         c_acctbal
 *     from
 *         customer
 *     where
 *         substring(c_phone from 1 for 2) in
 *             (13, 31, 23, 29, 30, 18, 17)
 *         and c_acctbal > (
 *             select
 *                 avg(c_acctbal)
 *             from
 *                 customer
 *             where
 *                 c_acctbal > 0.00
 *                 and substring (c_phone from 1 for 2) in
 *                     (13, 31, 23, 29, 30, 18, 17)
 *         )
 *         and not exists (
 *             select
 *                 *
 *             from
 *                 orders
 *             where
 *                 o_custkey = c_custkey
 *         )
 *     ) as custsale
 * group by
 *     cntrycode
 * order by
 *     cntrycode;
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q22(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace(
    "customer",
    read_parquet(sources.at("customer").make_source_info(), {"c_custkey", "c_phone", "c_acctbal"}));
  tables.emplace("orders", read_parquet(sources.at("orders").make_source_info(), {"o_custkey"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q22(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto customer = apply_projection(tables.at("customer"), {"c_custkey", "c_phone", "c_acctbal"});
  auto country_code =
    cudf::strings::slice_strings(cudf::strings_column_view{customer->column("c_phone")},
                                 std::optional<cudf::size_type>{0},
                                 std::optional<cudf::size_type>{2});
  customer->append(country_code, "cntrycode");

  auto const country_pattern = cudf::strings::regex_program::create("^(13|31|23|29|30|18|17)$");
  auto country_mask          = cudf::strings::contains_re(
    cudf::strings_column_view{customer->column("cntrycode")}, *country_pattern);
  auto selected = apply_mask(customer, country_mask);
  selected      = apply_projection(selected, {"c_acctbal", "c_custkey", "cntrycode"});

  auto const zero     = cudf::numeric_scalar<double>{0.0};
  auto positive_mask  = cudf::binary_operation(selected->column("c_acctbal"),
                                              zero,
                                              cudf::binary_operator::GREATER,
                                              cudf::data_type{cudf::type_id::BOOL8});
  auto const positive = apply_mask(selected, positive_mask);
  auto const mean     = cudf::make_mean_aggregation<cudf::reduce_aggregation>();
  auto average =
    cudf::reduce(positive->column("c_acctbal"), *mean, cudf::data_type{cudf::type_id::FLOAT64});

  auto const without_orders =
    apply_left_anti_join(selected, tables.at("orders"), {"c_custkey"}, {"o_custkey"});
  auto above_average_mask  = cudf::binary_operation(without_orders->column("c_acctbal"),
                                                   *average,
                                                   cudf::binary_operator::GREATER,
                                                   cudf::data_type{cudf::type_id::BOOL8});
  auto const above_average = apply_mask(without_orders, above_average_mask);

  auto const grouped =
    apply_groupby(above_average,
                  groupby_context_t{{"cntrycode"},
                                    {{"c_acctbal",
                                      {{cudf::aggregation::Kind::COUNT_ALL, "numcust"},
                                       {cudf::aggregation::Kind::SUM, "totacctbal"}}}}});
  return apply_orderby(grouped, {"cntrycode"}, {cudf::order::ASCENDING});
}

void ndsh_q22(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"customer", "orders"}, sources);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q22(sources); }
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::unique_ptr<table_with_names> result;
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q22(sources);
      result     = execute_ndsh_q22(input);
    } else {
      result = execute_ndsh_q22(tables);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  std::unique_ptr<table_with_names> result;
  if (mode == query_mode::END_TO_END) {
    auto input = load_ndsh_q22(sources);
    result     = execute_ndsh_q22(input);
  } else {
    result = execute_ndsh_q22(tables);
  }
  result->to_parquet("q22.parquet");
}

NVBENCH_BENCH(ndsh_q22)
  .set_name("ndsh_q22")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
