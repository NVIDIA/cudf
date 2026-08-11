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
#include <cudf/transform.hpp>

#include <nvbench/nvbench.cuh>

#include <array>
#include <memory>
#include <string_view>

/**
 * @file q19.cpp
 * @brief Implement query 19 of the NDS-H benchmark.
 */

std::unordered_map<std::string, std::unique_ptr<table_with_names>> load_ndsh_q19(
  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  tables.emplace("lineitem",
                 read_parquet(sources.at("lineitem").make_source_info(),
                              {"l_partkey",
                               "l_shipmode",
                               "l_shipinstruct",
                               "l_quantity",
                               "l_extendedprice",
                               "l_discount"}));
  tables.emplace("part",
                 read_parquet(sources.at("part").make_source_info(),
                              {"p_partkey", "p_brand", "p_container", "p_size"}));
  return tables;
}

std::unique_ptr<table_with_names> execute_ndsh_q19(
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> const& tables)
{
  auto const& lineitem = tables.at("lineitem");
  auto const& part     = tables.at("part");

  auto const shipmode_ref     = cudf::ast::column_reference(lineitem->column_id("l_shipmode"));
  auto const shipinstruct_ref = cudf::ast::column_reference(lineitem->column_id("l_shipinstruct"));
  auto air                    = cudf::string_scalar("AIR");
  auto air_reg                = cudf::string_scalar("AIR REG");
  auto deliver_in_person      = cudf::string_scalar("DELIVER IN PERSON");
  auto const air_literal      = cudf::ast::literal(air);
  auto const air_reg_literal  = cudf::ast::literal(air_reg);
  auto const instruct_literal = cudf::ast::literal(deliver_in_person);
  auto const air_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, shipmode_ref, air_literal);
  auto const air_reg_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, shipmode_ref, air_reg_literal);
  auto const shipmode_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, air_predicate, air_reg_predicate);
  auto const instruct_predicate =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, shipinstruct_ref, instruct_literal);
  auto const shipping_predicate = cudf::ast::operation(
    cudf::ast::ast_operator::LOGICAL_AND, shipmode_predicate, instruct_predicate);
  auto filtered_lineitem = apply_filter(lineitem, shipping_predicate);

  auto joined          = apply_inner_join(part, filtered_lineitem, {"p_partkey"}, {"l_partkey"});
  auto const brand_ref = cudf::ast::column_reference(joined->column_id("p_brand"));
  auto const container_ref = cudf::ast::column_reference(joined->column_id("p_container"));
  auto const quantity_ref  = cudf::ast::column_reference(joined->column_id("l_quantity"));
  auto const size_ref      = cudf::ast::column_reference(joined->column_id("p_size"));

  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::string_scalar>> strings;
  std::vector<std::unique_ptr<cudf::numeric_scalar<int8_t>>> numbers;
  auto equals = [&](cudf::ast::column_reference const& column,
                    std::string_view value) -> cudf::ast::expression const& {
    strings.push_back(std::make_unique<cudf::string_scalar>(value));
    auto& literal = tree.push(cudf::ast::literal(*strings.back()));
    return tree.push(cudf::ast::operation(cudf::ast::ast_operator::EQUAL, column, literal));
  };
  auto between = [&](cudf::ast::column_reference const& column,
                     int8_t lower,
                     int8_t upper) -> cudf::ast::expression const& {
    numbers.push_back(std::make_unique<cudf::numeric_scalar<int8_t>>(lower));
    auto& lower_literal = tree.push(cudf::ast::literal(*numbers.back()));
    numbers.push_back(std::make_unique<cudf::numeric_scalar<int8_t>>(upper));
    auto& upper_literal   = tree.push(cudf::ast::literal(*numbers.back()));
    auto& lower_predicate = tree.push(
      cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, column, lower_literal));
    auto& upper_predicate =
      tree.push(cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, column, upper_literal));
    return tree.push(
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, lower_predicate, upper_predicate));
  };
  auto branch = [&](std::string_view brand,
                    std::array<std::string_view, 4> const& containers,
                    int8_t quantity_lower,
                    int8_t quantity_upper,
                    int8_t size_upper) -> cudf::ast::expression const& {
    auto& brand_predicate = equals(brand_ref, brand);
    auto& container0      = equals(container_ref, containers[0]);
    auto& container1      = equals(container_ref, containers[1]);
    auto& container2      = equals(container_ref, containers[2]);
    auto& container3      = equals(container_ref, containers[3]);
    auto& containers01 =
      tree.push(cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, container0, container1));
    auto& containers23 =
      tree.push(cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, container2, container3));
    auto& container_predicate = tree.push(
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, containers01, containers23));
    auto& quantity_predicate = between(quantity_ref, quantity_lower, quantity_upper);
    auto& size_predicate     = between(size_ref, int8_t{1}, size_upper);
    auto& brand_container    = tree.push(cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, brand_predicate, container_predicate));
    auto& quantity_size      = tree.push(cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, quantity_predicate, size_predicate));
    return tree.push(
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, brand_container, quantity_size));
  };

  auto& branch1 = branch("Brand#12", {"SM CASE", "SM BOX", "SM PACK", "SM PKG"}, 1, 11, 5);
  auto& branch2 = branch("Brand#23", {"MED BAG", "MED BOX", "MED PKG", "MED PACK"}, 10, 20, 10);
  auto& branch3 = branch("Brand#34", {"LG CASE", "LG BOX", "LG PACK", "LG PKG"}, 20, 30, 15);
  auto& branches12 =
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, branch1, branch2));
  auto& branches =
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, branches12, branch3));
  auto filtered = apply_filter(joined, branches);

  auto const one          = cudf::numeric_scalar<double>(1);
  auto one_minus_discount = cudf::binary_operation(one,
                                                   filtered->column("l_discount"),
                                                   cudf::binary_operator::SUB,
                                                   cudf::data_type{cudf::type_id::FLOAT64});
  auto revenue            = cudf::binary_operation(filtered->column("l_extendedprice"),
                                        one_minus_discount->view(),
                                        cudf::binary_operator::MUL,
                                        cudf::data_type{cudf::type_id::FLOAT64});
  auto reduced = apply_reduction(revenue->view(), cudf::aggregation::Kind::SUM, "revenue");
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  auto rounded = cudf::round(reduced->column("revenue"), 2, cudf::rounding_method::HALF_UP);
#pragma GCC diagnostic pop
  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.push_back(std::move(rounded));
  return std::make_unique<table_with_names>(
    std::make_unique<cudf::table>(std::move(result_columns)), std::vector<std::string>{"revenue"});
}

void ndsh_q19(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const mode         = query_mode_from_string(state.get_string("mode"));
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"lineitem", "part"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  std::unordered_map<std::string, std::unique_ptr<table_with_names>> tables;
  if (mode == query_mode::COMPUTE_ONLY) { tables = load_ndsh_q19(sources); }
  std::unique_ptr<table_with_names> result;
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (mode == query_mode::END_TO_END) {
      auto input = load_ndsh_q19(sources);
      result     = execute_ndsh_q19(input);
    } else {
      result = execute_ndsh_q19(tables);
    }
  });
  result->to_parquet("q19.parquet");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q19)
  .set_name("ndsh_q19")
  .add_string_axis("mode", {"end_to_end", "compute_only"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});
