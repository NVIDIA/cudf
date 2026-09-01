/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#include <array>
#include <utility>

enum class engine_type : int32_t { BINARYOP = 0, AST = 1, TRANSFORM = 2 };

engine_type engine_from_string(std::string const& str)
{
  if (str == "binaryop") {
    return engine_type::BINARYOP;
  } else if (str == "ast") {
    return engine_type::AST;
  } else if (str == "transform") {
    return engine_type::TRANSFORM;
  } else {
    CUDF_FAIL("unrecognized engine enum: " + str);
  }
}

struct q9_data {
  std::unique_ptr<table_with_names> lineitem;
  std::unique_ptr<table_with_names> nation;
  std::unique_ptr<table_with_names> orders;
  std::unique_ptr<table_with_names> part;
  std::unique_ptr<table_with_names> partsupp;
  std::unique_ptr<table_with_names> supplier;
};

/**
 * @file q09.cpp
 * @brief Implement query 9 of the NDS-H benchmark.
 *
 * create view part as select * from '/tables/scale-1/part.parquet';
 * create view supplier as select * from '/tables/scale-1/supplier.parquet';
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 * create view partsupp as select * from '/tables/scale-1/partsupp.parquet';
 * create view orders as select * from '/tables/scale-1/orders.parquet';
 * create view nation as select * from '/tables/scale-1/nation.parquet';
 *
 * select
 *    nation,
 *    o_year,
 *    sum(amount) as sum_profit
 * from
 *     (
 *        select
 *            n_name as nation,
 *            extract(year from o_orderdate) as o_year,
 *            l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
 *        from
 *            part,
 *            supplier,
 *            lineitem,
 *            partsupp,
 *            orders,
 *            nation
 *        where
 *           s_suppkey = l_suppkey
 *           and ps_suppkey = l_suppkey
 *           and ps_partkey = l_partkey
 *           and p_partkey = l_partkey
 *           and o_orderkey = l_orderkey
 *           and s_nationkey = n_nationkey
 *           and p_name like '%green%'
 *     ) as profit
 * group by
 *     nation,
 *     o_year
 * order by
 *     nation,
 *     o_year desc;
 */

/**
 * @brief Calculate the amount column
 *
 * @param discount The discount column
 * @param extendedprice The extended price column
 * @param supplycost The supply cost column
 * @param quantity The quantity column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> compute_amount_binaryop(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  cudf::column_view const& supplycost,
  cudf::column_view const& quantity,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_BENCHMARK_RANGE();

  auto const one = cudf::numeric_scalar<double>(1);
  auto const one_minus_discount =
    cudf::binary_operation(one, discount, cudf::binary_operator::SUB, discount.type(), stream, mr);
  auto const extendedprice_discounted_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto const extendedprice_discounted      = cudf::binary_operation(extendedprice,
                                                               one_minus_discount->view(),
                                                               cudf::binary_operator::MUL,
                                                               extendedprice_discounted_type,
                                                               stream,
                                                               mr);
  auto const supplycost_quantity_type      = cudf::data_type{cudf::type_id::FLOAT64};
  auto const supplycost_quantity           = cudf::binary_operation(
    supplycost, quantity, cudf::binary_operator::MUL, supplycost_quantity_type, stream, mr);
  auto amount = cudf::binary_operation(extendedprice_discounted->view(),
                                       supplycost_quantity->view(),
                                       cudf::binary_operator::SUB,
                                       extendedprice_discounted->type(),
                                       stream,
                                       mr);
  return amount;
}

[[nodiscard]] std::unique_ptr<cudf::column> compute_amount_transform(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  cudf::column_view const& supplycost,
  cudf::column_view const& quantity,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_BENCHMARK_RANGE();

  std::string udf =
    R"***(
  void calculate_price(double * amount, double discount, double extended_price, double supply_cost, double quantity){
    *amount = extended_price * (1 - discount) - supply_cost * quantity;
  }
  )***";

  cudf::transform_input transform_inputs[] = {discount, extendedprice, supplycost, quantity};

  return std::move(
    cudf::transform(udf,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    transform_inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT64},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt,
                    stream,
                    mr)
      ->release()
      .front());
}

[[nodiscard]] std::unique_ptr<cudf::column> compute_amount_ast(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  cudf::column_view const& supplycost,
  cudf::column_view const& quantity,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  CUDF_BENCHMARK_RANGE();

  cudf::ast::tree tree;
  cudf::table_view table{std::vector{discount, extendedprice, supplycost, quantity}};

  auto& discount_ref       = tree.push(cudf::ast::column_reference{0});
  auto& extended_price_ref = tree.push(cudf::ast::column_reference{1});
  auto& supplycost_ref     = tree.push(cudf::ast::column_reference{2});
  auto& quantity_ref       = tree.push(cudf::ast::column_reference{3});

  auto& extended_price_mul_discount =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::MUL, extended_price_ref, discount_ref});

  // AST presently doesn't support literals on LHS, so we expand extended_price * (1 - discount)
  auto& extended_price_discounted = tree.push(cudf::ast::operation{
    cudf::ast::ast_operator::SUB, extended_price_ref, extended_price_mul_discount});

  auto& quantity_float64 =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::CAST_TO_FLOAT64, quantity_ref});

  auto& supply_cost_mul_quantity =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::MUL, supplycost_ref, quantity_float64});
  auto& result = tree.push(cudf::ast::operation{
    cudf::ast::ast_operator::SUB, extended_price_discounted, supply_cost_mul_quantity});

  return cudf::compute_column(table, result, stream, mr);
}

[[nodiscard]] std::unique_ptr<cudf::column> compute_amount(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  cudf::column_view const& supplycost,
  cudf::column_view const& quantity,
  engine_type engine,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  switch (engine) {
    case engine_type::BINARYOP:
      return compute_amount_binaryop(discount, extendedprice, supplycost, quantity, stream, mr);
    case engine_type::AST:
      return compute_amount_ast(discount, extendedprice, supplycost, quantity, stream, mr);
    case engine_type::TRANSFORM:
      return compute_amount_transform(discount, extendedprice, supplycost, quantity, stream, mr);
    default: CUDF_UNREACHABLE("invalid engine_type enum");
  }
}

q9_data load_data(std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  auto lineitem = read_parquet(
    sources.at("lineitem").make_source_info(),
    {"l_suppkey", "l_partkey", "l_orderkey", "l_extendedprice", "l_discount", "l_quantity"});
  auto nation = read_parquet(sources.at("nation").make_source_info(), {"n_nationkey", "n_name"});
  auto orders =
    read_parquet(sources.at("orders").make_source_info(), {"o_orderkey", "o_orderdate"});
  auto part     = read_parquet(sources.at("part").make_source_info(), {"p_partkey", "p_name"});
  auto partsupp = read_parquet(sources.at("partsupp").make_source_info(),
                               {"ps_suppkey", "ps_partkey", "ps_supplycost"});
  auto supplier =
    read_parquet(sources.at("supplier").make_source_info(), {"s_suppkey", "s_nationkey"});
  return q9_data{std::move(lineitem),
                 std::move(nation),
                 std::move(orders),
                 std::move(part),
                 std::move(partsupp),
                 std::move(supplier)};
}

std::unique_ptr<table_with_names> join_data(q9_data const& data)
{
  CUDF_BENCHMARK_RANGE();

  // Generating the `profit` table
  // Filter the part table using `p_name like '%green%'`
  auto const p_name        = data.part->table().column(1);
  auto const mask          = cudf::strings::like(cudf::strings_column_view(p_name), "%green%");
  auto const part_filtered = apply_mask(data.part, mask);

  // Perform the joins
  auto const join_a =
    apply_inner_join(data.supplier, data.nation, {"s_nationkey"}, {"n_nationkey"});
  auto const join_b = apply_inner_join(data.partsupp, join_a, {"ps_suppkey"}, {"s_suppkey"});
  auto const join_c = apply_inner_join(data.lineitem, part_filtered, {"l_partkey"}, {"p_partkey"});
  auto const join_d = apply_inner_join(data.orders, join_c, {"o_orderkey"}, {"l_orderkey"});
  return apply_inner_join(join_d, join_b, {"l_suppkey", "l_partkey"}, {"s_suppkey", "ps_partkey"});
}

std::unique_ptr<table_with_names> compute_profit(
  nvbench::state& state,
  engine_type engine,
  q9_data const& data,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto joined_table = join_data(data);
  // Calculate the `nation`, `o_year`, and `amount` columns
  auto n_name = std::make_unique<cudf::column>(joined_table->column("n_name"));
  auto o_year = cudf::datetime::extract_datetime_component(
    joined_table->column("o_orderdate"), cudf::datetime::datetime_component::YEAR);

  auto amount = compute_amount(joined_table->column("l_discount"),
                               joined_table->column("l_extendedprice"),
                               joined_table->column("ps_supplycost"),
                               joined_table->column("l_quantity"),
                               engine,
                               stream,
                               mr);

  // Put together the `profit` table
  std::vector<std::unique_ptr<cudf::column>> profit_columns;
  profit_columns.push_back(std::move(n_name));
  profit_columns.push_back(std::move(o_year));
  profit_columns.push_back(std::move(amount));

  auto profit_table = std::make_unique<cudf::table>(std::move(profit_columns));
  auto const profit = std::make_unique<table_with_names>(
    std::move(profit_table), std::vector<std::string>{"nation", "o_year", "amount"});

  // Perform the groupby operation
  auto const groupedby_table = apply_groupby(
    profit,
    groupby_context_t{{"nation", "o_year"},
                      {{"amount", {{cudf::groupby_aggregation::SUM, "sum_profit"}}}}});

  // Perform the orderby operation
  return apply_orderby(
    groupedby_table, {"nation", "o_year"}, {cudf::order::ASCENDING, cudf::order::DESCENDING});
}

void ndsh_q9(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const engine       = engine_from_string(state.get_string("engine"));

  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"part", "supplier", "lineitem", "partsupp", "orders", "nation"}, sources);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    q9_data const data = load_data(sources);
    auto const result  = compute_profit(state,
                                       engine,
                                       data,
                                       launch.get_stream().get_stream(),
                                       cudf::get_current_device_resource_ref());
    result->to_parquet("q9.parquet");
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void ndsh_q9_noio(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const engine       = engine_from_string(state.get_string("engine"));

  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"part", "supplier", "lineitem", "partsupp", "orders", "nation"}, sources);

  q9_data const data = load_data(sources);

  std::unique_ptr<table_with_names> result;

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    result = compute_profit(state,
                            engine,
                            data,
                            launch.get_stream().get_stream(),
                            cudf::get_current_device_resource_ref());
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  if (result) { result->to_parquet("q9_noio.parquet"); }
}

// unlike `ndsh_q9`, `ndsh_q9_amount` benchmarks only the amount calculation part of the benchmark
void ndsh_q9_amount(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const engine       = engine_from_string(state.get_string("engine"));

  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"part", "supplier", "lineitem", "partsupp", "orders", "nation"}, sources);

  q9_data const data      = load_data(sources);
  auto const joined_table = join_data(data);

  auto const size = joined_table->column("l_extendedprice").size();

  state.add_global_memory_reads<double>(size * 4);
  state.add_global_memory_writes<double>(size);
  state.add_element_count(size);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto amount = compute_amount(joined_table->column("l_discount"),
                                 joined_table->column("l_extendedprice"),
                                 joined_table->column("ps_supplycost"),
                                 joined_table->column("l_quantity"),
                                 engine,
                                 launch.get_stream().get_stream(),
                                 cudf::get_current_device_resource_ref());
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q9)
  .set_name("ndsh_q9")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1})
  .add_string_axis("engine", {"binaryop", "ast", "transform"});

NVBENCH_BENCH(ndsh_q9_noio)
  .set_name("ndsh_q9_noio")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1})
  .add_string_axis("engine", {"binaryop", "ast", "transform"});

NVBENCH_BENCH(ndsh_q9_amount)
  .set_name("ndsh_q9_amount")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1})
  .add_string_axis("engine", {"binaryop", "ast", "transform"});
