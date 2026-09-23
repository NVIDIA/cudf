/*
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet/parquet_io.hpp"
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
#include <map>
#include <utility>

#ifdef CUDF_WITH_VORTEX
#include "local_io.hpp"
#include "reference/q9_reference.hpp"
#include "vortex/vortex_io.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#endif

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

namespace {
std::vector<std::string> const q9_tables{
  "part", "supplier", "lineitem", "partsupp", "orders", "nation"};
std::map<std::string, std::vector<std::string>> const q9_projections{
  {"lineitem",
   {"l_suppkey", "l_partkey", "l_orderkey", "l_extendedprice", "l_discount", "l_quantity"}},
  {"nation", {"n_nationkey", "n_name"}},
  {"orders", {"o_orderkey", "o_orderdate"}},
  {"part", {"p_partkey", "p_name"}},
  {"partsupp", {"ps_suppkey", "ps_partkey", "ps_supplycost"}},
  {"supplier", {"s_suppkey", "s_nationkey"}}};
}  // namespace

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

template <typename Read>
q9_data load_data(Read&& read)
{
  return q9_data{read("lineitem", q9_projections.at("lineitem")),
                 read("nation", q9_projections.at("nation")),
                 read("orders", q9_projections.at("orders")),
                 read("part", q9_projections.at("part")),
                 read("partsupp", q9_projections.at("partsupp")),
                 read("supplier", q9_projections.at("supplier"))};
}

q9_data load_data(std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  return load_data([&](std::string const& name, std::vector<std::string> const& columns) {
    return read_parquet(sources.at(name).make_source_info(), columns);
  });
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
  generate_parquet_data_sources(scale_factor, q9_tables, sources);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    q9_data const data = load_data(sources);
    auto const result  = compute_profit(
      engine, data, launch.get_stream().get_stream(), cudf::get_current_device_resource_ref());
    write_parquet(*result, "q9.parquet");
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void ndsh_q9_noio(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const engine       = engine_from_string(state.get_string("engine"));

  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, q9_tables, sources);

  q9_data const data = load_data(sources);

  std::unique_ptr<table_with_names> result;

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    result = compute_profit(
      engine, data, launch.get_stream().get_stream(), cudf::get_current_device_resource_ref());
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  if (result) { write_parquet(*result, "q9_noio.parquet"); }
}

// unlike `ndsh_q9`, `ndsh_q9_amount` benchmarks only the amount calculation part of the benchmark
void ndsh_q9_amount(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const engine       = engine_from_string(state.get_string("engine"));

  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, q9_tables, sources);

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

#ifdef CUDF_WITH_VORTEX
namespace {

template <typename Read>
std::unique_ptr<table_with_names> execute_q9(
  engine_type engine,
  Read&& read,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const data = load_data(std::forward<Read>(read));
  return compute_profit(engine, data, stream, mr);
}

void check_q9_cases()
{
  cudf::test::fixed_width_column_wrapper<int8_t> nation_key{{10, 20}};
  cudf::test::strings_column_wrapper nation_name{"ALPHA", "ZULU"};
  cudf::test::fixed_width_column_wrapper<int32_t> supplier_key{{101, 102, 103}};
  cudf::test::fixed_width_column_wrapper<int8_t> supplier_nation{{10, 20, 99}};
  // The extra rows duplicate a matching key, a supplier with no nation, and a missing supplier.
  cudf::test::fixed_width_column_wrapper<int32_t> partsupp_supplier{
    {101, 102, 102, 101, 103, 101, 103, 999, 999}};
  cudf::test::fixed_width_column_wrapper<int32_t> partsupp_part{{1, 3, 2, 2, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> supply_cost{{10, 5, 20, 30, 40, 20, 50, 60, 70}};
  cudf::test::fixed_width_column_wrapper<int32_t> order_key{{1, 2, 3, 4}};
  // Epoch days: 1994-01-01, 1995-01-01, 1994-12-31, 1996-01-01.
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> order_date{
    {8766, 9131, 9130, 9496}};
  cudf::test::fixed_width_column_wrapper<int32_t> part_key{{1, 2, 3, 4}};
  cudf::test::strings_column_wrapper part_name{"forest green", "GREEN", "lightgreen", "blue"};
  cudf::test::fixed_width_column_wrapper<int32_t> line_supplier{
    {101, 101, 102, 101, 102, 101, 101, 999, 101, 103, 102, 101}};
  cudf::test::fixed_width_column_wrapper<int32_t> line_part{{1, 1, 3, 1, 2, 2, 3, 1, 1, 1, 1, 4}};
  cudf::test::fixed_width_column_wrapper<int32_t> line_order{
    {1, 1, 2, 4, 2, 3, 3, 1, 999, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> price{
    {100, 50, 200, 100, 500, 500, 500, 500, 500, 500, 500, 500}};
  cudf::test::fixed_width_column_wrapper<double> discount{
    {0.1, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  cudf::test::fixed_width_column_wrapper<int8_t> quantity{{2, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  CUDF_CUDA_TRY(cudaDeviceSynchronize());

  std::map<std::string, cudf::table_view> const input{
    {"nation", cudf::table_view{{nation_key, nation_name}}},
    {"supplier", cudf::table_view{{supplier_key, supplier_nation}}},
    {"partsupp", cudf::table_view{{partsupp_supplier, partsupp_part, supply_cost}}},
    {"orders", cudf::table_view{{order_key, order_date}}},
    {"part", cudf::table_view{{part_key, part_name}}},
    {"lineitem",
     cudf::table_view{{line_supplier, line_part, line_order, price, discount, quantity}}}};
  ndsh::q9_reference_result expected;
  expected.sum_profit[{"ALPHA", 1996}] = 90.0;
  expected.sum_profit[{"ALPHA", 1994}] = 110.0;
  expected.sum_profit[{"ZULU", 1995}]  = 130.0;
  expected.matched                     = 4;
  ndsh::q9_reference_result duplicate_expected;
  duplicate_expected.sum_profit[{"ALPHA", 1996}] = 170.0;
  duplicate_expected.sum_profit[{"ALPHA", 1994}] = 190.0;
  duplicate_expected.sum_profit[{"ZULU", 1995}]  = 130.0;
  duplicate_expected.matched                     = 7;
  cuda::stream_ref const stream                  = cudf::get_default_stream();
  for (auto const& partsupp : cudf::slice(input.at("partsupp"), {0, 5, 0, 9}, stream)) {
    // Full input, only rows rejected by filters/joins, and no rows, with and without duplicates.
    for (auto const& lines : cudf::slice(input.at("lineitem"), {0, 12, 4, 12, 0, 0}, stream)) {
      auto const want       = lines.num_rows() == 12
                                ? (partsupp.num_rows() == 5 ? expected : duplicate_expected)
                                : ndsh::q9_reference_result{};
      auto tables           = input;
      tables.at("partsupp") = partsupp;
      tables.at("lineitem") = lines;
      ndsh::q9_reference_builder builder;
      for (auto const& name : {"nation", "supplier", "partsupp", "orders", "part", "lineitem"}) {
        builder.add_table(name, tables.at(name), stream);
      }
      auto const cpu = builder.finish();
      CUDF_EXPECTS(cpu.matched == want.matched && cpu.sum_profit.size() == want.sum_profit.size(),
                   "Q9 CPU reference row/group regression");
      for (auto const& [key, value] : want.sum_profit) {
        auto const actual = cpu.sum_profit.find(key);
        CUDF_EXPECTS(
          actual != cpu.sum_profit.end() && ndsh::detail::reference_equal(actual->second, value),
          "Q9 CPU reference amount/year regression");
      }
      for (auto const engine : {engine_type::BINARYOP, engine_type::AST, engine_type::TRANSFORM}) {
        auto result =
          execute_q9(engine, [&](std::string const& name, std::vector<std::string> const& columns) {
            CUDF_EXPECTS(columns == q9_projections.at(name), "Q9 projection mismatch");
            return std::make_unique<table_with_names>(
              std::make_unique<cudf::table>(tables.at(name)), columns);
          });
        ndsh::check_q9_result(want, *result, stream);
      }
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

struct q9_files {
  ndsh::local_table_files tables;
  ndsh::q9_reference_result reference;

  explicit q9_files(double scale_factor)
  {
    check_q9_cases();
    ndsh::make_reference_files<ndsh::q9_reference_builder>(
      scale_factor,
      tables,
      reference,
      q9_tables,
      q9_projections,
      true,
      [&](auto&& read, cuda::stream_ref stream) {
        for (auto const engine :
             {engine_type::BINARYOP, engine_type::AST, engine_type::TRANSFORM}) {
          auto result = execute_q9(engine, read);
          ndsh::check_q9_result(reference, *result, stream);
        }
      });
  }
};

void ndsh_q9_local(nvbench::state& state)
{
  auto const engine = engine_from_string(state.get_string("engine"));
  auto const [use_vortex, read_only, cold, direct_io] = ndsh::local_options{state, 9};
  if (direct_io && !use_vortex) {
    state.skip("io=direct is supported only for Vortex");
    return;
  }
  auto const& files             = ndsh::local_fixture<q9_files>(state.get_float64("scale_factor"));
  cuda::stream_ref const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto memory = cudf::memory_stats_logger();
  ndsh::vortex_io io{stream.get()};
  auto read = [&](
                std::string const& name, std::vector<std::string> const& columns, auto const&...) {
    return ndsh::read_local_file(
      files.tables.path(name, use_vortex), use_vortex, io, columns, direct_io);
  };
  auto load_inputs = [&](bool verify = false) {
    if (!use_vortex) {
      return load_data([&](std::string const& name, std::vector<std::string> const& columns) {
        auto input = read(name, columns);
        if (verify) {
          ndsh::check_local_projection(
            files.tables.path(name, use_vortex), use_vortex, io, columns, *input);
        }
        return input;
      });
    }
    auto inputs = ndsh::read_local_tables(q9_tables, q9_projections, read, true);
    return load_data([&](std::string const& name, std::vector<std::string> const& columns) {
      auto const index =
        std::distance(q9_tables.begin(), std::find(q9_tables.begin(), q9_tables.end(), name));
      if (verify) {
        ndsh::check_local_projection(files.tables.path(name, use_vortex),
                                     use_vortex,
                                     io,
                                     columns,
                                     *inputs.at(index),
                                     direct_io);
      }
      return std::move(inputs.at(index));
    });
  };
  auto query = [&] {
    auto inputs = load_inputs();
    return compute_profit(engine, inputs, stream, cudf::get_current_device_resource_ref());
  };
  {
    auto inputs = load_inputs(true);
    auto result = compute_profit(engine, inputs, stream, cudf::get_current_device_resource_ref());
    ndsh::check_q9_result(files.reference, *result, stream);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  ndsh::warm_local_inputs(cold, [&] { return load_inputs(); });
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  memory.reset_counters();
  ndsh::exec_local_benchmark(state, files.tables, use_vortex, cold, [&] {
    if (read_only) {
      auto inputs = load_inputs();
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    } else {
      auto result = query();
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    }
  });
  state.add_buffer_size(files.tables.bytes(use_vortex), "file_size", "Total file size");
  state.add_buffer_size(memory.peak_memory_usage(), "rmm_peak", "RMM peak (excludes Vortex)");
  ndsh::add_count(state, "ndsh/q9/matched_rows", "Q9 matched rows", files.reference.matched);
  ndsh::add_count(
    state, "ndsh/q9/groups", "Q9 nation/year groups", files.reference.sum_profit.size());
}

}  // namespace

// NVBench varies the first axis fastest; keep scale last to reuse the one-scale fixture cache.
NVBENCH_BENCH(ndsh_q9_local)
  .set_name("ndsh_q9_local")
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q9"})
  .add_string_axis("engine", {"binaryop", "ast", "transform"})
  .add_string_axis("cache", {"warm", "cold"})
  .add_string_axis("io", {"buffered"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10});
#endif
