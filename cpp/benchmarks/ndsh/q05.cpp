/*
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet/parquet_io.hpp"
#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#include <map>

#ifdef CUDF_WITH_VORTEX
#include "local_io.hpp"
#include "parquet/parquet_fixture.hpp"
#include "reference/q5_reference.hpp"
#include "vortex/vortex_io.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>

#include <cstdint>
#endif

namespace {
std::map<std::string, std::vector<std::string>> const q5_projections{
  {"orders", {"o_custkey", "o_orderkey", "o_orderdate"}},
  {"customer", {"c_custkey", "c_nationkey"}},
  {"lineitem", {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"}},
  {"supplier", {"s_suppkey", "s_nationkey"}},
  {"nation", {"n_nationkey", "n_regionkey", "n_name"}},
  {"region", {"r_regionkey", "r_name"}}};
}  // namespace

/**
 * @file q05.cpp
 * @brief Implement query 5 of the NDS-H benchmark.
 *
 * create view customer as select * from '/tables/scale-1/customer.parquet';
 * create view orders as select * from '/tables/scale-1/orders.parquet';
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 * create view supplier as select * from '/tables/scale-1/supplier.parquet';
 * create view nation as select * from '/tables/scale-1/nation.parquet';
 * create view region as select * from '/tables/scale-1/region.parquet';
 *
 * select
 *    n_name,
 *    sum(l_extendedprice * (1 - l_discount)) as revenue
 * from
 *    customer,
 *    orders,
 *    lineitem,
 *    supplier,
 *    nation,
 *    region
 * where
 *     c_custkey = o_custkey
 *     and l_orderkey = o_orderkey
 *     and l_suppkey = s_suppkey
 *     and c_nationkey = s_nationkey
 *     and s_nationkey = n_nationkey
 *     and n_regionkey = r_regionkey
 *     and r_name = 'ASIA'
 *     and o_orderdate >= date '1994-01-01'
 *     and o_orderdate < date '1995-01-01'
 * group by
 *    n_name
 * order by
 *    revenue desc;
 */

/**
 * read returns owning projected tables, applying supplied predicates if filter_predicates is false.
 * Otherwise filtering happens here. consume receives the result owner by reference and may
 * move it out; its return value is forwarded. This helper adds no final stream synchronization.
 */
template <typename Read, typename Consume>
auto execute_q5(Read&& read, bool filter_predicates, Consume&& consume)
{
  // Define the column projection and filter predicate for the `orders` table
  auto const& orders_cols    = q5_projections.at("orders");
  auto const o_orderdate_ref = cudf::ast::column_reference(std::distance(
    orders_cols.begin(), std::find(orders_cols.begin(), orders_cols.end(), "o_orderdate")));
  auto o_orderdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
  auto const o_orderdate_lower_limit = cudf::ast::literal(o_orderdate_lower);
  auto const o_orderdate_pred_lower  = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, o_orderdate_ref, o_orderdate_lower_limit);
  auto o_orderdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
  auto const o_orderdate_upper_limit = cudf::ast::literal(o_orderdate_upper);
  auto const o_orderdate_pred_upper =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, o_orderdate_ref, o_orderdate_upper_limit);
  auto const orders_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, o_orderdate_pred_lower, o_orderdate_pred_upper);

  // Define the column projection and filter predicate for the `region` table
  auto const& region_cols   = q5_projections.at("region");
  auto const r_name_ref     = cudf::ast::column_reference(std::distance(
    region_cols.begin(), std::find(region_cols.begin(), region_cols.end(), "r_name")));
  auto r_name_value         = cudf::string_scalar("ASIA");
  auto const r_name_literal = cudf::ast::literal(r_name_value);
  auto const region_pred    = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, r_name_ref, r_name_literal);

  std::unique_ptr<cudf::ast::operation> const no_predicate;
  auto const customer = read("customer", q5_projections.at("customer"), no_predicate);
  auto orders         = read("orders", orders_cols, orders_pred);
  auto const lineitem = read("lineitem", q5_projections.at("lineitem"), no_predicate);
  auto const supplier = read("supplier", q5_projections.at("supplier"), no_predicate);
  auto const nation   = read("nation", q5_projections.at("nation"), no_predicate);
  auto region         = read("region", region_cols, region_pred);
  if (filter_predicates) {
    orders = apply_filter(orders, *orders_pred);
    region = apply_filter(region, *region_pred);
  }

  // Perform the joins
  auto const join_a = apply_inner_join(region, nation, {"r_regionkey"}, {"n_regionkey"});
  auto const join_b = apply_inner_join(join_a, customer, {"n_nationkey"}, {"c_nationkey"});
  auto const join_c = apply_inner_join(join_b, orders, {"c_custkey"}, {"o_custkey"});
  auto const join_d = apply_inner_join(join_c, lineitem, {"o_orderkey"}, {"l_orderkey"});
  auto joined_table =
    apply_inner_join(supplier, join_d, {"s_suppkey", "s_nationkey"}, {"l_suppkey", "n_nationkey"});

  // Calculate and append the `revenue` column
  auto revenue = calculate_discounted_revenue(joined_table->column("l_extendedprice"),
                                              joined_table->column("l_discount"));
  (*joined_table).append(revenue, "revenue");

  // Perform the groupby operation
  auto const groupedby_table =
    apply_groupby(joined_table,
                  groupby_context_t{{"n_name"},
                                    {
                                      {"revenue", {{cudf::aggregation::Kind::SUM, "revenue"}}},
                                    }});

  // Perform the order by operation
  auto orderedby_table = apply_orderby(groupedby_table, {"revenue"}, {cudf::order::DESCENDING});
  return consume(orderedby_table);
}

void run_ndsh_q5(nvbench::state& state,
                 std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  execute_q5(
    [&](std::string const& name,
        std::vector<std::string> const& columns,
        std::unique_ptr<cudf::ast::operation> const& predicate) {
      return read_parquet(sources.at(name).make_source_info(), columns, predicate);
    },
    false,
    [](auto const& result) { result->to_parquet("q5.parquet"); });
}

void ndsh_q5(nvbench::state& state)
{
  // Generate the required parquet files in device buffers
  double const scale_factor = state.get_float64("scale_factor");
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"customer", "orders", "lineitem", "supplier", "nation", "region"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { run_ndsh_q5(state, sources); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q5).set_name("ndsh_q5").add_float64_axis("scale_factor", {0.01, 0.1, 1});

#ifdef CUDF_WITH_VORTEX
namespace {

std::vector<std::string> const q5_tables{
  "customer", "orders", "lineitem", "supplier", "nation", "region"};

void check_q5_boundaries()
{
  cudf::test::fixed_width_column_wrapper<int32_t> customer_key{{1, 2, 3, 4, 5, 6}};
  cudf::test::fixed_width_column_wrapper<int8_t> customer_nation{{10, 20, 30, 40, 50, 99}};
  cudf::test::fixed_width_column_wrapper<int32_t> order_customer{
    {1, 1, 2, 1, 1, 3, 4, 999, 1, 1, 5, 6, 2, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> order_key{
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13}};
  // Epoch days: 1993-12-31, 1994-01-01, 1994-12-31, 1995-01-01.
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> order_date{
    {8766, 9130, 8766, 8765, 9131, 8766, 8766, 8766, 8766, 8766, 8766, 8766, 9130, 8766},
    {true, true, true, true, true, true, true, true, true, true, true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> line_order{
    {1, 1, 2, 3, 14, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 999}};
  cudf::test::fixed_width_column_wrapper<int32_t> line_supplier{
    {101, 101, 101, 102, 102, 101, 101, 103, 104, 101, 999, 102, 105, 106, 101, 101}};
  cudf::test::fixed_width_column_wrapper<double> price{{100.0,
                                                        40.0,
                                                        60.0,
                                                        200.0,
                                                        100.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0}};
  cudf::test::fixed_width_column_wrapper<double> discount{
    {0.1, 0.25, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  cudf::test::fixed_width_column_wrapper<int32_t> supplier_key{{101, 102, 103, 104, 105, 106}};
  cudf::test::fixed_width_column_wrapper<int8_t> supplier_nation{{10, 20, 30, 40, 50, 99}};
  cudf::test::fixed_width_column_wrapper<int8_t> nation_key{{10, 20, 30, 40, 50}};
  cudf::test::fixed_width_column_wrapper<int8_t> nation_region{{1, 1, 2, 3, 99}};
  cudf::test::strings_column_wrapper nation_name{"ALPHA", "ZULU", "LOWER", "EUROPE", "NO_REGION"};
  cudf::test::fixed_width_column_wrapper<int8_t> region_key{{1, 2, 3}};
  cudf::test::strings_column_wrapper region_name{"ASIA", "asia", "EUROPE"};
  CUDF_CUDA_TRY(cudaDeviceSynchronize());

  std::map<std::string, cudf::table_view> const input{
    {"customer", cudf::table_view{{customer_key, customer_nation}}},
    {"orders", cudf::table_view{{order_customer, order_key, order_date}}},
    {"lineitem", cudf::table_view{{line_order, line_supplier, price, discount}}},
    {"supplier", cudf::table_view{{supplier_key, supplier_nation}}},
    {"nation", cudf::table_view{{nation_key, nation_region, nation_name}}},
    {"region", cudf::table_view{{region_key, region_name}}}};
  // Five lines match; order 10 has a supplier from the wrong customer nation.
  ndsh::q5_reference_result const expected{{{"ALPHA", 150.0}, {"ZULU", 250.0}}, 5};
  cuda::stream_ref const stream = cudf::get_default_stream();
  // Full input, only rejected orders, and no orders. The CPU oracle accepts no nulls.
  auto const non_null_orders = cudf::slice(input.at("orders"), {0, 13}, stream).front();
  for (auto const& orders : cudf::slice(input.at("orders"), {0, 14, 3, 12, 0, 0}, stream)) {
    auto const want = orders.num_rows() == 14 ? expected : ndsh::q5_reference_result{};
    ndsh::q5_reference_builder builder;
    for (auto const& name : {"region", "nation", "supplier", "customer", "orders", "lineitem"}) {
      builder.add_table(name,
                        std::string{name} == "orders"
                          ? (orders.num_rows() == 14 ? non_null_orders : orders)
                          : input.at(name),
                        stream);
    }
    auto const cpu = builder.finish();
    CUDF_EXPECTS(cpu.matched == want.matched && cpu.revenue == want.revenue,
                 "Q5 CPU reference boundary/join regression");
    for (bool filter_predicates : {true, false}) {
      auto result = execute_q5(
        [&](std::string const& name,
            std::vector<std::string> const& columns,
            std::unique_ptr<cudf::ast::operation> const& predicate) {
          CUDF_EXPECTS(columns == q5_projections.at(name), "Q5 projection mismatch");
          auto const source = name == "orders" ? orders : input.at(name);
          if (!filter_predicates) { return ndsh::read_parquet_fixture(source, columns, predicate); }
          return std::make_unique<table_with_names>(std::make_unique<cudf::table>(source), columns);
        },
        filter_predicates,
        ndsh::take_result);
      ndsh::check_q5_result(want, *result, stream);
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

struct q5_files {
  ndsh::local_table_files tables;
  ndsh::q5_reference_result reference;

  explicit q5_files(double scale_factor)
  {
    check_q5_boundaries();
    ndsh::make_reference_files<ndsh::q5_reference_builder>(
      scale_factor,
      tables,
      reference,
      q5_tables,
      q5_projections,
      false,
      [&](auto&& read, cuda::stream_ref stream) {
        auto result = execute_q5(read, true, ndsh::take_result);
        ndsh::check_q5_result(reference, *result, stream);
      });
  }
};

void ndsh_q5_local(nvbench::state& state)
{
  auto const options = ndsh::local_options{state, 5};
  if (!options.supported(state)) { return; }
  auto const& files = ndsh::local_fixture<q5_files>(state.get_float64("scale_factor"));
  ndsh::local_benchmark benchmark{state, files.tables, options};
  auto read = [&](auto const& name, auto const& columns, auto const&) {
    return benchmark.read(name, columns);
  };
  auto read_inputs = [&] {
    return ndsh::read_local_tables(q5_tables, q5_projections, read, options.use_vortex);
  };
  auto query = [&] {
    if (!options.use_vortex) { return execute_q5(read, true, ndsh::take_result); }
    auto inputs = read_inputs();
    return execute_q5(
      [&](std::string const& name, auto const&...) {
        auto const index =
          std::distance(q5_tables.begin(), std::find(q5_tables.begin(), q5_tables.end(), name));
        return std::move(inputs.at(index));
      },
      true,
      ndsh::take_result);
  };
  {
    {
      auto inputs = read_inputs();
      for (std::size_t i = 0; i < q5_tables.size(); ++i) {
        auto const& name = q5_tables[i];
        benchmark.check_projection(name, q5_projections.at(name), *inputs[i], options.direct_io);
      }
      CUDF_CUDA_TRY(cudaStreamSynchronize(benchmark.stream.get()));
    }
    auto result = query();
    ndsh::check_q5_result(files.reference, *result, benchmark.stream);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  benchmark.exec(read_inputs, query);
  ndsh::add_count(state, "ndsh/q5/matched_rows", "Q5 matched rows", files.reference.matched);
  ndsh::add_count(state, "ndsh/q5/countries", "Q5 countries", files.reference.revenue.size());
}

}  // namespace

// NVBench varies the first axis fastest; keep scale last to reuse the one-scale fixture cache.
NVBENCH_BENCH(ndsh_q5_local)
  .set_name("ndsh_q5_local")
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q5"})
  .add_string_axis("cache", {"warm", "cold"})
  .add_string_axis("io", {"buffered"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10});
#endif
