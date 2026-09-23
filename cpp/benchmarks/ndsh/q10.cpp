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
#include "reference/q10_reference.hpp"
#include "vortex/vortex_io.hpp"

#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>

#include <cstdint>
#endif

namespace {
std::vector<std::string> const q10_tables{"customer", "orders", "lineitem", "nation"};
std::map<std::string, std::vector<std::string>> const q10_projections{
  {"customer",
   {"c_custkey", "c_name", "c_nationkey", "c_acctbal", "c_address", "c_phone", "c_comment"}},
  {"orders", {"o_custkey", "o_orderkey", "o_orderdate"}},
  {"lineitem", {"l_extendedprice", "l_discount", "l_orderkey", "l_returnflag"}},
  {"nation", {"n_name", "n_nationkey"}}};
}  // namespace

/**
 * @file q10.cpp
 * @brief Implement query 10 of the NDS-H benchmark.
 *
 * create view customer as select * from '/tables/scale-1/customer.parquet';
 * create view orders as select * from '/tables/scale-1/orders.parquet';
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 * create view nation as select * from '/tables/scale-1/nation.parquet';
 *
 * select
 *    c_custkey,
 *    c_name,
 *    sum(l_extendedprice * (1 - l_discount)) as revenue,
 *    c_acctbal,
 *    n_name,
 *    c_address,
 *    c_phone,
 *    c_comment
 * from
 *    customer,
 *    orders,
 *    lineitem,
 *    nation
 * where
 *     c_custkey = o_custkey
 *     and l_orderkey = o_orderkey
 *     and o_orderdate >= date '1993-10-01'
 *     and o_orderdate < date '1994-01-01'
 *     and l_returnflag = 'R'
 *     and c_nationkey = n_nationkey
 * group by
 *     c_custkey,
 *     c_name,
 *     c_acctbal,
 *     c_phone,
 *     n_name,
 *     c_address,
 *     c_comment
 * order by
 *     revenue desc;
 */

/**
 * read returns owning projected tables, applying supplied predicates if filter_predicates is false.
 * Otherwise filtering happens here. consume receives the result owner by reference and may
 * move it out; its return value is forwarded. This helper adds no final stream synchronization.
 */
template <typename Read, typename Consume>
auto execute_q10(Read&& read, bool filter_predicates, Consume&& consume)
{
  // Define the column projection and filter predicate for the `orders` table
  auto const& orders_cols    = q10_projections.at("orders");
  auto const o_orderdate_ref = cudf::ast::column_reference(std::distance(
    orders_cols.begin(), std::find(orders_cols.begin(), orders_cols.end(), "o_orderdate")));
  auto o_orderdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1993, 10, 1), true);
  auto const o_orderdate_lower_limit = cudf::ast::literal(o_orderdate_lower);
  auto const o_orderdate_pred_lower  = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, o_orderdate_ref, o_orderdate_lower_limit);
  auto o_orderdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
  auto const o_orderdate_upper_limit = cudf::ast::literal(o_orderdate_upper);
  auto const o_orderdate_pred_upper =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, o_orderdate_ref, o_orderdate_upper_limit);
  auto const orders_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, o_orderdate_pred_lower, o_orderdate_pred_upper);

  auto const l_returnflag_ref = cudf::ast::column_reference(3);
  auto r_scalar               = cudf::string_scalar("R");
  auto const r_literal        = cudf::ast::literal(r_scalar);
  auto const lineitem_pred    = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, l_returnflag_ref, r_literal);

  std::unique_ptr<cudf::ast::operation> const no_predicate;
  auto const customer = read("customer", q10_projections.at("customer"), no_predicate);
  auto orders         = read("orders", orders_cols, orders_pred);
  auto lineitem       = read("lineitem", q10_projections.at("lineitem"), lineitem_pred);
  auto const nation   = read("nation", q10_projections.at("nation"), no_predicate);
  if (filter_predicates) {
    orders   = apply_filter(orders, *orders_pred);
    lineitem = apply_filter(lineitem, *lineitem_pred);
  }

  // Perform the joins
  auto const join_a       = apply_inner_join(customer, nation, {"c_nationkey"}, {"n_nationkey"});
  auto const join_b       = apply_inner_join(lineitem, orders, {"l_orderkey"}, {"o_orderkey"});
  auto const joined_table = apply_inner_join(join_a, join_b, {"c_custkey"}, {"o_custkey"});

  // Calculate and append the `revenue` column
  auto revenue = calculate_discounted_revenue(joined_table->column("l_extendedprice"),
                                              joined_table->column("l_discount"));
  (*joined_table).append(revenue, "revenue");

  // Perform the groupby operation
  auto const groupedby_table = apply_groupby(
    joined_table,
    groupby_context_t{
      {"c_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address", "c_comment"},
      {
        {"revenue", {{cudf::aggregation::Kind::SUM, "revenue"}}},
      }});

  // Perform the order by operation
  auto orderedby_table = apply_orderby(groupedby_table, {"revenue"}, {cudf::order::DESCENDING});
  return consume(orderedby_table);
}

void run_ndsh_q10(nvbench::state& state,
                  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  execute_q10(
    [&](std::string const& name,
        std::vector<std::string> const& columns,
        std::unique_ptr<cudf::ast::operation> const& predicate) {
      return read_parquet(sources.at(name).make_source_info(), columns, predicate);
    },
    false,
    [](auto const& result) { result->to_parquet("q10.parquet"); });
}

void ndsh_q10(nvbench::state& state)
{
  // Generate the required parquet files in device buffers
  double const scale_factor = state.get_float64("scale_factor");
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, q10_tables, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { run_ndsh_q10(state, sources); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q10).set_name("ndsh_q10").add_float64_axis("scale_factor", {0.01, 0.1, 1});

#ifdef CUDF_WITH_VORTEX
namespace {

void check_q10_cases()
{
  cudf::test::fixed_width_column_wrapper<int8_t> nation_key{{10, 20}};
  cudf::test::strings_column_wrapper nation_name{"ALPHA", "ZULU"};
  cudf::test::fixed_width_column_wrapper<int32_t> customer_key{{1, 2, 3, 4}};
  cudf::test::strings_column_wrapper customer_name{"Alice", "Bob", "NoNation", "NoOrders"};
  cudf::test::fixed_width_column_wrapper<int8_t> customer_nation{{10, 20, 99, 10}};
  cudf::test::fixed_width_column_wrapper<double> account_balance{{100.0, -20.0, 0.0, 5.0}};
  cudf::test::strings_column_wrapper customer_address{"1 Main", "2 Main", "3 Main", "4 Main"};
  cudf::test::strings_column_wrapper customer_phone{"10-1", "20-2", "99-3", "10-4"};
  cudf::test::strings_column_wrapper customer_comment{"first", "second", "third", "fourth"};
  cudf::test::fixed_width_column_wrapper<int32_t> order_customer{{1, 1, 2, 2, 2, 3, 999, 2, 2}};
  cudf::test::fixed_width_column_wrapper<int32_t> order_key{
    {101, 102, 105, 103, 104, 106, 107, 109, 108}};
  // Epoch days include both date boundaries and a final null date.
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> order_date{
    {8674, 8765, 8705, 8766, 8673, 8705, 8705, 8705, 8674},
    {true, true, true, true, true, true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<double> price{{100.0,
                                                        50.0,
                                                        200.0,
                                                        80.0,
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
    {0.1, 0.2, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  cudf::test::fixed_width_column_wrapper<int32_t> line_order{
    {101, 101, 102, 105, 103, 104, 106, 107, 109, 109, 999, 108, 105}};
  cudf::test::strings_column_wrapper return_flag(
    {"R", "R", "R", "R", "R", "R", "R", "R", "r", "N", "R", "R", ""},
    {true, true, true, true, true, true, true, true, true, true, true, true, false});
  CUDF_CUDA_TRY(cudaDeviceSynchronize());

  std::map<std::string, cudf::table_view> const input{
    {"nation", cudf::table_view{{nation_name, nation_key}}},
    {"customer",
     cudf::table_view{{customer_key,
                       customer_name,
                       customer_nation,
                       account_balance,
                       customer_address,
                       customer_phone,
                       customer_comment}}},
    {"orders", cudf::table_view{{order_customer, order_key, order_date}}},
    {"lineitem", cudf::table_view{{price, discount, line_order, return_flag}}}};
  ndsh::q10_reference_result expected;
  expected.customers.emplace(
    1, ndsh::q10_customer_result{"Alice", 100.0, "ALPHA", "1 Main", "10-1", "first", 280.0});
  expected.customers.emplace(
    2, ndsh::q10_customer_result{"Bob", -20.0, "ZULU", "2 Main", "20-2", "second", 40.0});
  expected.matched              = 4;
  cuda::stream_ref const stream = cudf::get_default_stream();
  auto const order_cases        = cudf::slice(input.at("orders"), {0, 9, 3, 9, 0, 0}, stream);
  auto const cpu_order_cases    = cudf::slice(input.at("orders"), {0, 8, 3, 8, 0, 0}, stream);
  auto const cpu_lineitem       = cudf::slice(input.at("lineitem"), {0, 12}, stream).front();
  // Full input, only rows rejected by predicates/joins, and no orders.
  for (std::size_t case_index = 0; case_index < order_cases.size(); ++case_index) {
    auto const& orders     = order_cases[case_index];
    auto const& cpu_orders = cpu_order_cases[case_index];
    auto const want        = case_index == 0 ? expected : ndsh::q10_reference_result{};
    ndsh::q10_reference_builder builder;
    for (auto const& name : {"nation", "customer", "orders", "lineitem"}) {
      auto const source = std::string{name} == "orders"     ? cpu_orders
                          : std::string{name} == "lineitem" ? cpu_lineitem
                                                            : input.at(name);
      builder.add_table(name, source, stream);
    }
    auto const cpu = builder.finish();
    CUDF_EXPECTS(cpu.matched == want.matched && cpu.customers.size() == want.customers.size(),
                 "Q10 CPU reference row/customer regression");
    for (auto const& [key, expected_customer] : want.customers) {
      auto const customer = cpu.customers.find(key);
      CUDF_EXPECTS(customer != cpu.customers.end(), "Missing Q10 CPU reference customer");
      auto const& actual_customer = customer->second;
      CUDF_EXPECTS(
        actual_customer.name == expected_customer.name &&
          actual_customer.account_balance == expected_customer.account_balance &&
          actual_customer.nation == expected_customer.nation &&
          actual_customer.address == expected_customer.address &&
          actual_customer.phone == expected_customer.phone &&
          actual_customer.comment == expected_customer.comment &&
          ndsh::detail::reference_equal(actual_customer.revenue, expected_customer.revenue),
        "Q10 CPU reference predicate/join/revenue regression");
    }
    for (bool post_read_filters : {true, false}) {
      auto result = execute_q10(
        [&](std::string const& name,
            std::vector<std::string> const& columns,
            std::unique_ptr<cudf::ast::operation> const& predicate) {
          CUDF_EXPECTS(columns == q10_projections.at(name), "Q10 projection mismatch");
          auto const source = name == "orders" ? orders : input.at(name);
          if (!post_read_filters) { return ndsh::read_parquet_fixture(source, columns, predicate); }
          return std::make_unique<table_with_names>(std::make_unique<cudf::table>(source), columns);
        },
        post_read_filters,
        ndsh::take_result);
      ndsh::check_q10_result(want, *result, stream);
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

struct q10_files {
  ndsh::local_table_files tables;
  ndsh::q10_reference_result reference;

  explicit q10_files(double scale_factor)
  {
    check_q10_cases();
    ndsh::make_reference_files<ndsh::q10_reference_builder>(
      scale_factor,
      tables,
      reference,
      q10_tables,
      q10_projections,
      true,
      [&](auto&& read, cuda::stream_ref stream) {
        auto result = execute_q10(read, true, ndsh::take_result);
        ndsh::check_q10_result(reference, *result, stream);
      });
  }
};

void ndsh_q10_local(nvbench::state& state)
{
  auto const options = ndsh::local_options{state, 10};
  if (!options.supported(state)) { return; }
  auto const& files = ndsh::local_fixture<q10_files>(state.get_float64("scale_factor"));
  ndsh::local_benchmark benchmark{state, files.tables, options};
  auto read = [&](auto const& name, auto const& columns, auto const&) {
    return benchmark.read(name, columns);
  };
  auto read_inputs = [&] {
    return ndsh::read_local_tables(q10_tables, q10_projections, read, options.use_vortex);
  };
  auto query = [&] {
    if (!options.use_vortex) { return execute_q10(read, true, ndsh::take_result); }
    auto inputs = read_inputs();
    return execute_q10(
      [&](std::string const& name, auto const&...) {
        auto const index =
          std::distance(q10_tables.begin(), std::find(q10_tables.begin(), q10_tables.end(), name));
        return std::move(inputs.at(index));
      },
      true,
      ndsh::take_result);
  };
  {
    {
      auto inputs = read_inputs();
      for (std::size_t i = 0; i < q10_tables.size(); ++i) {
        auto const& name = q10_tables[i];
        benchmark.check_projection(name, q10_projections.at(name), *inputs[i], options.direct_io);
      }
      CUDF_CUDA_TRY(cudaStreamSynchronize(benchmark.stream.get()));
    }
    auto result = query();
    ndsh::check_q10_result(files.reference, *result, benchmark.stream);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  benchmark.exec(read_inputs, query, "Total file size", "ndsh_q10_local_timed");
  ndsh::add_count(state, "ndsh/q10/matched_rows", "Q10 matched rows", files.reference.matched);
  ndsh::add_count(state, "ndsh/q10/customers", "Q10 customers", files.reference.customers.size());
}

}  // namespace

// NVBench varies the first axis fastest; keep scale last to reuse the one-scale fixture cache.
NVBENCH_BENCH(ndsh_q10_local)
  .set_name("ndsh_q10_local")
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q10"})
  .add_string_axis("cache", {"warm", "cold"})
  .add_string_axis("io", {"buffered"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10});
#endif
