/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/copying.hpp>

#include <ndsh/parquet/parquet_fixture.hpp>
#include <ndsh/q10_query.hpp>
#include <ndsh/reference/q10_reference.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using ndsh::q10::execute_q10;
using ndsh::q10::q10_projections;

class NdshQ10Test : public cudf::test::BaseFixture {};

TEST_F(NdshQ10Test, Cases)
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
  cudf::test::fixed_width_column_wrapper<double> price{
    {100, 50, 200, 80, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000}};
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
  auto const cpu_lineitem       = cudf::slice(input.at("lineitem"), {0, 12}, stream).front();
  // Keep null predicate rows on the GPU, but exclude them from the CPU oracle.
  struct query_case {
    char const* name;
    cudf::table_view orders;
    cudf::table_view cpu_orders;
    ndsh::q10_reference_result expected;
  };
  auto const empty_orders = cudf::slice(input.at("orders"), {0, 0}, stream).front();
  query_case const cases[]{
    {"full", input.at("orders"), cudf::slice(input.at("orders"), {0, 8}, stream).front(), expected},
    {"rejected orders",
     cudf::slice(input.at("orders"), {3, 9}, stream).front(),
     cudf::slice(input.at("orders"), {3, 8}, stream).front(),
     {}},
    {"empty orders", empty_orders, empty_orders, {}}};
  for (auto const& test : cases) {
    SCOPED_TRACE(test.name);
    ndsh::q10_reference_builder builder;
    for (auto const& name : {"nation", "customer", "orders", "lineitem"}) {
      auto const source = std::string{name} == "orders"     ? test.cpu_orders
                          : std::string{name} == "lineitem" ? cpu_lineitem
                                                            : input.at(name);
      builder.add_table(name, source, stream);
    }
    auto const cpu = builder.finish();
    ASSERT_TRUE(cpu.matched == test.expected.matched &&
                cpu.customers.size() == test.expected.customers.size())
      << "Q10 CPU reference row/customer regression";
    for (auto const& [key, expected_customer] : test.expected.customers) {
      auto const customer = cpu.customers.find(key);
      ASSERT_TRUE(customer != cpu.customers.end()) << "Missing Q10 CPU reference customer";
      auto const& actual_customer = customer->second;
      ASSERT_TRUE(actual_customer.name == expected_customer.name &&
                  actual_customer.account_balance == expected_customer.account_balance &&
                  actual_customer.nation == expected_customer.nation &&
                  actual_customer.address == expected_customer.address &&
                  actual_customer.phone == expected_customer.phone &&
                  actual_customer.comment == expected_customer.comment &&
                  ndsh::detail::reference_equal(actual_customer.revenue, expected_customer.revenue))
        << "Q10 CPU reference predicate/join/revenue regression";
    }
    for (bool post_read_filters : {true, false}) {
      SCOPED_TRACE(::testing::Message{} << "post_read_filters=" << post_read_filters);
      std::unique_ptr<table_with_names> result;
      ASSERT_NO_THROW(result = execute_q10(
                        [&](std::string const& name,
                            std::vector<std::string> const& columns,
                            std::unique_ptr<cudf::ast::operation> const& predicate) {
                          CUDF_EXPECTS(columns == q10_projections.at(name),
                                       "Q10 projection mismatch");
                          auto const source = name == "orders" ? test.orders : input.at(name);
                          if (!post_read_filters) {
                            return ndsh::read_parquet_fixture(source, columns, predicate);
                          }
                          return std::make_unique<table_with_names>(
                            std::make_unique<cudf::table>(source), columns);
                        },
                        post_read_filters,
                        [](auto& result) { return std::move(result); }));
      ASSERT_NO_THROW(ndsh::check_q10_result(test.expected, *result, stream));
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace
