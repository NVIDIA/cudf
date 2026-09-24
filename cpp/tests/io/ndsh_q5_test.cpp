/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/copying.hpp>

#include <ndsh/parquet/parquet_fixture.hpp>
#include <ndsh/q05_query.hpp>
#include <ndsh/reference/q5_reference.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using ndsh::q5::execute_q5;
using ndsh::q5::q5_projections;

class NdshQ5Test : public cudf::test::BaseFixture {};

TEST_F(NdshQ5Test, Boundaries)
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
  cudf::test::fixed_width_column_wrapper<double> price{
    {100, 40, 60, 200, 100, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000}};
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
  // The CPU oracle accepts no nulls.
  struct boundary_case {
    char const* name;
    cudf::table_view orders;
    cudf::table_view cpu_orders;
    ndsh::q5_reference_result expected;
  };
  auto const rejected_orders = cudf::slice(input.at("orders"), {3, 12}, stream).front();
  auto const empty_orders    = cudf::slice(input.at("orders"), {0, 0}, stream).front();
  boundary_case const cases[]{{"full",
                               input.at("orders"),
                               cudf::slice(input.at("orders"), {0, 13}, stream).front(),
                               expected},
                              {"rejected orders", rejected_orders, rejected_orders, {}},
                              {"empty orders", empty_orders, empty_orders, {}}};
  for (auto const& test : cases) {
    SCOPED_TRACE(test.name);
    ndsh::q5_reference_builder builder;
    for (auto const& name : {"region", "nation", "supplier", "customer", "orders", "lineitem"}) {
      builder.add_table(
        name, std::string{name} == "orders" ? test.cpu_orders : input.at(name), stream);
    }
    auto const cpu = builder.finish();
    ASSERT_TRUE(cpu.matched == test.expected.matched && cpu.revenue == test.expected.revenue)
      << "Q5 CPU reference boundary/join regression";
    for (bool filter_predicates : {true, false}) {
      SCOPED_TRACE(::testing::Message{} << "filter_predicates=" << filter_predicates);
      std::unique_ptr<table_with_names> result;
      ASSERT_NO_THROW(result = execute_q5(
                        [&](std::string const& name,
                            std::vector<std::string> const& columns,
                            std::unique_ptr<cudf::ast::operation> const& predicate) {
                          CUDF_EXPECTS(columns == q5_projections.at(name),
                                       "Q5 projection mismatch");
                          auto const source = name == "orders" ? test.orders : input.at(name);
                          if (!filter_predicates) {
                            return ndsh::read_parquet_fixture(source, columns, predicate);
                          }
                          return std::make_unique<table_with_names>(
                            std::make_unique<cudf::table>(source), columns);
                        },
                        filter_predicates,
                        [](auto& result) { return std::move(result); }));
      ASSERT_NO_THROW(ndsh::check_q5_result(test.expected, *result, stream));
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace
