/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/copying.hpp>

#include <ndsh/q09_query.hpp>
#include <ndsh/reference/q9_reference.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using ndsh::q9::engine_type;
using ndsh::q9::execute_q9;
using ndsh::q9::q9_projections;

class NdshQ9Test : public cudf::test::BaseFixture {};

TEST_F(NdshQ9Test, Cases)
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
  struct query_case {
    char const* name;
    cudf::table_view partsupp;
    cudf::table_view lineitem;
    ndsh::q9_reference_result expected;
  };
  auto const unique_partsupp    = cudf::slice(input.at("partsupp"), {0, 5}, stream).front();
  auto const duplicate_partsupp = input.at("partsupp");
  auto const full_lines         = input.at("lineitem");
  auto const rejected_lines     = cudf::slice(full_lines, {4, 12}, stream).front();
  auto const empty_lines        = cudf::slice(full_lines, {0, 0}, stream).front();
  query_case const cases[]{
    {"full/unique partsupp", unique_partsupp, full_lines, expected},
    {"rejected/unique partsupp", unique_partsupp, rejected_lines, {}},
    {"empty/unique partsupp", unique_partsupp, empty_lines, {}},
    {"full/duplicate partsupp", duplicate_partsupp, full_lines, duplicate_expected},
    {"rejected/duplicate partsupp", duplicate_partsupp, rejected_lines, {}},
    {"empty/duplicate partsupp", duplicate_partsupp, empty_lines, {}}};
  for (auto const& test : cases) {
    SCOPED_TRACE(test.name);
    auto tables           = input;
    tables.at("partsupp") = test.partsupp;
    tables.at("lineitem") = test.lineitem;
    ndsh::q9_reference_builder builder;
    for (auto const& name : {"nation", "supplier", "partsupp", "orders", "part", "lineitem"}) {
      builder.add_table(name, tables.at(name), stream);
    }
    auto const cpu = builder.finish();
    ASSERT_TRUE(cpu.matched == test.expected.matched &&
                cpu.sum_profit.size() == test.expected.sum_profit.size())
      << "Q9 CPU reference row/group regression";
    for (auto const& [key, value] : test.expected.sum_profit) {
      auto const actual = cpu.sum_profit.find(key);
      ASSERT_TRUE(actual != cpu.sum_profit.end() &&
                  ndsh::detail::reference_equal(actual->second, value))
        << "Q9 CPU reference amount/year regression";
    }
    for (auto const engine : {engine_type::BINARYOP, engine_type::AST, engine_type::TRANSFORM}) {
      SCOPED_TRACE(::testing::Message{} << "engine=" << static_cast<int32_t>(engine));
      std::unique_ptr<table_with_names> result;
      ASSERT_NO_THROW(
        result =
          execute_q9(engine, [&](std::string const& name, std::vector<std::string> const& columns) {
            CUDF_EXPECTS(columns == q9_projections.at(name), "Q9 projection mismatch");
            return std::make_unique<table_with_names>(
              std::make_unique<cudf::table>(tables.at(name)), columns);
          }));
      ASSERT_NO_THROW(ndsh::check_q9_result(test.expected, *result, stream));
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace
