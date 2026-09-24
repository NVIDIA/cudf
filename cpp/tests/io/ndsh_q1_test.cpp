/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file ndsh_q1_test.cpp
 * @brief Verify NDS-H Q1 date filtering and grouped aggregates against reference results.
 *
 * Covers boundary, null, sliced, and empty inputs with post-read and Parquet predicate filtering.
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/copying.hpp>

#include <ndsh/parquet/parquet_fixture.hpp>
#include <ndsh/q01_query.hpp>
#include <ndsh/reference/q1_reference.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using ndsh::q1::execute_q1;

class NdshQ1Test : public cudf::test::BaseFixture {};

TEST_F(NdshQ1Test, Boundaries)
{
  cudf::test::strings_column_wrapper returnflag{"R", "A", "R", "A", "R", "X", "Z"};
  cudf::test::strings_column_wrapper linestatus{"O", "F", "F", "F", "O", "X", "Z"};
  cudf::test::fixed_width_column_wrapper<int8_t> quantity{{4, 2, 3, 6, 8, 50, 50}};
  cudf::test::fixed_width_column_wrapper<double> price{{100, 80, 40, 120, 200, 1000, 1000}};
  cudf::test::fixed_width_column_wrapper<double> discount{{0.25, 0.25, 0.5, 0.0, 0.5, 0.0, 0.0}};
  // Epoch days: 1998-09-01, inclusive cutoff 1998-09-02, 1998-09-03.
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> shipdate{
    {10470, 10471, 10470, 10470, 10471, 10472, 10471}, {true, true, true, true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> orderkey{{1, 2, 3, 4, 5, 6, 7}};
  cudf::test::fixed_width_column_wrapper<double> tax{{0.1, 0.2, 0.1, 0.0, 0.2, 0.0, 0.0}};
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  cudf::table_view const input{
    {returnflag, linestatus, quantity, price, discount, shipdate, orderkey, tax}};
  ndsh::q1_reference_result const expected{{{{"A", "F"}, {8, 200, 180, 192, 4, 100, 0.125, 2}},
                                            {{"R", "F"}, {3, 40, 20, 22, 3, 40, 0.5, 1}},
                                            {{"R", "O"}, {12, 300, 175, 202.5, 6, 150, 0.375, 2}}},
                                           5};
  cuda::stream_ref const stream = cudf::get_default_stream();
  auto const cpu        = ndsh::q1_cpu_reference(cudf::slice(input, {0, 6}).front(), stream);
  auto const sliced_cpu = ndsh::q1_cpu_reference(cudf::slice(input, {1, 6}).front(), stream);
  auto sliced_expected  = expected;
  sliced_expected.groups.at({"R", "O"}) = {8, 200, 100, 120, 8, 200, 0.5, 1};
  sliced_expected.matched               = 4;
  struct boundary_case {
    char const* name;
    cudf::table_view input;
    ndsh::q1_reference_result expected;
    ndsh::q1_reference_result const* cpu;
  };
  boundary_case const cases[]{
    {"full", input, expected, &cpu},
    {"offset", cudf::slice(input, {1, 7}).front(), sliced_expected, &sliced_cpu},
    {"rejected/null dates", cudf::slice(input, {5, 7}).front(), {}, nullptr},
    {"empty", cudf::slice(input, {0, 0}).front(), {}, nullptr}};
  for (auto const& test : cases) {
    SCOPED_TRACE(test.name);
    for (bool filter_shipdate : {true, false}) {
      SCOPED_TRACE(::testing::Message{} << "filter_shipdate=" << filter_shipdate);
      std::unique_ptr<table_with_names> result;
      ASSERT_NO_THROW(result = execute_q1(
                        [&](auto const& columns, auto const& predicate) {
                          if (!filter_shipdate) {
                            return ndsh::read_parquet_fixture(test.input, columns, predicate);
                          }
                          return std::make_unique<table_with_names>(
                            std::make_unique<cudf::table>(test.input), columns);
                        },
                        filter_shipdate,
                        [](auto& result) { return std::move(result); }));
      ASSERT_NO_THROW(ndsh::check_q1_result(test.expected, *result, stream));
      if (test.cpu) { ASSERT_NO_THROW(ndsh::check_q1_result(*test.cpu, *result, stream)); }
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace

CUDF_TEST_PROGRAM_MAIN()
