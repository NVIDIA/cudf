/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "memory_resources_test_utilities.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>

#include <vector>

using cudf::test::temporary_allocation_expectation;
using cudf::test::memory_resources_tests::expect_column_api_routes_memory_resources;
using cudf::test::memory_resources_tests::expect_table_api_routes_memory_resources;

namespace {

auto make_input()
{
  return cudf::test::fixed_width_column_wrapper<int32_t>{{5, 4, 3, 2, 1, 0, -1, -2},
                                                         {1, 1, 0, 1, 1, 0, 1, 1}};
}

auto make_strings()
{
  return cudf::test::strings_column_wrapper{{"a", "bb", "", "dddd", "eeeee", "", "g", "hh"},
                                            {1, 1, 0, 1, 1, 1, 0, 1}};
}

}  // namespace

struct ColumnMemoryResourcesTest : public cudf::test::BaseFixture {};

TEST_F(ColumnMemoryResourcesTest, MakeNumericColumn)
{
  expect_column_api_routes_memory_resources(
    [](cudf::memory_resources mr) {
      auto result = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                              100,
                                              cudf::mask_state::ALL_NULL,
                                              cudf::test::get_default_stream(),
                                              mr);
      return result;
    },
    temporary_allocation_expectation::NONE);
}

TEST_F(ColumnMemoryResourcesTest, MakeColumnFromScalar)
{
  auto const value = cudf::numeric_scalar<int32_t>{42};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::make_column_from_scalar(value, 100, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::NONE);
}

// strings::detail::make_strings_column allocates a null mask from the output resource and drops it
// when there are no nulls.
TEST_F(ColumnMemoryResourcesTest, DISABLED_MakeStringsColumnFromScalar)
{
  auto const value = cudf::string_scalar{"memory"};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::make_column_from_scalar(value, 100, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

TEST_F(ColumnMemoryResourcesTest, Gather)
{
  auto numbers     = make_input();
  auto strings     = make_strings();
  auto const input = cudf::table_view{{numbers, strings}};
  auto map         = cudf::test::fixed_width_column_wrapper<int32_t>{7, 0, 3, 3, 5, 1};
  expect_table_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::gather(
        input, map, cudf::out_of_bounds_policy::DONT_CHECK, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// Strings scatter creates its column_device_view with the current resource.
TEST_F(ColumnMemoryResourcesTest, DISABLED_Scatter)
{
  auto numbers      = make_input();
  auto strings      = make_strings();
  auto const target = cudf::table_view{{numbers, strings}};
  auto const source_numbers =
    cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30}, {1, 0, 1}};
  auto source_strings = cudf::test::strings_column_wrapper{{"x", "yy", "zzz"}, {1, 1, 0}};
  auto const source   = cudf::table_view{{source_numbers, source_strings}};
  auto map            = cudf::test::fixed_width_column_wrapper<int32_t>{6, 1, 2};
  expect_table_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::scatter(source, map, target, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// copy_if_else allocates its valid-count device_scalar from the current resource.
TEST_F(ColumnMemoryResourcesTest, DISABLED_CopyIfElse)
{
  auto lhs  = make_input();
  auto rhs  = cudf::test::fixed_width_column_wrapper<int32_t>{{9, 9, 9, 9, 9, 9, 9, 9},
                                                              {1, 0, 1, 0, 1, 0, 1, 0}};
  auto mask = cudf::test::fixed_width_column_wrapper<bool>{1, 0, 1, 0, 0, 1, 1, 0};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::copy_if_else(lhs, rhs, mask, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// Strings concatenate creates its device views with the current resource.
TEST_F(ColumnMemoryResourcesTest, DISABLED_ConcatenateStrings)
{
  auto lhs = make_strings();
  auto rhs = cudf::test::strings_column_wrapper{{"one", "two", "three"}, {false, true, true}};
  auto const columns = std::vector<cudf::column_view>{lhs, rhs};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::concatenate(columns, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

CUDF_TEST_PROGRAM_MAIN()
