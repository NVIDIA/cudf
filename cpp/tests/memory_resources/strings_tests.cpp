/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "memory_resources_test_utilities.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <optional>

using cudf::test::temporary_allocation_expectation;
using cudf::test::memory_resources_tests::expect_column_api_routes_memory_resources;

namespace {

auto make_strings()
{
  return cudf::test::strings_column_wrapper{
    {"The quick", "brown fox", "", "jumps over", "the lazy", "dog", "ÀÉÎõü", "tést"},
    {1, 1, 0, 1, 1, 1, 1, 1}};
}

}  // namespace

struct StringsMemoryResourcesTest : public cudf::test::BaseFixture {};

// Strings APIs create their column_device_view with the current resource.
TEST_F(StringsMemoryResourcesTest, DISABLED_ToUpper)
{
  auto input = make_strings();
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::strings::to_upper(
        cudf::strings_column_view{input}, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// Strings APIs create their column_device_view with the current resource.
TEST_F(StringsMemoryResourcesTest, DISABLED_Contains)
{
  auto input        = make_strings();
  auto const target = cudf::string_scalar{"o"};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::strings::contains(
        cudf::strings_column_view{input}, target, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// Strings APIs create their column_device_view with the current resource.
TEST_F(StringsMemoryResourcesTest, DISABLED_Replace)
{
  auto input        = make_strings();
  auto const target = cudf::string_scalar{"o"};
  auto const repl   = cudf::string_scalar{"00"};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::strings::replace(
        cudf::strings_column_view{input}, target, repl, -1, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// Strings APIs create their column_device_view with the current resource.
TEST_F(StringsMemoryResourcesTest, DISABLED_SliceStrings)
{
  auto input = make_strings();
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::strings::slice_strings(
        cudf::strings_column_view{input}, 1, std::nullopt, 2, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}

// Strings APIs create their column_device_view with the current resource.
TEST_F(StringsMemoryResourcesTest, DISABLED_JoinStrings)
{
  auto input           = make_strings();
  auto const separator = cudf::string_scalar{"|"};
  auto const narep     = cudf::string_scalar{"", false};
  expect_column_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::strings::join_strings(
        cudf::strings_column_view{input}, separator, narep, cudf::test::get_default_stream(), mr);
    },
    temporary_allocation_expectation::SOME);
}
