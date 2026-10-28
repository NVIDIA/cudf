/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

template <typename T>
struct TileTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TileTest, cudf::test::AllTypes);

TYPED_TEST(TileTest, NoColumns)
{
  cudf::table_view in(std::vector<cudf::column_view>{});

  auto expected = in;

  auto actual = cudf::tile(in, 10);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual->view());
}

TYPED_TEST(TileTest, NoRows)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> in_a({});
  cudf::table_view in(std::vector<cudf::column_view>{in_a});

  auto expected = in;

  auto actual = cudf::tile(in, 10);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual->view());
}

TYPED_TEST(TileTest, OneColumn)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T, int32_t> in_a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{in_a});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected_a({-1, 0, 1, -1, 0, 1});
  cudf::table_view expected(std::vector<cudf::column_view>{expected_a});

  auto actual = cudf::tile(in, 2);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual->view());
}

TYPED_TEST(TileTest, OneColumnNullable)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T, int32_t> in_a({-1, 0, 1}, {1, 0, 0});
  cudf::table_view in(std::vector<cudf::column_view>{in_a});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected_a({-1, 0, 1, -1, 0, 1},
                                                                {1, 0, 0, 1, 0, 0});
  cudf::table_view expected(std::vector<cudf::column_view>{expected_a});

  auto actual = cudf::tile(in, 2);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual->view());
}

TYPED_TEST(TileTest, OneColumnNegativeCount)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T, int32_t> in_a({-1, 0, 1}, {1, 0, 0});
  cudf::table_view in(std::vector<cudf::column_view>{in_a});

  EXPECT_THROW(cudf::tile(in, -1), cudf::logic_error);
}

TYPED_TEST(TileTest, OneColumnZeroCount)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T, int32_t> in_a({-1, 0, 1}, {1, 0, 0});
  cudf::table_view in(std::vector<cudf::column_view>{in_a});

  std::vector<T> vals{};
  std::vector<bool> mask{};

  cudf::test::fixed_width_column_wrapper<T> expected_a(vals.begin(), vals.end(), mask.begin());

  cudf::table_view expected(std::vector<cudf::column_view>{expected_a});

  auto actual = cudf::tile(in, 0);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual->view());
}

struct TileMemoryResourcesTest : cudf::test::BaseFixture {};

TEST_F(TileMemoryResourcesTest, NullableFixedWidthAndStrings)
{
  cudf::test::fixed_width_column_wrapper<int32_t> numbers{{1, 2, 3}, {1, 0, 1}};
  cudf::test::strings_column_wrapper strings{{"one", "two", "three"}, {1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_numbers{{1, 2, 3, 1, 2, 3},
                                                                   {1, 0, 1, 1, 0, 1}};
  cudf::test::strings_column_wrapper expected_strings{
    {"one", "two", "three", "one", "two", "three"}, {1, 0, 1, 1, 0, 1}};

  auto const input    = cudf::table_view{{numbers, strings}};
  auto const expected = cudf::table_view{{expected_numbers, expected_strings}};
  auto const stream   = cudf::test::get_default_stream();
  auto harness        = cudf::test::memory_resource_test_harness{this->mr()};
  cudf::test::expect_api_uses_memory_resources(
    harness,
    [&](auto resources) { return cudf::tile(input, 2, stream, resources); },
    [](auto const& result) { return result->alloc_size(); },
    [&](auto const& result) { CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view()); },
    {cudf::test::output_allocation_expectation::EXACT,
     cudf::test::temporary_allocation_expectation::SOME},
    stream);
}
