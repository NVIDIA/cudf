/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
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
