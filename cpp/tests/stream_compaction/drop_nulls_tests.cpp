/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <numeric>

struct DropNullsTest : public cudf::test::BaseFixture {};

TEST_F(DropNullsTest, WholeRowIsNull)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {true, true, false, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10},
                                                       {true, true, false, true, true, false}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10},
                                                      {true, true, false, true, true, false}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, false, false, true},
                                                                {true, true, true, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{10, 40, 5, 2},
                                                                {true, true, true, true}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{10, 40, 5, 2},
                                                               {true, true, true, true}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nulls(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(DropNullsTest, NoNull)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {true, true, true, true, true, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10},
                                                       {true, true, true, true, true, true}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10},
                                                      {true, true, true, true, true, true}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1, 2};

  auto got = cudf::drop_nulls(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(DropNullsTest, MixedSetOfRows)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {true, true, false, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10},
                                                       {true, true, false, true, true, false}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10},
                                                      {true, true, false, true, true, true}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, false, false, true},
                                                                {true, true, true, true}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{10, 40, 5, 2},
                                                                {true, true, true, true}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{10, 40, 5, 2},
                                                               {true, true, true, true}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nulls(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(DropNullsTest, LargeColumn)
{
  // This test is a C++ repro of the failing Python in this issue:
  // https://github.com/NVIDIA/cudf/issues/5456
  // Specifically, there are two large columns, one nullable, one non-nullable
  using T       = int32_t;
  using index_T = int64_t;
  constexpr cudf::size_type column_size{270000};
  std::vector<index_T> index(column_size);
  std::vector<T> data(column_size);
  std::vector<bool> mask_data(column_size);

  std::iota(index.begin(), index.end(), 0);
  std::generate_n(data.begin(), column_size, [x = 1]() mutable { return x++ % 3; });
  std::transform(data.begin(), data.end(), mask_data.begin(), [](auto const& x) { return x != 0; });

  std::vector<T> expected_data(column_size);
  // zeros are the null elements, remove them
  auto end           = std::remove_copy(data.begin(), data.end(), expected_data.begin(), 0);
  auto expected_size = std::distance(expected_data.begin(), end);
  expected_data.resize(expected_size);

  std::vector<index_T> expected_index(expected_size);
  std::copy_if(index.begin(), index.end(), expected_index.begin(), [](auto const& x) {
    return (x - 2) % 3 != 0;
  });

  // output null mask is all true
  std::vector<bool> expected_mask(expected_size, true);

  cudf::test::fixed_width_column_wrapper<T> col1(data.begin(), data.end(), mask_data.begin());
  cudf::test::fixed_width_column_wrapper<index_T> index1(index.begin(), index.end());
  cudf::table_view input{{index1, col1}};
  std::vector<cudf::size_type> keys{1};

  cudf::test::fixed_width_column_wrapper<T> exp1(
    expected_data.begin(), expected_data.end(), expected_mask.begin());
  cudf::test::fixed_width_column_wrapper<index_T> exp_index1(expected_index.begin(),
                                                             expected_index.end());
  cudf::table_view expected{{exp_index1, exp1}};

  auto got = cudf::drop_nulls(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(DropNullsTest, MixedSetOfRowsWithThreshold)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {true, true, false, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{10, 40, 70, 5, 2, 10},
                                                       {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<double> col3{{10, 40, 70, 5, 2, 10},
                                                      {true, true, true, true, true, true}};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{true, false, false, true, false},
                                                                {true, true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{{10, 40, 5, 2, 10},
                                                                {true, true, true, true, true}};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{{10, 40, 5, 2, 10},
                                                               {true, true, true, true, true}};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nulls(input, keys, keys.size() - 1);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(DropNullsTest, EmptyTable)
{
  cudf::table_view input{std::vector<cudf::column_view>()};
  cudf::table_view expected{std::vector<cudf::column_view>()};
  std::vector<cudf::size_type> keys{};

  auto got = cudf::drop_nulls(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(DropNullsTest, EmptyColumns)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{};
  cudf::test::fixed_width_column_wrapper<double> col3{};
  cudf::table_view input{{col1, col2, col3}};
  std::vector<cudf::size_type> keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{};
  cudf::test::fixed_width_column_wrapper<int32_t> col2_expected{};
  cudf::test::fixed_width_column_wrapper<double> col3_expected{};
  cudf::table_view expected{{col1_expected, col2_expected, col3_expected}};

  auto got = cudf::drop_nulls(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

TEST_F(DropNullsTest, EmptyKeys)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{true, false, true, false, true, false},
                                                       {true, true, false, true, true, false}};
  cudf::table_view input{{col1}};
  std::vector<cudf::size_type> keys{};

  auto got = cudf::drop_nulls(input, keys);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, got->view());
}

TEST_F(DropNullsTest, StringColWithNull)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1{{11, 12, 11, 13, 12, 15},
                                                       {true, true, false, true, false, true}};
  cudf::test::strings_column_wrapper col2{{"Hi", "Hello", "Hi", "No", "Hello", "Naive"},
                                          {true, true, false, true, false, true}};
  cudf::table_view input{{col1, col2}};
  std::vector<cudf::size_type> keys{0, 1};
  cudf::test::fixed_width_column_wrapper<int16_t> col1_expected{{11, 12, 13, 15},
                                                                {true, true, true, true}};
  cudf::test::strings_column_wrapper col2_expected{{"Hi", "Hello", "No", "Naive"},
                                                   {true, true, true, true}};
  cudf::table_view expected{{col1_expected, col2_expected}};

  auto got = cudf::drop_nulls(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}

template <typename T>
struct DropNullsTestAll : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(DropNullsTestAll, cudf::test::NumericTypes);

TYPED_TEST(DropNullsTestAll, AllNull)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T> key_col{{true, false, true, false, true, false},
                                                    {0, 0, 0, 0, 0, 0}};
  cudf::test::fixed_width_column_wrapper<T> col{{true, false, true, false, true, false},
                                                {1, 1, 1, 1, 1, 1}};
  cudf::table_view input{{key_col, col}};
  std::vector<cudf::size_type> keys{0};
  cudf::test::fixed_width_column_wrapper<T> expected_col{};
  cudf::column_view view = expected_col;
  cudf::table_view expected{{expected_col, expected_col}};

  auto got = cudf::drop_nulls(input, keys);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got->view());
}
