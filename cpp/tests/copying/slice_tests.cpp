/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/copying/slice_tests.cuh>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
struct SliceTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(SliceTest, cudf::test::NumericTypes);

struct SliceZeroColumnTest : public cudf::test::BaseFixture {};

TEST_F(SliceZeroColumnTest, PreservesRowCount)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 10};
  std::vector<cudf::size_type> indices{1, 4, 5, 9};
  auto const result = cudf::slice(input, indices);
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].num_columns(), 0);
  EXPECT_EQ(result[0].num_rows(), 3);
  EXPECT_EQ(result[1].num_columns(), 0);
  EXPECT_EQ(result[1].num_rows(), 4);
}

TEST_F(SliceZeroColumnTest, OutOfBoundsThrows)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 10};
  // end exceeds the row count
  EXPECT_THROW(cudf::slice(input, std::vector<cudf::size_type>{8, 12}), std::out_of_range);
  // end < begin
  EXPECT_THROW(cudf::slice(input, std::vector<cudf::size_type>{5, 3}), std::invalid_argument);
  // negative begin
  EXPECT_THROW(cudf::slice(input, std::vector<cudf::size_type>{-1, 4}), std::out_of_range);
}

TEST_F(SliceZeroColumnTest, EmptyTable)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 0};

  auto const result = cudf::slice(input, std::vector<cudf::size_type>{1, 4, 5, 9});
  ASSERT_EQ(result.size(), 2);
  for (auto const& t : result) {
    EXPECT_EQ(t.num_columns(), 0);
    EXPECT_EQ(t.num_rows(), 0);
  }

  // Unlike a zero-column table with rows, an empty table does not reject out-of-range indices.
  EXPECT_NO_THROW(cudf::slice(input, std::vector<cudf::size_type>{8, 12}));
}

TYPED_TEST(SliceTest, NumericColumnsWithNulls)
{
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, valids);

  std::vector<cudf::size_type> indices{1, 3, 2, 2, 5, 9};
  std::vector<cudf::test::fixed_width_column_wrapper<T>> expected =
    create_expected_columns<T>(indices, true);
  std::vector<cudf::column_view> result = cudf::slice(col, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result[index]);
  }
}

TYPED_TEST(SliceTest, NumericColumnsWithNullsAsColumn)
{
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<T> input = create_fixed_columns<T>(start, size, valids);

  std::vector<cudf::size_type> indices{1, 3, 2, 2, 5, 9};
  std::vector<cudf::test::fixed_width_column_wrapper<T>> expected =
    create_expected_columns<T>(indices, true);
  std::vector<cudf::column_view> result = cudf::slice(input, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    auto col = cudf::column(result[index]);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], col);
  }
}

struct SliceStringTest : public SliceTest<std::string> {};

TEST_F(SliceStringTest, StringWithNulls)
{
  std::vector<std::string> strings{
    "", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::strings_column_wrapper s(strings.begin(), strings.end(), valids);

  std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

  std::vector<cudf::test::strings_column_wrapper> expected =
    create_expected_string_columns(strings, indices, true);
  std::vector<cudf::column_view> result = cudf::slice(s, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result[index]);
  }
}

TEST_F(SliceStringTest, StringWithNullsAsColumn)
{
  std::vector<std::string> strings{
    "", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::strings_column_wrapper s(strings.begin(), strings.end(), valids);

  std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

  std::vector<cudf::test::strings_column_wrapper> expected =
    create_expected_string_columns(strings, indices, true);
  std::vector<cudf::column_view> result = cudf::slice(s, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    // this materializes a column to test slicing + materialization
    auto result_col = cudf::column(result[index]);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result_col);
  }
}

struct SliceListTest : public SliceTest<int> {};

TEST_F(SliceListTest, Lists)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  {
    cudf::test::lists_column_wrapper<int> list{{1, 2, 3},
                                               {4, 5},
                                               {6},
                                               {7, 8},
                                               {9, 10, 11},
                                               LCW{},
                                               LCW{},
                                               {-1, -2, -3, -4, -5},
                                               {-10},
                                               {-100, -200}};

    std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{4, 5}, {6}});
    expected.push_back(LCW{{6}, {7, 8}});
    expected.push_back(
      LCW{{4, 5}, {6}, {7, 8}, {9, 10, 11}, LCW{}, LCW{}, {-1, -2, -3, -4, -5}, {-10}});

    std::vector<cudf::column_view> result = cudf::slice(list, indices);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected[index], result[index]);
    }
  }

  {
    cudf::test::lists_column_wrapper<int> list{{{1, 2, 3}, {4, 5}},
                                               {LCW{}, LCW{}, {7, 8}, LCW{}},
                                               {{{6}}},  // NOLINT
                                               {{7, 8}, {9, 10, 11}, LCW{}},
                                               {LCW{}, {-1, -2, -3, -4, -5}},
                                               {LCW{}},
                                               {{-10}, {-100, -200}}};

    std::vector<cudf::size_type> indices{1, 3, 3, 6};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{LCW{}, LCW{}, {7, 8}, LCW{}}, {{{6}}}});  // NOLINT
    expected.push_back(LCW{{{7, 8}, {9, 10, 11}, LCW{}}, {LCW{}, {-1, -2, -3, -4, -5}}, {LCW{}}});

    std::vector<cudf::column_view> result = cudf::slice(list, indices);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected[index], result[index]);
    }
  }
}

TEST_F(SliceListTest, ListsWithNulls)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  {
    cudf::test::lists_column_wrapper<int> list{{1, 2, 3},
                                               {4, 5},
                                               {6},
                                               {{7, 8}, valids},
                                               {9, 10, 11},
                                               LCW{},
                                               LCW{},
                                               {{-1, -2, -3, -4, -5}, valids},
                                               {-10},
                                               {{-100, -200}, valids}};

    std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{4, 5}, {6}});
    expected.push_back(LCW{{6}, {{7, 8}, valids}});
    expected.push_back(LCW{{4, 5},
                           {6},
                           {{7, 8}, valids},
                           {9, 10, 11},
                           LCW{},
                           LCW{},
                           {{-1, -2, -3, -4, -5}, valids},
                           {-10}});

    std::vector<cudf::column_view> result = cudf::slice(list, indices);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected[index], result[index]);
    }
  }

  {
    cudf::test::lists_column_wrapper<int> list{{{{1, 2, 3}, valids}, {4, 5}},
                                               {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                               {{{6}}},  // NOLINT
                                               {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                               {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                               {LCW{}},
                                               {{-10}, {-100, -200}}};

    std::vector<cudf::size_type> indices{1, 3, 3, 6};

    std::vector<cudf::test::lists_column_wrapper<int>> expected;
    expected.push_back(LCW{{{LCW{}, LCW{}, {7, 8}, LCW{}}, valids}, {{{6}}}});  // NOLINT
    expected.push_back(LCW{{{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                           {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                           {LCW{}}});

    std::vector<cudf::column_view> result = cudf::slice(list, indices);
    EXPECT_EQ(expected.size(), result.size());

    for (unsigned long index = 0; index < result.size(); index++) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected[index], result[index]);
    }
  }
}

struct SliceCornerCases : public SliceTest<int8_t> {};

TEST_F(SliceCornerCases, EmptyColumn)
{
  cudf::column col{};
  std::vector<cudf::size_type> indices{0, 0, 0, 0, 0, 0};

  std::vector<cudf::column_view> result = cudf::slice(col.view(), indices);

  unsigned long expected = 3;

  EXPECT_EQ(expected, result.size());

  auto type_match_count = std::count_if(result.cbegin(), result.cend(), [](auto const& col) {
    return col.type().id() == cudf::type_id::EMPTY;
  });
  EXPECT_EQ(static_cast<std::size_t>(type_match_count), expected);
}

TEST_F(SliceCornerCases, EmptyIndices)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> indices{};

  std::vector<cudf::column_view> result = cudf::slice(col, indices);

  unsigned long expected = 0;

  EXPECT_EQ(expected, result.size());
}

TEST_F(SliceCornerCases, InvalidSetOfIndices)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> indices{11, 12};

  EXPECT_THROW(cudf::slice(col, indices), std::out_of_range);
}

TEST_F(SliceCornerCases, ImproperRange)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> indices{5, 4};

  EXPECT_THROW(cudf::slice(col, indices), std::invalid_argument);
}

TEST_F(SliceCornerCases, NegativeOffset)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> indices{-1, 4};

  EXPECT_THROW(cudf::slice(col, indices), std::out_of_range);
}

template <typename T>
struct SliceTableTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(SliceTableTest, cudf::test::NumericTypes);

TYPED_TEST(SliceTableTest, NumericColumnsWithNulls)
{
  using T = TypeParam;

  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<T>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> indices{1, 3, 2, 2, 5, 9};
  std::vector<cudf::table> expected = create_expected_tables<T>(num_cols, indices, true);

  std::vector<cudf::table_view> result = cudf::slice(src_table, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected[index], result[index]);
  }
}

struct SliceStringTableTest : public SliceTableTest<std::string> {};

TEST_F(SliceStringTableTest, StringWithNulls)
{
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  std::vector<std::vector<std::string>> strings{
    {{"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"},
     {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}}};
  std::array<cudf::test::strings_column_wrapper, 2> sw{
    {{strings[0].begin(), strings[0].end(), valids},
     {strings[1].begin(), strings[1].end(), valids}}};

  std::vector<std::unique_ptr<cudf::column>> scols;
  scols.push_back(sw[0].release());
  scols.push_back(sw[1].release());
  cudf::table src_table(std::move(scols));

  std::vector<cudf::size_type> indices{1, 3, 2, 4, 1, 9};

  std::vector<cudf::table> expected = create_expected_string_tables(strings, indices, true);

  std::vector<cudf::table_view> result = cudf::slice(src_table, indices);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(expected[index], result[index]);
  }
}

struct SliceTableCornerCases : public SliceTableTest<int8_t> {};

TEST_F(SliceTableCornerCases, EmptyTable)
{
  std::vector<cudf::size_type> indices{1, 3, 2, 4, 5, 9};

  cudf::table src_table{};
  std::vector<cudf::table_view> result = cudf::slice(src_table.view(), indices);

  unsigned long expected = 3;

  EXPECT_EQ(expected, result.size());
}

TEST_F(SliceTableCornerCases, EmptyIndices)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);
  std::vector<cudf::size_type> indices{};

  std::vector<cudf::table_view> result = cudf::slice(src_table, indices);

  unsigned long expected = 0;

  EXPECT_EQ(expected, result.size());
}

TEST_F(SliceTableCornerCases, InvalidSetOfIndices)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> indices{11, 12};

  EXPECT_THROW(cudf::slice(src_table, indices), std::out_of_range);
}

TEST_F(SliceTableCornerCases, ImproperRange)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> indices{5, 4};

  EXPECT_THROW(cudf::slice(src_table, indices), std::invalid_argument);
}

TEST_F(SliceTableCornerCases, NegativeOffset)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> indices{-1, 4};

  EXPECT_THROW(cudf::slice(src_table, indices), std::out_of_range);
}

TEST_F(SliceTableCornerCases, MiscOffset)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col2{
    {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
    {true, true, true, true, true, true, true, true, true, true,  true,  true, true,
     true, true, true, true, true, true, true, true, true, true,  true,  true, true,
     true, true, true, true, true, true, true, true, true, false, false, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> col3{
    {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
    {true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     true,
     false,
     false,
     false}};
  std::vector<cudf::size_type> indices{19, 38};
  std::vector<cudf::column_view> result = cudf::slice(col2, indices);
  cudf::column result_column(result[0]);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col3, result_column);
}

TEST_F(SliceTableCornerCases, PreSlicedInputs)
{
  {
    using LCW = cudf::test::lists_column_wrapper<float>;

    cudf::test::fixed_width_column_wrapper<int> a{
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      {true, true, false, true, true, true, false, false, true, false}};

    cudf::test::fixed_width_column_wrapper<int> b{
      {0, -1, -2, -3, -4, -5, -6, -7, -8, -9},
      {false, false, false, false, false, false, false, false, false, false}};

    cudf::test::strings_column_wrapper c{
      {"aa", "b", "", "ccc", "ddd", "e", "ff", "", "", "gggg"},
      {false, false, true, true, false, false, true, true, true, false}};

    std::vector<bool> list_validity{true, false, true, false, true, true, false, false, true, true};
    cudf::test::lists_column_wrapper<float> d{
      {{0, 1}, {2}, {3, 4, 5}, {6}, {7, 7}, {8, 9}, {10, 11}, {12, 13}, {}, {14, 15, 16}},
      list_validity.begin()};

    cudf::table_view t({a, b, c, d});

    auto pre_sliced = cudf::slice(t, {0, 4, 4, 10});

    auto result = cudf::slice(pre_sliced[1], {0, 1, 1, 6});

    cudf::test::fixed_width_column_wrapper<int> e0_a({4}, {true});
    cudf::test::fixed_width_column_wrapper<int> e0_b({-4}, {false});
    cudf::test::strings_column_wrapper e0_c({""}, {false});
    std::vector<bool> e0_list_validity{true};
    cudf::test::lists_column_wrapper<float> e0_d({LCW{7, 7}}, e0_list_validity.begin());
    cudf::table_view expected0({e0_a, e0_b, e0_c, e0_d});
    CUDF_TEST_EXPECT_TABLES_EQUAL(result[0], expected0);

    cudf::test::fixed_width_column_wrapper<int> e1_a{{5, 6, 7, 8, 9},
                                                     {true, false, false, true, false}};
    cudf::test::fixed_width_column_wrapper<int> e1_b{{-5, -6, -7, -8, -9},
                                                     {false, false, false, false, false}};
    cudf::test::strings_column_wrapper e1_c{{"e", "ff", "", "", "gggg"},
                                            {false, true, true, true, false}};
    std::vector<bool> e1_list_validity{true, false, false, true, true};
    cudf::test::lists_column_wrapper<float> e1_d{{{8, 9}, {10, 11}, {12, 13}, {}, {14, 15, 16}},
                                                 e1_list_validity.begin()};
    cudf::table_view expected1({e1_a, e1_b, e1_c, e1_d});
    CUDF_TEST_EXPECT_TABLES_EQUAL(result[1], expected1);
  }
}
