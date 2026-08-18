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
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/iterator>

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<cudf::size_type> splits_to_indices(std::vector<cudf::size_type> splits,
                                               cudf::size_type size)
{
  std::vector<cudf::size_type> indices{0};

  std::for_each(splits.begin(), splits.end(), [&indices](auto split) {
    indices.push_back(split);  // This for end
    indices.push_back(split);  // This for the start
  });

  indices.push_back(size);  // This to include rest of the elements

  return indices;
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns_for_splits(
  std::vector<cudf::size_type> const& splits, cudf::size_type size, bool nullable)
{
  // convert splits to slice indices
  std::vector<cudf::size_type> indices = splits_to_indices(splits, size);
  return create_expected_columns<T>(indices, nullable);
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns_for_splits(
  std::vector<cudf::size_type> const& splits, std::vector<T> const& elements, bool nullable)
{
  // convert splits to slice indices
  std::vector<cudf::size_type> indices = splits_to_indices(splits, elements.size());
  return create_expected_columns<T>(indices, elements.begin(), nullable);
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns_for_splits(
  std::vector<cudf::size_type> const& splits,
  cudf::size_type size,
  std::vector<bool> const& validity)
{
  // convert splits to slice indices
  std::vector<cudf::size_type> indices = splits_to_indices(splits, size);
  return create_expected_columns<T>(indices, validity);
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns_for_splits(
  std::vector<cudf::size_type> const& splits,
  std::vector<T> const& elements,
  std::vector<bool> const& validity)
{
  // convert splits to slice indices
  std::vector<cudf::size_type> indices = splits_to_indices(splits, elements.size());
  return create_expected_columns<T>(indices, elements.begin(), validity);
}

std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns_for_splits(
  std::vector<std::string> strings, std::vector<cudf::size_type> const& splits, bool nullable)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, strings.size());
  return create_expected_string_columns(strings, indices, nullable);
}

std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns_for_splits(
  std::vector<std::string> strings,
  std::vector<cudf::size_type> const& splits,
  std::vector<bool> const& validity)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, strings.size());
  return create_expected_string_columns(strings, indices, validity);
}

std::vector<std::vector<bool>> create_expected_validity(std::vector<cudf::size_type> const& splits,
                                                        std::vector<bool> const& validity)
{
  std::vector<std::vector<bool>> result = {};
  std::vector<cudf::size_type> indices  = splits_to_indices(splits, validity.size());

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    result.emplace_back(validity.begin() + indices[index], validity.begin() + indices[index + 1]);
  }

  return result;
}

template <typename T>
std::vector<cudf::table> create_expected_tables_for_splits(
  cudf::size_type num_cols,
  std::vector<cudf::size_type> const& splits,
  cudf::size_type col_size,
  bool nullable)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, col_size);
  return create_expected_tables<T>(num_cols, indices, nullable);
}

std::vector<cudf::table> create_expected_string_tables_for_splits(
  std::vector<std::vector<std::string>> const strings,
  std::vector<cudf::size_type> const& splits,
  bool nullable)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, strings[0].size());
  return create_expected_string_tables(strings, indices, nullable);
}

std::vector<cudf::table> create_expected_string_tables_for_splits(
  std::vector<std::vector<std::string>> const strings,
  std::vector<std::vector<bool>> const validity,
  std::vector<cudf::size_type> const& splits)
{
  std::vector<cudf::size_type> indices = splits_to_indices(splits, strings[0].size());
  auto ret_cols_0 = create_expected_string_columns(strings[0], indices, validity[0]);
  auto ret_cols_1 = create_expected_string_columns(strings[1], indices, validity[1]);

  std::vector<cudf::table> ret_tables;
  for (std::size_t i = 0; i < ret_cols_0.size(); ++i) {
    std::vector<std::unique_ptr<cudf::column>> scols;
    scols.push_back(ret_cols_0[i].release());
    scols.push_back(ret_cols_1[i].release());
    ret_tables.emplace_back(std::move(scols));
  }
  return ret_tables;
}

template <typename T>
struct SplitTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(SplitTest, cudf::test::NumericTypes);

TYPED_TEST(SplitTest, SplitEndLessThanSize)
{
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, valids);

  std::vector<cudf::size_type> splits{2, 5, 7};
  std::vector<cudf::test::fixed_width_column_wrapper<T>> expected =
    create_expected_columns_for_splits<T>(splits, size, true);
  std::vector<cudf::column_view> result = cudf::split(col, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result[index]);
  }
}

TYPED_TEST(SplitTest, SplitEndToSize)
{
  using T = TypeParam;

  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, valids);

  std::vector<cudf::size_type> splits{2, 5, 10, 10, 10, 10};
  std::vector<cudf::test::fixed_width_column_wrapper<T>> expected =
    create_expected_columns_for_splits<T>(splits, size, true);
  std::vector<cudf::column_view> result = cudf::split(col, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result[index]);
  }
}

// common functions for testing split/contiguous_split
template <typename T, typename SplitFunc, typename CompareFunc>
void split_custom_column(SplitFunc Split,
                         CompareFunc Compare,
                         int size,
                         std::vector<cudf::size_type> const& splits,
                         bool include_validity)
{
  // the intent here is to stress the various boundary conditions in contiguous_split -
  // especially the validity copying code.
  cudf::size_type start = 0;

  srand(824);

  std::vector<std::string> base_strings(
    {"banana", "pear", "apple", "pecans", "vanilla", "cat", "mouse", "green"});
  auto string_randomizer = cuda::transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0},
    [&base_strings](cudf::size_type i) { return base_strings[rand() % base_strings.size()]; });

  auto rvalids = cudf::detail::make_counting_transform_iterator(start, [include_validity](auto i) {
    return include_validity
             ? (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < 0.5f ? 0 : 1)
             : 0;
  });

  std::vector<bool> valids{rvalids, rvalids + size};
  cudf::test::fixed_width_column_wrapper<T> col =
    create_fixed_columns<T>(start, size, valids.begin());

  std::vector<bool> valids2{rvalids, rvalids + size};
  std::vector<std::string> strings(string_randomizer, string_randomizer + size);
  cudf::test::strings_column_wrapper col2(strings.begin(), strings.end(), valids2.begin());

  std::vector<cudf::table_view> expected;
  std::vector<cudf::test::fixed_width_column_wrapper<T>> expected_fixed =
    create_expected_columns_for_splits<T>(splits, size, valids);
  std::vector<cudf::test::strings_column_wrapper> expected_strings =
    create_expected_string_columns_for_splits(strings, splits, valids2);
  std::transform(cuda::counting_iterator{static_cast<size_t>(0)},
                 cuda::counting_iterator{expected_fixed.size()},
                 std::back_inserter(expected),
                 [&expected_fixed, &expected_strings](size_t i) {
                   return cudf::table_view({expected_fixed[i], expected_strings[i]});
                 });

  cudf::table_view tbl({col, col2});
  auto result = Split(tbl, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    Compare(expected[index], result[index]);
  }
}

TYPED_TEST(SplitTest, LongColumn)
{
  split_custom_column<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::table_view const& result) {
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{expected.num_columns()},
                    [&expected, &result](cudf::size_type i) {
                      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i), result.column(i));
                    });
    },
    10002,
    std::vector<cudf::size_type>{
      2, 16, 31, 35, 64, 97, 158, 190, 638, 899, 900, 901, 996, 4200, 7131, 8111},
    true);

  split_custom_column<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::table_view const& result) {
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{expected.num_columns()},
                    [&expected, &result](cudf::size_type i) {
                      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i), result.column(i));
                    });
    },
    10002,
    std::vector<cudf::size_type>{
      2, 16, 31, 35, 64, 97, 158, 190, 638, 899, 900, 901, 996, 4200, 7131, 8111},
    false);
}

struct SplitStringTest : public SplitTest<std::string> {};

TEST_F(SplitStringTest, StringWithInvalids)
{
  std::vector<std::string> strings{
    "", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::strings_column_wrapper s(strings.begin(), strings.end(), valids);

  std::vector<cudf::size_type> splits{2, 5, 9};

  std::vector<cudf::test::strings_column_wrapper> expected =
    create_expected_string_columns_for_splits(strings, splits, true);
  std::vector<cudf::column_view> result = cudf::split(s, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[index], result[index]);
  }
}

struct SplitCornerCases : public SplitTest<int8_t> {};

TEST_F(SplitCornerCases, EmptyColumn)
{
  cudf::column col{};
  std::vector<cudf::size_type> splits{2, 5, 9};

  std::vector<cudf::column_view> result = cudf::split(col.view(), splits);

  unsigned long expected = 1;

  EXPECT_EQ(expected, result.size());
}

TEST_F(SplitCornerCases, EmptyIndices)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> splits{};

  std::vector<cudf::column_view> result = cudf::split(col, splits);

  unsigned long expected = 1;

  EXPECT_EQ(expected, result.size());
}

TEST_F(SplitCornerCases, InvalidSetOfIndices)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> splits{11, 12};

  EXPECT_THROW(cudf::split(col, splits), std::out_of_range);
}

TEST_F(SplitCornerCases, ImproperRange)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> splits{5, 4};

  EXPECT_THROW(cudf::split(col, splits), std::invalid_argument);
}

TEST_F(SplitCornerCases, NegativeValue)
{
  cudf::size_type start = 0;
  cudf::size_type size  = 10;
  auto valids           = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::test::fixed_width_column_wrapper<int8_t> col =
    create_fixed_columns<int8_t>(start, size, valids);
  std::vector<cudf::size_type> splits{-1, 4};

  EXPECT_THROW(cudf::split(col, splits), std::invalid_argument);
}

// common functions for testing split/contiguous_split
template <typename T, typename SplitFunc, typename CompareFunc>
void split_end_less_than_size(SplitFunc Split, CompareFunc Compare)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<T>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{2, 5, 7};
  std::vector<cudf::table> expected =
    create_expected_tables_for_splits<T>(num_cols, splits, col_size, true);

  auto result = Split(src_table, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    Compare(expected[index], result[index]);
  }
}

template <typename T, typename SplitFunc, typename CompareFunc>
void split_end_to_size(SplitFunc Split, CompareFunc Compare)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<T>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{2, 5, 10};
  std::vector<cudf::table> expected =
    create_expected_tables_for_splits<T>(num_cols, splits, col_size, true);

  auto result = Split(src_table, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    Compare(expected[index], result[index]);
  }
}

template <typename SplitFunc>
void split_empty_table(SplitFunc Split, std::vector<cudf::size_type> const& splits = {2, 5, 6})
{
  cudf::table src_table{};
  auto result = Split(src_table, splits);

  unsigned long expected = 0;

  EXPECT_EQ(expected, result.size());
}

template <typename SplitFunc>
void split_empty_indices(SplitFunc Split)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);
  std::vector<cudf::size_type> splits{};

  auto result = Split(src_table, splits);

  unsigned long expected = 1;

  EXPECT_EQ(expected, result.size());
}

template <typename SplitFunc>
void split_invalid_indices(SplitFunc Split)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{11, 12};

  EXPECT_THROW(Split(src_table, splits), std::out_of_range);
}

template <typename SplitFunc>
void split_improper_range(SplitFunc Split)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{5, 4};

  EXPECT_THROW(Split(src_table, splits), std::invalid_argument);
}

template <typename SplitFunc>
void split_negative_value(SplitFunc Split)
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  std::vector<cudf::size_type> splits{-1, 4};

  EXPECT_THROW(Split(src_table, splits), std::invalid_argument);
}

template <typename SplitFunc, typename CompareFunc>
void split_empty_output_column_value(SplitFunc Split,
                                     CompareFunc Compare,
                                     std::vector<cudf::size_type> const& splits = {0, 2, 2})
{
  cudf::size_type start    = 0;
  cudf::size_type col_size = 10;
  auto valids              = cudf::test::iterators::valids_at_multiples_of(2);

  cudf::size_type num_cols = 5;
  cudf::table src_table    = create_fixed_table<int8_t>(num_cols, start, col_size, valids);

  EXPECT_NO_THROW(Split(src_table, splits));

  auto result = Split(src_table, splits);
  EXPECT_NO_THROW(Compare(result[0], num_cols));
}

// regular splits
template <typename T>
struct SplitTableTest : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(SplitTableTest, cudf::test::NumericTypes);

TYPED_TEST(SplitTableTest, SplitEndLessThanSize)
{
  split_end_less_than_size<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::table_view const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result);
    });
}

TYPED_TEST(SplitTableTest, SplitEndToSize)
{
  split_end_to_size<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::table_view const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result);
    });
}

struct SplitTableCornerCases : public SplitTest<int8_t> {};

TEST_F(SplitTableCornerCases, EmptyTable)
{
  split_empty_table([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::split(t, splits);
  });
}

TEST_F(SplitTableCornerCases, EmptyIndices)
{
  split_empty_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::split(t, splits);
  });
}

TEST_F(SplitTableCornerCases, InvalidSetOfIndices)
{
  split_invalid_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::split(t, splits);
  });
}

TEST_F(SplitTableCornerCases, ImproperRange)
{
  split_improper_range([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::split(t, splits);
  });
}

TEST_F(SplitTableCornerCases, NegativeValue)
{
  split_negative_value([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::split(t, splits);
  });
}

TEST_F(SplitTableCornerCases, EmptyOutputColumn)
{
  split_empty_output_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& t, int num_cols) { EXPECT_EQ(t.num_columns(), num_cols); });
}

template <typename SplitFunc, typename CompareFunc>
void split_string_with_invalids(SplitFunc Split,
                                CompareFunc Compare,
                                std::vector<cudf::size_type> splits = {2, 5, 9})
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

  std::vector<cudf::table> expected =
    create_expected_string_tables_for_splits(strings, splits, true);

  auto result = Split(src_table, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    Compare(expected[index], result[index]);
  }
}

template <typename SplitFunc, typename CompareFunc>
void split_empty_output_strings_column_value(SplitFunc Split,
                                             CompareFunc Compare,
                                             std::vector<cudf::size_type> const& splits = {0, 2, 2})
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

  cudf::size_type num_cols = 2;

  EXPECT_NO_THROW(Split(src_table, splits));

  auto result = Split(src_table, splits);
  EXPECT_NO_THROW(Compare(result[0], num_cols));
}

template <typename SplitFunc, typename CompareFunc>
void split_null_input_strings_column_value(SplitFunc Split, CompareFunc Compare)
{
  auto valids    = cudf::test::iterators::valids_at_multiples_of(2);
  auto no_valids = cudf::test::iterators::all_nulls();

  std::vector<std::vector<std::string>> strings{
    {{"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"},
     {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}}};

  std::vector<cudf::size_type> splits{2, 5, 9};

  {
    cudf::test::strings_column_wrapper empty_str_col{
      strings[0].begin(), strings[0].end(), no_valids};
    std::vector<std::unique_ptr<cudf::column>> scols;
    scols.push_back(empty_str_col.release());
    cudf::table empty_table(std::move(scols));
    EXPECT_NO_THROW(Split(empty_table, splits));
  }

  std::array<cudf::test::strings_column_wrapper, 2> sw{
    {{strings[0].begin(), strings[0].end(), no_valids},
     {strings[1].begin(), strings[1].end(), valids}}};
  std::vector<std::unique_ptr<cudf::column>> scols;
  scols.push_back(sw[0].release());
  scols.push_back(sw[1].release());
  cudf::table src_table(std::move(scols));
  auto result = Split(src_table, splits);

  std::vector<std::vector<bool>> validity_masks{std::vector<bool>(strings[0].size()),
                                                std::vector<bool>(strings[0].size())};
  std::generate(
    validity_masks[1].begin(), validity_masks[1].end(), [i = 0]() mutable { return i++ % 2 == 0; });

  auto expected = create_expected_string_tables_for_splits(strings, validity_masks, splits);

  for (std::size_t i = 0; i < result.size(); ++i) {
    Compare(expected[i], result[i]);
  }
}

// split with strings
struct SplitStringTableTest : public SplitTest<std::string> {};

TEST_F(SplitStringTableTest, StringWithInvalids)
{
  split_string_with_invalids(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::table_view const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result);
    });
}

TEST_F(SplitStringTableTest, EmptyOutputColumn)
{
  split_empty_output_strings_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table_view const& t, int num_cols) { EXPECT_EQ(t.num_columns(), num_cols); });
}

TEST_F(SplitStringTableTest, NullStringColumn)
{
  split_null_input_strings_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::table const& expected, cudf::table_view const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected.view(), result);
    });
}

struct SplitNestedTypesTest : public cudf::test::BaseFixture {};

// common functions for testing split/contiguous_split
template <typename T, typename SplitFunc, typename CompareFunc>
void split_lists(SplitFunc Split, CompareFunc Compare, bool split = true)
{
  using LCW = cudf::test::lists_column_wrapper<T>;

  {
    cudf::test::lists_column_wrapper<T> list{{1, 2, 3},
                                             {4, 5},
                                             {6},
                                             {7, 8},
                                             {9, 10, 11},
                                             LCW{},
                                             LCW{},
                                             {-1, -2, -3, -4, -5},
                                             {-10},
                                             {-100, -200}};

    if (split) {
      std::vector<cudf::size_type> splits{0, 1, 4, 5, 6, 9};

      std::vector<cudf::test::lists_column_wrapper<T>> expected;
      expected.push_back(LCW{});
      expected.push_back(LCW{{1, 2, 3}});
      expected.push_back(LCW{{4, 5}, {6}, {7, 8}});
      expected.push_back(LCW{{9, 10, 11}});
      expected.push_back(LCW{LCW{}});
      expected.push_back(LCW{LCW{}, {-1, -2, -3, -4, -5}, {-10}});
      expected.push_back(LCW{{-100, -200}});

      auto result = Split(list, splits);
      EXPECT_EQ(expected.size(), result.size());

      for (unsigned long index = 0; index < result.size(); index++) {
        Compare(expected[index], result[index]);
      }
    } else {
      auto result = Split(list, {});
      EXPECT_EQ(1, result.size());
      Compare(list, result[0]);
    }
  }

  {
    cudf::test::lists_column_wrapper<T> list{{{1, 2, 3}, {4, 5}},
                                             {LCW{}, LCW{}, {7, 8}, LCW{}},
                                             {LCW{6}},
                                             {{7, 8}, {9, 10, 11}, LCW{}},
                                             {LCW{}, {-1, -2, -3, -4, -5}},
                                             {LCW{}},
                                             {{-10}, {-100, -200}}};

    if (split) {
      std::vector<cudf::size_type> splits{1, 3, 4};

      std::vector<cudf::test::lists_column_wrapper<T>> expected;
      expected.push_back(LCW{{{1, 2, 3}, {4, 5}}});
      expected.push_back(LCW{{LCW{}, LCW{}, {7, 8}, LCW{}}, {LCW{6}}});
      expected.push_back(LCW{{{7, 8}, {9, 10, 11}, LCW{}}});
      expected.push_back(LCW{{LCW{}, {-1, -2, -3, -4, -5}}, {LCW{}}, {{-10}, {-100, -200}}});

      auto result = Split(list, splits);
      EXPECT_EQ(expected.size(), result.size());

      for (unsigned long index = 0; index < result.size(); index++) {
        Compare(expected[index], result[index]);
      }
    } else {
      auto result = Split(list, {});
      EXPECT_EQ(1, result.size());
      Compare(list, result[0]);
    }
  }
}

template <typename T, typename SplitFunc, typename CompareFunc>
void split_lists_with_nulls(SplitFunc Split, CompareFunc Compare, bool split = true)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  {
    cudf::test::lists_column_wrapper<T> list{{1, 2, 3},
                                             {4, 5},
                                             {6},
                                             {{7, 8}, valids},
                                             {9, 10, 11},
                                             LCW{},
                                             LCW{},
                                             {{-1, -2, -3, -4, -5}, valids},
                                             {-10},
                                             {{-100, -200}, valids}};

    if (split) {
      std::vector<cudf::size_type> splits{0, 1, 4, 5, 6, 9};

      std::vector<cudf::test::lists_column_wrapper<T>> expected;
      expected.push_back(LCW{});
      expected.push_back(LCW{{1, 2, 3}});
      expected.push_back(LCW{{4, 5}, {6}, {{7, 8}, valids}});
      expected.push_back(LCW{{9, 10, 11}});
      expected.push_back(LCW{LCW{}});
      expected.push_back(LCW{LCW{}, {{-1, -2, -3, -4, -5}, valids}, {-10}});
      expected.push_back(LCW{{{-100, -200}, valids}});

      auto result = Split(list, splits);
      EXPECT_EQ(expected.size(), result.size());

      for (unsigned long index = 0; index < result.size(); index++) {
        Compare(expected[index], result[index]);
      }
    } else {
      auto result = Split(list, {});
      EXPECT_EQ(1, result.size());
      Compare(list, result[0]);
    }
  }

  {
    cudf::test::lists_column_wrapper<T> list{{{{1, 2, 3}, valids}, {4, 5}},
                                             {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                             {{{6}}},
                                             {{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids},
                                             {{LCW{}, {-1, -2, -3, -4, -5}}, valids},
                                             {LCW{}},
                                             {{-10}, {-100, -200}}};

    if (split) {
      std::vector<cudf::size_type> splits{1, 3, 4};

      std::vector<cudf::test::lists_column_wrapper<T>> expected;
      expected.push_back(LCW{{{{1, 2, 3}, valids}, {4, 5}}});
      expected.push_back(LCW{{{LCW{}, LCW{}, {7, 8}, LCW{}}, valids}, {{{6}}}});
      expected.push_back(LCW{{{{7, 8}, {{9, 10, 11}, valids}, LCW{}}, valids}});
      expected.push_back(
        LCW{{{LCW{}, {-1, -2, -3, -4, -5}}, valids}, {LCW{}}, {{-10}, {-100, -200}}});

      auto result = Split(list, splits);
      EXPECT_EQ(expected.size(), result.size());

      for (unsigned long index = 0; index < result.size(); index++) {
        Compare(expected[index], result[index]);
      }
    } else {
      auto result = Split(list, {});
      EXPECT_EQ(1, result.size());
      Compare(list, result[0]);
    }
  }
}

template <typename SplitFunc, typename CompareFunc>
void split_structs(bool include_validity, SplitFunc Split, CompareFunc Compare, bool split = true)
{
  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin"};
  std::vector<bool> names_validity{true, true, true, true, true, true, true, true, true};
  cudf::test::strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102};
  std::vector<bool> ages_validity = {true, true, true, true, false, true, false, false, true};
  auto ages_column =
    include_validity
      ? cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin())
      : cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end());

  // 3. Boolean "is_human" column.
  std::vector<bool> is_human{true, true, false, false, false, false, true, true, true};
  std::vector<bool> is_human_validity{true, true, true, false, true, true, true, true, false};
  auto is_human_col =
    include_validity
      ? cudf::test::fixed_width_column_wrapper<bool>(
          is_human.begin(), is_human.end(), is_human_validity.begin())
      : cudf::test::fixed_width_column_wrapper<bool>(is_human.begin(), is_human.end());

  // Assemble struct column.
  auto const struct_validity =
    std::vector<bool>{true, true, true, true, true, false, false, true, false};
  auto struct_column =
    include_validity
      ? cudf::test::structs_column_wrapper({names_column, ages_column, is_human_col},
                                           struct_validity.begin())
      : cudf::test::structs_column_wrapper({names_column, ages_column, is_human_col});

  // split
  std::vector<cudf::size_type> splits;
  if (split) { splits = std::vector<cudf::size_type>({0, 1, 3, 8}); }
  auto result = Split(struct_column, splits);

  // expected outputs
  auto expected_names = include_validity
                          ? create_expected_string_columns_for_splits(names, splits, names_validity)
                          : create_expected_string_columns_for_splits(names, splits, false);
  auto expected_ages  = include_validity
                          ? create_expected_columns_for_splits<int>(splits, ages, ages_validity)
                          : create_expected_columns_for_splits<int>(splits, ages, false);
  auto expected_is_human =
    include_validity ? create_expected_columns_for_splits<bool>(splits, is_human, is_human_validity)
                     : create_expected_columns_for_splits<bool>(splits, is_human, false);

  auto expected_struct_validity = create_expected_validity(splits, struct_validity);

  EXPECT_EQ(expected_names.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    auto expected = include_validity
                      ? cudf::test::structs_column_wrapper(
                          {expected_names[index], expected_ages[index], expected_is_human[index]},
                          expected_struct_validity[index])
                      : cudf::test::structs_column_wrapper(
                          {expected_names[index], expected_ages[index], expected_is_human[index]});

    Compare(expected, result[index]);
  }
}

template <typename SplitFunc, typename CompareFunc>
void split_structs_no_children(SplitFunc Split, CompareFunc Compare, bool split = true)
{
  // no nulls
  {
    auto struct_column = cudf::make_structs_column(4, {}, 0, rmm::device_buffer{});
    if (split) {
      auto expected = cudf::make_structs_column(2, {}, 0, rmm::device_buffer{});

      // split
      std::vector<cudf::size_type> splits{2};
      auto result = Split(*struct_column, splits);

      EXPECT_EQ(result.size(), 2ul);
      Compare(*expected, result[0]);
      Compare(*expected, result[1]);
    } else {
      auto result = Split(*struct_column, {});
      EXPECT_EQ(1, result.size());
      Compare(*struct_column, result[0]);
    }
  }

  // all nulls
  {
    std::vector<bool> struct_validity{false, false, false, false};
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(struct_validity.begin(), struct_validity.end());
    auto struct_column = cudf::make_structs_column(4, {}, null_count, std::move(null_mask));

    if (split) {
      std::vector<bool> expected_validity{false, false};
      std::tie(null_mask, null_count) =
        cudf::test::detail::make_null_mask(expected_validity.begin(), expected_validity.end());
      auto expected = cudf::make_structs_column(2, {}, null_count, std::move(null_mask));

      // split
      std::vector<cudf::size_type> splits{2};
      auto result = Split(*struct_column, splits);

      EXPECT_EQ(result.size(), 2ul);
      Compare(*expected, result[0]);
      Compare(*expected, result[1]);
    } else {
      auto result = Split(*struct_column, {});
      EXPECT_EQ(1, result.size());
      Compare(*struct_column, result[0]);
    }
  }

  // no nulls, empty output column
  {
    auto struct_column = cudf::make_structs_column(4, {}, 0, rmm::device_buffer{});
    if (split) {
      auto expected0 = cudf::make_structs_column(4, {}, 0, rmm::device_buffer{});
      auto expected1 = cudf::make_structs_column(0, {}, 0, rmm::device_buffer{});

      // split
      std::vector<cudf::size_type> splits{4};
      auto result = Split(*struct_column, splits);

      EXPECT_EQ(result.size(), 2ul);
      Compare(*expected0, result[0]);
      Compare(*expected1, result[1]);
    } else {
      auto result = Split(*struct_column, {});
      EXPECT_EQ(1, result.size());
      Compare(*struct_column, result[0]);
    }
  }

  // all nulls, empty output column
  {
    std::vector<bool> struct_validity{false, false, false, false};
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(struct_validity.begin(), struct_validity.end());
    auto struct_column = cudf::make_structs_column(4, {}, null_count, std::move(null_mask));

    if (split) {
      std::vector<bool> expected_validity0{false, false, false, false};
      std::tie(null_mask, null_count) =
        cudf::test::detail::make_null_mask(expected_validity0.begin(), expected_validity0.end());
      auto expected0 = cudf::make_structs_column(4, {}, null_count, std::move(null_mask));

      auto expected1 = cudf::make_structs_column(0, {}, 0, rmm::device_buffer{});

      // split
      std::vector<cudf::size_type> splits{4};
      auto result = Split(*struct_column, splits);

      EXPECT_EQ(result.size(), 2ul);
      Compare(*expected0, result[0]);
      Compare(*expected1, result[1]);
    } else {
      auto result = Split(*struct_column, {});
      EXPECT_EQ(1, result.size());
      Compare(*struct_column, result[0]);
    }
  }
}

template <typename SplitFunc, typename CompareFunc>
void split_nested_struct_of_list(SplitFunc Split, CompareFunc Compare, bool split = true)
{
  // Struct<List<List>>
  using LCW = cudf::test::lists_column_wrapper<float>;

  // 1. String "names" column.
  std::vector<std::string> names{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred", "Todd", "Kevin"};
  std::vector<bool> names_validity{true, true, true, true, true, true, true, true, true};
  cudf::test::strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102};
  std::vector<bool> ages_validity = {true, true, true, true, false, true, false, false, true};
  auto ages_column =
    cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin());

  // 3. List column
  std::vector<bool> list_validity{true, true, true, true, true, false, true, false, true};
  cudf::test::lists_column_wrapper<float> list({{{1, 2, 3}, {4}},
                                                {{-1, -2}, LCW{}},
                                                LCW{},
                                                {{10}, {20, 30, 40}, {100, -100}},
                                                {LCW{}, LCW{}, {8, 9}},
                                                LCW{},
                                                {{8}, {10, 9, 8, 7, 6, 5}},
                                                {{5, 6}, LCW{}, {8}},
                                                {LCW{-3, 4, -5}}},
                                               list_validity.begin());

  // Assemble struct column.
  auto const struct_validity =
    std::vector<bool>{true, true, true, true, true, false, false, true, false};
  auto struct_column =
    cudf::test::structs_column_wrapper({names_column, ages_column, list}, struct_validity.begin());

  if (split) {
    std::vector<cudf::size_type> splits{1, 3, 8};
    auto result = Split(struct_column, splits);
    // expected results
    auto expected_names = create_expected_string_columns_for_splits(names, splits, names_validity);
    auto expected_ages  = create_expected_columns_for_splits<int>(splits, ages, ages_validity);
    std::vector<cudf::test::lists_column_wrapper<float>> expected_lists;
    expected_lists.push_back(LCW({{{1, 2, 3}, {4}}}));
    expected_lists.push_back(LCW({{{-1, -2}, LCW{}}, LCW{}}));
    std::vector<bool> ex_v{true, true, false, true, false};
    expected_lists.push_back(LCW({{{10}, {20, 30, 40}, {100, -100}},
                                  {LCW{}, LCW{}, {8, 9}},
                                  LCW{},
                                  {{8}, {10, 9, 8, 7, 6, 5}},
                                  {{5, 6}, LCW{}, {8}}},
                                 ex_v.begin()));
    expected_lists.push_back(LCW({{LCW{-3, 4, -5}}}));

    auto expected_struct_validity = create_expected_validity(splits, struct_validity);
    EXPECT_EQ(expected_names.size(), result.size());

    for (std::size_t index = 0; index < result.size(); index++) {
      auto expected = cudf::test::structs_column_wrapper(
        {expected_names[index], expected_ages[index], expected_lists[index]},
        expected_struct_validity[index]);
      Compare(expected, result[index]);
    }
  } else {
    auto result = Split(struct_column, {});
    Compare(struct_column, result[0]);
  }
}

template <typename SplitFunc, typename CompareFunc>
void split_nested_list_of_structs(SplitFunc Split, CompareFunc Compare, bool split = true)
{
  // List<Struct<List<>>
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  // 1. String "names" column.
  std::vector<std::string> names{"Vimes",
                                 "Carrot",
                                 "Angua",
                                 "Cheery",
                                 "Detritus",
                                 "Slant",
                                 "Fred",
                                 "Todd",
                                 "Kevin",
                                 "Jason",
                                 "Clark",
                                 "Bob",
                                 "Mithun",
                                 "Sameer",
                                 "Tim",
                                 "Mark",
                                 "Herman",
                                 "Will"};
  std::vector<bool> names_validity{true,
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
                                   true,
                                   true};
  cudf::test::strings_column_wrapper names_column(names.begin(), names.end());

  // 2. Numeric "ages" column.
  std::vector<int> ages{5, 10, 15, 20, 25, 30, 100, 101, 102, 26, 64, 12, 17, 16, 120, 44, 23, 50};
  std::vector<bool> ages_validity = {true,
                                     true,
                                     true,
                                     true,
                                     false,
                                     true,
                                     false,
                                     false,
                                     true,
                                     true,
                                     true,
                                     false,
                                     false,
                                     false,
                                     true,
                                     true,
                                     true,
                                     false};
  auto ages_column =
    cudf::test::fixed_width_column_wrapper<int>(ages.begin(), ages.end(), ages_validity.begin());

  // 3. List column
  std::vector<bool> list_validity{true,
                                  true,
                                  true,
                                  true,
                                  true,
                                  false,
                                  true,
                                  false,
                                  true,
                                  true,
                                  true,
                                  true,
                                  true,
                                  true,
                                  true,
                                  false,
                                  true,
                                  true};
  cudf::test::lists_column_wrapper<cudf::string_view> list(
    {{"ab", "cd", "ef"},
     LCW{"gh"},
     {"ijk", "lmn"},
     LCW{},
     LCW{"o"},
     {"pqr", "stu", "vwx"},
     {"yz", "aaaa"},
     LCW{"bbbb"},
     {"cccc", "ddd", "eee", "fff", "ggg", "hh"},
     {"b", "cdr", "efh", "um"},
     LCW{"gh", "iu"},
     {"lmn"},
     LCW{"org"},
     LCW{},
     {"stu", "vwx"},
     {"yz", "aaaa", "kem"},
     LCW{"bbbb"},
     {"cccc", "eee", "faff", "jiea", "fff", "ggg", "hh"}},
    list_validity.begin());

  // Assembly struct column
  auto const struct_validity = std::vector<bool>{true,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 false,
                                                 false,
                                                 true,
                                                 false,
                                                 false,
                                                 false,
                                                 false,
                                                 true,
                                                 true,
                                                 true,
                                                 true,
                                                 false,
                                                 true};
  auto struct_column =
    cudf::test::structs_column_wrapper({names_column, ages_column, list}, struct_validity.begin());

  // wrap in a list
  std::vector<int> outer_offsets{0, 3, 4, 8, 13, 16, 17, 18};
  cudf::test::fixed_width_column_wrapper<int> outer_offsets_col(outer_offsets.begin(),
                                                                outer_offsets.end());
  std::vector<bool> outer_validity{true, true, true, false, true, true, false};
  auto [outer_null_mask, outer_null_count] =
    cudf::test::detail::make_null_mask(outer_validity.begin(), outer_validity.end());
  auto outer_list = [&] {
    auto tmp = make_lists_column(static_cast<cudf::size_type>(outer_validity.size()),
                                 outer_offsets_col.release(),
                                 struct_column.release(),
                                 outer_null_count,
                                 std::move(outer_null_mask));
    return cudf::purge_nonempty_nulls(tmp->view());
  }();
  if (split) {
    std::vector<cudf::size_type> splits{1, 3, 7};
    cudf::table_view tbl({static_cast<cudf::column_view>(*outer_list)});

    // we are testing the results of contiguous_split against regular cudf::split, which may seem
    // weird. however, cudf::split() is a simple operation that just sets offsets at the topmost
    // output column, whereas contiguous_split is a deep copy of the data to contiguous output
    // buffers. so as long as we believe the comparison code (expect_columns_equivalent) can compare
    // these outputs correctly, this should be safe.
    auto result   = Split(*outer_list, splits);
    auto expected = cudf::split(static_cast<cudf::column_view>(*outer_list), splits);
    ASSERT_EQ(result.size(), expected.size());

    for (std::size_t index = 0; index < result.size(); index++) {
      Compare(expected[index], result[index]);
    }
  } else {
    auto result = Split(*outer_list, {});
    EXPECT_EQ(1, result.size());
    Compare(*outer_list, result[0]);
  }
}

TEST_F(SplitNestedTypesTest, Lists)
{
  split_lists<int>(
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

TEST_F(SplitNestedTypesTest, ListsWithNulls)
{
  split_lists_with_nulls<int>(
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

TEST_F(SplitNestedTypesTest, Structs)
{
  split_structs(
    false,
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

TEST_F(SplitNestedTypesTest, StructsWithNulls)
{
  split_structs(
    true,
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

TEST_F(SplitNestedTypesTest, StructsNoChildren)
{
  split_structs_no_children(
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

TEST_F(SplitNestedTypesTest, StructsOfList)
{
  split_nested_struct_of_list(
    [](cudf::column_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::column_view const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result);
    });
}

template <typename T>
struct ContiguousSplitTest : public cudf::test::BaseFixture {};

std::vector<cudf::packed_table> do_chunked_pack(cudf::table_view const& input)
{
  auto mr = cudf::get_current_device_resource_ref();

  rmm::device_buffer bounce_buff(1 * 1024 * 1024, cudf::get_default_stream(), mr);
  auto bounce_buff_span =
    cudf::device_span<uint8_t>(static_cast<uint8_t*>(bounce_buff.data()), bounce_buff.size());

  auto chunked_pack =
    cudf::chunked_pack::create(input, bounce_buff_span.size(), cudf::get_default_stream(), mr);

  // right size the final buffer
  rmm::device_buffer final_buff(
    chunked_pack->get_total_contiguous_size(), cudf::get_default_stream(), mr);

  std::size_t final_buff_offset = 0;
  while (chunked_pack->has_next()) {
    auto bytes_copied = chunked_pack->next(bounce_buff_span);
    cudaMemcpyAsync((uint8_t*)final_buff.data() + final_buff_offset,
                    bounce_buff.data(),
                    bytes_copied,
                    cudaMemcpyDefault,
                    cudf::get_default_stream());
    final_buff_offset += bytes_copied;
  }

  auto packed_column_metas = chunked_pack->build_metadata();
  // for chunked contig split, this is going to be a size 1 vector if we have
  // results, or a size 0 if the original table was empty (no columns)
  std::vector<cudf::packed_table> result;
  if (packed_column_metas) {
    result  = std::vector<cudf::packed_table>(1);
    auto pc = cudf::packed_columns(std::move(packed_column_metas),
                                   std::make_unique<rmm::device_buffer>(std::move(final_buff)));

    auto unpacked = cudf::unpack(pc);
    cudf::packed_table pt{std::move(unpacked), std::move(pc)};
    result[0] = std::move(pt);
  }
  return result;
}

// the various utility functions in slice_tests.cuh don't like the chrono types
using FixedWidthTypesWithoutChrono =
  cudf::test::Concat<cudf::test::NumericTypes, cudf::test::FixedPointTypes>;

TYPED_TEST_SUITE(ContiguousSplitTest, FixedWidthTypesWithoutChrono);

struct ContiguousSplitZeroColumnTest : public cudf::test::BaseFixture {};

TEST_F(ContiguousSplitZeroColumnTest, PreservesRowCount)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 7};

  // With splits: each partition's row count comes from the split boundaries, and
  // unpacking preserves it.
  auto const split = cudf::contiguous_split(input, {3});
  ASSERT_EQ(split.size(), 2);
  EXPECT_EQ(split[0].table.num_columns(), 0);
  EXPECT_EQ(split[0].table.num_rows(), 3);
  EXPECT_EQ(split[1].table.num_rows(), 4);
  EXPECT_EQ(cudf::unpack(split[0].data).num_rows(), 3);
  EXPECT_EQ(cudf::unpack(split[1].data).num_rows(), 4);

  // No splits: a single partition with all the rows.
  auto const whole = cudf::contiguous_split(input, {});
  ASSERT_EQ(whole.size(), 1);
  EXPECT_EQ(whole[0].table.num_rows(), 7);
  EXPECT_EQ(cudf::unpack(whole[0].data).num_rows(), 7);

  // A genuinely empty (0 columns, 0 rows) table still produces no outputs.
  cudf::table_view empty{std::vector<cudf::column_view>{}, 0};
  EXPECT_TRUE(cudf::contiguous_split(empty, {}).empty());
}

TEST_F(ContiguousSplitZeroColumnTest, InvalidSplitsThrow)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 7};
  EXPECT_THROW(cudf::contiguous_split(input, {8}), std::out_of_range);         // beyond row count
  EXPECT_THROW(cudf::contiguous_split(input, {5, 3}), std::invalid_argument);  // non-monotonic
}

TEST_F(ContiguousSplitZeroColumnTest, ChunkedPackPreservesRowCount)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 7};
  auto mr = cudf::get_current_device_resource_ref();
  auto cp = cudf::chunked_pack::create(input, 1 * 1024 * 1024, cudf::get_default_stream(), mr);
  // No device data to copy: zero total size and no chunks.
  EXPECT_EQ(cp->get_total_contiguous_size(), 0u);
  EXPECT_FALSE(cp->has_next());
  // Metadata is still emitted and unpacks to the original row count.
  auto const metadata = cp->build_metadata();
  ASSERT_NE(metadata, nullptr);
  auto const unpacked = cudf::unpack(metadata->data(), nullptr);
  EXPECT_EQ(unpacked.num_columns(), 0);
  EXPECT_EQ(unpacked.num_rows(), 7);
}

TYPED_TEST(ContiguousSplitTest, LongColumn)
{
  split_custom_column<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{expected.num_columns()},
                    [&expected, &result](cudf::size_type i) {
                      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i),
                                                          result.table.column(i));
                    });
    },
    10002,
    std::vector<cudf::size_type>{
      2, 16, 31, 35, 64, 97, 158, 190, 638, 899, 900, 901, 996, 4200, 7131, 8111},
    true);

  split_custom_column<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{expected.num_columns()},
                    [&expected, &result](cudf::size_type i) {
                      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i),
                                                          result.table.column(i));
                    });
    },
    10002,
    std::vector<cudf::size_type>{
      2, 16, 31, 35, 64, 97, 158, 190, 638, 899, 900, 901, 996, 4200, 7131, 8111},
    false);
}

TYPED_TEST(ContiguousSplitTest, LongColumnChunked)
{
  split_custom_column<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const&) {
      return do_chunked_pack(t);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{expected.num_columns()},
                    [&expected, &result](cudf::size_type i) {
                      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i),
                                                          result.table.column(i));
                    });
    },
    100002,
    {},
    true);

  split_custom_column<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const&) {
      return do_chunked_pack(t);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{expected.num_columns()},
                    [&expected, &result](cudf::size_type i) {
                      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i),
                                                          result.table.column(i));
                    });
    },
    100002,
    {},
    false);
}

TYPED_TEST(ContiguousSplitTest, LongColumnBigSplits)
{
  split_custom_column<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{expected.num_columns()},
                    [&expected, &result](cudf::size_type i) {
                      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i),
                                                          result.table.column(i));
                    });
    },
    10007,
    std::vector<cudf::size_type>{0, 3613, 7777, 10005, 10007},
    true);

  split_custom_column<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{expected.num_columns()},
                    [&expected, &result](cudf::size_type i) {
                      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i),
                                                          result.table.column(i));
                    });
    },
    10007,
    std::vector<cudf::size_type>{0, 3613, 7777, 10005, 10007},
    false);
}

// this is a useful test but a little too expensive to run all the time
/*
TYPED_TEST(ContiguousSplitTest, LongColumnTinySplits)
{
  std::vector<cudf::size_type> splits(cuda::counting_iterator<cudf::size_type>{0},
cuda::counting_iterator{10000});

  split_custom_column<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator{expected.num_columns()},
                    [&expected, &result](cudf::size_type i){

        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i), result.table.column(i));
      });
    },
    10002,
    splits
    );
}
*/
struct ContiguousSplitUntypedTest : public cudf::test::BaseFixture {};

TEST_F(ContiguousSplitUntypedTest, ProgressiveSizes)
{
  constexpr int col_size = 256;

  // stress test copying a wide amount of bytes.
  for (int idx = 0; idx < col_size; idx++) {
    split_custom_column<uint8_t>(
      [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
        return cudf::contiguous_split(t, splits);
      },
      [](cudf::table_view const& expected, cudf::packed_table const& result) {
        std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{expected.num_columns()},
                      [&expected, &result](cudf::size_type i) {
                        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i),
                                                            result.table.column(i));
                      });
      },
      col_size,
      std::vector<cudf::size_type>{idx},
      true);

    split_custom_column<uint8_t>(
      [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
        return cudf::contiguous_split(t, splits);
      },
      [](cudf::table_view const& expected, cudf::packed_table const& result) {
        std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{expected.num_columns()},
                      [&expected, &result](cudf::size_type i) {
                        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i),
                                                            result.table.column(i));
                      });
      },
      col_size,
      std::vector<cudf::size_type>{idx},
      false);
  }
}

TEST_F(ContiguousSplitUntypedTest, ProgressiveSizesChunked)
{
  constexpr int col_size = 4096;

  // stress test copying a wide amount of bytes.
  for (int idx = 2048; idx < col_size; idx += 128) {
    split_custom_column<uint64_t>(
      [](cudf::table_view const& t, std::vector<cudf::size_type> const&) {
        return do_chunked_pack(t);
      },
      [](cudf::table_view const& expected, cudf::packed_table const& result) {
        std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{expected.num_columns()},
                      [&expected, &result](cudf::size_type i) {
                        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i),
                                                            result.table.column(i));
                      });
      },
      col_size,
      {},
      true);

    split_custom_column<uint64_t>(
      [](cudf::table_view const& t, std::vector<cudf::size_type> const&) {
        return do_chunked_pack(t);
      },
      [](cudf::table_view const& expected, cudf::packed_table const& result) {
        std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{expected.num_columns()},
                      [&expected, &result](cudf::size_type i) {
                        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected.column(i),
                                                            result.table.column(i));
                      });
      },
      col_size,
      {},
      false);
  }
}

TEST_F(ContiguousSplitUntypedTest, ValidityRepartition)
{
  // it is tricky to actually get the internal repartitioning/load-balancing code to add new splits
  // inside a validity buffer.  Under almost all situations, the fraction of bytes that validity
  // represents is so small compared to the bytes for all other data, that those buffers end up not
  // getting subdivided. this test forces it happen by using a small, single column of int8's, which
  // keeps the overall fraction that validity takes up large enough to cause a repartition.
  srand(0);
  auto rvalids                   = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < 0.5f ? 0 : 1;
  });
  cudf::size_type const num_rows = 2000000;
  auto col                       = cudf::sequence(num_rows, cudf::numeric_scalar<int8_t>{0});
  auto [null_mask, null_count]   = cudf::test::detail::make_null_mask(rvalids, rvalids + num_rows);
  col->set_null_mask(std::move(null_mask), null_count);

  cudf::table_view t({*col});
  auto result   = cudf::contiguous_split(t, {num_rows / 2});
  auto expected = cudf::split(t, {num_rows / 2});
  ASSERT_EQ(result.size(), expected.size());

  for (size_t idx = 0; idx < result.size(); idx++) {
    CUDF_TEST_EXPECT_TABLES_EQUAL(result[idx].table, expected[idx]);
  }
}

TEST_F(ContiguousSplitUntypedTest, ValidityRepartitionChunked)
{
  srand(0);
  auto rvalids                   = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < 0.5f ? 0 : 1;
  });
  cudf::size_type const num_rows = 2000000;
  auto col                       = cudf::sequence(num_rows, cudf::numeric_scalar<int8_t>{0});
  auto [null_mask, null_count]   = cudf::test::detail::make_null_mask(rvalids, rvalids + num_rows);
  col->set_null_mask(std::move(null_mask), null_count);

  cudf::table_view t({*col});
  auto result    = do_chunked_pack(t);
  auto& expected = t;
  EXPECT_EQ(1, result.size());

  CUDF_TEST_EXPECT_TABLES_EQUAL(result[0].table, expected);
}

TEST_F(ContiguousSplitUntypedTest, ValidityEdgeCase)
{
  // tests an edge case where the splits cause the final validity data to be copied
  // to be < 32 full bits, making sure we don't unintentionally read past the end of the input
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 512, cudf::mask_state::ALL_VALID);
  auto result   = cudf::contiguous_split(cudf::table_view{{*col}}, {510});
  auto expected = cudf::split(cudf::table_view{{*col}}, {510});

  EXPECT_EQ(expected.size(), result.size());
  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected[index].column(0), result[index].table.column(0));
  }
}

// This test requires about 25GB of device memory when used with the arena allocator
TEST_F(ContiguousSplitUntypedTest, DISABLED_VeryLargeColumnTest)
{
  // tests an edge case where buf.elements * buf.element_size overflows an INT32.
  auto col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, 400 * 1024 * 1024, cudf::mask_state::UNALLOCATED);
  auto result = cudf::contiguous_split(cudf::table_view{{*col}}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, result[0].table.column(0));
}

// This test requires about 25GB of device memory when used with the arena allocator
TEST_F(ContiguousSplitUntypedTest, DISABLED_VeryLargeColumnTestChunked)
{
  // tests an edge case where buf.elements * buf.element_size overflows an INT32.
  auto col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, 400 * 1024 * 1024, cudf::mask_state::UNALLOCATED);
  auto result = do_chunked_pack(cudf::table_view{{*col}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, result[0].table.column(0));
}

// Disabled as this test requires about 7GB of device memory.
// See https://github.com/rapidsai/cudf/issues/20876 for more info.
TEST_F(ContiguousSplitUntypedTest, DISABLED_ChunkedPackNextReturnValueOver2GB)
{
  auto const mr = cudf::get_current_device_resource_ref();

  // 270M INT64 elements = 270 * 1024 * 1024 * 8 = 2,264,924,160 bytes > INT32_MAX
  constexpr cudf::size_type num_rows = 270 * 1024 * 1024;
  auto const col                     = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED);

  auto const expected_total_size = static_cast<std::size_t>(num_rows) * sizeof(int64_t);
  EXPECT_GT(expected_total_size, static_cast<std::size_t>(std::numeric_limits<int32_t>::max()));

  auto const tv = cudf::table_view{{*col}};

  // Bounce buffer large enough to hold all data in a single next() call.
  // The bug only manifests when a single next() call returns >2GB.
  auto const bounce_size = expected_total_size + (1024 * 1024);

  auto chunked_packer = cudf::chunked_pack::create(tv, bounce_size, cudf::get_default_stream(), mr);

  EXPECT_EQ(chunked_packer->get_total_contiguous_size(), expected_total_size);
  EXPECT_TRUE(chunked_packer->has_next());

  rmm::device_buffer bounce_buff(bounce_size, cudf::get_default_stream(), mr);
  auto const bounce_span =
    cudf::device_span<uint8_t>(static_cast<uint8_t*>(bounce_buff.data()), bounce_buff.size());

  auto const bytes_copied = chunked_packer->next(bounce_span);
  EXPECT_EQ(bytes_copied, expected_total_size);
  EXPECT_FALSE(chunked_packer->has_next());
}

TEST_F(ContiguousSplitUntypedTest, OffsetAlignment)
{
  // test copying offsets starting a various alignments (0, 4, 8, 12 bytes) and with
  // varying counts (1, 2, 3, 4 elements)
  cudf::test::strings_column_wrapper str{
    "aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh", "iiii", "jjjj", "kkkk", "llll"};

  std::vector<int> row_index_start{4, 5, 6, 7};
  std::vector<int> row_counts{1, 2, 3, 4};

  // 32 bit offsets
  {
    cudf::table_view tbl({str});
    for (size_t ridx = 0; ridx < row_index_start.size(); ridx++) {
      for (size_t cidx = 0; cidx < row_counts.size(); cidx++) {
        std::vector<int> splits{row_index_start[ridx], row_index_start[ridx] + row_counts[cidx]};

        auto res      = cudf::contiguous_split(tbl, splits);
        auto expected = cudf::split(tbl, splits);

        for (size_t idx = 0; idx < expected.size(); idx++) {
          CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[idx], res[idx].table);
        }
      }
    }
  }

  // 64 bit offsets
  {
    // reassemble with 64 bit offsets
    auto contents = str.release()->release();
    cudf::test::fixed_width_column_wrapper<int64_t> long_offsets{
      0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48};
    auto long_str =
      cudf::make_strings_column(12, long_offsets.release(), std::move(*contents.data), 0, {});
    cudf::strings_column_view scv(*long_str);
    CUDF_EXPECTS(scv.offsets().type().id() == cudf::type_id::INT64, "Unexpected short offset type");

    cudf::table_view tbl({*long_str});
    for (size_t ridx = 0; ridx < row_index_start.size(); ridx++) {
      for (size_t cidx = 0; cidx < row_counts.size(); cidx++) {
        std::vector<int> splits{row_index_start[ridx], row_index_start[ridx] + row_counts[cidx]};

        auto res      = cudf::contiguous_split(tbl, splits);
        auto expected = cudf::split(tbl, splits);

        for (size_t idx = 0; idx < expected.size(); idx++) {
          CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[idx], res[idx].table);
        }
      }
    }
  }
}

// contiguous split with strings
struct ContiguousSplitStringTableTest : public SplitTest<std::string> {};

TEST_F(ContiguousSplitStringTableTest, StringWithInvalids)
{
  split_string_with_invalids(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
    });
}

TEST_F(ContiguousSplitStringTableTest, StringWithInvalidsChunked)
{
  split_string_with_invalids(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const&) {
      return do_chunked_pack(t);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
    },
    {});
}

TEST_F(ContiguousSplitStringTableTest, EmptyInputColumn)
{
  // build a bunch of empty stuff
  cudf::test::strings_column_wrapper sw;
  cudf::test::lists_column_wrapper<int> lw;
  cudf::test::fixed_width_column_wrapper<float> fw;
  //
  cudf::test::strings_column_wrapper ssw;
  cudf::test::lists_column_wrapper<int> slw;
  cudf::test::fixed_width_column_wrapper<float> sfw;
  cudf::test::structs_column_wrapper st_w({sfw, ssw, slw});

  cudf::table_view src_table({sw, lw, fw, st_w});

  {
    std::vector<cudf::size_type> splits;
    auto result = cudf::contiguous_split(src_table, splits);
    ASSERT_EQ(result.size(), 1);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(src_table, result[0].table);
  }

  {
    auto result = do_chunked_pack(src_table);
    ASSERT_EQ(result.size(), 1);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(src_table, result[0].table);
  }

  {
    std::vector<cudf::size_type> splits{0, 0, 0, 0};
    auto result = cudf::contiguous_split(src_table, splits);
    ASSERT_EQ(result.size(), 5);

    for (auto& idx : result) {
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(src_table, idx.table);
    }
  }
}

TEST_F(ContiguousSplitStringTableTest, EmptyOutputColumn)
{
  split_empty_output_strings_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::packed_table const& t, int num_cols) { EXPECT_EQ(t.table.num_columns(), num_cols); });
}

TEST_F(ContiguousSplitStringTableTest, EmptyOutputColumnChunked)
{
  split_empty_output_strings_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const&) {
      return do_chunked_pack(t);
    },
    [](cudf::packed_table const& t, int num_cols) { EXPECT_EQ(t.table.num_columns(), num_cols); },
    {});
}

TEST_F(ContiguousSplitStringTableTest, NullStringColumn)
{
  split_null_input_strings_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected.view(), result.table);
    });
}

// contiguous splits
template <typename T>
struct ContiguousSplitTableTest : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(ContiguousSplitTableTest, FixedWidthTypesWithoutChrono);

TYPED_TEST(ContiguousSplitTableTest, SplitEndLessThanSize)
{
  split_end_less_than_size<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
    });
}

TYPED_TEST(ContiguousSplitTableTest, SplitEndToSize)
{
  split_end_to_size<TypeParam>(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::table_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
    });
}

struct ContiguousSplitTableCornerCases : public SplitTest<int8_t> {};

TEST_F(ContiguousSplitTableCornerCases, EmptyTable)
{
  split_empty_table([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::contiguous_split(t, splits);
  });
}

TEST_F(ContiguousSplitTableCornerCases, EmptyTableChunked)
{
  split_empty_table([](cudf::table_view const& t,
                       std::vector<cudf::size_type> const&) { return do_chunked_pack(t); },
                    {});
}

TEST_F(ContiguousSplitTableCornerCases, EmptyIndices)
{
  split_empty_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::contiguous_split(t, splits);
  });
}

TEST_F(ContiguousSplitTableCornerCases, InvalidSetOfIndices)
{
  split_invalid_indices([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::contiguous_split(t, splits);
  });
}

TEST_F(ContiguousSplitTableCornerCases, ImproperRange)
{
  split_improper_range([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::contiguous_split(t, splits);
  });
}

TEST_F(ContiguousSplitTableCornerCases, NegativeValue)
{
  split_negative_value([](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
    return cudf::contiguous_split(t, splits);
  });
}

TEST_F(ContiguousSplitTableCornerCases, EmptyOutputColumn)
{
  split_empty_output_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const& splits) {
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::packed_table const& t, int num_cols) { EXPECT_EQ(t.table.num_columns(), num_cols); });
}

TEST_F(ContiguousSplitTableCornerCases, EmptyOutputColumnChunked)
{
  split_empty_output_column_value(
    [](cudf::table_view const& t, std::vector<cudf::size_type> const&) {
      return do_chunked_pack(t);
    },
    [](cudf::packed_table const& t, int num_cols) { EXPECT_EQ(t.table.num_columns(), num_cols); },
    {});
}

TEST_F(ContiguousSplitTableCornerCases, MixedColumnTypes)
{
  cudf::size_type start = 0;
  auto valids = cudf::detail::make_counting_transform_iterator(start, [](auto i) { return true; });

  std::vector<std::vector<std::string>> strings{
    {{"", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"},
     {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}}};

  std::vector<std::unique_ptr<cudf::column>> cols;

  auto iter0 = cuda::counting_iterator{0};
  auto c0    = cudf::test::fixed_width_column_wrapper<int>(iter0, iter0 + 10, valids);
  cols.push_back(c0.release());

  auto iter1 = cuda::counting_iterator{10};
  auto c1    = cudf::test::fixed_width_column_wrapper<int>(iter1, iter1 + 10, valids);
  cols.push_back(c1.release());

  auto c2 = cudf::test::strings_column_wrapper(strings[0].begin(), strings[0].end(), valids);
  cols.push_back(c2.release());

  auto c3 = cudf::test::strings_column_wrapper(strings[1].begin(), strings[1].end(), valids);
  cols.push_back(c3.release());

  auto iter4 = cuda::counting_iterator{20};
  auto c4    = cudf::test::fixed_width_column_wrapper<int>(iter4, iter4 + 10, valids);
  cols.push_back(c4.release());

  auto tbl = cudf::table(std::move(cols));

  std::vector<cudf::size_type> splits{5};

  auto result   = cudf::contiguous_split(tbl, splits);
  auto expected = cudf::split(tbl, splits);

  for (unsigned long index = 0; index < expected.size(); index++) {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[index], result[index].table);
  }
}

TEST_F(ContiguousSplitTableCornerCases, MixedColumnTypesChunked)
{
  cudf::size_type start = 0;
  auto valids = cudf::detail::make_counting_transform_iterator(start, [](auto i) { return true; });

  std::size_t num_rows = 1000000;

  std::vector<std::string> strings1(num_rows);
  std::vector<std::string> strings2(num_rows);
  strings1[0] = "";
  strings2[0] = "";
  for (std::size_t i = 1; i < num_rows; ++i) {
    auto str    = std::to_string(i);
    strings1[i] = str;
    strings2[i] = str;
  }

  std::vector<std::unique_ptr<cudf::column>> cols;

  auto iter0 = cuda::counting_iterator{0};
  auto c0    = cudf::test::fixed_width_column_wrapper<int>(iter0, iter0 + num_rows, valids);
  cols.push_back(c0.release());

  auto iter1 = cudf::detail::make_counting_transform_iterator(10, [](auto i) { return (i); });
  auto c1    = cudf::test::fixed_width_column_wrapper<int>(iter1, iter1 + num_rows, valids);
  cols.push_back(c1.release());

  auto c2 = cudf::test::strings_column_wrapper(strings1.begin(), strings1.end(), valids);
  cols.push_back(c2.release());

  auto c3 = cudf::test::strings_column_wrapper(strings2.begin(), strings2.end(), valids);
  cols.push_back(c3.release());

  auto iter4 = cudf::detail::make_counting_transform_iterator(20, [](auto i) { return (i); });
  auto c4    = cudf::test::fixed_width_column_wrapper<int>(iter4, iter4 + num_rows, valids);
  cols.push_back(c4.release());

  auto tbl     = cudf::table(std::move(cols));
  auto results = do_chunked_pack(tbl.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(tbl, results[0].table);
}

TEST_F(ContiguousSplitTableCornerCases, MixedColumnTypesSingleRowChunked)
{
  cudf::size_type start = 0;
  auto valids = cudf::detail::make_counting_transform_iterator(start, [](auto i) { return true; });

  std::size_t num_rows = 1;

  std::vector<std::unique_ptr<cudf::column>> cols;

  auto iter0 = cuda::counting_iterator{0};
  auto c0    = cudf::test::fixed_width_column_wrapper<int32_t>(iter0, iter0 + num_rows, valids);
  cols.push_back(c0.release());

  auto iter1 = cudf::detail::make_counting_transform_iterator(1, [](auto i) { return (i); });
  auto c1    = cudf::test::fixed_width_column_wrapper<int64_t>(iter1, iter1 + num_rows);
  cols.push_back(c1.release());

  auto tbl     = cudf::table(std::move(cols));
  auto results = do_chunked_pack(tbl.view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(tbl, results[0].table);
}

TEST_F(ContiguousSplitTableCornerCases, PreSplitTable)
{
  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  using LCW = cudf::test::lists_column_wrapper<int>;

  cudf::test::lists_column_wrapper<int> col0{{{1, 2, 3}, {4, 5}},
                                             {{LCW{}, LCW{}, {7, 8}, LCW{}}, valids},
                                             {{{6}}},  // NOLINT
                                             {{{7, 8}, LCW{}, {{9, 10, 11}, valids}}, valids},
                                             {{{-1, -2, -3, -4, -5}, LCW{}}, valids},
                                             {LCW{}},
                                             {{-10}, {-100, -200}}};

  cudf::test::strings_column_wrapper col1{
    "Vimes", "Carrot", "Angua", "Cheery", "Detritus", "Slant", "Fred"};
  cudf::test::fixed_width_column_wrapper<float> col2{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::make_unique<cudf::column>(col2));
  children.push_back(std::make_unique<cudf::column>(col0));
  children.push_back(std::make_unique<cudf::column>(col1));
  auto col3 = cudf::make_structs_column(
    static_cast<cudf::column_view>(col0).size(), std::move(children), 0, rmm::device_buffer{});

  cudf::table_view t({col0, col1, col2, *col3});
  auto pre_split = cudf::split(t, {1});

  {
    std::vector<cudf::size_type> splits{1, 4};

    auto result   = cudf::contiguous_split(pre_split[1], splits);
    auto expected = cudf::split(pre_split[1], splits);

    for (size_t index = 0; index < expected.size(); index++) {
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[index], result[index].table);
    }
  }

  {
    std::vector<cudf::size_type> splits{0, 5};

    auto result   = cudf::contiguous_split(pre_split[1], splits);
    auto expected = cudf::split(pre_split[1], splits);

    for (size_t index = 0; index < expected.size(); index++) {
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[index], result[index].table);
    }
  }

  {
    auto result = do_chunked_pack(pre_split[1]);
    EXPECT_EQ(1, result.size());
    auto expected = pre_split[1];
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected, result[0].table);
  }
}

TEST_F(ContiguousSplitTableCornerCases, PreSplitTableLarge)
{
  // test splitting a table that is already split (has an offset)
  cudf::size_type start        = 0;
  cudf::size_type presplit_pos = 47;
  cudf::size_type size         = 10002;

  srand(824);
  auto rvalids = cudf::detail::make_counting_transform_iterator(start, [](auto i) {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < 0.5f ? 0 : 1;
  });

  std::vector<bool> pre_split_valids{rvalids, rvalids + size};
  cudf::test::fixed_width_column_wrapper<int> pre_split =
    create_fixed_columns<int>(start, size, pre_split_valids.begin());

  // pre-split this column
  auto split_cols = cudf::split(pre_split, {47});

  std::vector<cudf::size_type> splits{
    2, 16, 31, 35, 64, 97, 158, 190, 638, 899, 900, 901, 996, 4200, 7131, 8111};

  auto const post_split_start = start + presplit_pos;
  auto const post_split_size  = size - presplit_pos;
  auto el_iter                = cuda::counting_iterator{post_split_start};
  std::vector<int> post_split_elements{el_iter, el_iter + post_split_size};
  std::vector<bool> post_split_valids{
    pre_split_valids.begin() + post_split_start,
    pre_split_valids.begin() + post_split_start + post_split_size};

  std::vector<cudf::test::fixed_width_column_wrapper<int>> expected =
    create_expected_columns_for_splits<int>(splits, post_split_elements, post_split_valids);

  cudf::table_view t({split_cols[1]});
  auto result = cudf::contiguous_split(t, splits);

  EXPECT_EQ(expected.size(), result.size());

  for (unsigned long index = 0; index < result.size(); index++) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected[index], result[index].table.column(0));
  }
}

TEST_F(ContiguousSplitTableCornerCases, PreSplitList)
{
  // list<list<int>>
  {
    cudf::test::lists_column_wrapper<int> list{{{1, 2}, {3, 4}},
                                               {{5, 6}, {7}, {8, 9, 10}},
                                               {{11, 12}, {13}},
                                               {{14, 15, 16}, {17, 18}, {}},
                                               {{-1, -2, -3}, {-4, -5, -6, -7}},
                                               {{-8, -9}, {-10, -11}},
                                               {{-12, -13}, {-14}, {-15, -16}},
                                               {{-17, -18}, {}, {-19, -20}}};
    auto pre_split = cudf::split(list, {2});

    cudf::table_view t({pre_split[1]});
    auto result   = cudf::contiguous_split(t, {3, 4});
    auto expected = cudf::split(t, {3, 4});

    auto iter = cuda::counting_iterator<cudf::size_type>{0};
    std::for_each(iter, iter + expected.size(), [&](cudf::size_type index) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(result[index].table, expected[index]);
    });
  }

  // list<struct<float>>
  {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, 2, 5, 7, 10, 12, 14, 17, 20};
    cudf::test::fixed_width_column_wrapper<float> floats{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    cudf::test::structs_column_wrapper data({floats});

    auto list =
      cudf::make_lists_column(8, offsets.release(), data.release(), 0, rmm::device_buffer{});

    auto pre_split = cudf::split(*list, {2});

    cudf::table_view t({pre_split[1]});
    auto result   = cudf::contiguous_split(t, {3, 4});
    auto expected = cudf::split(t, {3, 4});

    auto iter = cuda::counting_iterator<cudf::size_type>{0};
    std::for_each(iter, iter + expected.size(), [&](cudf::size_type index) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(result[index].table, expected[index]);
    });
  }
}

TEST_F(ContiguousSplitTableCornerCases, PreSplitStructs)
{
  // includes struct<list>
  {
    cudf::test::fixed_width_column_wrapper<int> a{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    cudf::test::fixed_width_column_wrapper<float> b{
      {0, -1, -2, -3, -4, -5, -6, -7, -8, -9},
      {true, true, true, false, false, false, false, true, true, true}};
    cudf::test::strings_column_wrapper c{
      {"abc", "def", "ghi", "jkl", "mno", "", "st", "uvwx", "yy", "zzzz"},
      {false, false, true, true, true, true, true, true, true, true}};
    std::vector<bool> list_validity{true, false, true, false, true, false, true, true, true, true};
    cudf::test::lists_column_wrapper<int16_t> d{
      {{0, 1}, {2, 3, 4}, {5, 6}, {7}, {8, 9, 10}, {11, 12}, {}, {15, 16, 17}, {18, 19}, {20}},
      list_validity.begin()};
    cudf::test::fixed_width_column_wrapper<int> _a{10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    cudf::test::fixed_width_column_wrapper<float> _b{
      -10, -20, -30, -40, -50, -60, -70, -80, -90, -100};
    cudf::test::strings_column_wrapper _c{
      "aa", "", "ccc", "dddd", "eeeee", "f", "gg", "hhh", "i", "jjj"};
    cudf::test::structs_column_wrapper e(
      {_a, _b, _c}, {true, true, true, false, true, true, true, false, true, true});
    cudf::test::structs_column_wrapper s(
      {a, b, c, d, e}, {true, true, false, true, true, true, true, true, true, true});

    auto pre_split = cudf::split(s, {4});

    auto iter = cuda::counting_iterator<cudf::size_type>{0};
    std::for_each(iter, iter + pre_split.size(), [&](cudf::size_type index) {
      cudf::table_view t({pre_split[index]});
      auto result   = cudf::contiguous_split(t, {1});
      auto expected = cudf::split(t, {1});

      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result[0].table, expected[0]);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result[1].table, expected[1]);
    });
  }

  // struct<list<struct>>
  {
    cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, 2, 5, 7, 10, 12, 14, 17, 20};
    cudf::test::fixed_width_column_wrapper<float> floats{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    cudf::test::structs_column_wrapper data({floats});
    auto list =
      cudf::make_lists_column(8, offsets.release(), data.release(), 0, rmm::device_buffer{});
    cudf::test::strings_column_wrapper strings{"a", "bb", "ccc", "dddd", "", "e", "ff", "ggg"};

    std::vector<std::unique_ptr<cudf::column>> struct_children;
    struct_children.push_back(std::move(list));
    struct_children.push_back(strings.release());
    cudf::test::structs_column_wrapper col(std::move(struct_children));

    auto pre_split = cudf::split(col, {2});

    cudf::table_view t({pre_split[1]});
    auto result   = cudf::contiguous_split(t, {3, 4});
    auto expected = cudf::split(t, {3, 4});

    auto iter = cuda::counting_iterator<cudf::size_type>{0};
    std::for_each(iter, iter + expected.size(), [&](cudf::size_type index) {
      CUDF_TEST_EXPECT_TABLES_EQUAL(result[index].table, expected[index]);
    });
  }
}

TEST_F(ContiguousSplitTableCornerCases, NestedEmpty)
{
  // this produces an empty strings column with no children,
  // nested inside a list
  {
    auto empty_string = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    auto offsets      = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list         = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_string), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});

    std::vector<cudf::size_type> splits({0});
    EXPECT_NO_THROW(contiguous_split(src_table, splits));

    std::vector<cudf::size_type> splits2({1});
    EXPECT_NO_THROW(contiguous_split(src_table, splits2));

    EXPECT_NO_THROW(do_chunked_pack(src_table));
  }

  // this produces an empty strings column with children that have no data,
  // nested inside a list
  {
    cudf::test::strings_column_wrapper str{"abc"};
    auto empty_string = cudf::empty_like(str);
    auto offsets      = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list         = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_string), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});

    std::vector<cudf::size_type> splits({0});
    EXPECT_NO_THROW(contiguous_split(src_table, splits));

    std::vector<cudf::size_type> splits2({1});
    EXPECT_NO_THROW(contiguous_split(src_table, splits2));

    EXPECT_NO_THROW(do_chunked_pack(src_table));
  }

  // this produces an empty lists column with children that have no data,
  // nested inside a list
  {
    cudf::test::lists_column_wrapper<float> listw{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto empty_list = cudf::empty_like(listw);
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list =
      cudf::make_lists_column(1, offsets.release(), std::move(empty_list), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});

    std::vector<cudf::size_type> splits({0});
    EXPECT_NO_THROW(contiguous_split(src_table, splits));

    std::vector<cudf::size_type> splits2({1});
    EXPECT_NO_THROW(contiguous_split(src_table, splits2));

    EXPECT_NO_THROW(do_chunked_pack(src_table));
  }

  // this produces an empty lists column with children that have no data,
  // nested inside a list
  {
    cudf::test::lists_column_wrapper<float> listw{{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto empty_list = cudf::empty_like(listw);
    auto offsets    = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list =
      cudf::make_lists_column(1, offsets.release(), std::move(empty_list), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});

    std::vector<cudf::size_type> splits({0});
    EXPECT_NO_THROW(contiguous_split(src_table, splits));

    std::vector<cudf::size_type> splits2({1});
    EXPECT_NO_THROW(contiguous_split(src_table, splits2));

    EXPECT_NO_THROW(do_chunked_pack(src_table));
  }

  // this produces an empty struct column with children that have no data,
  // nested inside a list
  {
    cudf::test::fixed_width_column_wrapper<int> ints{0, 1, 2, 3, 4};
    cudf::test::fixed_width_column_wrapper<float> floats{4, 3, 2, 1, 0};
    auto struct_column = cudf::test::structs_column_wrapper({ints, floats});
    auto empty_struct  = cudf::empty_like(struct_column);
    auto offsets       = cudf::test::fixed_width_column_wrapper<int>({0, 0});
    auto list          = cudf::make_lists_column(
      1, offsets.release(), std::move(empty_struct), 0, rmm::device_buffer{});

    cudf::table_view src_table({static_cast<cudf::column_view>(*list)});

    std::vector<cudf::size_type> splits({0});
    EXPECT_NO_THROW(contiguous_split(src_table, splits));

    std::vector<cudf::size_type> splits2({1});
    EXPECT_NO_THROW(contiguous_split(src_table, splits2));

    EXPECT_NO_THROW(do_chunked_pack(src_table));
  }
}

TEST_F(ContiguousSplitTableCornerCases, SplitEmpty)
{
  // empty sliced column. this is specifically testing the corner case:
  // - a sliced column of size 0
  // - having children that are of size > 0
  //
  cudf::test::strings_column_wrapper a{"abc", "def", "ghi", "jkl", "mno", "", "st", "uvwx"};
  cudf::test::lists_column_wrapper<int> b{
    {0, 1}, {2}, {3, 4, 5}, {6, 7}, {8, 9}, {10}, {11, 12}, {13, 14}};
  cudf::test::fixed_width_column_wrapper<float> c{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::test::strings_column_wrapper _a{"abc", "def", "ghi", "jkl", "mno", "", "st", "uvwx"};
  cudf::test::lists_column_wrapper<float> _b{
    {0, 1}, {2}, {3, 4, 5}, {6, 7}, {8, 9}, {10}, {11, 12}, {13, 14}};
  cudf::test::fixed_width_column_wrapper<float> _c{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::test::structs_column_wrapper d({_a, _b, _c});

  cudf::table_view t({a, b, c, d});

  auto sliced = cudf::split(t, {0});

  {
    auto result = cudf::contiguous_split(sliced[0], {});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sliced[0], result[0].table);
  }

  {
    auto result = do_chunked_pack(sliced[0]);
    EXPECT_EQ(1, result.size());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sliced[0], result[0].table);
  }

  {
    auto result = cudf::contiguous_split(sliced[0], {0});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sliced[0], result[0].table);
  }

  {
    EXPECT_THROW(cudf::contiguous_split(sliced[0], {1}), std::out_of_range);
  }
}

TEST_F(ContiguousSplitTableCornerCases, OutBufferToSmall)
{
  // internally, contiguous split chunks GPU work in 1MB contiguous copies
  // so the output buffer must be 1MB or larger.
  EXPECT_THROW(auto _ = cudf::chunked_pack::create({}, 1 * 1024), cudf::logic_error);
}

TEST_F(ContiguousSplitTableCornerCases, ChunkSpanTooSmall)
{
  auto chunked_pack = cudf::chunked_pack::create({}, 1 * 1024 * 1024);
  rmm::device_buffer buff(
    1 * 1024, cudf::test::get_default_stream(), cudf::get_current_device_resource_ref());
  cudf::device_span<uint8_t> too_small(static_cast<uint8_t*>(buff.data()), buff.size());
  std::size_t copied = 0;
  // throws because we created chunked_contig_split with 1MB, but we are giving
  // it a 1KB span here
  EXPECT_THROW(copied = chunked_pack->next(too_small), cudf::logic_error);
  EXPECT_EQ(copied, 0);
}

TEST_F(ContiguousSplitTableCornerCases, EmptyTableHasNextFalse)
{
  auto chunked_pack = cudf::chunked_pack::create({}, 1 * 1024 * 1024);
  rmm::device_buffer buff(
    1 * 1024 * 1024, cudf::test::get_default_stream(), cudf::get_current_device_resource_ref());
  cudf::device_span<uint8_t> bounce_buff(static_cast<uint8_t*>(buff.data()), buff.size());
  EXPECT_EQ(chunked_pack->has_next(), false);  // empty input table
  std::size_t copied = 0;
  EXPECT_THROW(copied = chunked_pack->next(bounce_buff), cudf::logic_error);
  EXPECT_EQ(copied, 0);
}

TEST_F(ContiguousSplitTableCornerCases, ExhaustedHasNextFalse)
{
  cudf::test::strings_column_wrapper a{"abc", "def", "ghi", "jkl", "mno", "", "st", "uvwx"};
  cudf::table_view t({a});
  rmm::device_buffer buff(
    1 * 1024 * 1024, cudf::test::get_default_stream(), cudf::get_current_device_resource_ref());
  cudf::device_span<uint8_t> bounce_buff(static_cast<uint8_t*>(buff.data()), buff.size());
  auto chunked_pack = cudf::chunked_pack::create(t, buff.size());
  EXPECT_EQ(chunked_pack->has_next(), true);
  std::size_t copied = chunked_pack->next(bounce_buff);
  EXPECT_EQ(copied, chunked_pack->get_total_contiguous_size());
  EXPECT_EQ(chunked_pack->has_next(), false);
  copied = 0;
  EXPECT_THROW(copied = chunked_pack->next(bounce_buff), cudf::logic_error);
  EXPECT_EQ(copied, 0);
}

struct ContiguousSplitNestedTypesTest : public cudf::test::BaseFixture {};

TEST_F(ContiguousSplitNestedTypesTest, Lists)
{
  split_lists<int>(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, ListsChunked)
{
  split_lists<int>(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const&) {
      cudf::table_view t({c});
      return do_chunked_pack(t);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    },
    /*split*/ false);
}

TEST_F(ContiguousSplitNestedTypesTest, ListsWithNulls)
{
  split_lists_with_nulls<int>(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, ListsWithNullsChunked)
{
  split_lists_with_nulls<int>(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const&) {
      cudf::table_view t({c});
      return do_chunked_pack(t);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    },
    /*split*/ false);
}

TEST_F(ContiguousSplitNestedTypesTest, Structs)
{
  split_structs(
    false,
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, StructsChunked)
{
  split_structs(
    false,
    [](cudf::column_view const& c, std::vector<cudf::size_type> const&) {
      cudf::table_view t({c});
      return do_chunked_pack(t);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    },
    /*split*/ false);
}

TEST_F(ContiguousSplitNestedTypesTest, StructsWithNulls)
{
  split_structs(
    true,
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, StructsWithNullsChunked)
{
  split_structs(
    true,
    [](cudf::column_view const& c, std::vector<cudf::size_type> const&) {
      cudf::table_view t({c});
      return do_chunked_pack(t);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    },
    {});
}

TEST_F(ContiguousSplitNestedTypesTest, StructsNoChildren)
{
  split_structs_no_children(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, StructsNoChildrenChunked)
{
  split_structs_no_children(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const&) {
      cudf::table_view t({c});
      return do_chunked_pack(t);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    },
    /*split*/ false);
}

TEST_F(ContiguousSplitNestedTypesTest, StructsOfList)
{
  split_nested_struct_of_list(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, StructsOfListChunked)
{
  split_nested_struct_of_list(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const&) {
      cudf::table_view t({c});
      return do_chunked_pack(t);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    },
    /*split*/ false);
}

TEST_F(ContiguousSplitNestedTypesTest, ListOfStruct)
{
  split_nested_list_of_structs(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const& splits) {
      cudf::table_view t({c});
      return cudf::contiguous_split(t, splits);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    });
}

TEST_F(ContiguousSplitNestedTypesTest, ListOfStructChunked)
{
  split_nested_list_of_structs(
    [](cudf::column_view const& c, std::vector<cudf::size_type> const&) {
      cudf::table_view t({c});
      return do_chunked_pack(t);
    },
    [](cudf::column_view const& expected, cudf::packed_table const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result.table.column(0));
    },
    /*split*/ false);
}

struct ContiguousSplitLongStrings : public cudf::test::BaseFixture {};

TEST_F(ContiguousSplitLongStrings, LongOffsets)
{
  auto str = make_long_offsets_string_column();
  cudf::table_view tbl({*str});

  {
    auto res      = cudf::contiguous_split(tbl, {});
    auto expected = cudf::split(tbl, {});

    for (size_t idx = 0; idx < expected.size(); idx++) {
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[idx], res[idx].table);
    }
  }

  {
    std::vector<int> splits{3, 7};
    auto res      = cudf::contiguous_split(tbl, splits);
    auto expected = cudf::split(tbl, splits);

    for (size_t idx = 0; idx < expected.size(); idx++) {
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[idx], res[idx].table);
    }
  }
}

TEST_F(ContiguousSplitLongStrings, LongOffsetsNested)
{
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(make_long_offsets_string_column());
  children.push_back(make_long_offsets_string_column());
  cudf::test::structs_column_wrapper st(std::move(children));

  cudf::test::fixed_width_column_wrapper<int> offsets{0, 3, 5, 7, 9, 10};
  auto list = make_lists_column(5, offsets.release(), st.release(), 0, {});

  cudf::table_view tbl({*list});
  std::vector<int> splits{2, 3};
  auto res      = cudf::contiguous_split(tbl, splits);
  auto expected = cudf::split(tbl, splits);

  for (size_t idx = 0; idx < expected.size(); idx++) {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[idx], res[idx].table);
  }
}

TEST_F(ContiguousSplitLongStrings, DISABLED_LongOffsetsAndChars)
{
  auto str = make_long_offsets_and_chars_string_column();

  std::vector<int> splits{3, 7};
  cudf::table_view tbl({*str});
  auto res      = cudf::contiguous_split(tbl, splits);
  auto expected = cudf::split(tbl, splits);

  for (size_t idx = 0; idx < expected.size(); idx++) {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[idx], res[idx].table);
  }
}

TEST_F(ContiguousSplitLongStrings, DISABLED_LongOffsetsAndCharsNested)
{
  cudf::test::fixed_width_column_wrapper<int> offsets{0, 3, 5, 7, 9, 10};
  auto list =
    make_lists_column(5, offsets.release(), make_long_offsets_and_chars_string_column(), 0, {});

  cudf::table_view tbl({*list});
  std::vector<int> splits{2, 3};
  auto res      = cudf::contiguous_split(tbl, splits);
  auto expected = cudf::split(tbl, splits);

  for (size_t idx = 0; idx < expected.size(); idx++) {
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected[idx], res[idx].table);
  }
}
