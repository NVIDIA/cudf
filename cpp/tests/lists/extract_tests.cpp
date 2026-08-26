/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The usage of the lists_column_wrapper `LCW{LCW...` syntax in this file
// causes gcc14 to throw a maybe-uninitialized warning in the copy constructor
// of column_view_base. The same usage in every other test file causes no
// issues, so it seems highly likely to be an incorrect diagnostic.
// Unfortunately, because the warning is in an included file, neither inserting
// ignore pragmas around just the includes nor just around the calling code
// below that uses that syntax (of which there is a decent amount) seems to be
// sufficient to make the compiler happy, so for now the easiest option is to
// ignore the warning for the entire file.
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/lists/extract.hpp>

#include <vector>

struct ListsExtractTest : public cudf::test::BaseFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

template <typename T>
class ListsExtractNumericsTest : public ListsExtractTest {};

TYPED_TEST_SUITE(ListsExtractNumericsTest, NumericTypesNotBool);

TYPED_TEST(ListsExtractNumericsTest, ExtractElement)
{
  auto validity = cudf::detail::make_counting_transform_iterator(cudf::size_type{0},
                                                                 [](auto i) { return i != 1; });
  using LCW     = cudf::test::lists_column_wrapper<TypeParam>;
  LCW input({LCW{3, 2, 1}, LCW{}, LCW{30, 20, 10, 50}, LCW{100, 120}, LCW{0}}, validity);

  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 0);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({3, 0, 30, 100, 0}, {1, 0, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 1);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({2, 0, 20, 120, 0}, {1, 0, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 2);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({1, 0, 10, 0, 0}, {1, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 3);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({0, 0, 50, 0, 0}, {0, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 4);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({0, 0, 0, 0, 0}, {0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -1);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({1, 0, 50, 120, 0}, {1, 0, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -2);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({2, 0, 10, 100, 0}, {1, 0, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -3);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({3, 0, 20, 0, 0}, {1, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -4);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({0, 0, 30, 0, 0}, {0, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -5);
    cudf::test::fixed_width_column_wrapper<TypeParam> expected({0, 0, 0, 0, 0}, {0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TEST_F(ListsExtractTest, ExtractElementStrings)
{
  auto validity = cudf::detail::make_counting_transform_iterator(cudf::size_type{0},
                                                                 [](auto i) { return i != 1; });
  using LCW     = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW input(
    {LCW{"", "Héllo", "thesé"}, LCW{}, LCW{"are", "some", "", "z"}, LCW{"tést", "String"}, LCW{""}},
    validity);

  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 0);
    cudf::test::strings_column_wrapper expected({"", "", "are", "tést", ""}, {1, 0, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 1);
    cudf::test::strings_column_wrapper expected({"Héllo", "", "some", "String", ""},
                                                {1, 0, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 2);
    cudf::test::strings_column_wrapper expected({"thesé", "", "", "", ""}, {1, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 3);
    cudf::test::strings_column_wrapper expected({"", "", "z", "", ""}, {0, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 4);
    cudf::test::strings_column_wrapper expected({"", "", "", "", ""}, {0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -1);
    cudf::test::strings_column_wrapper expected({"thesé", "", "z", "String", ""}, {1, 0, 1, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -2);
    cudf::test::strings_column_wrapper expected({"Héllo", "", "", "tést", ""}, {1, 0, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -3);
    cudf::test::strings_column_wrapper expected({"", "", "some", "", ""}, {1, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -4);
    cudf::test::strings_column_wrapper expected({"", "", "are", "", ""}, {0, 0, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -5);
    cudf::test::strings_column_wrapper expected({"", "", "", "", ""}, {0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TYPED_TEST(ListsExtractNumericsTest, ExtractElementNestedLists)
{
  std::vector<int32_t> validity{1, 0, 1, 1};
  using LCW = cudf::test::lists_column_wrapper<TypeParam>;
  LCW list({LCW{LCW{2, 3}, LCW{4, 5}},
            LCW{LCW{}},
            LCW{LCW{6, 7, 8}, LCW{9, 10, 11}, LCW{12, 13, 14}},
            LCW{LCW{15, 16}, LCW{17, 18}, LCW{19, 20}, LCW{21, 22}, LCW{23, 24}}},
           validity.begin());

  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), 0);
    LCW expected({LCW{2, 3}, LCW{}, LCW{6, 7, 8}, LCW{15, 16}}, validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), 1);
    LCW expected({LCW{4, 5}, LCW{}, LCW{9, 10, 11}, LCW{17, 18}}, validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), 2);
    std::vector<int32_t> expected_validity{0, 0, 1, 1};
    LCW expected({LCW{}, LCW{}, LCW{12, 13, 14}, LCW{19, 20}}, expected_validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), 3);
    std::vector<int32_t> expected_validity{0, 0, 0, 1};
    LCW expected({LCW{}, LCW{}, LCW{}, LCW{21, 22}}, expected_validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), -1);
    LCW expected({LCW{4, 5}, LCW{}, LCW{12, 13, 14}, LCW{23, 24}}, validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), -2);
    LCW expected({LCW{2, 3}, LCW{}, LCW{9, 10, 11}, LCW{21, 22}}, validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(list), -3);
    std::vector<int32_t> expected_validity{0, 0, 1, 1};
    LCW expected({LCW{}, LCW{}, LCW{6, 7, 8}, LCW{19, 20}}, expected_validity.begin());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TEST_F(ListsExtractTest, ExtractElementEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  LCW empty{};
  auto result = cudf::lists::extract_list_element(cudf::lists_column_view(empty), 1);
  EXPECT_EQ(cudf::data_type{cudf::type_id::STRING}, result->type());
  EXPECT_EQ(0, result->size());

  LCW empty_strings({LCW{"", "", ""}});
  result = cudf::lists::extract_list_element(cudf::lists_column_view(empty_strings), 1);
  cudf::test::strings_column_wrapper expected({""});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);

  LCW null_strings({LCW{"", "", ""}}, cuda::make_constant_iterator<int32_t>(0));
  result = cudf::lists::extract_list_element(cudf::lists_column_view(null_strings), 1);
  cudf::test::strings_column_wrapper expected_null({""}, {0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_null, *result);
}

TEST_F(ListsExtractTest, ExtractElementWithNulls)
{
  auto validity = cudf::detail::make_counting_transform_iterator(cudf::size_type{0},
                                                                 [](auto i) { return i != 1; });
  using LCW     = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW input{
    {{"Héllo", "", "thesé"}, validity}, {"are"}, {{"some", ""}, validity}, {"tést", "strings"}};

  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 0);
    cudf::test::strings_column_wrapper expected({"Héllo", "are", "some", "tést"});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), 1);
    cudf::test::strings_column_wrapper expected({"", "", "", "strings"}, {0, 0, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
  {
    auto result = cudf::lists::extract_list_element(cudf::lists_column_view(input), -1);
    cudf::test::strings_column_wrapper expected({"thesé", "are", "", "strings"}, {1, 1, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
  }
}

struct ListsExtractColumnIndicesTest : ListsExtractTest {};

template <typename T>
struct ListsExtractColumnIndicesTypedTest : ListsExtractColumnIndicesTest {};

TYPED_TEST_SUITE(ListsExtractColumnIndicesTypedTest, cudf::test::FixedWidthTypes);

TYPED_TEST(ListsExtractColumnIndicesTypedTest, ExtractElement)
{
  using LCW     = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using FWCW    = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using indices = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  auto input_column = LCW({LCW{3, 2, 1}, LCW{}, LCW{30, 20, 10, 50}, LCW{100, 120}, LCW{0}, LCW{}},
                          cudf::test::iterators::null_at(1));
  auto input        = cudf::lists_column_view(input_column);

  {
    // Test fetching first element.
    auto result   = cudf::lists::extract_list_element(input, indices{0, 0, 0, 0, 0, 0});
    auto expected = FWCW({3, 0, 30, 100, 0, 0}, cudf::test::iterators::nulls_at({1, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test fetching last element.
    auto result   = cudf::lists::extract_list_element(input, indices{2, 0, 3, 1, 0, 0});
    auto expected = FWCW({1, 0, 50, 120, 0, 0}, cudf::test::iterators::nulls_at({1, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test fetching *all* out of bounds.
    auto result   = cudf::lists::extract_list_element(input, indices{9, 9, 9, 9, 9, 9});
    auto expected = FWCW({0, 0, 0, 0, 0, 0}, cudf::test::iterators::all_nulls());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test fetching first from the end.
    auto result   = cudf::lists::extract_list_element(input, indices{-1, -1, -1, -1, -1, -1});
    auto expected = FWCW({1, 0, 50, 120, 0, 0}, cudf::test::iterators::nulls_at({1, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test fetching last from the end.
    auto result   = cudf::lists::extract_list_element(input, indices{-3, 0, -4, -2, -1, 0});
    auto expected = FWCW({3, 0, 30, 100, 0, 0}, cudf::test::iterators::nulls_at({1, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test fetching *all* negative out of bounds.
    auto result   = cudf::lists::extract_list_element(input, indices{-9, -9, -9, -9, -9, -9});
    auto expected = FWCW({0, 0, 0, 0, 0, 0}, cudf::test::iterators::all_nulls());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test mixed indices.
    auto result   = cudf::lists::extract_list_element(input, indices{-2, 0, 3, -1, 0, 0});
    auto expected = FWCW({2, 0, 50, 120, 0, 0}, cudf::test::iterators::nulls_at({1, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test possibly null indices.
    auto result = cudf::lists::extract_list_element(
      input, indices{{-2, 0, 3, -1, 0, 0}, cudf::test::iterators::nulls_at({2, 4})});
    auto expected = FWCW({2, 0, 50, 120, 0, 0}, cudf::test::iterators::nulls_at({1, 2, 4, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TYPED_TEST(ListsExtractColumnIndicesTypedTest, FailureCases)
{
  using LCW     = cudf::test::lists_column_wrapper<TypeParam, int32_t>;
  using indices = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  {
    // Non-empty input, with mismatched size of indices.
    auto input_column =
      LCW({LCW{3, 2, 1}, LCW{}, LCW{30, 20, 10, 50}, LCW{100, 120}, LCW{0}, LCW{}},
          cudf::test::iterators::null_at(1));
    auto input = cudf::lists_column_view(input_column);

    EXPECT_THROW(cudf::lists::extract_list_element(input, indices{0, 1, 2}), cudf::logic_error);
  }
  {
    // Non-empty input, with empty indices.
    auto input_column =
      LCW({LCW{3, 2, 1}, LCW{}, LCW{30, 20, 10, 50}, LCW{100, 120}, LCW{0}, LCW{}},
          cudf::test::iterators::null_at(1));
    auto input = cudf::lists_column_view(input_column);

    EXPECT_THROW(cudf::lists::extract_list_element(input, indices{}), cudf::logic_error);
  }
  {
    // Empty input, with mismatched size of indices.
    auto input_column = LCW{};
    auto input        = cudf::lists_column_view(input_column);
    EXPECT_THROW(cudf::lists::extract_list_element(input, indices{0, 1, 2}), cudf::logic_error);
  }
}

TEST_F(ListsExtractColumnIndicesTest, ExtractStrings)
{
  using LCW     = cudf::test::lists_column_wrapper<cudf::string_view>;
  using strings = cudf::test::strings_column_wrapper;
  using indices = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  auto input_column = LCW(
    {LCW{"3", "2", "1"}, LCW{}, LCW{"30", "20", "10", "50"}, LCW{"100", "120"}, LCW{"0"}, LCW{}},
    cudf::test::iterators::null_at(1));
  auto input = cudf::lists_column_view(input_column);

  {
    // Test fetching first element.
    auto result = cudf::lists::extract_list_element(input, indices{0, 0, 0, 0, 0, 0});
    auto expected =
      strings({"3", "", "30", "100", "0", ""}, cudf::test::iterators::nulls_at({1, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test fetching last element.
    auto result = cudf::lists::extract_list_element(input, indices{2, 0, 3, 1, 0, 0});
    auto expected =
      strings({"1", "", "50", "120", "0", ""}, cudf::test::iterators::nulls_at({1, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test fetching *all* out of bounds.
    auto result   = cudf::lists::extract_list_element(input, indices{9, 9, 9, 9, 9, 9});
    auto expected = strings({"", "", "", "", "", ""}, cudf::test::iterators::all_nulls());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test fetching first from the end.
    auto result = cudf::lists::extract_list_element(input, indices{-1, -1, -1, -1, -1, -1});
    auto expected =
      strings({"1", "", "50", "120", "0", ""}, cudf::test::iterators::nulls_at({1, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test fetching last from the end.
    auto result = cudf::lists::extract_list_element(input, indices{-3, 0, -4, -2, -1, 0});
    auto expected =
      strings({"3", "", "30", "100", "0", ""}, cudf::test::iterators::nulls_at({1, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test fetching *all* negative out of bounds.
    auto result   = cudf::lists::extract_list_element(input, indices{-9, -9, -9, -9, -9, -9});
    auto expected = strings({"", "", "", "", "", ""}, cudf::test::iterators::all_nulls());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test mixed indices.
    auto result = cudf::lists::extract_list_element(input, indices{-2, 0, 3, -1, 0, 0});
    auto expected =
      strings({"2", "", "50", "120", "0", ""}, cudf::test::iterators::nulls_at({1, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    // Test possibly null indices.
    auto result = cudf::lists::extract_list_element(
      input, indices{{-2, 0, 3, -1, 0, 0}, cudf::test::iterators::nulls_at({2, 4})});
    auto expected =
      strings({"2", "", "50", "120", "", ""}, cudf::test::iterators::nulls_at({1, 2, 4, 5}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

CUDF_TEST_PROGRAM_MAIN()
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
