/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/combine.hpp>

#include <stdexcept>

using namespace cudf::test::iterators;

namespace {
using StrListsCol = cudf::test::lists_column_wrapper<cudf::string_view>;
using IntListsCol = cudf::test::lists_column_wrapper<int32_t>;
using IntCol      = cudf::test::fixed_width_column_wrapper<int32_t>;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};

}  // namespace

struct ConcatenateListElementsTest : public cudf::test::BaseFixture {};

TEST_F(ConcatenateListElementsTest, InvalidInput)
{
  // Input lists is not a 2-level depth lists column.
  {
    auto const col = IntCol{};
    EXPECT_THROW(cudf::lists::concatenate_list_elements(col), std::invalid_argument);
  }

  // Input lists is not at least 2-level depth lists column.
  {
    auto const col = IntListsCol{1, 2, 3};
    EXPECT_THROW(cudf::lists::concatenate_list_elements(col), std::invalid_argument);
  }
}

template <typename T>
struct ConcatenateListElementsTypedTest : public cudf::test::BaseFixture {};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(ConcatenateListElementsTypedTest, TypesForTest);

TYPED_TEST(ConcatenateListElementsTypedTest, SimpleInputNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col      = ListsCol{{{1, 2}, {3}, {4, 5, 6}}, {{}}, {{7, 8}, {9, 10}}};
  auto const results  = cudf::lists::concatenate_list_elements(col);
  auto const expected = ListsCol({{1, 2, 3, 4, 5, 6}, {}, {7, 8, 9, 10}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
}

TYPED_TEST(ConcatenateListElementsTypedTest, SimpleInputNestedManyLevelsNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col      = ListsCol{{{{1, 2}, {3}, {4, 5, 6}}, {{}}, {{7, 8}, {9, 10}}},
                                 {{{1, 2}, {3}, {4, 5, 6}}, {{}}, {{7, 8}, {9, 10}}},
                                 {{{1, 2}, {3}, {4, 5, 6}}, {{}}, {{7, 8}, {9, 10}}}};
  auto const results  = cudf::lists::concatenate_list_elements(col);
  auto const expected = ListsCol({{{1, 2}, {3}, {4, 5, 6}, {}, {7, 8}, {9, 10}},
                                  {{1, 2}, {3}, {4, 5, 6}, {}, {7, 8}, {9, 10}},
                                  {{1, 2}, {3}, {4, 5, 6}, {}, {7, 8}, {9, 10}}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
}

TEST_F(ConcatenateListElementsTest, SimpleInputStringsColumnNoNull)
{
  auto const col      = StrListsCol{{{"Tomato", "Apple"}, {"Orange"}},
                                    {{"Banana", "Kiwi", "Cherry"}, {"Lemon", "Peach"}},
                                    {{"Coconut"}, {}}};
  auto const results  = cudf::lists::concatenate_list_elements(col);
  auto const expected = StrListsCol(
    {{"Tomato", "Apple", "Orange"}, {"Banana", "Kiwi", "Cherry", "Lemon", "Peach"}, {"Coconut"}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
}

TYPED_TEST(ConcatenateListElementsTypedTest, SimpleInputWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;
  auto const col = ListsCol{
    {{{{1, null, 3, 4}, null_at(1)}, {{10, 11, 12, null}, null_at(3)}, {} /*NULL*/}, null_at(2)},
    {{{null, 2, 3, 4}, null_at(0)},
     {{13, 14, 15, 16, 17, null}, null_at(5)},
     {{20, null}, null_at(1)}},
    {{{{null, 2, 3, 4}, null_at(0)}, {} /*NULL*/, {{null, 21, null, null}, nulls_at({0, 2, 3})}},
     null_at(1)},
    {{{} /*NULL*/, {{null, 18}, null_at(0)}}, null_at(0)},
    {{{1, 2, null, 4}, null_at(2)}, {{19, 20, null}, null_at(2)}, {22, 23, 24, 25}},
    {{{1, 2, 3, null}, null_at(3)},
     {{null}, null_at(0)},
     {{null, null, null, null, null}, all_nulls()}},
    ListsCol::nested({{} /*NULL*/, {} /*NULL*/, {} /*NULL*/}, all_nulls())};

  // Ignore null list elements.
  {
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = ListsCol(
      {{{1, null, 3, 4, 10, 11, 12, null}, nulls_at({1, 7})},
       {{null, 2, 3, 4, 13, 14, 15, 16, 17, null, 20, null}, nulls_at({0, 9, 11})},
       {{null, 2, 3, 4, null, 21, null, null}, nulls_at({0, 4, 6, 7})},
       {{null, 18}, null_at(0)},
       {{1, 2, null, 4, 19, 20, null, 22, 23, 24, 25}, nulls_at({2, 6})},
       {{1, 2, 3, null, null, null, null, null, null, null}, nulls_at({3, 4, 5, 6, 7, 8, 9})},
       {} /*NULL*/},
      null_at(6));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }

  // Null lists result in null rows.
  {
    auto const results = cudf::lists::concatenate_list_elements(
      col, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected = ListsCol(
      {{} /*NULL*/,
       {{null, 2, 3, 4, 13, 14, 15, 16, 17, null, 20, null}, nulls_at({0, 9, 11})},
       {} /*NULL*/,
       {} /*NULL*/,
       {{1, 2, null, 4, 19, 20, null, 22, 23, 24, 25}, nulls_at({2, 6})},
       {{1, 2, 3, null, null, null, null, null, null, null}, nulls_at({3, 4, 5, 6, 7, 8, 9})},
       {} /*NULL*/},
      nulls_at({0, 2, 3, 6}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TYPED_TEST(ConcatenateListElementsTypedTest, SimpleInputNestedManyLevelsWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col =
    ListsCol{{{{{1, 2}, {3}, {4, 5, 6}}, {{}} /*NULL*/, {{7, 8}, {9, 10}}}, null_at(1)},
             {{{{1, 2}, {3}, {4, 5, 6} /*NULL*/}, null_at(2)}, {{}}, {{7, 8}, {9, 10}}},
             {{{1, 2}, {3}, {4, 5, 6}}, {{}}, {{{null, 8}, null_at(0)}, {9, 10}}}};

  // Ignore null list elements.
  {
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol({{{1, 2}, {3}, {4, 5, 6}, {7, 8}, {9, 10}},
                {{{1, 2}, {3}, {} /*NULL*/, {}, {7, 8}, {9, 10}}, null_at(2)},
                {{1, 2}, {3}, {4, 5, 6}, {}, {{null, 8}, null_at(0)}, {9, 10}}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }

  // Null lists result in null rows.
  {
    auto const results = cudf::lists::concatenate_list_elements(
      col, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected = ListsCol({ListsCol::nested({{}}), /*NULL*/
                                    {{{1, 2}, {3}, {} /*NULL*/, {}, {7, 8}, {9, 10}}, null_at(2)},
                                    {{1, 2}, {3}, {4, 5, 6}, {}, {{null, 8}, null_at(0)}, {9, 10}}},
                                   null_at(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TEST_F(ConcatenateListElementsTest, SimpleInputStringsColumnWithNulls)
{
  auto const col = StrListsCol{
    {{{"Tomato", "Bear" /*NULL*/, "Apple"}, null_at(1)},
     {{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/}, nulls_at({1, 2, 3})}},
    {{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/}, nulls_at({1, 4})},
     {"Lemon", "Peach"}},
    StrListsCol::nested({{"Coconut"}, {} /*NULL*/}, null_at(1))};

  // Ignore null list elements.
  {
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = StrListsCol(
      {{{"Tomato", "" /*NULL*/, "Apple", "Orange", "" /*NULL*/, "" /*NULL*/, ""
         /*NULL*/},
        nulls_at({1, 4, 5, 6})},
       {{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, "Lemon", "Peach"}, nulls_at({1, 4})},
       {"Coconut"}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }

  // Null lists result in null rows.
  {
    auto const results = cudf::lists::concatenate_list_elements(
      col, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected = StrListsCol(
      {{{"Tomato", "" /*NULL*/, "Apple", "Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/},
        nulls_at({1, 4, 5, 6})},
       {{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, "Lemon", "Peach"}, nulls_at({1, 4})},
       {} /*NULL*/},
      null_at(2));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}
TEST_F(ConcatenateListElementsTest, SimpleInputStringsColumnWithEmptyStringsAndNulls)
{
  auto const col = StrListsCol{
    {{"", "", ""}, {{"Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, nulls_at({1, 2, 3})}},
    {{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/}, nulls_at({1, 4})}, {""}},
    StrListsCol::nested({{"Coconut"}, {} /*NULL*/}, null_at(1))};

  // Ignore null list elements.
  {
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = StrListsCol(
      {{{"", "", "", "Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, nulls_at({4, 5, 6})},
       {{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, ""}, nulls_at({1, 4})},
       {"Coconut"}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }

  // Null lists result in null rows.
  {
    auto const results = cudf::lists::concatenate_list_elements(
      col, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected = StrListsCol(
      {{{"", "", "", "Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, nulls_at({4, 5, 6})},
       {{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, ""}, nulls_at({1, 4})},
       {} /*NULL*/},
      null_at(2));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TYPED_TEST(ConcatenateListElementsTypedTest, SlicedColumnsInputNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col_original = ListsCol({{{1, 2, 3}, {2, 3}},
                                      {{3, 4, 5, 6}, {5, 6}, {}, {7}},
                                      {{7, 7, 7}, {7, 8, 1, 0}, {1}},
                                      {{9, 10, 11}},
                                      {},
                                      {{12, 13, 14, 15}, {16}, {17}}});

  {
    auto const col     = cudf::slice(col_original, {0, 3})[0];
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol({{1, 2, 3, 2, 3}, {3, 4, 5, 6, 5, 6, 7}, {7, 7, 7, 7, 8, 1, 0, 1}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {1, 4})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = ListsCol({{3, 4, 5, 6, 5, 6, 7}, {7, 7, 7, 7, 8, 1, 0, 1}, {9, 10, 11}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {2, 5})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = ListsCol({{7, 7, 7, 7, 8, 1, 0, 1}, {9, 10, 11}, {}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {3, 6})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = ListsCol({{9, 10, 11}, {}, {12, 13, 14, 15, 16, 17}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TYPED_TEST(ConcatenateListElementsTypedTest, SlicedColumnsInputWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col_original = ListsCol{{{{null, 2, 3}, null_at(0)}, {2, 3}},
                                     {{{3, null, null, 6}, nulls_at({1, 2})},
                                      {{5, 6, null}, null_at(2)},
                                      {},
                                      {{7, null}, null_at(1)}},
                                     {{7, 7, 7}, {{7, 8, null, 0}, null_at(2)}, {1}},
                                     {{9, 10, 11}},
                                     {{}},
                                     {{{12, null, 14, 15}, null_at(1)}, {16}, {17}}};

  {
    auto const col     = cudf::slice(col_original, {0, 3})[0];
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol({{{null, 2, 3, 2, 3}, null_at(0)},
                {{3, null, null, 6, 5, 6, null, 7, null}, nulls_at({1, 2, 6, 8})},
                {{7, 7, 7, 7, 8, null, 0, 1}, null_at(5)}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col     = cudf::slice(col_original, {1, 4})[0];
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol({{{3, null, null, 6, 5, 6, null, 7, null}, nulls_at({1, 2, 6, 8})},
                {{7, 7, 7, 7, 8, null, 0, 1}, null_at(5)},
                {9, 10, 11}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {2, 5})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = ListsCol({{{7, 7, 7, 7, 8, null, 0, 1}, null_at(5)}, {9, 10, 11}, {}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {3, 6})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = ListsCol({{9, 10, 11}, {}, {{12, null, 14, 15, 16, 17}, null_at(1)}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TEST_F(ConcatenateListElementsTest, SlicedStringsColumnsInputWithNulls)
{
  auto const col_original = StrListsCol{
    {{{"Tomato", "Bear" /*NULL*/, "Apple"}, null_at(1)},
     {{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/}, nulls_at({1, 4})},
     {"Coconut"}},
    {{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/}, nulls_at({1, 4})},
     {"Coconut"},
     {{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/}, nulls_at({1, 2, 3})}},
    {{"Coconut"},
     {{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/}, nulls_at({1, 2, 3})},
     {"Lemon", "Peach"}},
    {{{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/}, nulls_at({1, 2, 3})},
      {"Lemon", "Peach"},
      {} /*NULL*/},
     null_at(2)}};

  {
    auto const col      = cudf::slice(col_original, {0, 2})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = StrListsCol({{{"Tomato",
                                         "" /*NULL*/,
                                         "Apple",
                                         "Banana",
                                         "" /*NULL*/,
                                         "Kiwi",
                                         "Cherry",
                                         "" /*NULL*/,
                                         "Coconut"},
                                        nulls_at({1, 4, 7})},
                                       {{"Banana",
                                         "" /*NULL*/,
                                         "Kiwi",
                                         "Cherry",
                                         "" /*NULL*/,
                                         "Coconut",
                                         "Orange",
                                         "" /*NULL*/,
                                         "" /*NULL*/,
                                         "" /*NULL*/},
                                        nulls_at({1, 4, 7, 8, 9})}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {1, 3})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = StrListsCol({{{"Banana",
                                         "" /*NULL*/,
                                         "Kiwi",
                                         "Cherry",
                                         "" /*NULL*/,
                                         "Coconut",
                                         "Orange",
                                         "" /*NULL*/,
                                         "" /*NULL*/,
                                         "" /*NULL*/},
                                        nulls_at({1, 4, 7, 8, 9})},
                                       {{"Coconut",
                                         "Orange",
                                         "" /*NULL*/,
                                         "" /*NULL*/,
                                         "", /*NULL*/
                                         "Lemon",
                                         "Peach"},
                                        nulls_at({2, 3, 4})}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {2, 4})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = StrListsCol({{{"Coconut",
                                         "Orange",
                                         "" /*NULL*/,
                                         "" /*NULL*/,
                                         "", /*NULL*/
                                         "Lemon",
                                         "Peach"},
                                        nulls_at({2, 3, 4})},
                                       {{"Orange",
                                         "" /*NULL*/,
                                         "" /*NULL*/,
                                         "", /*NULL*/
                                         "Lemon",
                                         "Peach"},
                                        nulls_at({1, 2, 3})}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col     = cudf::slice(col_original, {2, 4})[0];
    auto const results = cudf::lists::concatenate_list_elements(
      col, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected = StrListsCol({{{"Coconut",
                                         "Orange",
                                         "" /*NULL*/,
                                         "" /*NULL*/,
                                         "", /*NULL*/
                                         "Lemon",
                                         "Peach"},
                                        nulls_at({2, 3, 4})},
                                       {} /*NULL*/},
                                      null_at(1));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TEST_F(ConcatenateListElementsTest, ListsOfListsOfStructsNoNull)
{
  using structs_col = cudf::test::structs_column_wrapper;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using strings_col = cudf::test::strings_column_wrapper;

  // Input:
  // [ [{1, 11, "1"}, {2, 12, "2"}], [{3, 13, "3"}], [{4, 14, "4"}, {5, 15, "5"}, {6, 16, "6"}] ]
  // [ [] ]
  // [ [{7, 17, "7"}, {8, 18, "8"}], [{9, 19, "9"}, {10, 110, "10"}] ]
  auto const input = [] {
    auto child = [] {
      auto child1  = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      auto child2  = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
      auto child3  = strings_col{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
      auto structs = structs_col{{child1, child2, child3}};
      auto offsets = int32s_col{0, 2, 3, 6, 6, 8, 10};
      return cudf::make_lists_column(6, offsets.release(), structs.release(), 0, {});
    }();

    auto offsets = int32s_col{0, 3, 4, 6};
    return cudf::make_lists_column(3, offsets.release(), std::move(child), 0, {});
  }();

  // Output:
  // [{1, 11, "1"}, {2, 12, "2"}, {3, 13, "3"}, {4, 14, "4"}, {5, 15, "5"}, {6, 16, "6"}]
  // []
  // [{7, 17, "7"}, {8, 18, "8"}, {9, 19, "9"}, {10, 110, "10"}]
  auto const expected = [] {
    auto child1  = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto child2  = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
    auto child3  = strings_col{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
    auto structs = structs_col{{child1, child2, child3}};
    auto offsets = int32s_col{0, 6, 6, 10};
    return cudf::make_lists_column(3, offsets.release(), structs.release(), 0, {});
  }();

  auto const results = cudf::lists::concatenate_list_elements(*input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
}

TEST_F(ConcatenateListElementsTest, ListsOfListsOfStructsWithNull)
{
  using structs_col = cudf::test::structs_column_wrapper;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using strings_col = cudf::test::strings_column_wrapper;

  // Input:
  // [ [{1, 11, "1"}, {null, null, null}], [{3, 13, "3"}], NULL ]
  // [ [{4, 14, "4"}, {5, 15, "5"}, {null, null, null}] ]
  // [ [{7, 17, "7"}, {null, null, null}], [{9, 19, "9"}, {10, 110, "10"}] ]
  auto const input = [] {
    auto child = [] {
      auto child1                  = int32s_col{1, null, 3, 4, 5, null, 7, null, 9, 10};
      auto child2                  = int32s_col{11, null, 13, 14, 15, null, 17, null, 19, 110};
      auto child3                  = strings_col{"1", "", "3", "4", "5", "", "7", "", "9", "10"};
      auto structs                 = structs_col{{child1, child2, child3}, nulls_at({1, 5, 7})};
      auto offsets                 = int32s_col{0, 2, 3, 3, 6, 8, 10};
      auto const null_it           = null_at(2);  // null list
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 6);
      return cudf::make_lists_column(
        6, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto offsets = int32s_col{0, 3, 4, 6};
    return cudf::make_lists_column(3, offsets.release(), std::move(child), 0, {});
  }();

  // Concatenate with ignoring null lists.
  {
    // Output:
    // [{1, 11, "1"}, {null, null, null}, {3, 13, "3"}]
    // [{4, 14, "4"}, {5, 15, "5"}, {null, null, null}]
    // [{7, 17, "7"}, {null, null, null}, {9, 19, "9"}, {10, 110, "10"}]
    auto const expected = [] {
      auto child1  = int32s_col{1, null, 3, 4, 5, null, 7, null, 9, 10};
      auto child2  = int32s_col{11, null, 13, 14, 15, null, 17, null, 19, 110};
      auto child3  = strings_col{"1", "", "3", "4", "5", "", "7", "", "9", "10"};
      auto structs = structs_col{{child1, child2, child3}, nulls_at({1, 5, 7})};
      auto offsets = int32s_col{0, 3, 6, 10};
      return cudf::make_lists_column(3, offsets.release(), structs.release(), 0, {});
    }();

    auto const results = cudf::lists::concatenate_list_elements(*input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with ignoring null lists and sliced input.
  {
    // Output:
    // [{4, 14, "4"}, {5, 15, "5"}, {null, null, null}]
    auto const expected = [] {
      auto child1  = int32s_col{4, 5, null};
      auto child2  = int32s_col{14, 15, null};
      auto child3  = strings_col{"4", "5", ""};
      auto structs = structs_col{{child1, child2, child3}, null_at(2)};
      auto offsets = int32s_col{0, 3};
      return cudf::make_lists_column(1, offsets.release(), structs.release(), 0, {});
    }();

    auto const sliced_input = cudf::slice(*input, {1, 2})[0];
    auto const results      = cudf::lists::concatenate_list_elements(sliced_input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with `concatenate_null_policy::NULLIFY_OUTPUT_ROW`.
  {
    // Output:
    // NULL
    // [{4, 14, "4"}, {5, 15, "5"}, {null, null, null}]
    // [{7, 17, "7"}, {null, null, null}, {9, 19, "9"}, {10, 110, "10"}]
    auto const expected = [] {
      auto child1                  = int32s_col{4, 5, null, 7, null, 9, 10};
      auto child2                  = int32s_col{14, 15, null, 17, null, 19, 110};
      auto child3                  = strings_col{"4", "5", "", "7", "", "9", "10"};
      auto structs                 = structs_col{{child1, child2, child3}, nulls_at({2, 4})};
      auto offsets                 = int32s_col{0, 0, 3, 7};
      auto const null_it           = null_at(0);  // null row
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 3);
      return cudf::make_lists_column(
        3, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto const results = cudf::lists::concatenate_list_elements(
      *input, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with `concatenate_null_policy::NULLIFY_OUTPUT_ROW` and sliced input.
  {
    // Output:
    // NULL
    // [{4, 14, "4"}, {5, 15, "5"}, {null, null, null}]
    auto const expected = [] {
      auto child1                  = int32s_col{4, 5, null};
      auto child2                  = int32s_col{14, 15, null};
      auto child3                  = strings_col{"4", "5", ""};
      auto structs                 = structs_col{{child1, child2, child3}, null_at(2)};
      auto offsets                 = int32s_col{0, 0, 3};
      auto const null_it           = null_at(0);  // null row
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 2);
      return cudf::make_lists_column(
        2, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto const sliced_input = cudf::slice(*input, {0, 2})[0];
    auto const results      = cudf::lists::concatenate_list_elements(
      sliced_input, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }
}

TEST_F(ConcatenateListElementsTest, ListsOfListsOfStructsHavingListsNoNull)
{
  using structs_col = cudf::test::structs_column_wrapper;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using lists_col   = cudf::test::lists_column_wrapper<int32_t>;

  // clang-format off
  // Input:
  // [ [{1, 11, [1, 1]}, {2, 12, [2]}], [{3, 13, [3, 3]}], [{4, 14, []}, {5, 15, [5, 5, 5]}, {6, 16, [6, 6]}] ]
  // [ [] ]
  // [ [{7, 17, [7]}, {8, 18, [8]}], [{9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}] ]
  // clang-format on
  auto const input = [] {
    auto child = [] {
      auto child1 = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      auto child2 = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
      auto child3 =
        lists_col{{1, 1}, {2}, {3, 3}, {}, {5, 5, 5}, {6, 6}, {7}, {8}, {9, 9}, {10, 10, 10, 10}};
      auto structs = structs_col{{child1, child2, child3}};
      auto offsets = int32s_col{0, 2, 3, 6, 6, 8, 10};
      return cudf::make_lists_column(6, offsets.release(), structs.release(), 0, {});
    }();

    auto offsets = int32s_col{0, 3, 4, 6};
    return cudf::make_lists_column(3, offsets.release(), std::move(child), 0, {});
  }();

  // clang-format off
  // Output:
  // [{1, 11, [1, 1]}, {2, 12, [2]}, {3, 13, [3, 3]}, {4, 14, []}, {5, 15, [5, 5, 5]}, {6, 16, [6, 6]}]
  // []
  // [{7, 17, [7]}, {8, 18, [8]}, {9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}]
  // clang-format on
  auto const expected = [] {
    auto child1 = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto child2 = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
    auto child3 =
      lists_col{{1, 1}, {2}, {3, 3}, {}, {5, 5, 5}, {6, 6}, {7}, {8}, {9, 9}, {10, 10, 10, 10}};
    auto structs = structs_col{{child1, child2, child3}};
    auto offsets = int32s_col{0, 6, 6, 10};
    return cudf::make_lists_column(3, offsets.release(), structs.release(), 0, {});
  }();

  auto const results = cudf::lists::concatenate_list_elements(*input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
}

TEST_F(ConcatenateListElementsTest, ListsOfListsOfStructsHavingListsWithNulls)
{
  using structs_col = cudf::test::structs_column_wrapper;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using lists_col   = cudf::test::lists_column_wrapper<int32_t>;

  // Input:
  // [ [{1, 11, [1, 1]}, {2, 12, [2]}], [{3, 13, [3, 3]}] ]
  // [ [{4, 14, null}, {5, 15, [5, 5, 5]}, {6, 16, [6, 6]}], NULL ]
  // [ [{7, 17, [7]}, {8, 18, [8]}], [{9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}] ]
  auto const input = [] {
    auto child = [] {
      auto child1 = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      auto child2 = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
      auto child3 =
        lists_col{{{1, 1}, {2}, {3, 3}, {}, {5, 5, 5}, {6, 6}, {7}, {8}, {9, 9}, {10, 10, 10, 10}},
                  null_at(3)};
      auto structs                 = structs_col{{child1, child2, child3}};
      auto offsets                 = int32s_col{0, 2, 3, 6, 6, 8, 10};
      auto const null_it           = null_at(3);  // null list
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 6);
      return cudf::make_lists_column(
        6, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto offsets = int32s_col{0, 2, 4, 6};
    return cudf::make_lists_column(3, offsets.release(), std::move(child), 0, {});
  }();

  // Concatenate with ignoring null lists.
  {
    // Output:
    // [{1, 11, [1, 1]}, {2, 12, [2]}, {3, 13, [3, 3]}]
    // [{4, 14, null}, {5, 15, [5, 5, 5]}, {6, 16, [6, 6]}]
    // [{7, 17, [7]}, {8, 18, [8]}, {9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}]
    auto const expected = [] {
      auto child1 = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      auto child2 = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
      auto child3 =
        lists_col{{{1, 1}, {2}, {3, 3}, {}, {5, 5, 5}, {6, 6}, {7}, {8}, {9, 9}, {10, 10, 10, 10}},
                  null_at(3)};
      auto structs = structs_col{{child1, child2, child3}};
      auto offsets = int32s_col{0, 3, 6, 10};
      return cudf::make_lists_column(3, offsets.release(), structs.release(), 0, {});
    }();

    auto const results = cudf::lists::concatenate_list_elements(*input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with ignoring null lists and sliced input.
  {
    // Output:
    // [{4, 14, null}, {5, 15, [5, 5, 5]}, {6, 16, [6, 6]}]
    auto const expected = [] {
      auto child1  = int32s_col{4, 5, 6};
      auto child2  = int32s_col{14, 15, 16};
      auto child3  = lists_col{{{}, {5, 5, 5}, {6, 6}}, null_at(0)};
      auto structs = structs_col{{child1, child2, child3}};
      auto offsets = int32s_col{0, 3};
      return cudf::make_lists_column(1, offsets.release(), structs.release(), 0, {});
    }();

    auto const sliced_input = cudf::slice(*input, {1, 2})[0];
    auto const results      = cudf::lists::concatenate_list_elements(sliced_input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with `concatenate_null_policy::NULLIFY_OUTPUT_ROW`.
  {
    // Output:
    // [{1, 11, [1, 1]}, {2, 12, [2]}, {3, 13, [3, 3]}]
    // NULL
    // [{7, 17, [7]}, {8, 18, [8]}, {9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}]
    auto const expected = [] {
      auto child1 = int32s_col{1, 2, 3, 7, 8, 9, 10};
      auto child2 = int32s_col{11, 12, 13, 17, 18, 19, 110};
      auto child3 =
        lists_col{{{1, 1}, {2}, {3, 3}, {7}, {8}, {9, 9}, {10, 10, 10, 10}}, no_nulls()};
      auto structs                 = structs_col{{child1, child2, child3}};
      auto offsets                 = int32s_col{0, 3, 3, 7};
      auto const null_it           = null_at(1);  // null row
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 3);
      return cudf::make_lists_column(
        3, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto const results = cudf::lists::concatenate_list_elements(
      *input, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with `concatenate_null_policy::NULLIFY_OUTPUT_ROW` and sliced input.
  {
    // Output:
    // NULL
    // [{7, 17, [7]}, {8, 18, [8]}, {9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}]
    auto const expected = [] {
      auto child1                  = int32s_col{7, 8, 9, 10};
      auto child2                  = int32s_col{17, 18, 19, 110};
      auto child3                  = lists_col{{{7}, {8}, {9, 9}, {10, 10, 10, 10}}, no_nulls()};
      auto structs                 = structs_col{{child1, child2, child3}};
      auto offsets                 = int32s_col{0, 0, 4};
      auto const null_it           = null_at(0);  // null row
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 2);
      return cudf::make_lists_column(
        2, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto const sliced_input = cudf::slice(*input, {1, 3})[0];
    auto const results      = cudf::lists::concatenate_list_elements(
      sliced_input, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }
}

// Tests for the child.size()==0 early-exit guard added to fix NVIDIA/cudf#22146.
// When the inner list column has 0 rows, the offsets buffer may be unallocated;
// the guard must build the result without dereferencing that buffer.

// Subcase A: child.size()==1 (guard does NOT fire) — baseline check.
TEST_F(ConcatenateListElementsTest, EmptyInnerListColumnChildSizeOne)
{
  // list<list<int32>> with 1 outer row containing 1 empty inner list: [[]]
  // child.size() == 1; the new guard does not apply here.
  // Use brace nesting to create a proper list<list<int32>> (2-level nesting).
  // The child is a list<int32> with 1 empty row.
  auto const col      = IntListsCol{{{}}};  // list<list<int32>> with 1 outer row: [[]]
  auto const results  = cudf::lists::concatenate_list_elements(col);
  auto const expected = IntListsCol{{}};  // list<int32> with 1 empty row: [[]] → []
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
}

// Subcase B: child.size()==0, single valid-but-empty outer row — guard fires.
TEST_F(ConcatenateListElementsTest, EmptyInnerListColumnChildSizeZero)
{
  // list<list<int32>> with 1 outer row that is an empty list-of-lists (0 inner lists).
  // Outer offsets: {0, 0}, inner column: list<int32> with 0 rows.
  auto offsets        = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0};
  auto inner_offsets  = cudf::test::fixed_width_column_wrapper<int32_t>{0};
  auto inner_elements = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto inner_col =
    cudf::make_lists_column(0, inner_offsets.release(), inner_elements.release(), 0, {});
  auto const col = cudf::make_lists_column(1, offsets.release(), std::move(inner_col), 0, {});

  auto const results  = cudf::lists::concatenate_list_elements(*col);
  auto const expected = IntListsCol{{}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
}

// Subcase C: child.size()==0 with a null outer row — verifies null mask is preserved.
TEST_F(ConcatenateListElementsTest, EmptyInnerListColumnWithNullRow)
{
  // list<list<int32>> with 2 outer rows: row 0 is null, row 1 is a valid empty list.
  // child.size() == 0; the null mask must be copied to the output.
  auto offsets        = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0};
  auto inner_offsets  = cudf::test::fixed_width_column_wrapper<int32_t>{0};
  auto inner_elements = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto inner_col =
    cudf::make_lists_column(0, inner_offsets.release(), inner_elements.release(), 0, {});
  auto const null_it           = cudf::test::iterators::null_at(0);
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 2);
  auto const col               = cudf::make_lists_column(
    2, offsets.release(), std::move(inner_col), null_count, std::move(null_mask));

  auto const results = cudf::lists::concatenate_list_elements(*col);

  // Expected: list<int32> with 2 rows — null row 0, empty row 1.
  // concatenate_list_elements reduces list<list<int32>> → list<int32>.
  auto exp_offsets                     = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0};
  auto exp_elements                    = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto [exp_null_mask, exp_null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 2);
  auto const expected                  = cudf::make_lists_column(
    2, exp_offsets.release(), exp_elements.release(), exp_null_count, std::move(exp_null_mask));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
}

// Subcase D: child.size()==0 with a struct grandchild — verifies the guard preserves
// child column structure (field columns) when the grandchild is a struct type.
// Addresses review feedback on NVIDIA/cudf#22147 (pmattione-nvidia).
TEST_F(ConcatenateListElementsTest, EmptyInnerListColumnChildSizeZeroWithStructGrandchild)
{
  // list<list<struct<int32,int32>>> with 1 outer row that is an empty list-of-lists.
  // child.size() == 0; grandchild is struct<int32,int32> with 0 rows.
  // The guard must copy the grandchild (preserving its field columns) without
  // dereferencing the child offsets buffer.
  using structs_col = cudf::test::structs_column_wrapper;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto const col = [] {
    auto s_child1      = int32s_col{};
    auto s_child2      = int32s_col{};
    auto grandchild    = structs_col{{s_child1, s_child2}};  // struct<int32,int32>, 0 rows
    auto inner_offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0};
    auto inner_col     = cudf::make_lists_column(0,
                                             inner_offsets.release(),
                                             grandchild.release(),
                                             0,
                                                 {});  // list<struct<int32,int32>>, 0 rows
    auto outer_offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0};
    return cudf::make_lists_column(1, outer_offsets.release(), std::move(inner_col), 0, {});
  }();

  // Expected: list<struct<int32,int32>> with 1 empty row (the outer list flattened).
  auto const expected = [] {
    auto s_child1   = int32s_col{};
    auto s_child2   = int32s_col{};
    auto grandchild = structs_col{{s_child1, s_child2}};
    auto offsets    = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0};
    return cudf::make_lists_column(1, offsets.release(), grandchild.release(), 0, {});
  }();

  auto const results = cudf::lists::concatenate_list_elements(*col);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
}
