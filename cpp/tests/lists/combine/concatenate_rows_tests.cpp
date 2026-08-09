/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/utilities/error.hpp>

using namespace cudf::test::iterators;

namespace {
using StrListsCol = cudf::test::lists_column_wrapper<cudf::string_view>;
using IntListsCol = cudf::test::lists_column_wrapper<int32_t>;
using IntCol      = cudf::test::fixed_width_column_wrapper<int32_t>;
using TView       = cudf::table_view;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};
}  // namespace

struct ListConcatenateRowsTest : public cudf::test::BaseFixture {};

TEST_F(ListConcatenateRowsTest, InvalidInput)
{
  // Empty input table
  EXPECT_THROW(cudf::lists::concatenate_rows(TView{}), cudf::logic_error);

  // Input table contains non-list column
  {
    auto const col1 = IntCol{}.release();
    auto const col2 = IntListsCol{}.release();
    EXPECT_THROW(cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}}),
                 cudf::logic_error);
  }

  // Types mismatch
  {
    auto const col1 = IntListsCol{}.release();
    auto const col2 = StrListsCol{}.release();
    EXPECT_THROW(cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}}),
                 cudf::data_type_error);
  }
}

template <typename T>
struct ListConcatenateRowsTypedTest : public cudf::test::BaseFixture {};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(ListConcatenateRowsTypedTest, TypesForTest);

TYPED_TEST(ListConcatenateRowsTypedTest, ConcatenateEmptyColumns)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col     = ListsCol{}.release();
  auto const results = cudf::lists::concatenate_rows(TView{{col->view(), col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, verbosity);
}

TYPED_TEST(ListConcatenateRowsTypedTest, ConcatenateOneColumnNotNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col     = ListsCol{{1, 2}, {3, 4}, {5, 6}}.release();
  auto const results = cudf::lists::concatenate_rows(TView{{col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, verbosity);
}

TYPED_TEST(ListConcatenateRowsTypedTest, ConcatenateOneColumnWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col = ListsCol{{ListsCol{{1, 2, null}, null_at(2)},
                             ListsCol{} /*NULL*/,
                             ListsCol{{null, 3, 4, 4, 4, 4}, null_at(0)},
                             ListsCol{5, 6}},
                            null_at(1)}
                     .release();
  auto const results = cudf::lists::concatenate_rows(TView{{col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, verbosity);
}

TYPED_TEST(ListConcatenateRowsTypedTest, SimpleInputNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1        = ListsCol{{1, 2}, {3, 4}, {5, 6}}.release();
  auto const empty_lists = ListsCol{ListsCol{}, ListsCol{}, ListsCol{}}.release();
  auto const col2        = ListsCol{{7, 8}, {9, 10}, {11, 12}}.release();
  auto const expected    = ListsCol{{1, 2, 7, 8}, {3, 4, 9, 10}, {5, 6, 11, 12}}.release();
  auto const results =
    cudf::lists::concatenate_rows(TView{{col1->view(), empty_lists->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListConcatenateRowsTypedTest, SimpleInputWithNullableChild)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1        = ListsCol{{1, 2}, ListsCol{{null}, null_at(0)}, {5, 6}}.release();
  auto const empty_lists = ListsCol{{ListsCol{}, ListsCol{}, ListsCol{}}, null_at(2)}.release();
  auto const col2        = ListsCol{{7, 8}, {9, 10}, {11, 12}}.release();
  auto const expected =
    ListsCol{{1, 2, 7, 8}, ListsCol{{null, 9, 10}, null_at(0)}, {5, 6, 11, 12}}.release();
  auto const results =
    cudf::lists::concatenate_rows(TView{{col1->view(), empty_lists->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TEST_F(ListConcatenateRowsTest, SimpleInputStringsColumnsNoNull)
{
  auto const col1 = StrListsCol{StrListsCol{"Tomato", "Apple"},
                                StrListsCol{"Banana", "Kiwi", "Cherry"},
                                StrListsCol{"Coconut"}}
                      .release();
  auto const col2 =
    StrListsCol{StrListsCol{"Orange"}, StrListsCol{"Lemon", "Peach"}, StrListsCol{}}.release();
  auto const expected = StrListsCol{StrListsCol{"Tomato", "Apple", "Orange"},
                                    StrListsCol{"Banana", "Kiwi", "Cherry", "Lemon", "Peach"},
                                    StrListsCol{"Coconut"}}
                          .release();
  auto const results = cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TEST_F(ListConcatenateRowsTest, SimpleInputStringsColumnsWithNullableChild)
{
  auto const col1 = StrListsCol{StrListsCol{"Tomato", "Apple"},
                                StrListsCol{"Banana", "Kiwi", "Cherry"},
                                StrListsCol{"Coconut"}}
                      .release();
  auto const col2 = StrListsCol{
    StrListsCol{"Orange"},
    StrListsCol{{"Lemon", "Peach"}, null_at(1)},
    StrListsCol{}}.release();
  auto const expected =
    StrListsCol{StrListsCol{"Tomato", "Apple", "Orange"},
                StrListsCol{{"Banana", "Kiwi", "Cherry", "Lemon", "Peach"}, null_at(4)},
                StrListsCol{"Coconut"}}
      .release();
  auto const results = cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListConcatenateRowsTypedTest, SimpleInputWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1 = ListsCol{{ListsCol{{1, null, 3, 4}, null_at(1)},
                              ListsCol{{null, 2, 3, 4}, null_at(0)},
                              ListsCol{{null, 2, 3, 4}, null_at(0)},
                              ListsCol{} /*NULL*/,
                              ListsCol{{1, 2, null, 4}, null_at(2)},
                              ListsCol{{1, 2, 3, null}, null_at(3)},
                              ListsCol{} /*NULL*/},
                             nulls_at({3, 6})}
                      .release();
  auto const col2 = ListsCol{{ListsCol{{10, 11, 12, null}, null_at(3)},
                              ListsCol{{13, 14, 15, 16, 17, null}, null_at(5)},
                              ListsCol{} /*NULL*/,
                              ListsCol{{null, 18}, null_at(0)},
                              ListsCol{{19, 20, null}, null_at(2)},
                              ListsCol{{null}, null_at(0)},
                              ListsCol{} /*NULL*/},
                             nulls_at({2, 6})}
                      .release();
  auto const col3 = ListsCol{{ListsCol{} /*NULL*/,
                              ListsCol{{20, null}, null_at(1)},
                              ListsCol{{null, 21, null, null}, nulls_at({0, 2, 3})},
                              ListsCol{},
                              ListsCol{22, 23, 24, 25},
                              ListsCol{{null, null, null, null, null}, all_nulls()},
                              ListsCol{} /*NULL*/},
                             nulls_at({0, 6})}
                      .release();

  // Ignore null list elements
  {
    auto const results =
      cudf::lists::concatenate_rows(TView{{col1->view(), col2->view(), col3->view()}});
    auto const expected =
      ListsCol{{ListsCol{{1, null, 3, 4, 10, 11, 12, null}, nulls_at({1, 7})},
                ListsCol{{null, 2, 3, 4, 13, 14, 15, 16, 17, null, 20, null}, nulls_at({0, 9, 11})},
                ListsCol{{null, 2, 3, 4, null, 21, null, null}, nulls_at({0, 4, 6, 7})},
                ListsCol{{null, 18}, null_at(0)},
                ListsCol{{1, 2, null, 4, 19, 20, null, 22, 23, 24, 25}, nulls_at({2, 6})},
                ListsCol{{1, 2, 3, null, null, null, null, null, null, null},
                         nulls_at({3, 4, 5, 6, 7, 8, 9})},
                ListsCol{} /*NULL*/},
               null_at(6)}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
  }

  // Null list rows result in null list rows
  {
    auto const results =
      cudf::lists::concatenate_rows(TView{{col1->view(), col2->view(), col3->view()}},
                                    cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected =
      ListsCol{{ListsCol{} /*NULL*/,
                ListsCol{{null, 2, 3, 4, 13, 14, 15, 16, 17, null, 20, null}, nulls_at({0, 9, 11})},
                ListsCol{} /*NULL*/,
                ListsCol{} /*NULL*/,
                ListsCol{{1, 2, null, 4, 19, 20, null, 22, 23, 24, 25}, nulls_at({2, 6})},
                ListsCol{{1, 2, 3, null, null, null, null, null, null, null},
                         nulls_at({3, 4, 5, 6, 7, 8, 9})},
                ListsCol{} /*NULL*/},
               nulls_at({0, 2, 3, 6})}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
  }
}

TEST_F(ListConcatenateRowsTest, SimpleInputStringsColumnsWithNulls)
{
  auto const col1 =
    StrListsCol{
      StrListsCol{{"Tomato", "Bear" /*NULL*/, "Apple"}, null_at(1)},
      StrListsCol{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/}, nulls_at({1, 4})},
      StrListsCol{"Coconut"}}
      .release();
  auto const col2 =
    StrListsCol{
      {StrListsCol{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/},
                   nulls_at({1, 2, 3})},
       StrListsCol{"Lemon", "Peach"},
       StrListsCol{{"Deer" /*NULL*/, "Snake" /*NULL*/, "Horse" /*NULL*/}, all_nulls()}}, /*NULL*/
      null_at(2)}
      .release();

  // Ignore null list elements
  {
    auto const results = cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}});
    auto const expected =
      StrListsCol{
        StrListsCol{
          {"Tomato", "" /*NULL*/, "Apple", "Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/},
          nulls_at({1, 4, 5, 6})},
        StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, "Lemon", "Peach"},
                    nulls_at({1, 4})},
        StrListsCol{"Coconut"}}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
  }

  // Null list rows result in null list rows
  {
    auto const results =
      cudf::lists::concatenate_rows(TView{{col1->view(), col2->view()}},
                                    cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected =
      StrListsCol{
        {StrListsCol{
           {"Tomato", "" /*NULL*/, "Apple", "Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/},
           nulls_at({1, 4, 5, 6})},
         StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, "Lemon", "Peach"},
                     nulls_at({1, 4})},
         StrListsCol{""} /*NULL*/},
        null_at(2)}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
  }
}

TEST_F(ListConcatenateRowsTest, SimpleInputStringsColumnsWithEmptyLists)
{
  auto const col1 =
    StrListsCol{StrListsCol{{"" /*NULL*/}, null_at(0)}, StrListsCol{"One"}}.release();
  auto const col2 =
    StrListsCol{StrListsCol{{"Tomato", "" /*NULL*/, "Apple"}, null_at(1)}, StrListsCol{"Two"}}
      .release();
  auto const col3 =
    StrListsCol{{StrListsCol{"Lemon", "Peach"}, StrListsCol{"Three"} /*NULL*/}, null_at(1)}
      .release();

  // Ignore null list elements
  {
    auto const results =
      cudf::lists::concatenate_rows(TView{{col1->view(), col2->view(), col3->view()}});
    auto const expected =
      StrListsCol{StrListsCol{{"" /*NULL*/, "Tomato", "" /*NULL*/, "Apple", "Lemon", "Peach"},
                              nulls_at({0, 2})},
                  StrListsCol{"One", "Two"}}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
  }

  // Null list rows result in null list rows
  {
    auto const results =
      cudf::lists::concatenate_rows(TView{{col1->view(), col2->view(), col3->view()}},
                                    cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected =
      StrListsCol{{StrListsCol{{"" /*NULL*/, "Tomato", "" /*NULL*/, "Apple", "Lemon", "Peach"},
                               nulls_at({0, 2})},
                   StrListsCol{""} /*NULL*/},
                  null_at(1)}
        .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
  }
}

TYPED_TEST(ListConcatenateRowsTypedTest, SlicedColumnsInputNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col_original = ListsCol{{1, 2, 3}, {2, 3}, {3, 4, 5, 6}, {5, 6}, {}, {7}}.release();
  auto const col1         = cudf::slice(col_original->view(), {0, 3})[0];
  auto const col2         = cudf::slice(col_original->view(), {1, 4})[0];
  auto const col3         = cudf::slice(col_original->view(), {2, 5})[0];
  auto const col4         = cudf::slice(col_original->view(), {3, 6})[0];
  auto const expected =
    ListsCol{{1, 2, 3, 2, 3, 3, 4, 5, 6, 5, 6}, {2, 3, 3, 4, 5, 6, 5, 6}, {3, 4, 5, 6, 5, 6, 7}}
      .release();
  auto const results = cudf::lists::concatenate_rows(TView{{col1, col2, col3, col4}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListConcatenateRowsTypedTest, SlicedColumnsInputWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col_original = ListsCol{{ListsCol{{null, 2, 3}, null_at(0)},
                                      ListsCol{2, 3}, /*NULL*/
                                      ListsCol{{3, null, 5, 6}, null_at(1)},
                                      ListsCol{5, 6}, /*NULL*/
                                      ListsCol{},     /*NULL*/
                                      ListsCol{7},
                                      ListsCol{8, 9, 10}},
                                     nulls_at({1, 3, 4})}
                              .release();
  auto const col1     = cudf::slice(col_original->view(), {0, 3})[0];
  auto const col2     = cudf::slice(col_original->view(), {1, 4})[0];
  auto const col3     = cudf::slice(col_original->view(), {2, 5})[0];
  auto const col4     = cudf::slice(col_original->view(), {3, 6})[0];
  auto const col5     = cudf::slice(col_original->view(), {4, 7})[0];
  auto const expected = ListsCol{
    ListsCol{{null, 2, 3, 3, null, 5, 6}, nulls_at({0, 4})},
    ListsCol{{3, null, 5, 6, 7}, null_at(1)},
    ListsCol{{3, null, 5, 6, 7, 8, 9, 10},
             null_at(1)}}.release();
  auto const results = cudf::lists::concatenate_rows(TView{{col1, col2, col3, col4, col5}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TEST_F(ListConcatenateRowsTest, SlicedStringsColumnsInputWithNulls)
{
  auto const col =
    StrListsCol{
      {StrListsCol{{"Tomato", "Bear" /*NULL*/, "Apple"}, null_at(1)},
       StrListsCol{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/},
                   nulls_at({1, 4})},
       StrListsCol{"Coconut"},
       StrListsCol{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/},
                   nulls_at({1, 2, 3})},
       StrListsCol{"Lemon", "Peach"},
       StrListsCol{{"Deer" /*NULL*/, "Snake" /*NULL*/, "Horse" /*NULL*/}, all_nulls()}}, /*NULL*/
      null_at(5)}
      .release();
  auto const col1 = cudf::slice(col->view(), {0, 3})[0];
  auto const col2 = cudf::slice(col->view(), {1, 4})[0];
  auto const col3 = cudf::slice(col->view(), {2, 5})[0];
  auto const col4 = cudf::slice(col->view(), {3, 6})[0];

  {
    auto const results  = cudf::lists::concatenate_rows(TView{{col1, col2, col3, col4}});
    auto const expected = StrListsCol{StrListsCol{{"Tomato",
                                                   "" /*NULL*/,
                                                   "Apple",
                                                   "Banana",
                                                   "" /*NULL*/,
                                                   "Kiwi",
                                                   "Cherry",
                                                   "" /*NULL*/,
                                                   "Coconut",
                                                   "Orange",
                                                   "" /*NULL*/,
                                                   "" /*NULL*/,
                                                   "" /*NULL*/},
                                                  nulls_at({1, 4, 7, 10, 11, 12})},
                                      StrListsCol{{"Banana",
                                                   "" /*NULL*/,
                                                   "Kiwi",
                                                   "Cherry",
                                                   "" /*NULL*/,
                                                   "Coconut",
                                                   "Orange",
                                                   "" /*NULL*/,
                                                   "" /*NULL*/,
                                                   "", /*NULL*/
                                                   "Lemon",
                                                   "Peach"},
                                                  nulls_at({1, 4, 7, 8, 9})},
                                      StrListsCol{{
                                                    "Coconut",
                                                    "Orange",
                                                    "" /*NULL*/,
                                                    "" /*NULL*/,
                                                    "", /*NULL*/
                                                    "Lemon",
                                                    "Peach",
                                                  },
                                                  nulls_at({2, 3, 4})}}
                            .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
  }

  {
    auto const results = cudf::lists::concatenate_rows(
      TView{{col1, col2, col3, col4}}, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected = StrListsCol{{StrListsCol{{"Tomato",
                                                    "" /*NULL*/,
                                                    "Apple",
                                                    "Banana",
                                                    "" /*NULL*/,
                                                    "Kiwi",
                                                    "Cherry",
                                                    "" /*NULL*/,
                                                    "Coconut",
                                                    "Orange",
                                                    "" /*NULL*/,
                                                    "" /*NULL*/,
                                                    "" /*NULL*/},
                                                   nulls_at({1, 4, 7, 10, 11, 12})},
                                       StrListsCol{{"Banana",
                                                    "" /*NULL*/,
                                                    "Kiwi",
                                                    "Cherry",
                                                    "" /*NULL*/,
                                                    "Coconut",
                                                    "Orange",
                                                    "" /*NULL*/,
                                                    "" /*NULL*/,
                                                    "", /*NULL*/
                                                    "Lemon",
                                                    "Peach"},
                                                   nulls_at({1, 4, 7, 8, 9})},
                                       StrListsCol{} /*NULL*/},
                                      null_at(2)}
                            .release();
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
  }
}

TEST_F(ListConcatenateRowsTest, StringsColumnsWithEmptyListTest)
{
  auto const col1 = StrListsCol{{"1", "2", "3", "4"}}.release();
  auto const col2 = StrListsCol{{"a", "b", "c"}}.release();
  auto const col3 = StrListsCol{StrListsCol{}}.release();
  auto const col4 = StrListsCol{{"x", "y", "" /*NULL*/, "z"}, null_at(2)}.release();
  auto const col5 = StrListsCol{{StrListsCol{}}, null_at(0)}.release();
  auto const expected =
    StrListsCol{{"1", "2", "3", "4", "a", "b", "c", "x", "y", "" /*NULL*/, "z"}, null_at(9)}
      .release();
  auto const results = cudf::lists::concatenate_rows(
    TView{{col1->view(), col2->view(), col3->view(), col4->view(), col5->view()}});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

struct ListConcatenateRowsNestedTypesTest : public cudf::test::BaseFixture {};

TEST_F(ListConcatenateRowsNestedTypesTest, Identity)
{
  // list<list<string>>

  // col 0
  cudf::test::lists_column_wrapper<cudf::string_view> l0{
    {{{{"whee", "yay", "bananas"}, nulls_at({1})}, {}},
     {{}},
     {{{"abc"}, {"def", "g", "xyw", "ijk"}, {"x", "y", "", "column"}}, nulls_at({0, 2})},
     {{"f", "tesla"}},
     {{"phone"}, {"hack", "table", "car"}}},
    nulls_at({3, 4})};

  // perform the concatenate
  cudf::table_view t({l0});
  auto result = cudf::lists::concatenate_rows(t);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, l0);
}

TEST_F(ListConcatenateRowsNestedTypesTest, List)
{
  // list<list<string>>

  // col 0
  cudf::test::lists_column_wrapper<cudf::string_view> l0{
    {{"whee", "yay", "bananas"}, {}},
    {{}},
    {{"abc"}, {"def", "g", "xyw", "ijk"}, {"x", "y", "", "column"}},
    {{"f", "tesla"}},
    {{"phone"}, {"hack", "table", "car"}}};

  // col1
  cudf::test::lists_column_wrapper<cudf::string_view> l1{
    {{}},
    {{"arg"}, {"mno", "ampere"}, {"gpu"}, {"def"}},
    {{"", "hhh"}},
    {{"warp", "donuts", "parking"}, {"", "apply", "twelve", "mouse", "bbb"}, {"bbb", "pom"}, {}},
    {{}}};

  // perform the concatenate
  cudf::table_view t({l0, l1});
  auto result = cudf::lists::concatenate_rows(t);

  // expected
  cudf::test::lists_column_wrapper<cudf::string_view> expected{
    {{"whee", "yay", "bananas"}, {}, {}},
    {{}, {"arg"}, {"mno", "ampere"}, {"gpu"}, {"def"}},
    {{"abc"}, {"def", "g", "xyw", "ijk"}, {"x", "y", "", "column"}, {"", "hhh"}},
    {{"f", "tesla"},
     {"warp", "donuts", "parking"},
     {"", "apply", "twelve", "mouse", "bbb"},
     {"bbb", "pom"},
     {}},
    {{"phone"}, {"hack", "table", "car"}, {}}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(ListConcatenateRowsNestedTypesTest, ListWithNulls)
{
  // list<list<string>>

  // clang-format off

  // col 0
  cudf::test::lists_column_wrapper<cudf::string_view>
    l0{ {
          {{{"whee", "yay", "bananas"}, nulls_at({1})}, {}},
          {{}},
          {{{"abc"}, {"def", "g", "xyw", "ijk"}, {"x", "y", "", "column"}},       nulls_at({0, 2})},
          {{"f", "tesla"}},
          {{"phone"}, {"hack", "table", "car"}}
        }, nulls_at({3, 4}) };

  // col1
  cudf::test::lists_column_wrapper<cudf::string_view>
    l1{ {
          {{}},
          {{"arg"}, {"mno", "ampere"}, {"gpu"}, {"def"}},
          {{{{"", "hhh"}, nulls_at({0})}, {"www"}},                               nulls_at({1})},
          {{"warp", "donuts", "parking"}, { "", "apply", "twelve", "mouse", "bbb"}, {"bbb", "pom"}, {}},
          {{}}
        }, nulls_at({4}) };

  // col2
  cudf::test::lists_column_wrapper<cudf::string_view>
    l2{ {
          {{"monitor", "sugar"}},
          {{"spurs", "garlic"}, {"onion", "shallot", "carrot"}},
          {{"cars", "trucks", "planes"}, {"abc"}, {"mno", "pqr"}},
          {{}, {"ram", "cpu", "disk"}, {}},
          {{"round"}, {"square"}}
        }, nulls_at({0, 4}) };

  // concatenate_policy::IGNORE_NULLS
  {
    // perform the concatenate
    cudf::table_view t({l0, l1, l2});
    auto result = cudf::lists::concatenate_rows(t, cudf::lists::concatenate_null_policy::IGNORE);

    // expected
    cudf::test::lists_column_wrapper<cudf::string_view>
      expected{ {
                  {{{"whee", "yay", "bananas"}, nulls_at({1})}, {}, {}},
                  {{}, {"arg"}, {"mno", "ampere"}, {"gpu"}, {"def"}, {"spurs", "garlic"}, {"onion", "shallot", "carrot"}},
                  {{{"abc"}, {"def", "g", "xyw", "ijk"}, {"x", "y", "", "column"},
                    {{"", "hhh"}, nulls_at({0})}, {"www"}, {"cars", "trucks", "planes"}, {"abc"}, {"mno", "pqr"}},
                      nulls_at({0, 2, 4}) },
                  {{"warp", "donuts", "parking"}, { "", "apply", "twelve", "mouse", "bbb"}, {"bbb", "pom"}, {}, {}, {"ram", "cpu", "disk"}, {}},
                  {{}}
                }, nulls_at({4}) };

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  // concatenate_policy::NULLIFY_OUTPUT_ROW
  {
    // perform the concatenate
    cudf::table_view t({l0, l1, l2});
    auto result = cudf::lists::concatenate_rows(t, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);

    // expected
    cudf::test::lists_column_wrapper<cudf::string_view>
      expected{ {
                  {{}},
                  {{}, {"arg"}, {"mno", "ampere"}, {"gpu"}, {"def"}, {"spurs", "garlic"}, {"onion", "shallot", "carrot"}},
                  {{{"abc"}, {"def", "g", "xyw", "ijk"}, {"x", "y", "", "column"},
                    {{"", "hhh"}, nulls_at({0})}, {"www"}, {"cars", "trucks", "planes"}, {"abc"}, {"mno", "pqr"}},
                      nulls_at({0, 2, 4}) },
                  {{}},
                  {{}}
                }, nulls_at({0, 3, 4}) };

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  // clang-format on
}

TEST_F(ListConcatenateRowsNestedTypesTest, ListWithNullsSliced)
{
  // list<list<string>>

  // clang-format off

  // col 0
  cudf::test::lists_column_wrapper<cudf::string_view>
    unsliced_l0{ {
          {{{"whee", "yay", "bananas"}, nulls_at({1})}, {}},
          {{}},
          {{{"abc"}, {"def", "g", "xyw", "ijk"}, {"x", "y", "", "column"}},       nulls_at({0, 2})},
          {{"f", "tesla"}},
          {{"phone"}, {"hack", "table", "car"}}
        }, nulls_at({3, 4}) };
  auto l0 = cudf::split(unsliced_l0, {2})[1];

  // col1
  cudf::test::lists_column_wrapper<cudf::string_view>
    unsliced_l1{ {
          {{}},
          {{"arg"}, {"mno", "ampere"}, {"gpu"}, {"def"}},
          {{{{"", "hhh"}, nulls_at({0})}, {"www"}},                               nulls_at({1})},
          {{"warp", "donuts", "parking"}, { "", "apply", "twelve", "mouse", "bbb"}, {"bbb", "pom"}, {}},
          {{}}
        }, nulls_at({4}) };
  auto l1 = cudf::split(unsliced_l1, {2})[1];

  // concatenate_policy::IGNORE_NULLS
  {
    // perform the concatenate
    cudf::table_view t({l0, l1});
    auto result = cudf::lists::concatenate_rows(t, cudf::lists::concatenate_null_policy::IGNORE);

    // expected
    cudf::test::lists_column_wrapper<cudf::string_view>
      expected{ { {{{"abc"}, {"def", "g", "xyw", "ijk"}, {"x", "y", "", "column"},
                    {{"", "hhh"}, nulls_at({0})}, {"www"}},                           nulls_at({0, 2, 4}) },
                  {{"warp", "donuts", "parking"}, { "", "apply", "twelve", "mouse", "bbb"}, {"bbb", "pom"}, {}},
                  {{}}
                }, nulls_at({2}) };

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  // concatenate_policy::NULLIFY_OUTPUT_ROW
  {
    // perform the concatenate
    cudf::table_view t({l0, l1});
    auto result = cudf::lists::concatenate_rows(t, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);

    // expected
    cudf::test::lists_column_wrapper<cudf::string_view>
      expected{ { {{{"abc"}, {"def", "g", "xyw", "ijk"}, {"x", "y", "", "column"},
                    {{"", "hhh"}, nulls_at({0})}, {"www"}},                           nulls_at({0, 2, 4}) },
                  {{}},
                  {{}}
                }, nulls_at({1, 2}) };

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }

  // clang-format on
}

TEST_F(ListConcatenateRowsNestedTypesTest, Struct)
{
  // list<struct<int, string>>

  // col 0
  cudf::test::fixed_width_column_wrapper<int> s0_0{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::test::strings_column_wrapper s0_1{
    "whee", "yay", "bananas", "abc", "def", "g", "xyw", "ijk"};
  std::vector<std::unique_ptr<cudf::column>> s0_children;
  s0_children.push_back(s0_0.release());
  s0_children.push_back(s0_1.release());
  cudf::test::structs_column_wrapper s0(std::move(s0_children));
  cudf::test::fixed_width_column_wrapper<cudf::size_type> l0_offsets{0, 2, 2, 5, 6, 8};
  auto const l0_size = static_cast<cudf::column_view>(l0_offsets).size() - 1;
  auto l0            = cudf::make_lists_column(l0_size, l0_offsets.release(), s0.release(), 0, {});

  // col1
  cudf::test::fixed_width_column_wrapper<int> s1_0{
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  cudf::test::strings_column_wrapper s1_1{"arg",
                                          "mno",
                                          "ampere",
                                          "gpu",
                                          "",
                                          "hhh",
                                          "warp",
                                          "donuts",
                                          "parking",
                                          "",
                                          "apply",
                                          "twelve",
                                          "mouse",
                                          "bbb",
                                          "pom"};
  std::vector<std::unique_ptr<cudf::column>> s1_children;
  s1_children.push_back(s1_0.release());
  s1_children.push_back(s1_1.release());
  cudf::test::structs_column_wrapper s1(std::move(s1_children));
  cudf::test::fixed_width_column_wrapper<cudf::size_type> l1_offsets{0, 0, 4, 7, 15, 15};
  auto const l1_size = static_cast<cudf::column_view>(l1_offsets).size() - 1;
  auto l1            = cudf::make_lists_column(l1_size, l1_offsets.release(), s1.release(), 0, {});

  // perform the concatenate
  cudf::table_view t({*l0, *l1});
  auto result = cudf::lists::concatenate_rows(t);

  // expected
  cudf::test::fixed_width_column_wrapper<int> se_0{0, 1,  10, 11, 12, 13, 2,  3,  4,  14, 15, 16,
                                                   5, 17, 18, 19, 20, 21, 22, 23, 24, 6,  7};
  cudf::test::strings_column_wrapper se_1{"whee",    "yay",    "arg",     "mno", "ampere", "gpu",
                                          "bananas", "abc",    "def",     "",    "hhh",    "warp",
                                          "g",       "donuts", "parking", "",    "apply",  "twelve",
                                          "mouse",   "bbb",    "pom",     "xyw", "ijk"};
  std::vector<std::unique_ptr<cudf::column>> se_children;
  se_children.push_back(se_0.release());
  se_children.push_back(se_1.release());
  cudf::test::structs_column_wrapper se(std::move(se_children));
  cudf::test::fixed_width_column_wrapper<cudf::size_type> le_offsets{0, 2, 6, 12, 21, 23};
  auto const le_size = static_cast<cudf::column_view>(le_offsets).size() - 1;
  auto expected      = cudf::make_lists_column(le_size, le_offsets.release(), se.release(), 0, {});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(ListConcatenateRowsNestedTypesTest, StructWithNulls)
{
  // list<struct<int, string>>

  // col 0
  cudf::test::fixed_width_column_wrapper<int> s0_0{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::test::strings_column_wrapper s0_1{
    {"whee", "yay", "bananas", "abc", "def", "g", "xyw", "ijk"}, nulls_at({1, 3, 4})};
  std::vector<std::unique_ptr<cudf::column>> s0_children;
  s0_children.push_back(s0_0.release());
  s0_children.push_back(s0_1.release());
  cudf::test::structs_column_wrapper s0(std::move(s0_children));
  cudf::test::fixed_width_column_wrapper<cudf::size_type> l0_offsets{0, 2, 2, 5, 6, 8};
  auto const l0_size = static_cast<cudf::column_view>(l0_offsets).size() - 1;
  std::vector<bool> l0_validity{false, true, true, false, true};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(l0_validity.begin(), l0_validity.end());
  auto l0 = cudf::make_lists_column(
    l0_size, l0_offsets.release(), s0.release(), null_count, std::move(null_mask));
  l0 = cudf::purge_nonempty_nulls(l0->view());

  // col1
  cudf::test::fixed_width_column_wrapper<int> s1_0{
    {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, nulls_at({14})};
  cudf::test::strings_column_wrapper s1_1{"arg",
                                          "mno",
                                          "ampere",
                                          "gpu",
                                          "",
                                          "hhh",
                                          "warp",
                                          "donuts",
                                          "parking",
                                          "",
                                          "apply",
                                          "twelve",
                                          "mouse",
                                          "bbb",
                                          "pom"};
  std::vector<std::unique_ptr<cudf::column>> s1_children;
  s1_children.push_back(s1_0.release());
  s1_children.push_back(s1_1.release());
  cudf::test::structs_column_wrapper s1(std::move(s1_children));
  cudf::test::fixed_width_column_wrapper<cudf::size_type> l1_offsets{0, 0, 4, 7, 15, 15};
  auto const l1_size = static_cast<cudf::column_view>(l1_offsets).size() - 1;
  std::vector<bool> l1_validity{false, true, true, true, true};
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(l1_validity.begin(), l1_validity.end());
  auto l1 = cudf::make_lists_column(
    l1_size, l1_offsets.release(), s1.release(), null_count, std::move(null_mask));

  // concatenate_policy::IGNORE_NULLS
  {
    // perform the concatenate
    cudf::table_view t({*l0, *l1});
    auto result = cudf::lists::concatenate_rows(t, cudf::lists::concatenate_null_policy::IGNORE);

    // expected
    cudf::test::fixed_width_column_wrapper<int> se_0{
      {10, 11, 12, 13, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 6, 7}, nulls_at({17})};
    cudf::test::strings_column_wrapper se_1{
      {"arg",    "mno",     "ampere", "gpu",   "bananas", "",      "",    "",    "hhh", "warp",
       "donuts", "parking", "",       "apply", "twelve",  "mouse", "bbb", "pom", "xyw", "ijk"},
      nulls_at({5, 6})};
    std::vector<std::unique_ptr<cudf::column>> se_children;
    se_children.push_back(se_0.release());
    se_children.push_back(se_1.release());
    cudf::test::structs_column_wrapper se(std::move(se_children));
    cudf::test::fixed_width_column_wrapper<cudf::size_type> le_offsets{0, 0, 4, 10, 18, 20};
    auto const le_size = static_cast<cudf::column_view>(le_offsets).size() - 1;
    std::vector<bool> le_validity{false, true, true, true, true};
    std::tie(null_mask, null_count) =
      cudf::test::detail::make_null_mask(le_validity.begin(), le_validity.end());
    auto expected = cudf::make_lists_column(
      le_size, le_offsets.release(), se.release(), null_count, std::move(null_mask));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
  }

  // concatenate_policy::NULLIFY_OUTPUT_ROW
  {
    // perform the concatenate
    cudf::table_view t({*l0, *l1});
    auto result =
      cudf::lists::concatenate_rows(t, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);

    // expected
    cudf::test::fixed_width_column_wrapper<int> se_0{{10, 11, 12, 13, 2, 3, 4, 14, 15, 16, 6, 7},
                                                     nulls_at({})};
    cudf::test::strings_column_wrapper se_1{
      {"arg", "mno", "ampere", "gpu", "bananas", "", "", "", "hhh", "warp", "xyw", "ijk"},
      nulls_at({5, 6})};
    std::vector<std::unique_ptr<cudf::column>> se_children;
    se_children.push_back(se_0.release());
    se_children.push_back(se_1.release());
    cudf::test::structs_column_wrapper se(std::move(se_children));
    cudf::test::fixed_width_column_wrapper<cudf::size_type> le_offsets{0, 0, 4, 10, 10, 12};
    auto const le_size = static_cast<cudf::column_view>(le_offsets).size() - 1;
    std::vector<bool> le_validity{false, true, true, false, true};
    std::tie(null_mask, null_count) =
      cudf::test::detail::make_null_mask(le_validity.begin(), le_validity.end());
    auto expected = cudf::make_lists_column(
      le_size, le_offsets.release(), se.release(), null_count, std::move(null_mask));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
  }
}

TEST_F(ListConcatenateRowsNestedTypesTest, StructWithNullsSliced)
{
  // list<struct<int, string>>

  // col 0
  cudf::test::fixed_width_column_wrapper<int> s0_0{0, 1, 2, 3, 4, 5, 6, 7};
  cudf::test::strings_column_wrapper s0_1{
    {"whee", "yay", "bananas", "abc", "def", "g", "xyw", "ijk"}, nulls_at({1, 3, 4})};
  std::vector<std::unique_ptr<cudf::column>> s0_children;
  s0_children.push_back(s0_0.release());
  s0_children.push_back(s0_1.release());
  cudf::test::structs_column_wrapper s0(std::move(s0_children));
  cudf::test::fixed_width_column_wrapper<cudf::size_type> l0_offsets{0, 2, 2, 5, 6, 8};
  auto const l0_size = static_cast<cudf::column_view>(l0_offsets).size() - 1;
  std::vector<bool> l0_validity{false, true, false, false, true};
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(l0_validity.begin(), l0_validity.end());
  auto l0_unsliced = cudf::make_lists_column(
    l0_size, l0_offsets.release(), s0.release(), null_count, std::move(null_mask));
  l0_unsliced = cudf::purge_nonempty_nulls(l0_unsliced->view());
  auto l0     = cudf::split(*l0_unsliced, {2})[1];

  // col1
  cudf::test::fixed_width_column_wrapper<int> s1_0{
    {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, nulls_at({14})};
  cudf::test::strings_column_wrapper s1_1{"arg",
                                          "mno",
                                          "ampere",
                                          "gpu",
                                          "",
                                          "hhh",
                                          "warp",
                                          "donuts",
                                          "parking",
                                          "",
                                          "apply",
                                          "twelve",
                                          "mouse",
                                          "bbb",
                                          "pom"};
  std::vector<std::unique_ptr<cudf::column>> s1_children;
  s1_children.push_back(s1_0.release());
  s1_children.push_back(s1_1.release());
  cudf::test::structs_column_wrapper s1(std::move(s1_children));
  cudf::test::fixed_width_column_wrapper<cudf::size_type> l1_offsets{0, 0, 4, 7, 15, 15};
  auto const l1_size = static_cast<cudf::column_view>(l1_offsets).size() - 1;
  std::vector<bool> l1_validity{false, true, false, true, true};
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(l1_validity.begin(), l1_validity.end());
  auto l1_unsliced = cudf::make_lists_column(
    l1_size, l1_offsets.release(), s1.release(), null_count, std::move(null_mask));
  l1_unsliced = cudf::purge_nonempty_nulls(l1_unsliced->view());
  auto l1     = cudf::split(*l1_unsliced, {2})[1];

  // concatenate_policy::IGNORE_NULLS
  {
    // perform the concatenate
    cudf::table_view t({l0, l1});
    auto result = cudf::lists::concatenate_rows(t, cudf::lists::concatenate_null_policy::IGNORE);

    // expected
    cudf::test::fixed_width_column_wrapper<int> se_0{{17, 18, 19, 20, 21, 22, 23, 24, 6, 7},
                                                     nulls_at({7})};
    cudf::test::strings_column_wrapper se_1{
      {"donuts", "parking", "", "apply", "twelve", "mouse", "bbb", "pom", "xyw", "ijk"}};
    std::vector<std::unique_ptr<cudf::column>> se_children;
    se_children.push_back(se_0.release());
    se_children.push_back(se_1.release());
    cudf::test::structs_column_wrapper se(std::move(se_children));
    cudf::test::fixed_width_column_wrapper<cudf::size_type> le_offsets{0, 0, 8, 10};
    auto const le_size = static_cast<cudf::column_view>(le_offsets).size() - 1;
    std::vector<bool> le_validity{false, true, true};
    std::tie(null_mask, null_count) =
      cudf::test::detail::make_null_mask(le_validity.begin(), le_validity.end());
    auto expected = cudf::make_lists_column(
      le_size, le_offsets.release(), se.release(), null_count, std::move(null_mask));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
  }

  // concatenate_policy::NULLIFY_OUTPUT_ROW
  {
    // perform the concatenate
    cudf::table_view t({l0, l1});
    auto result =
      cudf::lists::concatenate_rows(t, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);

    // expected
    cudf::test::fixed_width_column_wrapper<int> se_0{{6, 7}, nulls_at({})};
    cudf::test::strings_column_wrapper se_1{"xyw", "ijk"};
    std::vector<std::unique_ptr<cudf::column>> se_children;
    se_children.push_back(se_0.release());
    se_children.push_back(se_1.release());
    cudf::test::structs_column_wrapper se(std::move(se_children));
    cudf::test::fixed_width_column_wrapper<cudf::size_type> le_offsets{0, 0, 0, 2};
    auto const le_size = static_cast<cudf::column_view>(le_offsets).size() - 1;
    std::vector<bool> le_validity{false, false, true};
    std::tie(null_mask, null_count) =
      cudf::test::detail::make_null_mask(le_validity.begin(), le_validity.end());
    auto expected = cudf::make_lists_column(
      le_size, le_offsets.release(), se.release(), null_count, std::move(null_mask));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
  }
}
