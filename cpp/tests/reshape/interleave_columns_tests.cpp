/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/reshape.hpp>

using namespace cudf::test::iterators;

namespace {
constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
using TView       = cudf::table_view;
using IntCol      = cudf::test::fixed_width_column_wrapper<int32_t>;
using StructsCol  = cudf::test::structs_column_wrapper;
using StringsCol  = cudf::test::strings_column_wrapper;
using StrListsCol = cudf::test::lists_column_wrapper<cudf::string_view>;
using IntListsCol = cudf::test::lists_column_wrapper<int32_t>;

constexpr int32_t null{0};      // mark for null elements
constexpr int32_t NOT_USE{-1};  // mark for elements that we don't care
}  // namespace

template <typename T>
struct InterleaveColumnsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(InterleaveColumnsTest, cudf::test::FixedWidthTypes);

TYPED_TEST(InterleaveColumnsTest, NoColumns)
{
  cudf::table_view in(std::vector<cudf::column_view>{});

  EXPECT_THROW(cudf::interleave_columns(in), cudf::logic_error);
}

TYPED_TEST(InterleaveColumnsTest, OneColumn)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T, int32_t> a({-1, 0, 1});

  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto expected = cudf::test::fixed_width_column_wrapper<T, int32_t>({-1, 0, 1});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, TwoColumns)
{
  using T = TypeParam;

  auto a = cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 2});
  auto b = cudf::test::fixed_width_column_wrapper<T, int32_t>({1, 3});

  cudf::table_view in(std::vector<cudf::column_view>{a, b});

  auto expected = cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 1, 2, 3});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, ThreeColumns)
{
  using T = TypeParam;

  auto a = cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 3, 6});
  auto b = cudf::test::fixed_width_column_wrapper<T, int32_t>({1, 4, 7});
  auto c = cudf::test::fixed_width_column_wrapper<T, int32_t>({2, 5, 8});

  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});

  auto expected = cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, OneColumnEmpty)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> a({});

  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, ThreeColumnsEmpty)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> a({});
  cudf::test::fixed_width_column_wrapper<T> b({});
  cudf::test::fixed_width_column_wrapper<T> c({});

  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});

  auto expected = cudf::test::fixed_width_column_wrapper<T>({});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, OneColumnNullable)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T, int32_t> a({1, 2, 3}, {0, 1, 0});

  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto expected = cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 2, 0}, {0, 1, 0});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, TwoColumnNullable)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T, int32_t> a({1, 2, 3}, {0, 1, 0});
  cudf::test::fixed_width_column_wrapper<T, int32_t> b({4, 5, 6}, {1, 0, 1});

  cudf::table_view in(std::vector<cudf::column_view>{a, b});

  auto expected =
    cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 4, 2, 0, 0, 6}, {0, 1, 1, 0, 0, 1});
  auto actual = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, ThreeColumnsNullable)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T, int32_t> a({1, 4, 7}, {1, 0, 1});
  cudf::test::fixed_width_column_wrapper<T, int32_t> b({2, 5, 8}, {0, 1, 0});
  cudf::test::fixed_width_column_wrapper<T, int32_t> c({3, 6, 9}, {1, 0, 1});

  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});

  auto expected = cudf::test::fixed_width_column_wrapper<T, int32_t>({1, 0, 3, 0, 5, 0, 7, 0, 9},
                                                                     {1, 0, 1, 0, 1, 0, 1, 0, 1});
  auto actual   = cudf::interleave_columns(in);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
}

TYPED_TEST(InterleaveColumnsTest, MismatchedDtypes)
{
  using T = TypeParam;

  if (not std::is_same_v<int, T> and not cudf::is_fixed_point<T>()) {
    cudf::test::fixed_width_column_wrapper<int32_t> input_a({1, 4, 7}, {1, 0, 1});
    cudf::test::fixed_width_column_wrapper<T, int32_t> input_b({2, 5, 8}, {0, 1, 0});

    cudf::table_view input(std::vector<cudf::column_view>{input_a, input_b});

    EXPECT_THROW(cudf::interleave_columns(input), cudf::logic_error);
  }
}

struct InterleaveStringsColumnsTest : public cudf::test::BaseFixture {};

TEST_F(InterleaveStringsColumnsTest, ZeroSizedColumns)
{
  auto const col0 = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto results = cudf::interleave_columns(cudf::table_view{{col0}});
  cudf::test::expect_column_empty(results->view());
}

TEST_F(InterleaveStringsColumnsTest, SingleColumn)
{
  auto col0 = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});

  auto results = cudf::interleave_columns(cudf::table_view{{col0}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, col0, verbosity);
}

TEST_F(InterleaveStringsColumnsTest, MultiColumnNullAndEmpty)
{
  auto col0 = cudf::test::strings_column_wrapper({"", "", "", ""}, {false, true, true, false});
  auto col1 = cudf::test::strings_column_wrapper({"", "", "", ""}, {true, false, true, false});

  auto exp_results = cudf::test::strings_column_wrapper(
    {"", "", "", "", "", "", "", ""}, {false, true, true, false, true, true, false, false});

  auto results = cudf::interleave_columns(cudf::table_view{{col0, col1}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(InterleaveStringsColumnsTest, MultiColumnEmptyNonNullable)
{
  auto col0 = cudf::test::strings_column_wrapper({"", "", "", ""});
  auto col1 = cudf::test::strings_column_wrapper({"", "", "", ""});

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", "", "", "", ""});

  auto results = cudf::interleave_columns(cudf::table_view{{col0, col1}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(InterleaveStringsColumnsTest, MultiColumnStringMix)
{
  auto col0 = cudf::test::strings_column_wrapper({"null", "null", "", "valid", "", "valid"},
                                                 {false, false, true, true, true, true});
  auto col1 = cudf::test::strings_column_wrapper({"", "valid", "null", "null", "valid", ""},
                                                 {true, true, false, false, true, true});
  auto col2 = cudf::test::strings_column_wrapper({"valid", "", "valid", "", "null", "null"},
                                                 {true, true, true, true, false, false});

  auto exp_results = cudf::test::strings_column_wrapper({"null",
                                                         "",
                                                         "valid",
                                                         "null",
                                                         "valid",
                                                         "",
                                                         "",
                                                         "null",
                                                         "valid",
                                                         "valid",
                                                         "null",
                                                         "",
                                                         "",
                                                         "valid",
                                                         "null",
                                                         "valid",
                                                         "",
                                                         "null"},
                                                        {false,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         false});

  auto results = cudf::interleave_columns(cudf::table_view{{col0, col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(InterleaveStringsColumnsTest, MultiColumnStringMixNonNullable)
{
  auto col0 = cudf::test::strings_column_wrapper({"c00", "c01", "", "valid", "", "valid"});
  auto col1 = cudf::test::strings_column_wrapper({"", "valid", "c13", "c14", "valid", ""});
  auto col2 = cudf::test::strings_column_wrapper({"valid", "", "valid", "", "c24", "c25"});

  auto exp_results = cudf::test::strings_column_wrapper({"c00",
                                                         "",
                                                         "valid",
                                                         "c01",
                                                         "valid",
                                                         "",
                                                         "",
                                                         "c13",
                                                         "valid",
                                                         "valid",
                                                         "c14",
                                                         "",
                                                         "",
                                                         "valid",
                                                         "c24",
                                                         "valid",
                                                         "",
                                                         "c25"});

  auto results = cudf::interleave_columns(cudf::table_view{{col0, col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

TEST_F(InterleaveStringsColumnsTest, MultiColumnStringMixNullableMix)
{
  auto col0 = cudf::test::strings_column_wrapper({"c00", "c01", "", "valid", "", "valid"});
  auto col1 = cudf::test::strings_column_wrapper({"", "valid", "null", "null", "valid", ""},
                                                 {true, true, false, false, true, true});
  auto col2 = cudf::test::strings_column_wrapper({"valid", "", "valid", "", "c24", "c25"});

  auto exp_results = cudf::test::strings_column_wrapper({"c00",
                                                         "",
                                                         "valid",
                                                         "c01",
                                                         "valid",
                                                         "",
                                                         "",
                                                         "null",
                                                         "valid",
                                                         "valid",
                                                         "null",
                                                         "",
                                                         "",
                                                         "valid",
                                                         "c24",
                                                         "valid",
                                                         "",
                                                         "c25"},
                                                        {true,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         false,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         true,
                                                         true});

  auto results = cudf::interleave_columns(cudf::table_view{{col0, col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results, verbosity);
}

template <typename T>
struct FixedPointTestAllReps : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTestAllReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestAllReps, FixedPointInterleave)
{
  using namespace numeric;
  using decimalXX = TypeParam;
  using RepType   = typename decimalXX::rep;

  for (int i = 0; i > -4; --i) {
    auto const a = cudf::test::fixed_point_column_wrapper<RepType>({1, 4}, scale_type{i});
    auto const b = cudf::test::fixed_point_column_wrapper<RepType>({2, 5}, scale_type{i});

    auto const input = cudf::table_view{std::vector<cudf::column_view>{a, b}};
    auto const expected =
      cudf::test::fixed_point_column_wrapper<RepType>({1, 2, 4, 5}, scale_type{i});
    auto const actual = cudf::interleave_columns(input);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
  }
}

struct ListsColumnsInterleaveTest : public cudf::test::BaseFixture {};

TEST_F(ListsColumnsInterleaveTest, InvalidInput)
{
  // Input table contains non-list column
  {
    auto const col1 = IntCol{}.release();
    auto const col2 = IntListsCol{}.release();
    EXPECT_THROW(cudf::interleave_columns(TView{{col1->view(), col2->view()}}), cudf::logic_error);
  }

  // Types mismatch
  {
    auto const col1 = IntListsCol{}.release();
    auto const col2 = StrListsCol{}.release();
    EXPECT_THROW(cudf::interleave_columns(TView{{col1->view(), col2->view()}}), cudf::logic_error);
  }
}

template <typename T>
struct ListsColumnsInterleaveTypedTest : public cudf::test::BaseFixture {};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(ListsColumnsInterleaveTypedTest, TypesForTest);

TYPED_TEST(ListsColumnsInterleaveTypedTest, InterleaveEmptyColumns)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col     = ListsCol{}.release();
  auto const results = cudf::interleave_columns(TView{{col->view(), col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, InterleaveOneColumnNotNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col     = ListsCol{{1, 2}, {3, 4}, {5, 6}}.release();
  auto const results = cudf::interleave_columns(TView{{col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, InterleaveOneColumnWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col = ListsCol{{ListsCol{{1, 2, null}, null_at(2)},
                             ListsCol{} /*NULL*/,
                             ListsCol{{null, 3, 4, 4, 4, 4}, null_at(0)},
                             ListsCol{5, 6}},
                            null_at(1)}
                     .release();
  auto const results = cudf::interleave_columns(TView{{col->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*col, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SimpleInputNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1     = ListsCol{{1, 2}, {3, 4}, {5, 6}}.release();
  auto const col2     = ListsCol{{7, 8}, {9, 10}, {11, 12}}.release();
  auto const expected = ListsCol{{1, 2}, {7, 8}, {3, 4}, {9, 10}, {5, 6}, {11, 12}}.release();
  auto const results  = cudf::interleave_columns(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TEST_F(ListsColumnsInterleaveTest, SimpleInputStringsColumnsNoNull)
{
  auto const col1 = StrListsCol{StrListsCol{"Tomato", "Apple"},
                                StrListsCol{"Banana", "Kiwi", "Cherry"},
                                StrListsCol{"Coconut"}}
                      .release();
  auto const col2 =
    StrListsCol{StrListsCol{"Orange"}, StrListsCol{"Lemon", "Peach"}, StrListsCol{}}.release();
  auto const expected = StrListsCol{
    StrListsCol{"Tomato", "Apple"},
    StrListsCol{"Orange"},
    StrListsCol{"Banana", "Kiwi", "Cherry"},
    StrListsCol{"Lemon", "Peach"},
    StrListsCol{"Coconut"},
    StrListsCol{}}.release();
  auto const results = cudf::interleave_columns(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SimpleInputWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1 = ListsCol{{ListsCol{{1, null, 3, 4}, null_at(1)},
                              ListsCol{{null, 2, 3, 4}, null_at(0)},
                              ListsCol{{null, 2, 3, 4}, null_at(0)},
                              ListsCol{} /*NULL*/,
                              ListsCol{{1, 2, null, 4}, null_at(2)},
                              ListsCol{{1, 2, 3, null}, null_at(3)}},
                             null_at(3)}
                      .release();
  auto const col2 = ListsCol{{ListsCol{{10, 11, 12, null}, null_at(3)},
                              ListsCol{{13, 14, 15, 16, 17, null}, null_at(5)},
                              ListsCol{} /*NULL*/,
                              ListsCol{{null, 18}, null_at(0)},
                              ListsCol{{19, 20, null}, null_at(2)},
                              ListsCol{{null}, null_at(0)}},
                             null_at(2)}
                      .release();
  auto const col3 = ListsCol{{ListsCol{} /*NULL*/,
                              ListsCol{{20, null}, null_at(1)},
                              ListsCol{{null, 21, null, null}, nulls_at({0, 2, 3})},
                              ListsCol{},
                              ListsCol{22, 23, 24, 25},
                              ListsCol{{null, null, null, null, null}, all_nulls()}},
                             null_at(0)}
                      .release();
  auto const expected = ListsCol{{ListsCol{{1, null, 3, 4}, null_at(1)},
                                  ListsCol{{10, 11, 12, null}, null_at(3)},
                                  ListsCol{} /*NULL*/,
                                  ListsCol{{null, 2, 3, 4}, null_at(0)},
                                  ListsCol{{13, 14, 15, 16, 17, null}, null_at(5)},
                                  ListsCol{{20, null}, null_at(1)},
                                  ListsCol{{null, 2, 3, 4}, null_at(0)},
                                  ListsCol{} /*NULL*/,
                                  ListsCol{{null, 21, null, null}, nulls_at({0, 2, 3})},
                                  ListsCol{} /*NULL*/,
                                  ListsCol{{null, 18}, null_at(0)},
                                  ListsCol{},
                                  ListsCol{{1, 2, null, 4}, null_at(2)},
                                  ListsCol{{19, 20, null}, null_at(2)},
                                  ListsCol{22, 23, 24, 25},
                                  ListsCol{{1, 2, 3, null}, null_at(3)},
                                  ListsCol{{null}, null_at(0)},
                                  ListsCol{{null, null, null, null, null}, all_nulls()}},
                                 nulls_at({2, 7, 9})}
                          .release();
  auto const results = cudf::interleave_columns(TView{{col1->view(), col2->view(), col3->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SimpleInputWithNullableChild)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1 = ListsCol{{1, 2}, {3, 4}}.release();
  auto const col2 = ListsCol{{5, 6}, {7, 8}}.release();
  auto const col3 = ListsCol{{9, 10}, ListsCol{{null, 12}, null_at(0)}}.release();
  auto const expected =
    ListsCol{{1, 2}, {5, 6}, {9, 10}, {3, 4}, {7, 8}, ListsCol{{null, 12}, null_at(0)}}.release();
  auto const results = cudf::interleave_columns(TView{{col1->view(), col2->view(), col3->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TEST_F(ListsColumnsInterleaveTest, SimpleInputStringsColumnsWithNulls)
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

  auto const expected =
    StrListsCol{
      {StrListsCol{{"Tomato", "" /*NULL*/, "Apple"}, null_at(1)},
       StrListsCol{{"Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, nulls_at({1, 2, 3})},
       StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/}, nulls_at({1, 4})},
       StrListsCol{"Lemon", "Peach"},
       StrListsCol{"Coconut"},
       StrListsCol{}}, /*NULL*/
      null_at(5)}
      .release();
  auto const results = cudf::interleave_columns(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TEST_F(ListsColumnsInterleaveTest, SimpleInputStringsColumnsWithNullableChild)
{
  auto const col1 = StrListsCol{StrListsCol{"Tomato", "Bear", "Apple"},
                                StrListsCol{"Banana", "Pig", "Kiwi", "Cherry", "Whale"},
                                StrListsCol{"Coconut"}}
                      .release();
  auto const col2 = StrListsCol{
    StrListsCol{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/}, nulls_at({1, 2, 3})},
    StrListsCol{"Lemon", "Peach"},
    StrListsCol{
      {"Deer" /*NULL*/, "Snake" /*NULL*/, "Horse" /*NULL*/},
      all_nulls()}}.release();

  auto const expected = StrListsCol{
    StrListsCol{"Tomato", "Bear", "Apple"},
    StrListsCol{{"Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, nulls_at({1, 2, 3})},
    StrListsCol{"Banana", "Pig", "Kiwi", "Cherry", "Whale"},
    StrListsCol{"Lemon", "Peach"},
    StrListsCol{"Coconut"},
    StrListsCol{
      {"Deer" /*NULL*/, "Snake" /*NULL*/, "Horse" /*NULL*/},
      all_nulls()}}.release();
  auto const results = cudf::interleave_columns(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SlicedColumnsInputNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col      = ListsCol{{1, 2, 3}, {2, 3}, {3, 4, 5, 6}, {5, 6}, {}, {7}}.release();
  auto const col1     = cudf::slice(col->view(), {0, 3})[0];
  auto const col2     = cudf::slice(col->view(), {1, 4})[0];
  auto const col3     = cudf::slice(col->view(), {2, 5})[0];
  auto const col4     = cudf::slice(col->view(), {3, 6})[0];
  auto const expected = ListsCol{ListsCol{1, 2, 3},
                                 ListsCol{2, 3},
                                 ListsCol{3, 4, 5, 6},
                                 ListsCol{5, 6},
                                 ListsCol{2, 3},
                                 ListsCol{3, 4, 5, 6},
                                 ListsCol{5, 6},
                                 ListsCol{},
                                 ListsCol{3, 4, 5, 6},
                                 ListsCol{5, 6},
                                 ListsCol{},
                                 ListsCol{7}}
                          .release();
  auto const results = cudf::interleave_columns(TView{{col1, col2, col3, col4}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SlicedColumnsInputWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col = ListsCol{{ListsCol{{null, 2, 3}, null_at(0)},
                             ListsCol{2, 3}, /*NULL*/
                             ListsCol{{3, null, 5, 6}, null_at(1)},
                             ListsCol{5, 6}, /*NULL*/
                             ListsCol{},     /*NULL*/
                             ListsCol{7},
                             ListsCol{8, 9, 10}},
                            nulls_at({1, 3, 4})}
                     .release();
  auto const col1     = cudf::slice(col->view(), {0, 3})[0];
  auto const col2     = cudf::slice(col->view(), {1, 4})[0];
  auto const col3     = cudf::slice(col->view(), {2, 5})[0];
  auto const col4     = cudf::slice(col->view(), {3, 6})[0];
  auto const col5     = cudf::slice(col->view(), {4, 7})[0];
  auto const expected = ListsCol{{ListsCol{{null, 2, 3}, null_at(0)},
                                  ListsCol{}, /*NULL*/
                                  ListsCol{{3, null, 5, 6}, null_at(1)},
                                  ListsCol{}, /*NULL*/
                                  ListsCol{}, /*NULL*/
                                  ListsCol{}, /*NULL*/
                                  ListsCol{{3, null, 5, 6}, null_at(1)},
                                  ListsCol{}, /*NULL*/
                                  ListsCol{}, /*NULL*/
                                  ListsCol{7},
                                  ListsCol{{3, null, 5, 6}, null_at(1)},
                                  ListsCol{}, /*NULL*/
                                  ListsCol{}, /*NULL*/
                                  ListsCol{7},
                                  ListsCol{8, 9, 10}},
                                 nulls_at({1, 3, 4, 5, 7, 8, 11, 12})}
                          .release();
  auto const results = cudf::interleave_columns(TView{{col1, col2, col3, col4, col5}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SlicedColumnsInputNullableChild)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col =
    ListsCol{{1, 2, 3}, ListsCol{{null, 3}, null_at(0)}, {3, 4, 5, 6}, {5, 6}, {}, {7}}.release();
  auto const col1     = cudf::slice(col->view(), {0, 3})[0];
  auto const col2     = cudf::slice(col->view(), {1, 4})[0];
  auto const col3     = cudf::slice(col->view(), {2, 5})[0];
  auto const col4     = cudf::slice(col->view(), {3, 6})[0];
  auto const expected = ListsCol{ListsCol{1, 2, 3},
                                 ListsCol{{null, 3}, null_at(0)},
                                 ListsCol{3, 4, 5, 6},
                                 ListsCol{5, 6},
                                 ListsCol{{null, 3}, null_at(0)},
                                 ListsCol{3, 4, 5, 6},
                                 ListsCol{5, 6},
                                 ListsCol{},
                                 ListsCol{3, 4, 5, 6},
                                 ListsCol{5, 6},
                                 ListsCol{},
                                 ListsCol{7}}
                          .release();
  auto const results = cudf::interleave_columns(TView{{col1, col2, col3, col4}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, InputListsOfListsNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1 = ListsCol{ListsCol{ListsCol{1, 2, 3}, ListsCol{4, 5, 6}},
                             ListsCol{ListsCol{7, 8}, ListsCol{9, 10}},
                             ListsCol{ListsCol{11, 12, 13}, ListsCol{14, 15}, ListsCol{16, 17}}};
  auto const col2 =
    ListsCol{ListsCol{ListsCol{11, 12, 13}, ListsCol{14, 15, 16}},
             ListsCol{ListsCol{17, 18}, ListsCol{19, 110}},
             ListsCol{ListsCol{111, 112, 13}, ListsCol{114, 115}, ListsCol{116, 117}}};
  auto const expected =
    ListsCol{ListsCol{ListsCol{1, 2, 3}, ListsCol{4, 5, 6}},
             ListsCol{ListsCol{11, 12, 13}, ListsCol{14, 15, 16}},
             ListsCol{ListsCol{7, 8}, ListsCol{9, 10}},
             ListsCol{ListsCol{17, 18}, ListsCol{19, 110}},
             ListsCol{ListsCol{11, 12, 13}, ListsCol{14, 15}, ListsCol{16, 17}},
             ListsCol{ListsCol{111, 112, 13}, ListsCol{114, 115}, ListsCol{116, 117}}}
      .release();
  auto const results = cudf::interleave_columns(TView{{col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, InputListsOfListsWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1 = ListsCol{
    ListsCol{ListsCol{{null, 2, 3}, null_at(0)}, ListsCol{{4, null, null}, nulls_at({1, 2})}},
    ListsCol{{ListsCol{7, 8}, ListsCol{9, 10}, ListsCol{null, null, null} /*NULL*/}, null_at(2)},
    ListsCol{ListsCol{11, 12, 13}, ListsCol{{14, null}, null_at(1)}, ListsCol{16, 17}}};
  auto const col2 =
    ListsCol{ListsCol{{ListsCol{11, 12, 13}, ListsCol{null, null} /*NULL*/}, null_at(1)},
             ListsCol{ListsCol{17, 18}, ListsCol{{19, 110, null}, null_at(2)}},
             ListsCol{ListsCol{111, 112, 13}, ListsCol{114, 115}, ListsCol{116, 117}}};
  auto const expected = ListsCol{
    ListsCol{ListsCol{{null, 2, 3}, null_at(0)}, ListsCol{{4, null, null}, nulls_at({1, 2})}},
    ListsCol{{ListsCol{11, 12, 13}, ListsCol{null, null} /*NULL*/}, null_at(1)},
    ListsCol{{ListsCol{7, 8}, ListsCol{9, 10}, ListsCol{null, null, null} /*NULL*/}, null_at(2)},
    ListsCol{ListsCol{17, 18}, ListsCol{{19, 110, null}, null_at(2)}},
    ListsCol{ListsCol{11, 12, 13}, ListsCol{{14, null}, null_at(1)}, ListsCol{16, 17}},
    ListsCol{
      ListsCol{111, 112, 13},
      ListsCol{114, 115},
      ListsCol{116, 117}}}.release();
  auto const results = cudf::interleave_columns(TView{{col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SlicedInputListsOfListsNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1_original = ListsCol{
    ListsCol{ListsCol{11, 11, 11}, ListsCol{22}, ListsCol{33, 33, 33}},  // don't care
    ListsCol{ListsCol{11, 11, 11}, ListsCol{22}, ListsCol{33, 33, 33}},  // don't care
    //
    ListsCol{ListsCol{1, 2, 3}, ListsCol{4, 5, 6}},
    ListsCol{ListsCol{7, 8}, ListsCol{9, 10}},
    ListsCol{ListsCol{11, 12, 13}, ListsCol{14, 15}, ListsCol{16, 17}},
    //
    ListsCol{ListsCol{11, 11, 11}, ListsCol{22}, ListsCol{33, 33, 33}},  // don't care
    ListsCol{ListsCol{11, 11, 11}, ListsCol{22}, ListsCol{33, 33, 33}}   // don't care
  };
  auto const col2_original = ListsCol{
    ListsCol{ListsCol{11, 12, 13}, ListsCol{14, 15, 16}},
    ListsCol{ListsCol{17, 18}, ListsCol{19, 110}},
    ListsCol{ListsCol{111, 112, 13}, ListsCol{114, 115}, ListsCol{116, 117}},
    //
    ListsCol{ListsCol{11, 11, 11}, ListsCol{22}, ListsCol{33, 33, 33}}  // don't care
  };

  auto const col1 = cudf::slice(col1_original, {2, 5})[0];
  auto const col2 = cudf::slice(col2_original, {0, 3})[0];
  auto const expected =
    ListsCol{ListsCol{ListsCol{1, 2, 3}, ListsCol{4, 5, 6}},
             ListsCol{ListsCol{11, 12, 13}, ListsCol{14, 15, 16}},
             ListsCol{ListsCol{7, 8}, ListsCol{9, 10}},
             ListsCol{ListsCol{17, 18}, ListsCol{19, 110}},
             ListsCol{ListsCol{11, 12, 13}, ListsCol{14, 15}, ListsCol{16, 17}},
             ListsCol{ListsCol{111, 112, 13}, ListsCol{114, 115}, ListsCol{116, 117}}}
      .release();
  auto const results = cudf::interleave_columns(TView{{col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SlicedInputListsOfListsWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col1_original = ListsCol{
    {
      ListsCol{ListsCol{{null, 11}, null_at(0)},
               ListsCol{{22, null, null}, nulls_at({1, 2})}},  // don't care
      ListsCol{ListsCol{{null, 11}, null_at(0)},
               ListsCol{{22, null, null}, nulls_at({1, 2})}},  // don't care
      ListsCol{ListsCol{{null, 11}, null_at(0)},
               ListsCol{{22, null, null}, nulls_at({1, 2})}},  // don't care
      //
      ListsCol{ListsCol{{null, 2, 3}, null_at(0)}, ListsCol{{4, null, null}, nulls_at({1, 2})}},
      ListsCol{{ListsCol{7, 8}, ListsCol{9, 10}, ListsCol{null, null, null} /*NULL*/}, null_at(2)},
      ListsCol{ListsCol{11, 12, 13}, ListsCol{{14, null}, null_at(1)}, ListsCol{16, 17}},
      //
      ListsCol{ListsCol{{null, 11}, null_at(0)},
               ListsCol{{22, null, null}, nulls_at({1, 2})}}  // don't care
    },
    nulls_at({0, 2, 3})};
  auto const col2_original = ListsCol{
    ListsCol{ListsCol{{null, 11}, null_at(0)},
             ListsCol{{22, null, null}, nulls_at({1, 2})}},  // don't care
    ListsCol{ListsCol{{null, 11}, null_at(0)},
             ListsCol{{22, null, null}, nulls_at({1, 2})}},  // don't care
                                                             //
    ListsCol{{ListsCol{11, 12, 13}, ListsCol{null, null} /*NULL*/}, null_at(1)},
    ListsCol{ListsCol{17, 18}, ListsCol{{19, 110, null}, null_at(2)}},
    ListsCol{ListsCol{111, 112, 13}, ListsCol{114, 115}, ListsCol{116, 117}},
    ListsCol{ListsCol{{null, 11}, null_at(0)},
             //
             ListsCol{{22, null, null}, nulls_at({1, 2})}},  // don't care
    ListsCol{ListsCol{{null, 11}, null_at(0)},
             ListsCol{{22, null, null}, nulls_at({1, 2})}},  // don't care
    ListsCol{ListsCol{{null, 11}, null_at(0)},
             ListsCol{{22, null, null}, nulls_at({1, 2})}}  // don't care
  };

  auto const col1 = cudf::slice(col1_original, {3, 6})[0];
  auto const col2 = cudf::slice(col2_original, {2, 5})[0];
  auto const expected =
    ListsCol{
      {ListsCol{ListsCol{{null, 2, 3}, null_at(0)}, ListsCol{{4, null, null}, nulls_at({1, 2})}},
       ListsCol{{ListsCol{11, 12, 13}, ListsCol{null, null} /*NULL*/}, null_at(1)},
       ListsCol{{ListsCol{7, 8}, ListsCol{9, 10}, ListsCol{null, null, null} /*NULL*/}, null_at(2)},
       ListsCol{ListsCol{17, 18}, ListsCol{{19, 110, null}, null_at(2)}},
       ListsCol{ListsCol{11, 12, 13}, ListsCol{{14, null}, null_at(1)}, ListsCol{16, 17}},
       ListsCol{ListsCol{111, 112, 13}, ListsCol{114, 115}, ListsCol{116, 117}}},
      null_at(0)}
      .release();
  auto const results = cudf::interleave_columns(TView{{col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, InputListsOfStructsNoNull)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto structs1 = [] {
    auto child1 = ColWrapper{1, 2, 3, 4, 5};
    auto child2 = ColWrapper{6, 7, 8, 9, 10};
    auto child3 = StringsCol{"Banana", "Mango", "Apple", "Cherry", "Kiwi"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto structs2 = [] {
    auto child1 = ColWrapper{11, 12, 13, 14, 15};
    auto child2 = ColWrapper{16, 17, 18, 19, 110};
    auto child3 = StringsCol{"Bear", "Duck", "Cat", "Dog", "Panda"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto structs_expected = [] {
    auto child1 = ColWrapper{1, 11, 12, 13, 2, 3, 14, 4, 5, 15};
    auto child2 = ColWrapper{6, 16, 17, 18, 7, 8, 19, 9, 10, 110};
    auto child3 = StringsCol{
      "Banana", "Bear", "Duck", "Cat", "Mango", "Apple", "Dog", "Cherry", "Kiwi", "Panda"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto const col1 =
    cudf::make_lists_column(3, IntCol{0, 1, 3, 5}.release(), structs1.release(), 0, {});
  auto const col2 =
    cudf::make_lists_column(3, IntCol{0, 3, 4, 5}.release(), structs2.release(), 0, {});
  auto const expected = cudf::make_lists_column(
    6, IntCol{0, 1, 4, 6, 7, 9, 10}.release(), structs_expected.release(), 0, {});
  auto const results = cudf::interleave_columns(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, InputListsOfStructsWithNulls)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto structs1 = [] {
    auto child1 = ColWrapper{{1, 2, null, 4, 5}, null_at(2)};
    auto child2 = ColWrapper{{6, 7, 8, 9, null}, null_at(4)};
    auto child3 = StringsCol{"Banana", "Mango", "Apple", "Cherry", "Kiwi"};
    return StructsCol{{child1, child2, child3}, null_at(0)};
  }();

  auto structs2 = [] {
    auto child1 = ColWrapper{11, 12, 13, 14, 15};
    auto child2 = ColWrapper{{null, 17, 18, 19, 110}, null_at(0)};
    auto child3 = StringsCol{{"" /*NULL*/, "Duck", "Cat", "Dog", "" /*NULL*/}, nulls_at({0, 4})};
    return StructsCol{{child1, child2, child3}};
  }();

  auto structs_expected = [] {
    auto child1 = ColWrapper{{1, 11, 12, 13, 2, null, 14, 4, 5, 15}, null_at(5)};
    auto child2 = ColWrapper{{6, null, 17, 18, 7, 8, 19, 9, null, 110}, nulls_at({1, 8})};
    auto child3 = StringsCol{{"Banana",
                              "" /*NULL*/,
                              "Duck",
                              "Cat",
                              "Mango",
                              "Apple",
                              "Dog",
                              "Cherry",
                              "Kiwi",
                              "" /*NULL*/},
                             nulls_at({1, 9})};
    return StructsCol{{child1, child2, child3}, null_at(0)};
  }();

  auto const col1 =
    cudf::make_lists_column(3, IntCol{0, 1, 3, 5}.release(), structs1.release(), 0, {});
  auto const col2 =
    cudf::make_lists_column(3, IntCol{0, 3, 4, 5}.release(), structs2.release(), 0, {});
  auto const expected = cudf::make_lists_column(
    6, IntCol{0, 1, 4, 6, 7, 9, 10}.release(), structs_expected.release(), 0, {});
  auto const results = cudf::interleave_columns(TView{{col1->view(), col2->view()}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SlicedInputListsOfStructsNoNull)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto structs1 = [] {
    auto child1 = ColWrapper{NOT_USE, NOT_USE, 1, 2, 3, 4, 5, NOT_USE};
    auto child2 = ColWrapper{NOT_USE, NOT_USE, 6, 7, 8, 9, 10, NOT_USE};
    auto child3 =
      StringsCol{"NOT_USE", "NOT_USE", "Banana", "Mango", "Apple", "Cherry", "Kiwi", "NOT_USE"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto structs2 = [] {
    auto child1 = ColWrapper{11, 12, 13, 14, 15, NOT_USE, NOT_USE};
    auto child2 = ColWrapper{16, 17, 18, 19, 110, NOT_USE, NOT_USE};
    auto child3 = StringsCol{"Bear", "Duck", "Cat", "Dog", "Panda", "NOT_USE", "NOT_USE"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto structs_expected = [] {
    auto child1 = ColWrapper{1, 11, 12, 13, 2, 3, 14, 4, 5, 15};
    auto child2 = ColWrapper{6, 16, 17, 18, 7, 8, 19, 9, 10, 110};
    auto child3 = StringsCol{
      "Banana", "Bear", "Duck", "Cat", "Mango", "Apple", "Dog", "Cherry", "Kiwi", "Panda"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto const col1_original =
    cudf::make_lists_column(5, IntCol{0, 2, 3, 5, 7, 8}.release(), structs1.release(), 0, {});
  auto const col2_original =
    cudf::make_lists_column(4, IntCol{0, 3, 4, 5, 7}.release(), structs2.release(), 0, {});
  auto const expected = cudf::make_lists_column(
    6, IntCol{0, 1, 4, 6, 7, 9, 10}.release(), structs_expected.release(), 0, {});

  auto const col1    = cudf::slice(col1_original->view(), {1, 4})[0];
  auto const col2    = cudf::slice(col2_original->view(), {0, 3})[0];
  auto const results = cudf::interleave_columns(TView{{col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TYPED_TEST(ListsColumnsInterleaveTypedTest, SlicedInputListsOfStructsWithNulls)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto structs1 = [] {
    auto child1 = ColWrapper{{NOT_USE, 1, 2, null, 4, 5, NOT_USE}, nulls_at({0, 3})};
    auto child2 = ColWrapper{{NOT_USE, 6, 7, 8, 9, null, NOT_USE}, null_at(5)};
    auto child3 = StringsCol{"NOT_USE", "Banana", "Mango", "Apple", "Cherry", "Kiwi", "NOT_USE"};
    return StructsCol{{child1, child2, child3}, nulls_at({1, 6})};
  }();

  auto structs2 = [] {
    auto child1 = ColWrapper{{NOT_USE, 11, 12, 13, 14, 15}, null_at(0)};
    auto child2 = ColWrapper{{NOT_USE, null, 17, 18, 19, 110}, null_at(1)};
    auto child3 =
      StringsCol{{"NOT_USE", "" /*NULL*/, "Duck", "Cat", "Dog", "" /*NULL*/}, nulls_at({0, 1, 5})};
    return StructsCol{{child1, child2, child3}};
  }();

  auto structs_expected = [] {
    auto child1 = ColWrapper{{1, 11, 12, 13, 2, null, 14, 4, 5, 15}, null_at(5)};
    auto child2 = ColWrapper{{6, null, 17, 18, 7, 8, 19, 9, null, 110}, nulls_at({1, 8})};
    auto child3 = StringsCol{{"Banana",
                              "" /*NULL*/,
                              "Duck",
                              "Cat",
                              "Mango",
                              "Apple",
                              "Dog",
                              "Cherry",
                              "Kiwi",
                              "" /*NULL*/},
                             nulls_at({1, 9})};
    return StructsCol{{child1, child2, child3}, null_at(0)};
  }();

  auto const col1_original =
    cudf::make_lists_column(5, IntCol{0, 1, 2, 4, 6, 7}.release(), structs1.release(), 0, {});
  auto const col2_original =
    cudf::make_lists_column(4, IntCol{0, 1, 4, 5, 6}.release(), structs2.release(), 0, {});

  auto const col1     = cudf::slice(col1_original->view(), {1, 4})[0];
  auto const col2     = cudf::slice(col2_original->view(), {1, 4})[0];
  auto const expected = cudf::make_lists_column(
    6, IntCol{0, 1, 4, 6, 7, 9, 10}.release(), structs_expected.release(), 0, {});
  auto const results = cudf::interleave_columns(TView{{col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

TEST_F(ListsColumnsInterleaveTest, SlicedStringsColumnsInputWithNulls)
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
  auto const expected =
    StrListsCol{
      {StrListsCol{{"Tomato", "" /*NULL*/, "Apple"}, null_at(1)},
       StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/}, nulls_at({1, 4})},
       StrListsCol{"Coconut"},
       StrListsCol{{"Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, nulls_at({1, 2, 3})},
       StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/}, nulls_at({1, 4})},
       StrListsCol{"Coconut"},
       StrListsCol{{"Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, nulls_at({1, 2, 3})},
       StrListsCol{"Lemon", "Peach"},
       StrListsCol{"Coconut"},
       StrListsCol{{"Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, nulls_at({1, 2, 3})},
       StrListsCol{"Lemon", "Peach"},
       StrListsCol{}}, /*NULL*/
      null_at(11)}
      .release();
  auto const results = cudf::interleave_columns(TView{{col1, col2, col3, col4}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results, verbosity);
}

struct StructsColumnsInterleaveTest : public cudf::test::BaseFixture {};

TEST_F(StructsColumnsInterleaveTest, InvalidInput)
{
  // Input table contains non-structs column
  {
    auto const col1 = IntCol{};
    auto const col2 = StructsCol{};
    EXPECT_THROW(cudf::interleave_columns(TView{{col1, col2}}), cudf::logic_error);
  }

  // Types mismatch
  {
    auto const structs1 = [] {
      auto child1 = IntCol{1, 2, 3};
      auto child2 = IntCol{4, 5, 6};
      return StructsCol{{child1, child2}};
    }();

    auto const structs2 = [] {
      auto child1 = IntCol{7, 8, 9};
      auto child2 = StringsCol{"", "abc", "123"};
      return StructsCol{{child1, child2}};
    }();

    EXPECT_THROW(cudf::interleave_columns(TView{{structs1, structs2}}), cudf::logic_error);
  }

  // Numbers of children mismatch
  {
    auto const structs1 = [] {
      auto child1 = IntCol{1, 2, 3};
      auto child2 = IntCol{4, 5, 6};
      return StructsCol{{child1, child2}};
    }();

    auto const structs2 = [] {
      auto child1 = IntCol{7, 8, 9};
      auto child2 = IntCol{10, 11, 12};
      auto child3 = IntCol{13, 14, 15};
      return StructsCol{{child1, child2, child3}};
    }();

    EXPECT_THROW(cudf::interleave_columns(TView{{structs1, structs2}}), cudf::logic_error);
  }
}

TEST_F(StructsColumnsInterleaveTest, InterleaveEmptyColumns)
{
  auto const structs = StructsCol{};
  auto const results = cudf::interleave_columns(TView{{structs, structs}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(structs, *results, verbosity);
}

template <typename T>
struct StructsColumnsInterleaveTypedTest : public cudf::test::BaseFixture {};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(StructsColumnsInterleaveTypedTest, TypesForTest);

TYPED_TEST(StructsColumnsInterleaveTypedTest, InterleaveOneColumnNotNull)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const structs = [] {
    auto child1 = ColWrapper{1, 2, 3};
    auto child2 = ColWrapper{4, 5, 6};
    auto child3 = StringsCol{"Banana", "Mango", "Apple"};
    return StructsCol{{child1, child2, child3}};
  }();
  auto const results = cudf::interleave_columns(TView{{structs}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(structs, *results, verbosity);
}

TYPED_TEST(StructsColumnsInterleaveTypedTest, InterleaveOneColumnWithNulls)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const structs = [] {
    auto child1 = ColWrapper{{1, 2, null, 3}, null_at(2)};
    auto child2 = ColWrapper{{4, null, 5, 6}, null_at(1)};
    auto child3 = StringsCol{{"" /*NULL*/, "Banana", "Mango", "Apple"}, null_at(0)};
    return StructsCol{{child1, child2, child3}, null_at(3)};
  }();
  auto const results = cudf::interleave_columns(TView{{structs}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(structs, *results, verbosity);
}

TYPED_TEST(StructsColumnsInterleaveTypedTest, SimpleInputNoNull)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const structs1 = [] {
    auto child1 = ColWrapper{1, 2, 3};
    auto child2 = ColWrapper{4, 5, 6};
    auto child3 = StringsCol{"Banana", "Mango", "Apple"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto const structs2 = [] {
    auto child1 = ColWrapper{7, 8, 9};
    auto child2 = ColWrapper{10, 11, 12};
    auto child3 = StringsCol{"Bear", "Duck", "Cat"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto const expected = [] {
    auto child1 = ColWrapper{1, 7, 2, 8, 3, 9};
    auto child2 = ColWrapper{4, 10, 5, 11, 6, 12};
    auto child3 = StringsCol{"Banana", "Bear", "Mango", "Duck", "Apple", "Cat"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto const results = cudf::interleave_columns(TView{{structs1, structs2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results, verbosity);
}

TYPED_TEST(StructsColumnsInterleaveTypedTest, SimpleInputWithNulls)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const structs1 = [] {
    auto child1 = ColWrapper{{1, 2, null, 3, 4}, null_at(2)};
    auto child2 = ColWrapper{{4, null, 5, 6, 7}, null_at(1)};
    auto child3 = StringsCol{{"" /*NULL*/, "Banana", "Mango", "Apple", "Cherry"}, null_at(0)};
    return StructsCol{{child1, child2, child3}, null_at(0)};
  }();

  auto const structs2 = [] {
    auto child1 = ColWrapper{{7, null, null, 8, 9}, nulls_at({1, 2})};
    auto child2 = ColWrapper{{10, 11, 12, null, 14}, null_at(3)};
    auto child3 = StringsCol{"Bear", "Duck", "Cat", "Dog", "Panda"};
    return StructsCol{{child1, child2, child3}, null_at(4)};
  }();

  auto const structs3 = [] {
    auto child1 = ColWrapper{{-1, -2, -3, 0, null}, null_at(4)};
    auto child2 = ColWrapper{{-5, 0, null, -1, -10}, null_at(2)};
    auto child3 = StringsCol{"111", "Bànànà", "abcxyz", "é á í", "zzz"};
    return StructsCol{{child1, child2, child3}, null_at(1)};
  }();

  auto const expected = [] {
    auto child1 = ColWrapper{{1, 7, -1, 2, null, -2, null, null, -3, 3, 8, 0, 4, 9, null},
                             nulls_at({4, 6, 7, 14})};
    auto child2 = ColWrapper{{4, 10, -5, null, 11, 0, 5, 12, null, 6, null, -1, 7, 14, -10},
                             nulls_at({3, 8, 10})};
    auto child3 = StringsCol{{"" /*NULL*/,
                              "Bear",
                              "111",
                              "Banana",
                              "Duck",
                              "Bànànà",
                              "Mango",
                              "Cat",
                              "abcxyz",
                              "Apple",
                              "Dog",
                              "é á í",
                              "Cherry",
                              "Panda",
                              "zzz"},
                             null_at(0)};
    return StructsCol{{child1, child2, child3}, nulls_at({0, 5, 13})};
  }();

  auto const results = cudf::interleave_columns(TView{{structs1, structs2, structs3}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results, verbosity);
}

TYPED_TEST(StructsColumnsInterleaveTypedTest, NestedInputStructsColumns)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const structs1 = [] {
    auto child_structs1 = [] {
      auto child1 = ColWrapper{{null, 2, 3, 4, 5}, null_at(0)};
      auto child2 = ColWrapper{{6, 7, 8, null, 10}, null_at(3)};
      return StructsCol{{child1, child2}, null_at(0)};
    }();

    auto child_structs2 = [] {
      auto child1 = ColWrapper{{11, null, 13, 14, 15}, null_at(1)};
      auto child2 = ColWrapper{{null, 17, 18, 19, 20}, null_at(0)};
      return StructsCol{{child1, child2}, nulls_at({0, 1})};
    }();

    auto child_strings = [] { return StringsCol{"Banana", "Mango", "Apple", "Cherry", "Kiwi"}; }();

    return StructsCol{{child_structs1, child_structs2, child_strings}, null_at(0)};
  }();

  auto const structs2 = [] {
    auto child_structs1 = [] {
      auto child1 = ColWrapper{{-1, null, -3, -4, -5}, null_at(1)};
      auto child2 = ColWrapper{{-6, -7, -8, null, -10}, null_at(3)};
      return StructsCol{{child1, child2}};
    }();

    auto child_structs2 = [] {
      auto child1 = ColWrapper{{-11, -12, null, -14, -15}, null_at(2)};
      auto child2 = ColWrapper{{-16, -17, -18, -19, null}, null_at(4)};
      return StructsCol{{child1, child2}, null_at(2)};
    }();

    auto child_strings = [] { return StringsCol{"Bear", "Duck", "Cat", "Dog", "Rabbit"}; }();

    return StructsCol{{child_structs1, child_structs2, child_strings}, null_at(2)};
  }();

  auto const expected = [] {
    auto child_structs1 = [] {
      auto child1 = ColWrapper{{null, -1, 2, null, 3, -3, 4, -4, 5, -5}, nulls_at({0, 3})};
      auto child2 = ColWrapper{{6, -6, 7, -7, 8, -8, null, null, 10, -10}, nulls_at({6, 7})};
      return StructsCol{{child1, child2}, null_at(0)};
    }();

    auto child_structs2 = [] {
      auto child1 = ColWrapper{{11, -11, null, -12, 13, null, 14, -14, 15, -15}, nulls_at({2, 5})};
      auto child2 = ColWrapper{{null, -16, 17, -17, 18, -18, 19, -19, 20, null}, nulls_at({0, 9})};
      return StructsCol{{child1, child2}, nulls_at({0, 2, 5})};
    }();

    auto child_strings = [] {
      return StringsCol{
        "Banana", "Bear", "Mango", "Duck", "Apple", "Cat", "Cherry", "Dog", "Kiwi", "Rabbit"};
    }();

    return StructsCol{{child_structs1, child_structs2, child_strings}, nulls_at({0, 5})};
  }();

  auto const results = cudf::interleave_columns(TView{{structs1, structs2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results, verbosity);
}

TYPED_TEST(StructsColumnsInterleaveTypedTest, SlicedColumnsInputNoNull)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const structs1_original = [] {
    auto child1 = ColWrapper{NOT_USE, NOT_USE, 1, 2, 3, NOT_USE};
    auto child2 = ColWrapper{NOT_USE, NOT_USE, 4, 5, 6, NOT_USE};
    auto child3 = StringsCol{"NOT_USE", "NOT_USE", "Banana", "Mango", "Apple", "NOT_USE"};
    return StructsCol{{child1, child2, child3}};
  }();

  // structs2 has more rows than structs1
  auto const structs2_original = [] {
    auto child1 = ColWrapper{NOT_USE, 7, 8, 9, NOT_USE, NOT_USE, NOT_USE};
    auto child2 = ColWrapper{NOT_USE, 10, 11, 12, NOT_USE, NOT_USE, NOT_USE};
    auto child3 = StringsCol{"NOT_USE", "Bear", "Duck", "Cat", "NOT_USE", "NOT_USE", "NOT_USE"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto const expected = [] {
    auto child1 = ColWrapper{1, 7, 2, 8, 3, 9};
    auto child2 = ColWrapper{4, 10, 5, 11, 6, 12};
    auto child3 = StringsCol{"Banana", "Bear", "Mango", "Duck", "Apple", "Cat"};
    return StructsCol{{child1, child2, child3}};
  }();

  auto const structs1 = cudf::slice(structs1_original, {2, 5})[0];
  auto const structs2 = cudf::slice(structs2_original, {1, 4})[0];
  auto const results  = cudf::interleave_columns(TView{{structs1, structs2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results, verbosity);
}

TYPED_TEST(StructsColumnsInterleaveTypedTest, SlicedColumnsInputWithNulls)
{
  using ColWrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  constexpr int32_t NOT_USE{-1};  // mark for elements that we don't care

  auto const structs1_original = [] {
    auto child1 = ColWrapper{{NOT_USE, NOT_USE, 1, 2, null, 3, 4, NOT_USE}, null_at(4)};
    auto child2 = ColWrapper{{NOT_USE, NOT_USE, 4, null, 5, 6, 7, NOT_USE}, null_at(3)};
    auto child3 = StringsCol{
      {"NOT_USE", "NOT_USE", "" /*NULL*/, "Banana", "Mango", "Apple", "Cherry", "NOT_USE"},
      null_at(2)};
    return StructsCol{{child1, child2, child3}, null_at(2)};
  }();

  auto const structs2_original = [] {
    auto child1 = ColWrapper{{7, null, null, 8, 9, NOT_USE, NOT_USE}, nulls_at({1, 2})};
    auto child2 = ColWrapper{{10, 11, 12, null, 14, NOT_USE, NOT_USE}, null_at(3)};
    auto child3 = StringsCol{"Bear", "Duck", "Cat", "Dog", "Panda", "NOT_USE", "NOT_USE"};
    return StructsCol{{child1, child2, child3}, null_at(4)};
  }();

  auto const structs3_original = [] {
    auto child1 = ColWrapper{{NOT_USE, NOT_USE, NOT_USE, -1, -2, -3, 0, null}, null_at(7)};
    auto child2 = ColWrapper{{NOT_USE, NOT_USE, NOT_USE, -5, 0, null, -1, -10}, null_at(5)};
    auto child3 =
      StringsCol{"NOT_USE", "NOT_USE", "NOT_USE", "111", "Bànànà", "abcxyz", "é á í", "zzz"};
    return StructsCol{{child1, child2, child3}, null_at(4)};
  }();

  auto const expected = [] {
    auto child1 = ColWrapper{{1, 7, -1, 2, null, -2, null, null, -3, 3, 8, 0, 4, 9, null},
                             nulls_at({4, 6, 7, 14})};
    auto child2 = ColWrapper{{4, 10, -5, null, 11, 0, 5, 12, null, 6, null, -1, 7, 14, -10},
                             nulls_at({3, 8, 10})};
    auto child3 = StringsCol{{"" /*NULL*/,
                              "Bear",
                              "111",
                              "Banana",
                              "Duck",
                              "Bànànà",
                              "Mango",
                              "Cat",
                              "abcxyz",
                              "Apple",
                              "Dog",
                              "é á í",
                              "Cherry",
                              "Panda",
                              "zzz"},
                             null_at(0)};
    return StructsCol{{child1, child2, child3}, nulls_at({0, 5, 13})};
  }();

  auto const structs1 = cudf::slice(structs1_original, {2, 7})[0];
  auto const structs2 = cudf::slice(structs2_original, {0, 5})[0];
  auto const structs3 = cudf::slice(structs3_original, {3, 8})[0];
  auto const results  = cudf::interleave_columns(TView{{structs1, structs2, structs3}});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results, verbosity);
}

CUDF_TEST_PROGRAM_MAIN()
