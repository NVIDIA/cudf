/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>

using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

using bools_col   = cudf::test::fixed_width_column_wrapper<bool>;
using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using structs_col = cudf::test::structs_column_wrapper;
using strings_col = cudf::test::strings_column_wrapper;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};  // Mark for null child elements at the current level

using TestTypes = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                     cudf::test::FloatingPointTypes,
                                     cudf::test::DurationTypes,
                                     cudf::test::TimestampTypes>;

template <typename T>
struct TypedListsContainsTestScalarNeedle : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(TypedListsContainsTestScalarNeedle, TestTypes);

TYPED_TEST(TypedListsContainsTestScalarNeedle, EmptyInput)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack = lists_col{};

  auto const needle1 = [] {
    auto child = tdata_col{};
    return cudf::list_scalar(child);
  }();
  auto const needle2 = [] {
    auto child = tdata_col{1, 2, 3};
    return cudf::list_scalar(child);
  }();

  EXPECT_FALSE(cudf::contains(haystack, needle1));
  EXPECT_FALSE(cudf::contains(haystack, needle2));
}

TYPED_TEST(TypedListsContainsTestScalarNeedle, TrivialInput)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack = lists_col{{1, 2}, {1}, {}, {1, 3}, {4}, {1, 1}};

  auto const needle1 = [] {
    auto child = tdata_col{1, 2};
    return cudf::list_scalar(child);
  }();
  auto const needle2 = [] {
    auto child = tdata_col{2, 1};
    return cudf::list_scalar(child);
  }();

  EXPECT_TRUE(cudf::contains(haystack, needle1));

  // Lists are order-sensitive.
  EXPECT_FALSE(cudf::contains(haystack, needle2));
}

TYPED_TEST(TypedListsContainsTestScalarNeedle, SlicedColumnInput)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack_original = lists_col{{0, 0}, {0}, {1, 2}, {1}, {}, {1, 3}, {0, 0}};
  auto const haystack          = cudf::slice(haystack_original, {2, 6})[0];

  auto const needle1 = [] {
    auto child = tdata_col{1, 2};
    return cudf::list_scalar(child);
  }();
  auto const needle2 = [] {
    auto child = tdata_col{};
    return cudf::list_scalar(child);
  }();
  auto const needle3 = [] {
    auto child = tdata_col{0, 0};
    return cudf::list_scalar(child);
  }();

  EXPECT_TRUE(cudf::contains(haystack, needle1));
  EXPECT_TRUE(cudf::contains(haystack, needle2));
  EXPECT_FALSE(cudf::contains(haystack, needle3));
}

TYPED_TEST(TypedListsContainsTestScalarNeedle, SimpleInputWithNulls)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Test with invalid scalar.
  {
    auto const haystack = lists_col{{1, 2}, {1}, {}, {1, 3}, {4}, {}, {1, 1}};
    auto const needle   = [] {
      auto child = tdata_col{};
      return cudf::list_scalar(child, false);
    }();

    EXPECT_FALSE(cudf::contains(haystack, needle));
  }

  // Test with nulls at the top level.
  {
    auto const haystack =
      lists_col{{{1, 2}, {1}, {} /*NULL*/, {1, 3}, {4}, {} /*NULL*/, {1, 1}}, nulls_at({2, 5})};

    auto const needle1 = [] {
      auto child = tdata_col{1, 2};
      return cudf::list_scalar(child);
    }();
    auto const needle2 = [] {
      auto child = tdata_col{};
      return cudf::list_scalar(child, false);
    }();

    EXPECT_TRUE(cudf::contains(haystack, needle1));
    EXPECT_TRUE(cudf::contains(haystack, needle2));
  }

  // Test with nulls at the children level.
  {
    auto const haystack = lists_col{{lists_col{1, 2},
                                     lists_col{1},
                                     lists_col{{1, null}, null_at(1)},
                                     lists_col{} /*NULL*/,
                                     lists_col{1, 3},
                                     lists_col{1, 4},
                                     lists_col{4},
                                     lists_col{} /*NULL*/,
                                     lists_col{1, 1}},
                                    nulls_at({3, 7})};

    auto const needle1 = [] {
      auto child = tdata_col{{1, null}, null_at(1)};
      return cudf::list_scalar(child);
    }();
    auto const needle2 = [] {
      auto child = tdata_col{{null, 1}, null_at(0)};
      return cudf::list_scalar(child);
    }();
    auto const needle3 = [] {
      auto child = tdata_col{1, 0};
      return cudf::list_scalar(child);
    }();

    EXPECT_TRUE(cudf::contains(haystack, needle1));
    EXPECT_FALSE(cudf::contains(haystack, needle2));
    EXPECT_FALSE(cudf::contains(haystack, needle3));
  }
}

TYPED_TEST(TypedListsContainsTestScalarNeedle, SlicedInputHavingNulls)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack_original = lists_col{{{0, 0},
                                            {0} /*NULL*/,
                                            lists_col{{1, null}, null_at(1)},
                                            {1},
                                            {} /*NULL*/,
                                            {1, 3},
                                            {4},
                                            {} /*NULL*/,
                                            {1, 1},
                                            {0}},
                                           nulls_at({1, 4, 7})};
  auto const haystack          = cudf::slice(haystack_original, {2, 9})[0];

  auto const needle1 = [] {
    auto child = tdata_col{{1, null}, null_at(1)};
    return cudf::list_scalar(child);
  }();
  auto const needle2 = [] {
    auto child = tdata_col{};
    return cudf::list_scalar(child);
  }();
  auto const needle3 = [] {
    auto child = tdata_col{0, 0};
    return cudf::list_scalar(child);
  }();

  EXPECT_TRUE(cudf::contains(haystack, needle1));
  EXPECT_FALSE(cudf::contains(haystack, needle2));
  EXPECT_FALSE(cudf::contains(haystack, needle3));
}

template <typename T>
struct TypedListContainsTestColumnNeedles : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedListContainsTestColumnNeedles, TestTypes);

TYPED_TEST(TypedListContainsTestColumnNeedles, EmptyInput)
{
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack = lists_col{};
  auto const needles  = lists_col{};
  auto const expected = bools_col{};
  auto const result   = cudf::contains(haystack, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

TYPED_TEST(TypedListContainsTestColumnNeedles, TrivialInput)
{
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack = lists_col{{0, 1}, {2}, {3, 4, 5}, {2, 3, 4}, {}, {0, 2, 0}};
  auto const needles  = lists_col{{0, 1}, {1}, {3, 5, 4}, {}};

  auto const expected = bools_col{1, 0, 0, 1};
  auto const result   = cudf::contains(haystack, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

TYPED_TEST(TypedListContainsTestColumnNeedles, SlicedInputNoNulls)
{
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack_original =
    lists_col{{0, 0}, {0}, {0, 1}, {2}, {3, 4, 5}, {2, 3, 4}, {}, {0, 2, 0}};
  auto const haystack = cudf::slice(haystack_original, {2, 8})[0];

  auto const needles_original = lists_col{{0}, {0, 1}, {0, 0}, {3, 5, 4}, {}, {0, 0}, {} /*0*/};
  auto const needles          = cudf::slice(needles_original, {1, 5})[0];

  auto const expected = bools_col{1, 0, 0, 1};
  auto const result   = cudf::contains(haystack, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

TYPED_TEST(TypedListContainsTestColumnNeedles, SlicedInputHavingNulls)
{
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const haystack_original = lists_col{{{0, 0},
                                            {0} /*NULL*/,
                                            lists_col{{1, null}, null_at(1)},
                                            {1},
                                            {} /*NULL*/,
                                            {1, 3},
                                            {4},
                                            {} /*NULL*/,
                                            {1, 1},
                                            {0}},
                                           nulls_at({1, 4, 7})};
  auto const haystack          = cudf::slice(haystack_original, {2, 9})[0];

  auto const needles_original = lists_col{{{0, 0},
                                           {0} /*NULL*/,
                                           lists_col{{1, null}, null_at(1)},
                                           {1},
                                           {} /*NULL*/,
                                           {1, 3, 1},
                                           {4},
                                           {} /*NULL*/,
                                           {},
                                           {0}},
                                          nulls_at({1, 4, 7})};
  auto const needles          = cudf::slice(needles_original, {2, 9})[0];

  auto const expected = bools_col{{1, 1, null, 0, 1, null, 0}, nulls_at({2, 5})};
  auto const result   = cudf::contains(haystack, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

TYPED_TEST(TypedListContainsTestColumnNeedles, ListsOfStructs)
{
  using tdata_col = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto const haystack = [] {
    auto offsets = int32s_col{0, 2, 3, 5, 8, 10};
    // clang-format off
    auto data1 = tdata_col{1, 2,     //
                           1,        //
                           0, 1,     //
                           1, 3, 4,  //
                           0, 0      //
    };
    auto data2 = tdata_col{1, 3,     //
                           2,        //
                           1, 1,     //
                           0, 2, 0,  //
                           1, 2      //
    };
    // clang-format on
    auto child = structs_col{{data1, data2}};
    return cudf::make_lists_column(5, offsets.release(), child.release(), 0, {});
  }();

  auto const needles = [] {
    auto offsets = int32s_col{0, 3, 4, 6, 9, 11};
    // clang-format off
    auto data1 = tdata_col{1, 2, 1,  //
                           1,        //
                           0, 1,     //
                           1, 3, 4,  //
                           0, 0      //
    };
    auto data2 = tdata_col{1, 3, 0,  //
                           2,        //
                           1, 1,     //
                           0, 2, 2,  //
                           1, 1      //
    };
    // clang-format on
    auto child = structs_col{{data1, data2}};
    return cudf::make_lists_column(5, offsets.release(), child.release(), 0, {});
  }();

  auto const expected = bools_col{0, 1, 1, 0, 0};
  auto const result   = cudf::contains(*haystack, *needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

auto search_bounds(cudf::table_view const& t,
                   cudf::table_view const& values,
                   std::vector<cudf::order> const& column_order,
                   std::vector<cudf::null_order> const& null_precedence = {
                     cudf::null_order::BEFORE})
{
  auto result_lower_bound = cudf::lower_bound(t, values, column_order, null_precedence);
  auto result_upper_bound = cudf::upper_bound(t, values, column_order, null_precedence);
  return std::pair(std::move(result_lower_bound), std::move(result_upper_bound));
}

struct ListBinarySearch : public cudf::test::BaseFixture {};

TEST_F(ListBinarySearch, ListWithNulls)
{
  {
    using lcw           = cudf::test::lists_column_wrapper<double>;
    auto const haystack = lcw{
      lcw{-3.45967821e+12},  // 0
      lcw{-3.6912186e-32},   // 1
      lcw{9.721175},         // 2
    };

    auto const needles = lcw{
      lcw{{null, 4.22671e+32}, null_at(0)},
    };

    auto const expected = int32s_col{0};
    auto const [result_lower_bound, result_upper_bound] =
      search_bounds(cudf::table_view{{haystack}},
                    cudf::table_view{{needles}},
                    {cudf::order::ASCENDING},
                    {cudf::null_order::BEFORE});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result_lower_bound, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result_upper_bound, verbosity);
  }

  {
    using lcw       = cudf::test::lists_column_wrapper<int32_t, int32_t>;
    auto const col1 = lcw{
      lcw{{null}, null_at(0)},  // 0
      lcw{-80},                 // 1
      lcw{-17},                 // 2
    };

    auto const col2 = lcw{
      lcw{27},                  // 0
      lcw{{null}, null_at(0)},  // 1
      lcw{},                    // 2
    };

    auto const val1 = lcw{
      lcw{87},
    };

    auto const val2 = lcw{
      lcw{},
    };

    cudf::table_view input{{col1, col2}};
    cudf::table_view values{{val1, val2}};
    std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::DESCENDING};
    std::vector<cudf::null_order> null_order_flags{cudf::null_order::BEFORE,
                                                   cudf::null_order::BEFORE};

    auto const expected                                 = int32s_col{3};
    auto const [result_lower_bound, result_upper_bound] = search_bounds(
      cudf::table_view{{input}}, cudf::table_view{{values}}, column_order, null_order_flags);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result_lower_bound, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result_upper_bound, verbosity);
  }
}

TEST_F(ListBinarySearch, ListsOfStructs)
{
  // Haystack must be pre-sorted.
  auto const haystack = [] {
    auto offsets = int32s_col{0, 2, 3, 4, 5, 7, 10, 13, 16, 18};
    // clang-format off
    auto data1 = int32s_col{1, 2,
                            3,
                            3,
                            3,
                            4, 5,
                            4, 5, 4,
                            4, 5, 4,
                            4, 5, 4,
                            4, 6
    };
    auto data2 = int32s_col{1, 2,
                            3,
                            3,
                            3,
                            4, 5,
                            4, 5, 4,
                            4, 5, 4,
                            4, 5, 4,
                            5, 1
    };
    // clang-format on
    auto child = structs_col{{data1, data2}};
    return cudf::make_lists_column(9, offsets.release(), child.release(), 0, {});
  }();

  auto const needles = [] {
    auto offsets = int32s_col{0, 3, 4, 6, 8, 10, 13, 14, 15};
    // clang-format off
    auto data1 = int32s_col{1, 2, 1,
                            3,
                            4, 1,
                            0, 1,
                            1, 0,
                            1, 3, 5,
                            3,
                            3
    };
    auto data2 = int32s_col{1, 3, 0,
                            3,
                            1, 2,
                            1, 1,
                            1, 2,
                            0, 2, 2,
                            3,
                            3
    };
    // clang-format on
    auto child = structs_col{{data1, data2}};
    return cudf::make_lists_column(8, offsets.release(), child.release(), 0, {});
  }();

  auto const [result_lower_bound, result_upper_bound] = search_bounds(
    cudf::table_view{{*haystack}}, cudf::table_view{{*needles}}, {cudf::order::ASCENDING});
  auto const expected_lower_bound = int32s_col{1, 1, 4, 0, 0, 0, 1, 1};
  auto const expected_upper_bound = int32s_col{1, 4, 4, 0, 0, 0, 4, 4};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, *result_lower_bound, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, *result_upper_bound, verbosity);
}

TEST_F(ListBinarySearch, ListsOfEqualStructsInTwoTables)
{
  // Haystack must be pre-sorted.
  auto const haystack = [] {
    auto offsets = int32s_col{0, 2, 3, 4, 5, 7, 10, 13, 16, 18};
    // clang-format off
    auto data1 = int32s_col{1, 2,
                            3,
                            3,
                            3,
                            4, 5,
                            4, 5, 4,
                            4, 5, 4,
                            4, 5, 4,
                            4, 6
    };
    auto data2 = int32s_col{1, 2,
                            3,
                            3,
                            3,
                            4, 5,
                            4, 5, 4,
                            4, 5, 4,
                            4, 5, 4,
                            5, 1
    };
    // clang-format on
    auto child = structs_col{{data1, data2}};
    return cudf::make_lists_column(9, offsets.release(), child.release(), 0, {});
  }();

  auto const needles = [] {
    auto offsets = int32s_col{0, 2, 3, 4, 5, 7, 10, 13, 15, 17};
    // clang-format off
    auto data1 = int32s_col{1, 2,
                            3,
                            4,
                            5,
                            4, 5,
                            5, 5, 4,
                            4, 5, 4,
                            4, 4,
                            4, 6
    };
    auto data2 = int32s_col{1, 2,
                            3,
                            4,
                            5,
                            4, 5,
                            5, 5, 4,
                            4, 5, 4,
                            4, 4,
                            5, 1
    };
    // clang-format on
    auto child = structs_col{{data1, data2}};
    return cudf::make_lists_column(9, offsets.release(), child.release(), 0, {});
  }();

  // In this search, the two table have many equal structs.
  // This help to verify the internal implementation of two-table lex comparator in which the
  // structs column of two input tables are concatenated, ranked, then split.
  auto const [result_lower_bound, result_upper_bound] = search_bounds(
    cudf::table_view{{*haystack}}, cudf::table_view{{*needles}}, {cudf::order::ASCENDING});
  auto const expected_lower_bound = int32s_col{0, 1, 4, 9, 4, 9, 5, 4, 8};
  auto const expected_upper_bound = int32s_col{1, 4, 4, 9, 5, 9, 8, 4, 9};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, *result_lower_bound, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, *result_upper_bound, verbosity);
}

TEST_F(ListBinarySearch, CrazyListTest)
{
  // Data type: List<List<Struct<Struct<List<Struct<int, int>>>>>>

  // Haystack must be pre-sorted.
  auto const haystack = [] {
    auto lists_of_structs_of_ints = [] {
      auto offsets = int32s_col{0, 2, 3, 4, 5, 7, 10, 13, 16, 18};
      // clang-format off
      auto data1 = int32s_col{1, 2,
        3,
        3,
        3,
        4, 5,
        4, 5, 4,
        4, 5, 4,
        4, 5, 4,
        4, 6
      };
      auto data2 = int32s_col{1, 2,
        3,
        3,
        3,
        4, 5,
        4, 5, 4,
        4, 5, 4,
        4, 5, 4,
        5, 1
      };
      // clang-format on
      auto child = structs_col{{data1, data2}};
      return cudf::make_lists_column(9, offsets.release(), child.release(), 0, {});
    }();

    auto struct_nested0 = [&] {
      std::vector<std::unique_ptr<cudf::column>> child_columns;
      child_columns.emplace_back(std::move(lists_of_structs_of_ints));
      return cudf::make_structs_column(9, std::move(child_columns), 0, {});
    }();

    auto struct_nested1 = [&] {
      std::vector<std::unique_ptr<cudf::column>> child_columns;
      child_columns.emplace_back(std::move(struct_nested0));
      return cudf::make_structs_column(9, std::move(child_columns), 0, {});
    }();

    auto list_nested0 = [&] {
      auto offsets = int32s_col{0, 3, 3, 4, 6, 9};
      return cudf::make_lists_column(5, offsets.release(), std::move(struct_nested1), 0, {});
    }();

    auto offsets = int32s_col{0, 0, 2, 4, 5, 5};
    return cudf::make_lists_column(5, offsets.release(), std::move(list_nested0), 0, {});
  }();

  auto const needles = [] {
    auto lists_of_structs_of_ints = [] {
      auto offsets = int32s_col{0, 2, 3, 4, 5, 7, 10, 13, 15, 17};
      // clang-format off
      auto data1 = int32s_col{1, 2,
        3,
        4,
        5,
        4, 5,
        5, 5, 4,
        4, 5, 4,
        4, 4,
        4, 6
      };
      auto data2 = int32s_col{1, 2,
        3,
        4,
        5,
        4, 5,
        5, 5, 4,
        4, 5, 4,
        4, 4,
        5, 1
      };
      // clang-format on
      auto child = structs_col{{data1, data2}};
      return cudf::make_lists_column(9, offsets.release(), child.release(), 0, {});
    }();

    auto struct_nested0 = [&] {
      std::vector<std::unique_ptr<cudf::column>> child_columns;
      child_columns.emplace_back(std::move(lists_of_structs_of_ints));
      return cudf::make_structs_column(9, std::move(child_columns), 0, {});
    }();

    auto struct_nested1 = [&] {
      std::vector<std::unique_ptr<cudf::column>> child_columns;
      child_columns.emplace_back(std::move(struct_nested0));
      return cudf::make_structs_column(9, std::move(child_columns), 0, {});
    }();

    auto list_nested0 = [&] {
      auto offsets = int32s_col{0, 3, 3, 4, 6, 9};
      return cudf::make_lists_column(5, offsets.release(), std::move(struct_nested1), 0, {});
    }();

    auto offsets = int32s_col{0, 2, 2, 4, 4, 5};
    return cudf::make_lists_column(5, offsets.release(), std::move(list_nested0), 0, {});
  }();

  // In this search, the two table have many equal structs.
  // This help to verify the internal implementation of two-table lex comparator in which the
  // structs column of two input tables are concatenated, ranked, then split.
  auto const [result_lower_bound, result_upper_bound] = search_bounds(
    cudf::table_view{{*haystack}}, cudf::table_view{{*needles}}, {cudf::order::ASCENDING});

  auto const expected_lower_bound = int32s_col{2, 0, 5, 0, 5};
  auto const expected_upper_bound = int32s_col{2, 1, 5, 1, 5};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_bound, *result_lower_bound, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_upper_bound, *result_upper_bound, verbosity);
}
