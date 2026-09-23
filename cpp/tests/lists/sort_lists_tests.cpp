/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/sorting.hpp>

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

auto generate_sorted_lists(cudf::lists_column_view const& input,
                           cudf::order column_order,
                           cudf::null_order null_precedence)
{
  return std::pair{cudf::lists::sort_lists(input, column_order, null_precedence),
                   cudf::lists::stable_sort_lists(input, column_order, null_precedence)};
}

template <typename T>
struct SortLists : public cudf::test::BaseFixture {};

using TypesForTest = cudf::test::Concat<cudf::test::NumericTypes, cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(SortLists, TypesForTest);

TYPED_TEST(SortLists, NoNull)
{
  using T = TypeParam;

  // List<T>
  LCW<T> list{{3, 2, 1, 4}, {5}, {10, 8, 9}, {6, 7}};

  // Ascending
  // LCW<int>  order{{2, 1, 0, 3}, {0}, {1, 2, 0},  {0, 1}};
  LCW<T> expected{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  LCW<T> expected2{{4, 3, 2, 1}, {5}, {10, 9, 8}, {7, 6}};
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected2);
  }
  {
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected2);
  }
}

TYPED_TEST(SortLists, Null)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) return;
  std::vector<bool> valids_o{true, true, false, true};
  std::vector<bool> valids_a{true, true, true, false};
  std::vector<bool> valids_b{false, true, true, true};

  // List<T>
  LCW<T> list{{{3, 2, 4, 1}, valids_o.begin()}, {5}, {10, 8, 9}, {6, 7}};
  // LCW<int>  order{{2, 1, 3, 0}, {0}, {1, 2, 0},  {0, 1}};

  {
    LCW<T> expected{{{1, 2, 3, 4}, valids_a.begin()}, {5}, {8, 9, 10}, {6, 7}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    LCW<T> expected{{{4, 1, 2, 3}, valids_b.begin()}, {5}, {8, 9, 10}, {6, 7}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  // Descending
  // LCW<int>  order{{3, 0, 1, 2}, {0}, {0, 1, 2},  {1, 0}};
  {
    LCW<T> expected{{{4, 3, 2, 1}, valids_b.begin()}, {5}, {10, 9, 8}, {7, 6}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    LCW<T> expected{{{3, 2, 1, 4}, valids_a.begin()}, {5}, {10, 9, 8}, {7, 6}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{list}, cudf::order::DESCENDING, cudf::null_order::BEFORE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

using SortListsInt = SortLists<int>;

TEST_F(SortListsInt, Empty)
{
  using T = int;

  {
    LCW<T> l{};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{LCW<T>{}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{LCW<T>{}, LCW<T>{}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
}

TEST_F(SortListsInt, Single)
{
  using T = int;

  {
    LCW<T> l{1};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
  {
    LCW<T> l{{1, 2, 3}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
  }
}

TEST_F(SortListsInt, NullRows)
{
  using T = int;
  std::vector<int> valids{0, 1, 0};
  LCW<T> l{{{1, 2, 3}, {4, 5, 6}, {7}}, valids.begin()};  // offset 0, 0, 3, 3

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{l}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), l);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), l);
}

TEST_F(SortListsInt, NestedListElement)
{
  using T = int;
  // Column of LIST<LIST<int>>: each row's inner lists are reordered as whole elements. The third
  // row's inner lists tie on their first element, so ordering falls through to the second.
  LCW<T> input{LCW<T>{{3, 1}, {2, 0}}, LCW<T>{{5, 5}, {4, 9}}, LCW<T>{{1, 3}, {1, 2}}};
  {
    // Ascending.
    LCW<T> expected{LCW<T>{{2, 0}, {3, 1}}, LCW<T>{{4, 9}, {5, 5}}, LCW<T>{{1, 2}, {1, 3}}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{input}, cudf::order::ASCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
  {
    // Descending reverses each row's ascending order.
    LCW<T> expected{LCW<T>{{3, 1}, {2, 0}}, LCW<T>{{5, 5}, {4, 9}}, LCW<T>{{1, 3}, {1, 2}}};
    auto const [sorted_lists, stable_sorted_lists] = generate_sorted_lists(
      cudf::lists_column_view{input}, cudf::order::DESCENDING, cudf::null_order::AFTER);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

// E = LIST<STRUCT<int, int>>: a list-of-struct element type sorts; struct ranks are computed
// internally before the lexicographic comparison.
TEST_F(SortListsInt, ListOfStructElement)
{
  // One row with two elements [{3, 30}] and [{1, 10}]; ascending reorders to [{1, 10}], [{3, 30}].
  cudf::test::fixed_width_column_wrapper<int> in_f0{3, 1};
  cudf::test::fixed_width_column_wrapper<int> in_f1{30, 10};
  cudf::test::structs_column_wrapper in_structs{{in_f0, in_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_inner_off{0, 1, 2};
  auto in_inner = cudf::make_lists_column(2, in_inner_off.release(), in_structs.release(), 0, {});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_outer_off{0, 2};
  auto in_outer = cudf::make_lists_column(1, in_outer_off.release(), std::move(in_inner), 0, {});

  cudf::test::fixed_width_column_wrapper<int> ex_f0{1, 3};
  cudf::test::fixed_width_column_wrapper<int> ex_f1{10, 30};
  cudf::test::structs_column_wrapper ex_structs{{ex_f0, ex_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_inner_off{0, 1, 2};
  auto ex_inner = cudf::make_lists_column(2, ex_inner_off.release(), ex_structs.release(), 0, {});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_outer_off{0, 2};
  auto ex_outer = cudf::make_lists_column(1, ex_outer_off.release(), std::move(ex_inner), 0, {});

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{in_outer->view()}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), ex_outer->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), ex_outer->view());
}

// E = STRUCT<int, LIST<int>>: a struct-with-list-field element type sorts.
TEST_F(SortListsInt, StructOfListElement)
{
  // One row with two struct elements {2, [9, 0]} and {1, [8, 7]}; ascending reorders them to
  // {1, [8, 7]}, {2, [9, 0]}.
  cudf::test::fixed_width_column_wrapper<int> in_f0{2, 1};
  cudf::test::lists_column_wrapper<int, int32_t> in_f1{{9, 0}, {8, 7}};
  cudf::test::structs_column_wrapper in_structs{{in_f0, in_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> in_off{0, 2};
  auto in_list = cudf::make_lists_column(1, in_off.release(), in_structs.release(), 0, {});

  cudf::test::fixed_width_column_wrapper<int> ex_f0{1, 2};
  cudf::test::lists_column_wrapper<int, int32_t> ex_f1{{8, 7}, {9, 0}};
  cudf::test::structs_column_wrapper ex_structs{{ex_f0, ex_f1}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ex_off{0, 2};
  auto ex_list = cudf::make_lists_column(1, ex_off.release(), ex_structs.release(), 0, {});

  auto const [sorted_lists, stable_sorted_lists] =
    generate_sorted_lists(cudf::lists_column_view{in_list->view()}, {}, {});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), ex_list->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), ex_list->view());
}

TEST_F(SortListsInt, Sliced)
{
  using T = int;
  LCW<T> l{{3, 2, 1, 4}, {7, 5, 6}, {8, 9}, {10}};

  {
    auto const sliced_list = cudf::slice(l, {0, 4})[0];
    auto const expected    = LCW<T>{{1, 2, 3, 4}, {5, 6, 7}, {8, 9}, {10}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {1, 4})[0];
    auto const expected    = LCW<T>{{5, 6, 7}, {8, 9}, {10}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {1, 2})[0];
    auto const expected    = LCW<T>{{5, 6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }

  {
    auto const sliced_list = cudf::slice(l, {0, 2})[0];
    auto const expected    = LCW<T>{{1, 2, 3, 4}, {5, 6, 7}};
    auto const [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{sliced_list}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(stable_sorted_lists->view(), expected);
  }
}

using SortListsDouble = SortLists<double>;
TEST_F(SortListsDouble, InfinityAndNaN)
{
  auto constexpr NaN = std::numeric_limits<double>::quiet_NaN();
  auto constexpr Inf = std::numeric_limits<double>::infinity();

  using LCW = cudf::test::lists_column_wrapper<double>;
  {
    LCW input{-0.0, -NaN, -NaN, NaN, Inf, -Inf, 7, 5, 6, NaN, Inf, -Inf, -NaN, -NaN, -0.0};
    auto [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
    LCW expected{-Inf, -Inf, -0, -0, 5, 6, 7, Inf, Inf, -NaN, -NaN, NaN, NaN, -NaN, -NaN};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted_lists->view(), expected);
  }
  // This data includes a row with over 200 elements to test the
  // radix sort is not used in the logic path in segmented_sort.
  // Technically radix sort is not expected to be used in either case.
  {
    // clang-format off
    LCW input{0.0, -0.0, -NaN, -NaN, NaN, Inf, -Inf,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
               1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
              NaN, Inf, -Inf, -NaN, -NaN, -0.0, 0.0};
    LCW expected{-Inf, -Inf, 0.0, -0.0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0, 0,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
               4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
               5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
               6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
               7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
               8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
               9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
              Inf, Inf, -NaN, -NaN, NaN, NaN, -NaN, -NaN};
    // clang-format on
    auto [sorted_lists, stable_sorted_lists] =
      generate_sorted_lists(cudf::lists_column_view{input}, {}, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_lists->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(stable_sorted_lists->view(), expected);
  }
}
