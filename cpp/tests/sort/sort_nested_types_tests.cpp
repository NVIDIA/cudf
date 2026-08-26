/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

using int32s_lists = cudf::test::lists_column_wrapper<int32_t>;
using int32s_col   = cudf::test::fixed_width_column_wrapper<int32_t>;
using strings_col  = cudf::test::strings_column_wrapper;
using structs_col  = cudf::test::structs_column_wrapper;

using namespace cudf::test::iterators;

constexpr auto null{0};

struct NestedStructTest : public cudf::test::BaseFixture {};

TEST_F(NestedStructTest, SimpleStructsOfListsNoNulls)
{
  auto const input = [] {
    auto child = int32s_lists{{4, 2, 0}, {2}, {0, 5}, {1, 5}, {4, 1}};
    return structs_col{{child}};
  }();

  {
    auto const expected_order = int32s_col{2, 3, 1, 4, 0};
    auto const order          = cudf::sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{0, 4, 1, 3, 2};
    auto const order = cudf::sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, SimpleStructsOfListsWithNulls)
{
  auto const input = [] {
    auto child =
      int32s_lists{{{4, 2, null}, null_at(2)}, {2}, {{null, 5}, null_at(0)}, {0, 5}, {4, 1}};
    return structs_col{{child}};
  }();

  {
    auto const expected_order = int32s_col{2, 3, 1, 4, 0};
    auto const order          = cudf::sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{0, 4, 1, 3, 2};
    auto const order = cudf::sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, StructsHaveListsNoNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  auto const input = [] {
    auto child0 = int32s_lists{{4, 2, 0}, {}, {5}, {4, 1}, {4, 0}, {}, {}};
    auto child1 = int32s_col{1, 2, 5, 0, 3, 3, 4};
    return structs_col{{child0, child1}};
  }();

  {
    auto const expected_order = int32s_col{1, 5, 6, 4, 3, 0, 2};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{2, 0, 3, 4, 6, 5, 1};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, StructsHaveListsWithNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  auto const input = [] {
    auto child0 =
      int32s_lists{{{4, 2, null}, null_at(2)}, {}, {} /*NULL*/, {5}, {4, 1}, {4, 0}, {}, {}};
    auto child1 = int32s_col{{1, 2, null, 5, null, 3, 3, 4}, nulls_at({2, 4})};
    return structs_col{{child0, child1}, null_at(2)};
  }();

  {
    auto const expected_order = int32s_col{2, 1, 6, 7, 5, 4, 0, 3};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{3, 0, 4, 5, 7, 6, 1, 2};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, SlicedStructsHaveListsNoNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  // The original input has 3 first elements repeated at the beginning and the end.
  auto const input_original = [] {
    auto child0 = int32s_lists{
      {4, 2, 0}, {}, {5}, {4, 2, 0}, {}, {5}, {4, 1}, {4, 0}, {}, {}, {4, 2, 0}, {}, {5}};
    auto child1 = int32s_col{1, 2, 5, 1, 2, 5, 0, 3, 3, 4, 1, 2, 5};
    return structs_col{{child0, child1}};
  }();

  auto const input = cudf::slice(input_original, {3, 10})[0];

  {
    auto const expected_order = int32s_col{1, 5, 6, 4, 3, 0, 2};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{2, 0, 3, 4, 6, 5, 1};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, SlicedStructsHaveListsWithNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  // The original input has 2 first elements repeated at the beginning and the end.
  auto const input_original = [] {
    auto child0 = int32s_lists{{{4, 2, null}, null_at(2)},
                               {},
                               {{4, 2, null}, null_at(2)},
                               {},
                               {} /*NULL*/,
                               {5},
                               {4, 1},
                               {4, 0},
                               {},
                               {},
                               {{4, 2, null}, null_at(2)},
                               {}};
    auto child1 = int32s_col{{1, 2, 1, 2, null, 5, null, 3, 3, 4, 1, 2}, nulls_at({4, 6})};
    return structs_col{{child0, child1}, null_at(4)};
  }();

  auto const input = cudf::slice(input_original, {2, 10})[0];

  {
    auto const expected_order = int32s_col{2, 1, 6, 7, 5, 4, 0, 3};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{3, 0, 4, 5, 7, 6, 1, 2};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, StructsOfStructsHaveListsNoNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  auto const input = [] {
    auto child0 = [] {
      auto child0 = int32s_lists{{4, 2, 0}, {}, {5}, {4, 1}, {4, 0}, {}, {}};
      auto child1 = int32s_col{1, 2, 5, 0, 3, 3, 4};
      return structs_col{{child0, child1}};
    }();
    auto child1 = int32s_lists{{4, 2, 0}, {}, {5}, {4, 1}, {4, 0}, {}, {}};
    auto child2 = int32s_col{1, 2, 5, 0, 3, 3, 4};
    return structs_col{{child0, child1, child2}};
  }();

  {
    auto const expected_order = int32s_col{1, 5, 6, 4, 3, 0, 2};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{2, 0, 3, 4, 6, 5, 1};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, StructsOfStructsHaveListsWithNulls)
{
  // Input has equal elements, thus needs to be tested by stable sort.
  auto const input = [] {
    auto child0 = [] {
      auto child0 =
        int32s_lists{{{4, 2, null}, null_at(2)}, {}, {} /*NULL*/, {5}, {4, 1}, {4, 0}, {}, {}};
      auto child1 = int32s_col{{1, 2, null, 5, null, 3, 3, 4}, nulls_at({2, 4})};
      return structs_col{{child0, child1}, null_at(2)};
    }();
    auto child1 =
      int32s_lists{{{4, 2, null}, null_at(2)}, {}, {} /*NULL*/, {5}, {4, 1}, {4, 0}, {}, {}};
    auto child2 = int32s_col{{1, 2, null, 5, null, 3, 3, 4}, nulls_at({2, 4})};
    return structs_col{{child0, child1, child2}, null_at(2)};
  }();

  {
    auto const expected_order = int32s_col{2, 1, 6, 7, 5, 4, 0, 3};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{3, 0, 4, 5, 7, 6, 1, 2};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedStructTest, SimpleStructsOfListsOfStructsNoNulls)
{
  auto const input = [] {
    auto const make_lists_of_structs = [] {
      auto const get_structs = [] {
        auto child0 = int32s_col{3, 2, 3, 3, 4, 2, 4, 4, 1, 0, 3, 0, 2, 5, 4};
        auto child1 = int32s_col{0, 4, 3, 2, 1, 1, 5, 1, 5, 5, 4, 2, 4, 1, 3};
        return structs_col{{child0, child1}};
      };
      return cudf::make_lists_column(
        8, int32s_col{0, 3, 5, 6, 6, 8, 10, 12, 15}.release(), get_structs().release(), 0, {});
    };

    std::vector<std::unique_ptr<cudf::column>> children;
    children.emplace_back(make_lists_of_structs());
    children.emplace_back(make_lists_of_structs());

    return cudf::make_structs_column(8, std::move(children), 0, {});
  }();

  {
    auto const expected_order = int32s_col{3, 5, 2, 7, 0, 1, 6, 4};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{*input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{4, 6, 1, 0, 7, 2, 5, 3};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{*input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

struct NestedListTest : public cudf::test::BaseFixture {};

TEST_F(NestedListTest, SimpleListsOfStructsNoNulls)
{
  auto const input = [] {
    auto const get_structs = [] {
      auto child0 = int32s_col{3, 2, 3, 3, 4, 2, 4, 4, 1, 0, 3, 0, 2, 5, 4};
      auto child1 = int32s_col{0, 4, 3, 2, 1, 1, 5, 1, 5, 5, 4, 2, 4, 1, 3};
      return structs_col{{child0, child1}};
    };
    return cudf::make_lists_column(
      8, int32s_col{0, 3, 5, 6, 6, 8, 10, 12, 15}.release(), get_structs().release(), 0, {});
  }();

  {
    auto const expected_order = int32s_col{3, 5, 2, 7, 0, 1, 6, 4};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{*input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{4, 6, 1, 0, 7, 2, 5, 3};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{*input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedListTest, SlicedListsOfStructsNoNulls)
{
  auto const input_original = [] {
    auto const get_structs = [] {
      auto child0 = int32s_col{0, 0, 3, 2, 3, 3, 4, 2, 4, 4, 1, 0, 3, 0, 2, 5, 4, 0};
      auto child1 = int32s_col{0, 0, 0, 4, 3, 2, 1, 1, 5, 1, 5, 5, 4, 2, 4, 1, 3, 0};
      return structs_col{{child0, child1}};
    };
    return cudf::make_lists_column(11,
                                   int32s_col{0, 1, 2, 5, 7, 8, 8, 10, 12, 14, 17, 18}.release(),
                                   get_structs().release(),
                                   0,
                                   {});
  }();
  auto const input = cudf::slice(*input_original, {2, 10})[0];

  {
    auto const expected_order = int32s_col{3, 5, 2, 7, 0, 1, 6, 4};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{4, 6, 1, 0, 7, 2, 5, 3};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedListTest, ListsOfEqualStructsNoNulls)
{
  auto const input = [] {
    auto const get_structs = [] {
      auto child0 = int32s_col{0, 3, 0, 1};
      auto child1 = strings_col{"a", "c", "a", "b"};
      return structs_col{{child0, child1}};
    };
    return cudf::make_lists_column(
      2, int32s_col{0, 2, 4}.release(), get_structs().release(), 0, {});
  }();

  {
    auto const expected_order = int32s_col{1, 0};
    auto const order          = cudf::sorted_order(cudf::table_view{{*input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{0, 1};
    auto const order = cudf::sorted_order(cudf::table_view{{*input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedListTest, SimpleListsOfStructsWithNulls)
{
  // [ {null, 2},    {null, null}, {1, 2} ]     | 0
  // []                                         | 1
  // [ {null, null}, {4, 2} ]                   | 2
  // []                                         | 3
  // [ {3, 5},       {null, 4}            ]     | 4
  // []                                         | 5
  // [ {5, 3},       {5, 0},       {1, 1} ]     | 6
  // [ {null, 3},    {5, 2},       {4, 2} ]     | 7
  auto const input = [] {
    auto const get_structs = [] {
      auto child0 = int32s_col{{null, null, 1, null, 4, 3, null, 5, 5, 1, null, 5, 4},
                               nulls_at({0, 1, 3, 6, 10})};
      auto child1 = int32s_col{{2, null, 2, null, 2, 5, 4, 3, 0, 1, 3, 2, 2}, nulls_at({1, 3})};
      return structs_col{{child0, child1}, nulls_at({1, 3})};
    };
    return cudf::make_lists_column(
      8, int32s_col{0, 3, 3, 5, 5, 7, 7, 10, 13}.release(), get_structs().release(), 0, {});
  }();

  {
    auto const expected_order = int32s_col{1, 3, 5, 2, 0, 7, 4, 6};
    auto const order          = cudf::stable_sorted_order(
      cudf::table_view{{*input}}, {cudf::order::ASCENDING}, {cudf::null_order::BEFORE});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{6, 4, 7, 0, 2, 1, 3, 5};
    auto const order          = cudf::stable_sorted_order(
      cudf::table_view{{*input}}, {cudf::order::DESCENDING}, {cudf::null_order::BEFORE});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{1, 3, 5, 2, 4, 6, 0, 7};
    auto const order          = cudf::stable_sorted_order(
      cudf::table_view{{*input}}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{7, 0, 6, 4, 2, 1, 3, 5};
    auto const order          = cudf::stable_sorted_order(
      cudf::table_view{{*input}}, {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedListTest, ListsOfListsOfStructsNoNulls)
{
  auto const input = [] {
    auto const get_structs = [] {
      auto child0 = int32s_col{0, 7, 4, 9, 2, 9, 4, 1, 5, 5, 3, 7, 0, 6, 3, 1, 9};
      auto child1 = int32s_col{4, 6, 7, 3, 1, 2, 1, 10, 7, 9, 8, 7, 1, 10, 5, 3, 3};
      return structs_col{{child0, child1}};
    };
    auto lists_of_structs =
      cudf::make_lists_column(13,
                              int32s_col{0, 1, 3, 4, 5, 7, 9, 10, 12, 12, 14, 15, 17, 17}.release(),
                              get_structs().release(),
                              0,
                              {});
    return cudf::make_lists_column(
      8, int32s_col{0, 3, 4, 6, 6, 8, 10, 10, 13}.release(), std::move(lists_of_structs), 0, {});
  }();

  {
    auto const expected_order = int32s_col{3, 6, 5, 0, 1, 7, 4, 2};
    auto const order          = cudf::stable_sorted_order(cudf::table_view{{*input}});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }

  {
    auto const expected_order = int32s_col{2, 4, 7, 1, 0, 5, 3, 6};
    auto const order =
      cudf::stable_sorted_order(cudf::table_view{{*input}}, {cudf::order::DESCENDING});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
  }
}

TEST_F(NestedListTest, MultipleListsColumnsWithNulls)
{
  // A STRUCT<LIST<INT>> column with all nulls.
  auto const col0 = [] {
    auto child = int32s_lists{{int32s_lists{}, int32s_lists{}}, all_nulls()};
    return structs_col{{child}, all_nulls()};
  }();

  auto const col1 = [] {
    auto child = int32s_lists{{0, 1, 2}, {10, 11, 12}};
    return structs_col{{child}};
  }();

  auto const col2 = int32s_col{1, 0};

  auto const expected_order = int32s_col{0, 1};
  auto const order          = cudf::sorted_order(cudf::table_view{{col0, col1, col2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_order, order->view());
}
