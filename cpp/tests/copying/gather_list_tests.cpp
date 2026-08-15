/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// TODO: enable fail_on_current after lists_column_initializer (nested LCW st/mr),
// valid_if, get_sliced_child/slice, and concatenate MR ports land.

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

template <typename T>
class GatherTestListTyped : public cudf::test::BaseFixtureWithHarness {};
using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FixedPointTypes,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::DurationTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_SUITE(GatherTestListTyped, FixedWidthTypesNotBool);

class GatherTestList : public cudf::test::BaseFixtureWithHarness {};

// to disambiguate between {} == 0 and {} == List{0}
// Also, see note about compiler issues when declaring nested
// empty lists in lists_column_wrapper documentation
template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

TYPED_TEST(GatherTestListTyped, Gather)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  using T = TypeParam;

  // List<T>
  LCW<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
  cudf::test::fixed_width_column_wrapper<int> gather_map{{0, 2}, st, mr};

  cudf::table_view source_table({list});
  auto results =
    cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

  LCW<T> expected{{1, 2, 3, 4}, {6, 7}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
}

TYPED_TEST(GatherTestListTyped, GatherNothing)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  using T = TypeParam;

  // List<T>
  {
    LCW<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{};

    cudf::table_view source_table({list});
    auto results =
      cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

    LCW<T> expected;

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
  }

  // List<T>
  {
    cudf::test::lists_column_wrapper<int> list{{{{1, 2, 3, 4}, {5}}}, {{{6, 7}, {8, 9, 10}}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{};

    cudf::table_view source_table({list});
    auto result =
      cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

    // the result should preserve the full List<List<List<int>>> hierarchy
    // even though it is empty past the first level
    cudf::lists_column_view lcv(result->view().column(0));
    EXPECT_EQ(lcv.size(), 0);
    EXPECT_EQ(lcv.child().type().id(), cudf::type_id::LIST);
    EXPECT_EQ(lcv.child().size(), 0);
    EXPECT_EQ(cudf::lists_column_view(lcv.child()).child().type().id(), cudf::type_id::LIST);
    EXPECT_EQ(cudf::lists_column_view(lcv.child()).child().size(), 0);
    EXPECT_EQ(
      cudf::lists_column_view(cudf::lists_column_view(lcv.child()).child()).child().type().id(),
      cudf::type_id::INT32);
    EXPECT_EQ(cudf::lists_column_view(cudf::lists_column_view(lcv.child()).child()).child().size(),
              0);
  }
}

TYPED_TEST(GatherTestListTyped, GatherNulls)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  using T = TypeParam;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<T>
  LCW<T> list{{{1, 2, 3, 4}, valids}, {5}, {{6, 7}, valids}, {{8, 9, 10}, valids}};
  cudf::test::fixed_width_column_wrapper<int> gather_map{{0, 2}, st, mr};

  cudf::table_view source_table({list});
  auto results =
    cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

  LCW<T> expected{{{1, 2, 3, 4}, valids}, {{6, 7}, valids}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
}

TYPED_TEST(GatherTestListTyped, GatherNested)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  using T = TypeParam;

  // List<List<T>>
  {
    LCW<T> list{{{2, 3}, {4, 5}},
                {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{{0, 2}, st, mr};

    cudf::table_view source_table({list});
    auto results =
      cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

    LCW<T> expected{{{2, 3}, {4, 5}}, {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
  }

  // List<List<List<T>>>
  {
    LCW<T> list{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                {{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
                {{LCW<T>{0}}},
                {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
                 {{0, 1, 3}, {5}},
                 {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
                {{{10, 20}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{{1, 2, 4}, st, mr};

    cudf::table_view source_table({list});
    auto results =
      cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

    LCW<T> expected{{{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
                    {{LCW<T>{0}}},
                    {{{10, 20}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
  }
}

TYPED_TEST(GatherTestListTyped, GatherOutOfOrder)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  using T = TypeParam;

  // List<List<T>>
  {
    LCW<T> list{{{2, 3}, {4, 5}},
                {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{{1, 2, 0}, st, mr};

    cudf::table_view source_table({list});
    auto results =
      cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

    LCW<T> expected{{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                    {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}},
                    {{2, 3}, {4, 5}}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
  }
}

TYPED_TEST(GatherTestListTyped, GatherNestedNulls)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  using T = TypeParam;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<List<T>>
  {
    LCW<T> list{{{{2, 3}, valids}, {4, 5}},
                {{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, valids},
                {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}},
                {{{{25, 26}, valids}, {27, 28}, {{29, 30}, valids}, {31, 32}, {33, 34}}, valids}};

    cudf::test::fixed_width_column_wrapper<int> gather_map{{0, 1, 3}, st, mr};

    cudf::table_view source_table({list});
    auto results =
      cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

    LCW<T> expected{
      {{{2, 3}, valids}, {4, 5}},
      {{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, valids},
      {{{{25, 26}, valids}, {27, 28}, {{29, 30}, valids}, {31, 32}, {33, 34}}, valids}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
  }

  // List<List<List<T>>>
  {
    LCW<T> list{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
                {{LCW<T>{0}}},
                {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
                 {{0, 1, 3}, {5}},
                 {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
                {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids}};

    cudf::test::fixed_width_column_wrapper<int> gather_map{{1, 2, 4}, st, mr};

    cudf::table_view source_table({list});
    auto results =
      cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

    LCW<T> expected{{{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
                    {{LCW<T>{0}}},
                    {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
  }
}

TYPED_TEST(GatherTestListTyped, GatherNestedWithEmpties)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  using T = TypeParam;

  LCW<T> list{{{2, 3}, LCW<T>{}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, {LCW<T>{}}};
  cudf::test::fixed_width_column_wrapper<int> gather_map{{0, 2}, st, mr};

  cudf::table_view source_table({list});
  auto results =
    cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

  LCW<T> expected{{{2, 3}, LCW<T>{}}, {LCW<T>{}}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
}

TYPED_TEST(GatherTestListTyped, GatherDetailInvalidIndex)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  using T = TypeParam;

  // List<List<T>>
  {
    LCW<T> list{{{2, 3}, {4, 5}},
                {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    cudf::test::fixed_width_column_wrapper<int> gather_map{{0, 15, 16, 2}, st, mr};

    cudf::table_view source_table({list});
    auto results =
      cudf::gather(source_table, gather_map, cudf::out_of_bounds_policy::NULLIFY, st, mr);

    std::vector<int32_t> expected_validity{1, 0, 0, 1};
    LCW<T> expected{{{{2, 3}, {4, 5}},
                     {LCW<T>{}},
                     {LCW<T>{}},
                     {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
                    expected_validity.begin()};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view().column(0), expected, cudf::test::debug_output_level::FIRST_ERROR, st, mr);
  }
}

TEST_F(GatherTestList, GatherIncompleteHierarchies)
{
  auto const st = this->stream();
  auto const mr = this->resources();

  using LCW = cudf::test::lists_column_wrapper<int32_t>;

  {
    // List<List<List<int>, but rows 1 and 2 are empty at the very top.
    // We expect to get back a "full" hierarchy of type List<List<List<int>> anyway.
    cudf::test::lists_column_wrapper<int32_t> list{{{{1, 2}}}, LCW{}, LCW{}};

    cudf::table_view source_table({list});

    cudf::test::fixed_width_column_wrapper<int32_t> row1_map{{1}, st, mr};
    auto result =
      cudf::gather(source_table, row1_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

    // the result should preserve the full List<List<List<int>>> hierarchy
    // even though it is empty past the first level
    cudf::lists_column_view lcv(result->view().column(0));
    EXPECT_EQ(lcv.size(), 1);
    EXPECT_EQ(lcv.child().type().id(), cudf::type_id::LIST);
    EXPECT_EQ(lcv.child().size(), 0);
    EXPECT_EQ(cudf::lists_column_view(lcv.child()).child().type().id(), cudf::type_id::LIST);
    EXPECT_EQ(cudf::lists_column_view(lcv.child()).child().size(), 0);
    EXPECT_EQ(
      cudf::lists_column_view(cudf::lists_column_view(lcv.child()).child()).child().type().id(),
      cudf::type_id::INT32);
    EXPECT_EQ(cudf::lists_column_view(cudf::lists_column_view(lcv.child()).child()).child().size(),
              0);
  }

  {
    // List<List<List<int>, gathering nothing.
    // We expect to get back a "full" hierarchy of type List<List<List<int>> anyway.
    cudf::test::lists_column_wrapper<int32_t> list{{{{1, 2}}}, LCW{}};

    cudf::table_view source_table({list});

    cudf::test::fixed_width_column_wrapper<int32_t> empty_map{};
    auto result =
      cudf::gather(source_table, empty_map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);

    // the result should preserve the full List<List<List<int>>> hierarchy
    // even though it is empty past the first level
    cudf::lists_column_view lcv(result->view().column(0));
    EXPECT_EQ(lcv.size(), 0);
    EXPECT_EQ(lcv.child().type().id(), cudf::type_id::LIST);
    EXPECT_EQ(lcv.child().size(), 0);
    EXPECT_EQ(cudf::lists_column_view(lcv.child()).child().type().id(), cudf::type_id::LIST);
    EXPECT_EQ(cudf::lists_column_view(lcv.child()).child().size(), 0);
    EXPECT_EQ(
      cudf::lists_column_view(cudf::lists_column_view(lcv.child()).child()).child().type().id(),
      cudf::type_id::INT32);
    EXPECT_EQ(cudf::lists_column_view(cudf::lists_column_view(lcv.child()).child()).child().size(),
              0);
  }
}

TYPED_TEST(GatherTestListTyped, GatherSliced)
{
  // TODO: cudf::split/slice still use current MR (no mr parameter yet)
  auto const st = this->stream();
  auto const mr = this->resources();

  using T = TypeParam;
  {
    LCW<T> a{
      {{1, 1, 1}, {2, 2}, {3, 3}},
      {{4, 4, 4}, {5, 5}, {6, 6}},
      {{7, 7, 7}, {8, 8}, {9, 9}},
      {{10, 10, 10}, {11, 11}, {12, 12}},
      {{20, 20, 20, 20}, {25}},
      {{30, 30, 30, 30}, {40}},
      {{50, 50, 50, 50}, {6, 13}},
      {{70, 70, 70, 70}, {80}},
    };
    auto split_a = cudf::split(a, {3}, st);
    cudf::table_view tbl0({split_a[0]});
    cudf::table_view tbl1({split_a[1]});

    auto result0 = cudf::gather(tbl0,
                                cudf::test::fixed_width_column_wrapper<int>{{1, 2}, st, mr},
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                st,
                                mr);
    LCW<T> expected0{
      {{4, 4, 4}, {5, 5}, {6, 6}},
      {{7, 7, 7}, {8, 8}, {9, 9}},
    };
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      expected0, result0->get_column(0).view(), cudf::test::debug_output_level::FIRST_ERROR, st, mr);

    auto result1 = cudf::gather(tbl1,
                                cudf::test::fixed_width_column_wrapper<int>{{0, 3}, st, mr},
                                cudf::out_of_bounds_policy::DONT_CHECK,
                                st,
                                mr);
    LCW<T> expected1{
      {{10, 10, 10}, {11, 11}, {12, 12}},
      {{50, 50, 50, 50}, {6, 13}},
    };
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      expected1, result1->get_column(0).view(), cudf::test::debug_output_level::FIRST_ERROR, st, mr);
  }

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<List<List<T>>>
  {
    LCW<T> list{
      // slice 0
      {{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},

      {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}},
       {{11, 12}, {{42, 43, 44}, valids}, {{77, 78}, valids}}},

      // slice 1
      {{LCW<T>{0}}},
      {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
       {{0, 1, 3}, {5}},
       {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
      {{{{1, 6}, {60, 70, 80, 100}}, {{10, 11, 13}, {15}}, {{11, 12, 13, 14, 15}}}, valids},

      // slice 2
      {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids},
      {{{{10, 20, 30}}, {LCW<T>{30}}, {{{20, 30}, valids}, {62, 72, 82}}}, valids}};

    auto sliced = cudf::slice(list, {0, 1, 2, 5, 5, 7}, st);

    // gather from slice 0
    {
      cudf::table_view tbl({sliced[0]});

      cudf::test::fixed_width_column_wrapper<int> map{{0}, st, mr};
      auto result = cudf::gather(tbl, map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);
      LCW<T> expected{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}}};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected,
                                          result->get_column(0).view(),
                                          cudf::test::debug_output_level::FIRST_ERROR,
                                          cudf::test::default_ulp,
                                          st,
                                          mr);
    }

    // gather from slice 1
    {
      cudf::table_view tbl({sliced[1]});

      cudf::test::fixed_width_column_wrapper<int> map{{1, 2, 0, 1}, st, mr};
      auto result = cudf::gather(tbl, map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);
      LCW<T> expected{
        {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
         {{0, 1, 3}, {5}},
         {{11, 12, 13, 14, 15}, {16, 17}, {0}}},

        {{{{1, 6}, {60, 70, 80, 100}}, {{10, 11, 13}, {15}}, {{11, 12, 13, 14, 15}}}, valids},

        {{LCW<T>{0}}},

        {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
         {{0, 1, 3}, {5}},
         {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
      };
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected,
                                          result->get_column(0).view(),
                                          cudf::test::debug_output_level::FIRST_ERROR,
                                          cudf::test::default_ulp,
                                          st,
                                          mr);
    }

    // gather from slice 2
    {
      cudf::table_view tbl({sliced[2]});

      cudf::test::fixed_width_column_wrapper<int> map{{1, 0, 0, 1, 1, 0}, st, mr};
      auto result = cudf::gather(tbl, map, cudf::out_of_bounds_policy::DONT_CHECK, st, mr);
      LCW<T> expected{{{{{10, 20, 30}}, {LCW<T>{30}}, {{{20, 30}, valids}, {62, 72, 82}}}, valids},
                      {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids},
                      {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids},
                      {{{{10, 20, 30}}, {LCW<T>{30}}, {{{20, 30}, valids}, {62, 72, 82}}}, valids},
                      {{{{10, 20, 30}}, {LCW<T>{30}}, {{{20, 30}, valids}, {62, 72, 82}}}, valids},
                      {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids}};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected,
                                          result->get_column(0).view(),
                                          cudf::test::debug_output_level::FIRST_ERROR,
                                          cudf::test::default_ulp,
                                          st,
                                          mr);
    }
  }
}
