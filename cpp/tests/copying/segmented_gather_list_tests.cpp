/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/gather.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <stdexcept>

template <typename T>
class SegmentedGatherTest : public cudf::test::BaseFixture {};
using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::DurationTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_SUITE(SegmentedGatherTest, FixedWidthTypesNotBool);

// to disambiguate between {} == 0 and {} == List{0}
// Also, see note about compiler issues when declaring nested
// empty lists in lists_column_wrapper documentation
template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;
using namespace cudf::test::iterators;
auto constexpr NULLIFY = cudf::out_of_bounds_policy::NULLIFY;

TYPED_TEST(SegmentedGatherTest, Gather)
{
  using T = TypeParam;

  // List<T>
  LCW<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};

  {
    // Straight-line case.
    auto const gather_map = LCW<int>{{3, 2, 1, 0}, {0}, {0, 1}, {0, 2, 1}};
    auto const expected   = LCW<T>{{4, 3, 2, 1}, {5}, {6, 7}, {8, 10, 9}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }

  {
    // Nullify out-of-bounds values.
    auto const gather_map = LCW<int>{{3, 2, 4, 0}, {0}, {0, -3}, {0, 2, 1}};
    auto const expected = LCW<T>{{{4, 3, 2, 1}, null_at(2)}, {5}, {{6, 7}, null_at(1)}, {8, 10, 9}};
    auto const results  = cudf::lists::segmented_gather(
      cudf::lists_column_view{list}, cudf::lists_column_view{gather_map}, NULLIFY);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNothing)
{
  using T = TypeParam;

  // List<T>
  {
    auto const list       = LCW<T>{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
    auto const gather_map = LCW<int>{LCW<int>{}, LCW<int>{}, LCW<int>{}, LCW<int>{}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map});
    auto const expected   = LCW<T>{LCW<T>{}, LCW<T>{}, LCW<T>{}, LCW<T>{}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  // List<List<T>>
  {
    auto const list       = LCW<T>{{{1, 2, 3, 4}, {5}}, {{6, 7}}, {{}, {8, 9, 10}}};
    auto const gather_map = LCW<int>{LCW<int>{}, LCW<int>{}, LCW<int>{}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map});

    // hack to get column of empty list of list
    auto const expected_dummy = LCW<T>{{{1, 2, 3, 4}, {5}}, LCW<T>{}, LCW<T>{}, LCW<T>{}};
    auto const expected       = cudf::split(expected_dummy, {1})[1];
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  // List<List<List<T>>>
  {
    auto const list       = LCW<T>{{{{1, 2, 3, 4}, {5}}}, {{{6, 7}, {8, 9, 10}}}};
    auto const gather_map = LCW<int>{LCW<int>{}, LCW<int>{}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map});
    // hack to get column of empty list of list of list
    auto const expected_dummy = LCW<T>{{{{1, 2, 3, 4}}}, LCW<T>{}, LCW<T>{}};
    auto const expected       = cudf::split(expected_dummy, {1})[1];
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    // the result should preserve the full List<List<List<int>>> hierarchy
    // even though it is empty past the first level
    cudf::lists_column_view lcv(results->view());
    EXPECT_EQ(lcv.size(), 2);
    EXPECT_EQ(lcv.child().type().id(), cudf::type_id::LIST);
    EXPECT_EQ(lcv.child().size(), 0);
    EXPECT_EQ(cudf::lists_column_view(lcv.child()).child().type().id(), cudf::type_id::LIST);
    EXPECT_EQ(cudf::lists_column_view(lcv.child()).child().size(), 0);
    EXPECT_EQ(
      cudf::lists_column_view(cudf::lists_column_view(lcv.child()).child()).child().type().id(),
      cudf::type_to_id<T>());
    EXPECT_EQ(cudf::lists_column_view(cudf::lists_column_view(lcv.child()).child()).child().size(),
              0);
  }
}

using SegmentedGatherTestSingle = SegmentedGatherTest<int32_t>;
TEST_F(SegmentedGatherTestSingle, GatherEmpty)
{
  auto const list       = LCW<int32_t>{};
  auto const gather_map = LCW<cudf::size_type>{};
  auto const expected   = LCW<int32_t>{};
  auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                     cudf::lists_column_view{gather_map});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(SegmentedGatherTest, GatherNulls)
{
  using T = TypeParam;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<T>
  auto const list = LCW<T>{{{1, 2, 3, 4}, valids}, {5}, {{6, 7}, valids}, {{8, 9, 10}, valids}};

  {
    // Test gathering on lists that contain nulls.
    auto const gather_map = LCW<int>{{0, 1}, LCW<int>{}, {1}, {2, 1, 0}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map});
    auto const expected =
      LCW<T>{{{1, 2}, valids}, LCW<T>{}, {{7}, valids + 1}, {{10, 9, 8}, valids}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
  {
    // Test gathering on lists that contain nulls, with out-of-bounds indices.
    auto const gather_map = LCW<int>{{10, -10}, LCW<int>{}, {1}, {2, -10, 0}};
    auto const results    = cudf::lists::segmented_gather(
      cudf::lists_column_view{list}, cudf::lists_column_view{gather_map}, NULLIFY);
    auto const expected =
      LCW<T>{{{0, 0}, nulls_at({0, 1})}, LCW<T>{}, {{7}, valids + 1}, {{10, 0, 8}, null_at(1)}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNested)
{
  using T = TypeParam;

  // List<List<T>>
  {
    // clang-format off
    auto const list       = LCW<T>{{{2, 3}, {4, 5}},
                                   {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                   {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {-17, -18}}};
    auto const gather_map = LCW<int>{{0, -2, -2}, {1}, {1, 0, -1, -5}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list}, cudf::lists_column_view{gather_map});
    auto const expected   = LCW<T>{{{2, 3}, {2, 3}, {2, 3}},
                                   {{9, 10, 11}},
                                   {{17, 18}, {15, 16}, {-17, -18}, {15, 16}}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    // clang-format on
  }

  // List<List<T>>, with out-of-bounds gather indices.
  {
    // clang-format off
    auto const list       = LCW<T>{{{2, 3}, {4, 5}},
                                   {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                   {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {-17, -18}}};
    auto const gather_map = LCW<int>{{0, 2, -2}, {1}, {1, 0, -1, -6}};
    auto const results =
      cudf::lists::segmented_gather(cudf::lists_column_view{list}, cudf::lists_column_view{gather_map}, NULLIFY);
    auto const expected = LCW<T>{{{{2, 3}, LCW<T>{}, {2, 3}}, null_at(1)},
                                 {{9, 10, 11}},
                                 {{{17, 18}, {15, 16}, {-17, -18}, LCW<T>{}}, null_at(3)}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    // clang-format on
  }

  // List<List<List<T>>>
  {
    // clang-format off
    auto const list = LCW<T>{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                             {{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
                             {{LCW<T>{0}}},
                             {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
                              {{0, 1, 3}, {5}},
                              {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
                             {{{10, 20}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}};
    auto const gather_map = LCW<int>{{1}, LCW<int>{}, {0}, {1}, {0, -1, 1}};
    auto const results = cudf::lists::segmented_gather(cudf::lists_column_view{list}, cudf::lists_column_view{gather_map});
    auto const expected = LCW<T>{{{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                                 LCW<T>{},
                                 {{LCW<T>{0}}},
                                 {{{0, 1, 3}, {5}}},
                                 {{{10, 20}}, {{40, 50}, {60, 70, 80}}, {LCW<T>{30}}}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    // clang-format on
  }

  // List<List<List<T>>>, with out-of-bounds gather indices.
  {
    auto const list       = LCW<T>{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                                   {{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
                                   {{LCW<T>{0}}},
                                   {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
                                    {{0, 1, 3}, {5}},
                                    {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
                                   {{{10, 20}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}};
    auto const gather_map = LCW<int>{{1}, LCW<int>{}, {0}, {1}, {0, -1, 3, -4}};
    auto const results    = cudf::lists::segmented_gather(
      cudf::lists_column_view{list}, cudf::lists_column_view{gather_map}, NULLIFY);
    auto const expected =
      LCW<T>{{{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
             LCW<T>{},
             {{LCW<T>{0}}},
             {{{0, 1, 3}, {5}}},
             {{{{10, 20}}, {{40, 50}, {60, 70, 80}}, LCW<T>{}, LCW<T>{}}, nulls_at({2, 3})}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    // clang-format on
  }
}

TYPED_TEST(SegmentedGatherTest, GatherOutOfOrder)
{
  using T = TypeParam;

  // List<List<T>>
  {
    // clang-format off
    auto const list       = LCW<T>{{{2, 3}, {4, 5}},
                                   {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                   {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    auto const gather_map = LCW<int>{{1, 0}, {1, 2, 0}, {4, 3, 2, 1, 0}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list}, cudf::lists_column_view{gather_map});
    auto const expected   = LCW<T>{{{4, 5}, {2, 3}},
                                   {{9, 10, 11}, {12, 13, 14}, {6, 7, 8}},
                                   {{17, 18}, {17, 18}, {17, 18}, {17, 18}, {15, 16}}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    // clang-format on
  }

  // List<List<T>>, with out-of-bounds gather indices.
  {
    // clang-format off
    auto const list       = LCW<T>{{{2, 3}, {4, 5}},
                                   {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                   {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    auto const gather_map = LCW<int>{{1, 0}, {3, -1, -4}, {5, 4, 3, 2, 1, 0}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list}, cudf::lists_column_view{gather_map}, NULLIFY);
    auto const expected   = LCW<T>{{{4, 5}, {2, 3}},
                                   {{LCW<T>{}, {12, 13, 14}, LCW<T>{}}, nulls_at({0, 2})},
                                   {{LCW<T>{}, {17, 18}, {17, 18}, {17, 18}, {17, 18}, {15, 16}}, null_at(0)}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    // clang-format on
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNegatives)
{
  using T = TypeParam;

  // List<List<T>>
  {
    // clang-format off
    auto const list       = LCW<T>{{{2, 3}, {4, 5}},
                                   {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                   {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    auto const gather_map = LCW<int>{{-1, 0}, {-2, -1, 0}, {-5, -4, -3, -2, -1, 0}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list}, cudf::lists_column_view{gather_map});
    auto const expected   = LCW<T>{{{4, 5}, {2, 3}},
                                   {{9, 10, 11}, {12, 13, 14}, {6, 7, 8}},
                                   {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}, {15, 16}}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    // clang-format on
  }
  // List<List<T>>, with out-of-bounds gather indices.
  {
    // clang-format off
    auto const list       = LCW<T>{{{2, 3}, {4, 5}},
                                   {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
                                   {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}};
    auto const gather_map = LCW<int>{{-1, 0}, {-2, -1, -4}, {-6, -4, -3, -2, -1, 0}};
    auto const results    =
      cudf::lists::segmented_gather(cudf::lists_column_view{list}, cudf::lists_column_view{gather_map}, NULLIFY);
    auto const expected   = LCW<T>{{{4, 5}, {2, 3}},
                                   {{{9, 10, 11}, {12, 13, 14}, LCW<T>{}}, null_at(2)},
                                   {{LCW<T>{}, {17, 18}, {17, 18}, {17, 18}, {17, 18}, {15, 16}}, null_at(0)}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    // clang-format on
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNestedNulls)
{
  using T = TypeParam;

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<List<T>>
  {
    // clang-format off
    auto const list = LCW<T>{{{{2, 3}, valids}, {4, 5}},
                             {{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, valids},
                             {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}},
                             {{{{25, 26}, valids}, {27, 28}, {{29, 30}, valids}, {31, 32}, {33, 34}}, valids}};
    auto const gather_map = LCW<int>{{0, 1}, {0, 2}, LCW<int>{}, {0, 1, 4}};
    auto const results  = cudf::lists::segmented_gather(cudf::lists_column_view{list}, cudf::lists_column_view{gather_map});
    auto const expected = LCW<T>{{{{2, 3}, valids}, {4, 5}},
                                 {{{6, 7, 8}, {12, 13, 14}}, no_nulls()},
                                 LCW<T>{},
                                 {{{{25, 26}, valids}, {27, 28}, {33, 34}}, valids}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    // clang-format on
  }

  // List<List<List<List<T>>>>
  {
    // clang-format off
    auto const list = LCW<T>{{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                              {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
                              {{LCW<T>{0}}},
                              {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
                                {{0, 1, 3}, {5}},
                                {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
                              {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids}}};
    auto const gather_map = LCW<int>{{1, 2, 4}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list}, cudf::lists_column_view{gather_map});
    auto const expected   = LCW<T>{{{{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
                                    {{LCW<T>{0}}},
                                    {{{{{10, 20}, valids}}, {LCW<T>{30}}, {{40, 50}, {60, 70, 80}}}, valids}}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    // clang-format on
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNestedWithEmpties)
{
  using T = TypeParam;

  auto const list = LCW<T>{{{2, 3}, LCW<T>{}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, {LCW<T>{}}};
  auto const gather_map = LCW<int>{LCW<int>{0}, LCW<int>{0}, LCW<int>{0}};
  auto results          = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                               cudf::lists_column_view{gather_map});
  auto const expected =
    LCW<T>{{{2, 3}}, {{6, 7, 8}}, {LCW<T>{}}};  // skip one null, gather one null.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TYPED_TEST(SegmentedGatherTest, GatherSliced)
{
  using T = TypeParam;
  {
    auto const a = LCW<T>{
      {{1, 1, 1}, {2, 2}, {3, 3}},
      {{4, 4, 4}, {5, 5}, {6, 6}},
      {{7, 7, 7}, {8, 8}, {9, 9}},
      {{10, 10, 10}, {11, 11}, {12, 12}},
      {{20, 20, 20, 20}, {25}},
      {{30, 30, 30, 30}, {40}},
      {{50, 50, 50, 50}, {6, 13}},
      {{70, 70, 70, 70}, {80}},
    };
    auto const split_a = cudf::split(a, {3});

    {
      auto const list       = LCW<int>{{1, 2}, {0, 2}, {0, 1}};
      auto const gather_map = cudf::lists_column_view{list};
      auto const result =
        cudf::lists::segmented_gather(cudf::lists_column_view{split_a[0]}, gather_map);
      auto const expected = LCW<T>{
        {{2, 2}, {3, 3}},
        {{4, 4, 4}, {6, 6}},
        {{7, 7, 7}, {8, 8}},
      };
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
    }

    {
      auto const list       = LCW<int>{{0, 1}, LCW<int>{}, LCW<int>{}, {0, 1}, LCW<int>{}};
      auto const gather_map = cudf::lists_column_view{list};
      auto const result =
        cudf::lists::segmented_gather(cudf::lists_column_view{split_a[1]}, gather_map);
      auto const expected =
        LCW<T>{{{10, 10, 10}, {11, 11}}, LCW<T>{}, LCW<T>{}, {{50, 50, 50, 50}, {6, 13}}, LCW<T>{}};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
    }
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

    auto sliced = cudf::slice(list, {0, 1, 2, 5, 5, 7});

    // gather from slice 0
    {
      LCW<int> map{{0, 1}};
      auto result = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[0]},
                                                  cudf::lists_column_view{map});
      LCW<T> expected{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}}};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
    }

    // gather from slice 1
    {
      LCW<int16_t> map{{0}, {1, 2, 0, 1}, {0, 1, 2}};
      auto result = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[1]},
                                                  cudf::lists_column_view{map});
      LCW<T> expected{
        {{LCW<T>{0}}},

        {{{0, 1, 3}, {5}},
         {{11, 12, 13, 14, 15}, {16, 17}, {0}},
         {{10}, {20, 30, 40, 50}, {60, 70, 80}},
         {{0, 1, 3}, {5}}},

        {{{{1, 6}, {60, 70, 80, 100}}, {{10, 11, 13}, {15}}, {{11, 12, 13, 14, 15}}}, valids},
      };
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
    }

    // gather from slice 2
    {
      LCW<int> map{{1, 0, 0, 1, 1, 0}, {1, 0, 0, 1, 1, 2}};
      auto result = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[2]},
                                                  cudf::lists_column_view{map});
      std::vector<bool> expected_valids = {false, true, true, false, false, true};

      LCW<T> expected{{{{LCW<T>{30}},
                        {{{10, 20}, valids}},
                        {{{10, 20}, valids}},
                        {LCW<T>{30}},
                        {LCW<T>{30}},
                        {{{10, 20}, valids}}},
                       expected_valids.begin()},
                      {{{LCW<T>{30}},
                        {{10, 20, 30}},
                        {{10, 20, 30}},
                        {LCW<T>{30}},
                        {LCW<T>{30}},
                        {{{20, 30}, valids}, {62, 72, 82}}},
                       expected_valids.begin()}};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
    }
  }
}

using SegmentedGatherTestString = SegmentedGatherTest<cudf::string_view>;
TEST_F(SegmentedGatherTestString, StringGather)
{
  using T = cudf::string_view;
  // List<T>
  {
    auto const list       = LCW<T>{{"a", "b", "c", "d"}, {"1", "22", "333", "4"}, {"x", "y", "z"}};
    auto const gather_map = LCW<int8_t>{{0, 1, 3, 2}, {1, 0, 3, 2}, LCW<int8_t>{}};
    auto const expected   = LCW<T>{{"a", "b", "d", "c"}, {"22", "1", "4", "333"}, LCW<T>{}};
    auto const result     = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                      cudf::lists_column_view{gather_map});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  // List<T>, with out-of-order gather indices.
  {
    auto const list       = LCW<T>{{"a", "b", "c", "d"}, {"1", "22", "333", "4"}, {"x", "y", "z"}};
    auto const gather_map = LCW<int8_t>{{0, 1, 3, 4}, {1, -5, 3, 2}, LCW<int8_t>{}};
    auto const expected   = LCW<T>{{{"a", "b", "d", "c"}, cudf::test::iterators::null_at(3)},
                                   {{"22", "1", "4", "333"}, cudf::test::iterators::null_at(1)},
                                   LCW<T>{}};
    auto result           = cudf::lists::segmented_gather(
      cudf::lists_column_view{list}, cudf::lists_column_view{gather_map}, NULLIFY);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

using SegmentedGatherTestFloat = SegmentedGatherTest<float>;
TEST_F(SegmentedGatherTestFloat, GatherMapSliced)
{
  using T = float;

  // List<T>
  {
    auto const list = LCW<T>{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}, {11, 12}, {13, 14, 15, 16}};
    auto const gather_map = LCW<int>{{3, 2, 1, 0}, {0}, {0, 1}, {0, 2, 1}, {0}, {1}};
    // gather_map.offset: 0, 4, 5, 7, 10, 11, 12
    auto const expected = LCW<T>{{4, 3, 2, 1}, {5}, {6, 7}, {8, 10, 9}, {11}, {14}};
    auto const results  = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

    auto const sliced  = cudf::split(list, {1, 4});
    auto const split_m = cudf::split(gather_map, {1, 4});
    auto const split_e = cudf::split(expected, {1, 4});

    auto result0 = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[0]},
                                                 cudf::lists_column_view{split_m[0]});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(split_e[0], result0->view());
    auto result1 = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[1]},
                                                 cudf::lists_column_view{split_m[1]});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(split_e[1], result1->view());
    auto result2 = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[2]},
                                                 cudf::lists_column_view{split_m[2]});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(split_e[2], result2->view());
  }

  // List<T>, with out-of-bounds gather indices.
  {
    auto const list = LCW<T>{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}, {11, 12}, {13, 14, 15, 16}};
    auto const gather_map = LCW<int>{{3, -5, 1, 0}, {0}, {0, 1}, {0, 2, 3}, {0}, {1}};
    // gather_map.offset: 0, 4, 5, 7, 10, 11, 12
    auto const expected =
      LCW<T>{{{4, 0, 2, 1}, null_at(1)}, {5}, {6, 7}, {{8, 10, 9}, null_at(2)}, {11}, {14}};
    auto results = cudf::lists::segmented_gather(
      cudf::lists_column_view{list}, cudf::lists_column_view{gather_map}, NULLIFY);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);

    auto const sliced  = cudf::split(list, {1, 4});
    auto const split_m = cudf::split(gather_map, {1, 4});
    auto const split_e = cudf::split(expected, {1, 4});

    auto const result0 = cudf::lists::segmented_gather(
      cudf::lists_column_view{sliced[0]}, cudf::lists_column_view{split_m[0]}, NULLIFY);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(split_e[0], result0->view());
    auto const result1 = cudf::lists::segmented_gather(
      cudf::lists_column_view{sliced[1]}, cudf::lists_column_view{split_m[1]}, NULLIFY);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(split_e[1], result1->view());
    auto const result2 = cudf::lists::segmented_gather(
      cudf::lists_column_view{sliced[2]}, cudf::lists_column_view{split_m[2]}, NULLIFY);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(split_e[2], result2->view());
  }
}

TEST_F(SegmentedGatherTestFloat, Fails)
{
  using T = float;
  // List<T>
  LCW<T> list{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}};
  LCW<int8_t> size_mismatch_map{{3, 2, 1, 0}, {0}, {0, 1}};
  cudf::test::fixed_width_column_wrapper<int> nonlist_map0{1, 2, 0, 1};
  cudf::test::strings_column_wrapper nonlist_map1{"1", "2", "0", "1"};
  LCW<cudf::string_view> nonlist_map2{{"1", "2", "0", "1"}};

  // Input must be a list of integer indices. It should fail for integers,
  // strings, or lists containing anything other than integers.
  EXPECT_THROW(cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                             cudf::lists_column_view{nonlist_map0}),
               cudf::logic_error);

  EXPECT_THROW(cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                             cudf::lists_column_view{nonlist_map1}),
               cudf::logic_error);

  EXPECT_THROW(cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                             cudf::lists_column_view{nonlist_map2}),
               cudf::logic_error);

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);
  LCW<int8_t> nulls_map{{{3, 2, 1, 0}, {0}, {0}, {0, 1}}, valids};

  // Nulls are not supported in the gather map.
  EXPECT_THROW(cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                             cudf::lists_column_view{nulls_map}),
               std::invalid_argument);

  // Gather map and list column sizes must be the same.
  EXPECT_THROW(cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                             cudf::lists_column_view{size_mismatch_map}),
               cudf::logic_error);
}
