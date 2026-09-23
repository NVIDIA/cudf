/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Nested `LCW{LCW...` constructions (including empty-list columns with explicit stream/mr)
// trip gcc14's -Wmaybe-uninitialized on column_view_base's copy constructor. Same diagnostic
// as lists/extract_tests.cpp; ignore for the whole file because the warning fires in headers.
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
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
class SegmentedGatherTest : public cudf::test::BaseFixtureWithHarness {};
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

  auto const stream = this->stream();
  auto const mr     = this->resources();

  // List<T>
  LCW<T> list{{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}}, stream, mr};

  {
    // Straight-line case.
    LCW<int> gather_map{{{3, 2, 1, 0}, {0}, {0, 1}, {0, 2, 1}}, stream, mr};
    LCW<T> expected{{{4, 3, 2, 1}, {5}, {6, 7}, {8, 10, 9}}, stream, mr};
    auto const results = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map},
                                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                                       stream,
                                                       mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view(), expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }

  {
    // Nullify out-of-bounds values.
    LCW<int> gather_map{{{3, 2, 4, 0}, {0}, {0, -3}, {0, 2, 1}}, stream, mr};
    LCW<T> expected{
      {{{4, 3, 2, 1}, null_at(2)}, {5}, {{6, 7}, null_at(1)}, {8, 10, 9}}, stream, mr};
    auto const results = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map},
                                                       NULLIFY,
                                                       stream,
                                                       mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view(), expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNothing)
{
  using T = TypeParam;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  // List<T>
  {
    LCW<T> list{{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}}, stream, mr};
    auto const gather_map = LCW<int>{{}, {}, {}, {}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map},
                                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                                       stream,
                                                       mr.get_output_mr());
    auto const expected   = LCW<T>{{}, {}, {}, {}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      *results, expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }
  // List<List<T>>
  {
    LCW<T> list{{{{1, 2, 3, 4}, {5}}, {{6, 7}}, {{}, {8, 9, 10}}}, stream, mr};
    auto const gather_map = LCW<int>{{}, {}, {}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map},
                                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                                       stream,
                                                       mr.get_output_mr());

    // hack to get column of empty list of list
    LCW<T> col{{{{1, 2, 3, 4}, {5}}, {}, {}, {}}, stream, mr};

    auto const expected = cudf::split(col, {1}, stream)[1];
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      *results, expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }
  // List<List<List<T>>>
  {
    LCW<T> list{{{{{1, 2, 3, 4}, {5}}}, {{{6, 7}, {8, 9, 10}}}}, stream, mr};
    auto const gather_map = LCW<int>{{}, {}};
    auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map},
                                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                                       stream,
                                                       mr.get_output_mr());
    // hack to get column of empty list of list of list
    LCW<T> col{{{{{1, 2, 3, 4}}}, {}, {}}, stream, mr};

    auto const expected = cudf::split(col, {1}, stream)[1];
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      *results, expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);

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
  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto const list       = LCW<int32_t>{};
  auto const gather_map = LCW<cudf::size_type>{};
  auto const expected   = LCW<int32_t>{};
  auto const results    = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                     cudf::lists_column_view{gather_map},
                                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                                     stream,
                                                     mr.get_output_mr());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *results, expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TYPED_TEST(SegmentedGatherTest, GatherNulls)
{
  using T = TypeParam;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<T>
  LCW<T> list{{{{1, 2, 3, 4}, valids}, {5}, {{6, 7}, valids}, {{8, 9, 10}, valids}}, stream, mr};

  {
    // Test gathering on lists that contain nulls.
    LCW<int> gather_map{{{0, 1}, {}, {1}, {2, 1, 0}}, stream, mr};
    auto const results = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map},
                                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                                       stream,
                                                       mr.get_output_mr());
    LCW<T> expected{{{{1, 2}, valids}, {}, {{7}, valids + 1}, {{10, 9, 8}, valids}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view(), expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }
  {
    // Test gathering on lists that contain nulls, with out-of-bounds indices.
    LCW<int> gather_map{{{10, -10}, {}, {1}, {2, -10, 0}}, stream, mr};
    auto const results = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map},
                                                       NULLIFY,
                                                       stream,
                                                       mr.get_output_mr());
    LCW<T> expected{
      {{{0, 0}, nulls_at({0, 1})}, {}, {{7}, valids + 1}, {{10, 0, 8}, null_at(1)}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view(), expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNested)
{
  using T = TypeParam;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  // List<List<T>>
  {
    // clang-format off
    LCW<T> list{{{{2, 3}, {4, 5}},
              {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
              {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {-17, -18}}}, stream, mr};
    LCW<int> gather_map{{{0, -2, -2}, {1}, {1, 0, -1, -5}}, stream, mr};
    auto const results =
      cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                    cudf::lists_column_view{gather_map},
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    stream,
                                    mr.get_output_mr());
    LCW<T> expected{{{{2, 3}, {2, 3}, {2, 3}},
                  {{9, 10, 11}},
                  {{17, 18}, {15, 16}, {-17, -18}, {15, 16}}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(),
                                   expected,
                                   cudf::test::debug_output_level::FIRST_ERROR,
                                   stream,
                                   mr);
    // clang-format on
  }

  // List<List<T>>, with out-of-bounds gather indices.
  {
    // clang-format off
    LCW<T> list{{{{2, 3}, {4, 5}},
              {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
              {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {-17, -18}}}, stream, mr};
    LCW<int> gather_map{{{0, 2, -2}, {1}, {1, 0, -1, -6}}, stream, mr};
    auto const results =
      cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                    cudf::lists_column_view{gather_map},
                                    NULLIFY,
                                    stream,
                                    mr.get_output_mr());
    LCW<T> expected{{{{{2, 3}, {}, {2, 3}}, null_at(1)},
                  {{9, 10, 11}},
                  {{{17, 18}, {15, 16}, {-17, -18}, {}}, null_at(3)}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(),
                                   expected,
                                   cudf::test::debug_output_level::FIRST_ERROR,
                                   stream,
                                   mr);
    // clang-format on
  }

  // List<List<List<T>>>
  {
    // clang-format off
    LCW<T> list{{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
              {{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
              {{{0}}},
              {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
               {{0, 1, 3}, {5}},
               {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
              {{{10, 20}}, {{30}}, {{40, 50}, {60, 70, 80}}}}, stream, mr};
    LCW<int> gather_map{{{1}, {}, {0}, {1}, {0, -1, 1}}, stream, mr};
    auto const results =
      cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                    cudf::lists_column_view{gather_map},
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    stream,
                                    mr.get_output_mr());
    LCW<T> expected{{{{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                  {},
                  {{{0}}},
                  {{{0, 1, 3}, {5}}},
                  {{{10, 20}}, {{40, 50}, {60, 70, 80}}, {{30}}}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(),
                                   expected,
                                   cudf::test::debug_output_level::FIRST_ERROR,
                                   stream,
                                   mr);
    // clang-format on
  }

  // List<List<List<T>>>, with out-of-bounds gather indices.
  {
    LCW<T> list{{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                 {{{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}},
                 {{{0}}},
                 {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
                  {{0, 1, 3}, {5}},
                  {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
                 {{{10, 20}}, {{30}}, {{40, 50}, {60, 70, 80}}}},
                stream,
                mr};
    LCW<int> gather_map{{{1}, {}, {0}, {1}, {0, -1, 3, -4}}, stream, mr};
    auto const results = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                       cudf::lists_column_view{gather_map},
                                                       NULLIFY,
                                                       stream,
                                                       mr.get_output_mr());
    LCW<T> expected{{{{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
                     {},
                     {{{0}}},
                     {{{0, 1, 3}, {5}}},
                     {{{{10, 20}}, {{40, 50}, {60, 70, 80}}, {}, {}}, nulls_at({2, 3})}},
                    stream,
                    mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view(), expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }
}

TYPED_TEST(SegmentedGatherTest, GatherOutOfOrder)
{
  using T = TypeParam;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  // List<List<T>>
  {
    // clang-format off
    LCW<T> list{{{{2, 3}, {4, 5}},
              {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
              {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}}, stream, mr};
    LCW<int> gather_map{{{1, 0}, {1, 2, 0}, {4, 3, 2, 1, 0}}, stream, mr};
    auto const results =
      cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                    cudf::lists_column_view{gather_map},
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    stream,
                                    mr.get_output_mr());
    LCW<T> expected{{{{4, 5}, {2, 3}},
                  {{9, 10, 11}, {12, 13, 14}, {6, 7, 8}},
                  {{17, 18}, {17, 18}, {17, 18}, {17, 18}, {15, 16}}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(),
                                   expected,
                                   cudf::test::debug_output_level::FIRST_ERROR,
                                   stream,
                                   mr);
    // clang-format on
  }

  // List<List<T>>, with out-of-bounds gather indices.
  {
    // clang-format off
    LCW<T> list{{{{2, 3}, {4, 5}},
              {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
              {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}}, stream, mr};
    LCW<int> gather_map{{{1, 0}, {3, -1, -4}, {5, 4, 3, 2, 1, 0}}, stream, mr};
    auto const results =
      cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                    cudf::lists_column_view{gather_map},
                                    NULLIFY,
                                    stream,
                                    mr.get_output_mr());
    LCW<T> expected{{{{4, 5}, {2, 3}},
                  {{{}, {12, 13, 14}, {}}, nulls_at({0, 2})},
                  {{{}, {17, 18}, {17, 18}, {17, 18}, {17, 18}, {15, 16}}, null_at(0)}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(),
                                   expected,
                                   cudf::test::debug_output_level::FIRST_ERROR,
                                   stream,
                                   mr);
    // clang-format on
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNegatives)
{
  using T = TypeParam;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  // List<List<T>>
  {
    // clang-format off
    LCW<T> list{{{{2, 3}, {4, 5}},
              {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
              {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}}, stream, mr};
    LCW<int> gather_map{{{-1, 0}, {-2, -1, 0}, {-5, -4, -3, -2, -1, 0}}, stream, mr};
    auto const results =
      cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                    cudf::lists_column_view{gather_map},
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    stream,
                                    mr.get_output_mr());
    LCW<T> expected{{{{4, 5}, {2, 3}},
                  {{9, 10, 11}, {12, 13, 14}, {6, 7, 8}},
                  {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}, {15, 16}}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(),
                                   expected,
                                   cudf::test::debug_output_level::FIRST_ERROR,
                                   stream,
                                   mr);
    // clang-format on
  }
  // List<List<T>>, with out-of-bounds gather indices.
  {
    // clang-format off
    LCW<T> list{{{{2, 3}, {4, 5}},
              {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}},
              {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}}}, stream, mr};
    LCW<int> gather_map{{{-1, 0}, {-2, -1, -4}, {-6, -4, -3, -2, -1, 0}}, stream, mr};
    auto const results =
      cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                    cudf::lists_column_view{gather_map},
                                    NULLIFY,
                                    stream,
                                    mr.get_output_mr());
    LCW<T> expected{{{{4, 5}, {2, 3}},
                  {{{9, 10, 11}, {12, 13, 14}, {}}, null_at(2)},
                  {{{}, {17, 18}, {17, 18}, {17, 18}, {17, 18}, {15, 16}}, null_at(0)}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(),
                                   expected,
                                   cudf::test::debug_output_level::FIRST_ERROR,
                                   stream,
                                   mr);
    // clang-format on
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNestedNulls)
{
  using T = TypeParam;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<List<T>>
  {
    // clang-format off
    LCW<T> list{{{{{2, 3}, valids}, {4, 5}},
              {{{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, valids},
              {{15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}},
              {{{{25, 26}, valids}, {27, 28}, {{29, 30}, valids}, {31, 32}, {33, 34}}, valids}}, stream, mr};
    LCW<int> gather_map{{{0, 1}, {0, 2}, {}, {0, 1, 4}}, stream, mr};
    auto const results =
      cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                    cudf::lists_column_view{gather_map},
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    stream,
                                    mr.get_output_mr());
    LCW<T> expected{{{{{2, 3}, valids}, {4, 5}},
                  {{{6, 7, 8}, {12, 13, 14}}, no_nulls()},
                  {},
                  {{{{25, 26}, valids}, {27, 28}, {33, 34}}, valids}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(),
                                   expected,
                                   cudf::test::debug_output_level::FIRST_ERROR,
                                   stream,
                                   mr);
    // clang-format on
  }

  // List<List<List<List<T>>>>
  {
    // clang-format off
    LCW<T> list{{{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},
               {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
               {{{0}}},
               {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
                 {{0, 1, 3}, {5}},
                 {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
               {{{{{10, 20}, valids}}, {{30}}, {{40, 50}, {60, 70, 80}}}, valids}}}, stream, mr};
    LCW<int> gather_map{{{1, 2, 4}}, stream, mr};
    auto const results =
      cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                    cudf::lists_column_view{gather_map},
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    stream,
                                    mr.get_output_mr());
    LCW<T> expected{{{{{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}}},
                   {{{0}}},
                   {{{{{10, 20}, valids}}, {{30}}, {{40, 50}, {60, 70, 80}}}, valids}}}, stream, mr};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(),
                                   expected,
                                   cudf::test::debug_output_level::FIRST_ERROR,
                                   stream,
                                   mr);
    // clang-format on
  }
}

TYPED_TEST(SegmentedGatherTest, GatherNestedWithEmpties)
{
  using T = TypeParam;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  LCW<T> list{
    LCW<T>::nested({{{2, 3}, {}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}, {{}}}), stream, mr};
  // Per-row singleton lists: {{0},{0},{0}} flattens to one list of three.
  LCW<int> gather_map({{0}, {0}, {0}}, stream, mr);
  auto results = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                               cudf::lists_column_view{gather_map},
                                               cudf::out_of_bounds_policy::DONT_CHECK,
                                               stream,
                                               mr.get_output_mr());
  LCW<T> expected{LCW<T>::nested({{{2, 3}}, {{6, 7, 8}}, {{}}}),  // skip one null, gather one null.
                  stream,
                  mr};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    results->view(), expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TYPED_TEST(SegmentedGatherTest, GatherSliced)
{
  using T = TypeParam;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  {
    LCW<T> col{{
                 {{1, 1, 1}, {2, 2}, {3, 3}},
                 {{4, 4, 4}, {5, 5}, {6, 6}},
                 {{7, 7, 7}, {8, 8}, {9, 9}},
                 {{10, 10, 10}, {11, 11}, {12, 12}},
                 {{20, 20, 20, 20}, {25}},
                 {{30, 30, 30, 30}, {40}},
                 {{50, 50, 50, 50}, {6, 13}},
                 {{70, 70, 70, 70}, {80}},
               },
               stream,
               mr};

    auto const split_a = cudf::split(col, {3}, stream);

    {
      LCW<int> map_col{{{1, 2}, {0, 2}, {0, 1}}, stream, mr};

      auto const gather_map = cudf::lists_column_view{map_col};
      auto const result     = cudf::lists::segmented_gather(cudf::lists_column_view{split_a[0]},
                                                        gather_map,
                                                        cudf::out_of_bounds_policy::DONT_CHECK,
                                                        stream,
                                                        mr.get_output_mr());
      LCW<T> expected{{
                        {{2, 2}, {3, 3}},
                        {{4, 4, 4}, {6, 6}},
                        {{7, 7, 7}, {8, 8}},
                      },
                      stream,
                      mr};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(
        expected, result->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
    }

    {
      LCW<int> map_col{{{0, 1}, {}, {}, {0, 1}, {}}, stream, mr};

      auto const gather_map = cudf::lists_column_view{map_col};
      auto const result     = cudf::lists::segmented_gather(cudf::lists_column_view{split_a[1]},
                                                        gather_map,
                                                        cudf::out_of_bounds_policy::DONT_CHECK,
                                                        stream,
                                                        mr.get_output_mr());
      LCW<T> expected{
        {{{10, 10, 10}, {11, 11}}, {}, {}, {{50, 50, 50, 50}, {6, 13}}, {}}, stream, mr};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(
        expected, result->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
    }
  }

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);

  // List<List<List<T>>>
  {
    LCW<T> col{
      {// slice 0
       {{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}},

       {{{15, 16}, {{27, 28}, valids}, {{37, 38}, valids}, {47, 48}, {57, 58}},
        {{11, 12}, {{42, 43, 44}, valids}, {{77, 78}, valids}}},

       // slice 1
       {{{0}}},
       {{{10}, {20, 30, 40, 50}, {60, 70, 80}},
        {{0, 1, 3}, {5}},
        {{11, 12, 13, 14, 15}, {16, 17}, {0}}},
       {{{{1, 6}, {60, 70, 80, 100}}, {{10, 11, 13}, {15}}, {{11, 12, 13, 14, 15}}}, valids},

       // slice 2
       {{{{{10, 20}, valids}}, {{30}}, {{40, 50}, {60, 70, 80}}}, valids},
       {{{{10, 20, 30}}, {{30}}, {{{20, 30}, valids}, {62, 72, 82}}}, valids}},
      stream,
      mr};

    auto sliced = cudf::slice(col, {0, 1, 2, 5, 5, 7}, stream);

    // gather from slice 0
    {
      LCW<int> map{{{0, 1}}, stream, mr};
      auto result = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[0]},
                                                  cudf::lists_column_view{map},
                                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                                  stream,
                                                  mr.get_output_mr());
      LCW<T> expected{{{{{2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10, 11}, {12, 13, 14}}}}, stream, mr};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected,
                                          result->view(),
                                          cudf::test::debug_output_level::FIRST_ERROR,
                                          cudf::test::default_ulp,
                                          stream,
                                          mr);
    }

    // gather from slice 1
    {
      LCW<int16_t> map{{{0}, {1, 2, 0, 1}, {0, 1, 2}}, stream, mr};
      auto result = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[1]},
                                                  cudf::lists_column_view{map},
                                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                                  stream,
                                                  mr.get_output_mr());
      LCW<T> expected{
        {
          {{{0}}},

          {{{0, 1, 3}, {5}},
           {{11, 12, 13, 14, 15}, {16, 17}, {0}},
           {{10}, {20, 30, 40, 50}, {60, 70, 80}},
           {{0, 1, 3}, {5}}},

          {{{{1, 6}, {60, 70, 80, 100}}, {{10, 11, 13}, {15}}, {{11, 12, 13, 14, 15}}}, valids},
        },
        stream,
        mr};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected,
                                          result->view(),
                                          cudf::test::debug_output_level::FIRST_ERROR,
                                          cudf::test::default_ulp,
                                          stream,
                                          mr);
    }

    // gather from slice 2
    {
      LCW<int> map{{{1, 0, 0, 1, 1, 0}, {1, 0, 0, 1, 1, 2}}, stream, mr};
      auto result = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[2]},
                                                  cudf::lists_column_view{map},
                                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                                  stream,
                                                  mr.get_output_mr());
      std::vector<bool> expected_valids = {false, true, true, false, false, true};

      LCW<T> expected{{{{{{30}},
                         {{{10, 20}, valids}},
                         {{{10, 20}, valids}},
                         {{30}},
                         {{30}},
                         {{{10, 20}, valids}}},
                        expected_valids.begin()},
                       {{{{30}},
                         {{10, 20, 30}},
                         {{10, 20, 30}},
                         {{30}},
                         {{30}},
                         {{{20, 30}, valids}, {62, 72, 82}}},
                        expected_valids.begin()}},
                      stream,
                      mr};
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected,
                                          result->view(),
                                          cudf::test::debug_output_level::FIRST_ERROR,
                                          cudf::test::default_ulp,
                                          stream,
                                          mr);
    }
  }
}

using SegmentedGatherTestString = SegmentedGatherTest<cudf::string_view>;
TEST_F(SegmentedGatherTestString, StringGather)
{
  using T = cudf::string_view;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  // List<T>
  {
    LCW<T> list{{{"a", "b", "c", "d"}, {"1", "22", "333", "4"}, {"x", "y", "z"}}, stream, mr};
    cudf::test::lists_column_wrapper<int8_t> gather_map{
      {{0, 1, 3, 2}, {1, 0, 3, 2}, {}}, stream, mr};
    LCW<T> expected{{{"a", "b", "d", "c"}, {"22", "1", "4", "333"}, {}}, stream, mr};
    auto const result = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                      cudf::lists_column_view{gather_map},
                                                      cudf::out_of_bounds_policy::DONT_CHECK,
                                                      stream,
                                                      mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      expected, result->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }

  // List<T>, with out-of-order gather indices.
  {
    LCW<T> list{{{"a", "b", "c", "d"}, {"1", "22", "333", "4"}, {"x", "y", "z"}}, stream, mr};
    cudf::test::lists_column_wrapper<int8_t> gather_map{
      {{0, 1, 3, 4}, {1, -5, 3, 2}, {}}, stream, mr};
    LCW<T> expected{{{{"a", "b", "d", "c"}, cudf::test::iterators::null_at(3)},
                     {{"22", "1", "4", "333"}, cudf::test::iterators::null_at(1)},
                     {}},
                    stream,
                    mr};
    auto result = cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                                cudf::lists_column_view{gather_map},
                                                NULLIFY,
                                                stream,
                                                mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      expected, result->view(), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }
}

using SegmentedGatherTestFloat = SegmentedGatherTest<float>;
TEST_F(SegmentedGatherTestFloat, GatherMapSliced)
{
  using T = float;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  // List<T>
  {
    LCW<T> list_col{
      {{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}, {11, 12}, {13, 14, 15, 16}}, stream, mr};
    LCW<int> gather_map_col{{{3, 2, 1, 0}, {0}, {0, 1}, {0, 2, 1}, {0}, {1}}, stream, mr};
    // gather_map.offset: 0, 4, 5, 7, 10, 11, 12
    LCW<T> expected_col{{{4, 3, 2, 1}, {5}, {6, 7}, {8, 10, 9}, {11}, {14}}, stream, mr};

    auto const results = cudf::lists::segmented_gather(cudf::lists_column_view{list_col},
                                                       cudf::lists_column_view{gather_map_col},
                                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                                       stream,
                                                       mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view(), expected_col, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);

    auto const sliced  = cudf::split(list_col, {1, 4}, stream);
    auto const split_m = cudf::split(gather_map_col, {1, 4}, stream);
    auto const split_e = cudf::split(expected_col, {1, 4}, stream);

    auto result0 = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[0]},
                                                 cudf::lists_column_view{split_m[0]},
                                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                                 stream,
                                                 mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(split_e[0],
                                        result0->view(),
                                        cudf::test::debug_output_level::FIRST_ERROR,
                                        cudf::test::default_ulp,
                                        stream,
                                        mr);
    auto result1 = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[1]},
                                                 cudf::lists_column_view{split_m[1]},
                                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                                 stream,
                                                 mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(split_e[1],
                                        result1->view(),
                                        cudf::test::debug_output_level::FIRST_ERROR,
                                        cudf::test::default_ulp,
                                        stream,
                                        mr);
    auto result2 = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[2]},
                                                 cudf::lists_column_view{split_m[2]},
                                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                                 stream,
                                                 mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(split_e[2],
                                        result2->view(),
                                        cudf::test::debug_output_level::FIRST_ERROR,
                                        cudf::test::default_ulp,
                                        stream,
                                        mr);
  }

  // List<T>, with out-of-bounds gather indices.
  {
    LCW<T> list_col{
      {{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}, {11, 12}, {13, 14, 15, 16}}, stream, mr};
    LCW<int> gather_map_col{{{3, -5, 1, 0}, {0}, {0, 1}, {0, 2, 3}, {0}, {1}}, stream, mr};
    // gather_map.offset: 0, 4, 5, 7, 10, 11, 12
    LCW<T> expected_col{
      {{{4, 0, 2, 1}, null_at(1)}, {5}, {6, 7}, {{8, 10, 9}, null_at(2)}, {11}, {14}}, stream, mr};

    auto results = cudf::lists::segmented_gather(cudf::lists_column_view{list_col},
                                                 cudf::lists_column_view{gather_map_col},
                                                 NULLIFY,
                                                 stream,
                                                 mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results->view(), expected_col, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);

    auto const sliced  = cudf::split(list_col, {1, 4}, stream);
    auto const split_m = cudf::split(gather_map_col, {1, 4}, stream);
    auto const split_e = cudf::split(expected_col, {1, 4}, stream);

    auto const result0 = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[0]},
                                                       cudf::lists_column_view{split_m[0]},
                                                       NULLIFY,
                                                       stream,
                                                       mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(split_e[0],
                                        result0->view(),
                                        cudf::test::debug_output_level::FIRST_ERROR,
                                        cudf::test::default_ulp,
                                        stream,
                                        mr);
    auto const result1 = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[1]},
                                                       cudf::lists_column_view{split_m[1]},
                                                       NULLIFY,
                                                       stream,
                                                       mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(split_e[1],
                                        result1->view(),
                                        cudf::test::debug_output_level::FIRST_ERROR,
                                        cudf::test::default_ulp,
                                        stream,
                                        mr);
    auto const result2 = cudf::lists::segmented_gather(cudf::lists_column_view{sliced[2]},
                                                       cudf::lists_column_view{split_m[2]},
                                                       NULLIFY,
                                                       stream,
                                                       mr.get_output_mr());
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(split_e[2],
                                        result2->view(),
                                        cudf::test::debug_output_level::FIRST_ERROR,
                                        cudf::test::default_ulp,
                                        stream,
                                        mr);
  }
}

TEST_F(SegmentedGatherTestFloat, Fails)
{
  using T = float;

  auto const stream = this->stream();
  auto const mr     = this->resources();

  // List<T>
  LCW<T> list{{{1, 2, 3, 4}, {5}, {6, 7}, {8, 9, 10}}, stream, mr};
  cudf::test::lists_column_wrapper<int8_t> size_mismatch_map{
    {{3, 2, 1, 0}, {0}, {0, 1}}, stream, mr};
  cudf::test::fixed_width_column_wrapper<int> nonlist_map0{{1, 2, 0, 1}, stream, mr};
  cudf::test::strings_column_wrapper nonlist_map1{{"1", "2", "0", "1"}, stream, mr};
  LCW<cudf::string_view> nonlist_map2{{{"1", "2", "0", "1"}}, stream, mr};

  // Input must be a list of integer indices. It should fail for integers,
  // strings, or lists containing anything other than integers.
  EXPECT_THROW(cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                             cudf::lists_column_view{nonlist_map0},
                                             cudf::out_of_bounds_policy::DONT_CHECK,
                                             stream,
                                             mr.get_output_mr()),
               cudf::logic_error);

  EXPECT_THROW(cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                             cudf::lists_column_view{nonlist_map1},
                                             cudf::out_of_bounds_policy::DONT_CHECK,
                                             stream,
                                             mr.get_output_mr()),
               cudf::logic_error);

  EXPECT_THROW(cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                             cudf::lists_column_view{nonlist_map2},
                                             cudf::out_of_bounds_policy::DONT_CHECK,
                                             stream,
                                             mr.get_output_mr()),
               cudf::logic_error);

  auto valids = cudf::test::iterators::valids_at_multiples_of(2);
  cudf::test::lists_column_wrapper<int8_t> nulls_map{
    {{{3, 2, 1, 0}, {0}, {0}, {0, 1}}, valids}, stream, mr};

  // Nulls are not supported in the gather map.
  EXPECT_THROW(cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                             cudf::lists_column_view{nulls_map},
                                             cudf::out_of_bounds_policy::DONT_CHECK,
                                             stream,
                                             mr.get_output_mr()),
               std::invalid_argument);

  // Gather map and list column sizes must be the same.
  EXPECT_THROW(cudf::lists::segmented_gather(cudf::lists_column_view{list},
                                             cudf::lists_column_view{size_mismatch_map},
                                             cudf::out_of_bounds_policy::DONT_CHECK,
                                             stream,
                                             mr.get_output_mr()),
               cudf::logic_error);
}

#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
