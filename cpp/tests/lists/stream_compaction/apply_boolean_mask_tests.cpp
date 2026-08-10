/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/stream_compaction.hpp>

namespace cudf::test {

using namespace iterators;
using cudf::lists_column_view;
using cudf::lists::apply_boolean_mask;

template <typename T>
using lists    = lists_column_wrapper<T, int32_t>;
using filter_t = lists_column_wrapper<bool, int32_t>;

template <typename T>
using fwcw    = fixed_width_column_wrapper<T, int32_t>;
using offsets = fwcw<int32_t>;
using strings = strings_column_wrapper;

auto constexpr X = int32_t{0};  // Placeholder for NULL.

struct ApplyBooleanMaskTest : public BaseFixture {};

template <typename T>
struct ApplyBooleanMaskTypedTest : ApplyBooleanMaskTest {};

TYPED_TEST_SUITE(ApplyBooleanMaskTypedTest, cudf::test::NumericTypes);

TYPED_TEST(ApplyBooleanMaskTypedTest, StraightLine)
{
  using T    = TypeParam;
  auto input = lists<T>{{0, 1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {0, 1}, {2, 3, 4, 5}, {6, 7}}.release();
  auto filter = filter_t{{1, 0, 1, 0}, {1, 0}, {1, 0, 1, 0}, {1, 0}, {1, 0, 1, 0}, {1, 0}};

  {
    // Unsliced.
    auto filtered = apply_boolean_mask(lists_column_view{*input}, lists_column_view{filter});
    auto expected = lists<T>{{0, 2}, {4}, {6, 8}, {0}, {2, 4}, {6}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
  {
    // Sliced input: Remove the first row.
    auto sliced = cudf::slice(*input, {1, input->size()}).front();
    //           == lists_t {{4, 5}, {6, 7, 8, 9}, {0, 1}, {2, 3, 4, 5}, {6, 7}};
    auto filter   = filter_t{{0, 1}, {0, 1, 0, 1}, {1, 1}, {0, 1, 0, 1}, {0, 0}};
    auto filtered = apply_boolean_mask(lists_column_view{sliced}, lists_column_view{filter});
    auto expected = lists<T>{{5}, {7, 9}, {0, 1}, {3, 5}, {}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
}

TYPED_TEST(ApplyBooleanMaskTypedTest, NullElementsInTheListRows)
{
  using T = TypeParam;
  auto input =
    lists<T>{
      {0, 1, 2, 3},
      lists<T>{{X, 5}, null_at(0)},
      {6, 7, 8, 9},
      {0, 1},
      lists<T>{{X, 3, 4, X}, nulls_at({0, 3})},
      lists<T>{{X, X}, nulls_at({0, 1})},
    }
      .release();
  auto filter = filter_t{{1, 0, 1, 0}, {1, 0}, {1, 0, 1, 0}, {1, 0}, {1, 0, 1, 0}, {1, 0}};

  {
    // Unsliced.
    auto filtered = apply_boolean_mask(lists_column_view{*input}, lists_column_view{filter});
    auto expected = lists<T>{{0, 2},
                             lists<T>{{X}, null_at(0)},
                             {6, 8},
                             {0},
                             lists<T>{{X, 4}, null_at(0)},
                             lists<T>{{X}, null_at(0)}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
  {
    // Sliced input: Remove the first row.
    auto sliced = cudf::slice(*input, {1, input->size()}).front();
    //           == lists_t {{X, 5}, {6, 7, 8, 9}, {0, 1}, {X, 3, 4, X}, {X, X}};
    auto filter   = filter_t{{0, 1}, {0, 1, 0, 1}, {1, 1}, {0, 1, 0, 1}, {0, 0}};
    auto filtered = apply_boolean_mask(lists_column_view{sliced}, lists_column_view{filter});
    auto expected = lists<T>{{5}, {7, 9}, {0, 1}, lists<T>{{3, X}, null_at(1)}, {}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
}

TYPED_TEST(ApplyBooleanMaskTypedTest, NullListRowsInTheInputColumn)
{
  using T = TypeParam;
  auto input =
    lists<T>{{{0, 1, 2, 3}, {}, {6, 7, 8, 9}, {}, {2, 3, 4, 5}, {6, 7}}, nulls_at({1, 3})}
      .release();
  auto filter = filter_t{{1, 0, 1, 0}, {}, {1, 0, 1, 0}, {}, {1, 0, 1, 0}, {1, 0}};

  {
    // Unsliced.
    auto filtered = apply_boolean_mask(lists_column_view{*input}, lists_column_view{filter});
    auto expected = lists<T>{{{0, 2}, {}, {6, 8}, {}, {2, 4}, {6}}, nulls_at({1, 3})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
  {
    // Sliced input: Remove the first row.
    auto sliced = cudf::slice(*input, {1, input->size()}).front();
    //           == lists_t{{{}, {6, 7, 8, 9}, {}, {2, 3, 4, 5}, {6, 7}}, nulls_at({0,2})};
    auto filter   = filter_t{{}, {0, 1, 0, 1}, {}, {0, 1, 0, 1}, {0, 0}};
    auto filtered = apply_boolean_mask(lists_column_view{sliced}, lists_column_view{filter});
    auto expected = lists<T>{{{}, {7, 9}, {}, {3, 5}, {}}, nulls_at({0, 2})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
  {
    // Sliced input: Remove the first two rows.
    auto sliced = cudf::slice(*input, {2, input->size()}).front();
    //           == lists_t{{{6, 7, 8, 9}, {}, {2, 3, 4, 5}, {6, 7}}, null_at(1)};
    auto filter   = filter_t{{0, 1, 0, 1}, {}, {0, 1, 0, 1}, {0, 0}};
    auto filtered = apply_boolean_mask(lists_column_view{sliced}, lists_column_view{filter});
    auto expected = lists<T>{{{7, 9}, {}, {3, 5}, {}}, null_at(1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
}

TYPED_TEST(ApplyBooleanMaskTypedTest, StructInput)
{
  using T    = TypeParam;
  using fwcw = fwcw<T>;

  auto constexpr num_input_rows = 7;
  auto const input              = [] {
    auto child_num               = fwcw{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto child_str               = strings{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    auto const null_mask_begin   = null_at(5);
    auto const null_mask_end     = null_mask_begin + num_input_rows;
    auto [null_mask, null_count] = detail::make_null_mask(null_mask_begin, null_mask_end);
    return cudf::make_lists_column(num_input_rows,
                                   offsets{0, 2, 3, 6, 6, 8, 8, 10}.release(),
                                   structs_column_wrapper{{child_num, child_str}}.release(),
                                   null_count,
                                   std::move(null_mask));
  }();
  {
    // Unsliced.
    // The input should now look as follows: (String child dropped for brevity.)
    // Input:                     {[0, 1], [2], [3, 4, 5], [], [6, 7], [], [8, 9]}
    auto const filter   = filter_t{{1, 1}, {0}, {0, 1, 0}, {}, {1, 0}, {}, {0, 1}};
    auto const result   = apply_boolean_mask(lists_column_view{*input}, lists_column_view{filter});
    auto const expected = [] {
      auto child_num               = fwcw{0, 1, 4, 6, 9};
      auto child_str               = strings{"0", "1", "4", "6", "9"};
      auto const null_mask_begin   = null_at(5);
      auto const null_mask_end     = null_mask_begin + num_input_rows;
      auto [null_mask, null_count] = detail::make_null_mask(null_mask_begin, null_mask_end);
      return cudf::make_lists_column(num_input_rows,
                                     offsets{0, 2, 2, 3, 3, 4, 4, 5}.release(),
                                     structs_column_wrapper{{child_num, child_str}}.release(),
                                     null_count,
                                     std::move(null_mask));
    }();
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
  }
  {
    // Sliced. Remove the first row.
    auto const sliced_input = cudf::slice(*input, {1, input->size()}).front();
    // The input should now look as follows: (String child dropped for brevity.)
    // Input:                   {[2], [3, 4, 5], [], [6, 7], [], [8, 9]}
    auto const filter = filter_t{{0}, {0, 1, 0}, {}, {1, 0}, {}, {0, 1}};
    auto const result =
      apply_boolean_mask(lists_column_view{sliced_input}, lists_column_view{filter});
    auto const expected = [] {
      auto child_num               = fwcw{4, 6, 9};
      auto child_str               = strings{"4", "6", "9"};
      auto const null_mask_begin   = null_at(4);
      auto const null_mask_end     = null_mask_begin + num_input_rows;
      auto [null_mask, null_count] = detail::make_null_mask(null_mask_begin, null_mask_end);
      return cudf::make_lists_column(num_input_rows - 1,
                                     offsets{0, 0, 1, 1, 2, 2, 3}.release(),
                                     structs_column_wrapper{{child_num, child_str}}.release(),
                                     null_count,
                                     std::move(null_mask));
    }();
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
  }
}

TYPED_TEST(ApplyBooleanMaskTypedTest, NullsInBooleanMask)
{
  using T    = TypeParam;
  auto input = lists<T>{{10, 20, 30}, {40, 50}, {60, 70, 80, 90}};
  // mask child (flattened):      {T, null, T,  F, null,  null, F, T, F}
  // mask offsets:                {0,           3,        5,             9}
  // kept (valid AND true):       {10, 30},     {},       {80}
  auto mask_child = fwcw<bool>{{1, X, 1, 0, X, X, 0, 1, 0}, nulls_at({1, 4, 5})};
  auto mask =
    cudf::make_lists_column(3, offsets{0, 3, 5, 9}.release(), mask_child.release(), 0, {});
  auto filtered = apply_boolean_mask(lists_column_view{input}, lists_column_view{*mask});
  auto expected = lists<T>{{10, 30}, lists<T>{}, {80}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
}

TEST_F(ApplyBooleanMaskTest, Trivial)
{
  auto const input  = lists<int32_t>{};
  auto const filter = filter_t{};
  auto const result = apply_boolean_mask(lists_column_view{input}, lists_column_view{filter});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, lists<int32_t>{});
}

TEST_F(ApplyBooleanMaskTest, Failure)
{
  {
    // Invalid mask type.
    auto const input  = lists<int32_t>{{1, 2, 3}, {4, 5, 6}};
    auto const filter = lists<int32_t>{{0, 0, 0}};
    EXPECT_THROW(apply_boolean_mask(lists_column_view{input}, lists_column_view{filter}),
                 cudf::logic_error);
  }
  {
    // Mismatched number of rows.
    auto const input  = lists<int32_t>{{1, 2, 3}, {4, 5, 6}};
    auto const filter = filter_t{{0, 0, 0}};
    EXPECT_THROW(apply_boolean_mask(lists_column_view{input}, lists_column_view{filter}),
                 cudf::logic_error);
  }
}
struct ApplyDeletionMaskTest : public BaseFixture {};

template <typename T>
struct ApplyDeletionMaskTypedTest : ApplyDeletionMaskTest {};

TYPED_TEST_SUITE(ApplyDeletionMaskTypedTest, cudf::test::NumericTypes);

using cudf::lists::apply_deletion_mask;

TYPED_TEST(ApplyDeletionMaskTypedTest, StraightLine)
{
  using T    = TypeParam;
  auto input = lists<T>{{0, 1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {0, 1}, {2, 3, 4, 5}, {6, 7}}.release();
  auto filter = filter_t{{1, 0, 1, 0}, {1, 0}, {1, 0, 1, 0}, {1, 0}, {1, 0, 1, 0}, {1, 0}};

  {
    auto filtered = apply_deletion_mask(lists_column_view{*input}, lists_column_view{filter});
    auto expected = lists<T>{{1, 3}, {5}, {7, 9}, {1}, {3, 5}, {7}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
}

TYPED_TEST(ApplyDeletionMaskTypedTest, AllTrue)
{
  using T    = TypeParam;
  auto input = lists<T>{{0, 1, 2}, {3, 4}};
  auto mask  = filter_t{{1, 1, 1}, {1, 1}};

  auto filtered = apply_deletion_mask(lists_column_view{input}, lists_column_view{mask});
  auto expected = lists<T>{lists<T>{}, lists<T>{}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
}

TYPED_TEST(ApplyDeletionMaskTypedTest, AllFalse)
{
  using T     = TypeParam;
  auto input  = lists<T>{{0, 1, 2}, {3, 4}};
  auto filter = filter_t{{0, 0, 0}, {0, 0}};

  auto filtered = apply_deletion_mask(lists_column_view{input}, lists_column_view{filter});
  auto expected = lists<T>{{0, 1, 2}, {3, 4}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
}

TYPED_TEST(ApplyDeletionMaskTypedTest, NullElementsInTheListRows)
{
  using T = TypeParam;
  auto input =
    lists<T>{
      {0, 1, 2, 3},
      lists<T>{{X, 5}, null_at(0)},
      {6, 7, 8, 9},
      {0, 1},
      lists<T>{{X, 3, 4, X}, nulls_at({0, 3})},
      lists<T>{{X, X}, nulls_at({0, 1})},
    }
      .release();
  auto filter = filter_t{{1, 0, 1, 0}, {1, 0}, {1, 0, 1, 0}, {1, 0}, {1, 0, 1, 0}, {1, 0}};

  {
    auto filtered = apply_deletion_mask(lists_column_view{*input}, lists_column_view{filter});
    auto expected =
      lists<T>{{1, 3}, {5}, {7, 9}, {1}, lists<T>{{3, X}, null_at(1)}, lists<T>{{X}, null_at(0)}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
  {
    auto sliced   = cudf::slice(*input, {1, input->size()}).front();
    auto filter   = filter_t{{0, 1}, {0, 1, 0, 1}, {1, 1}, {0, 1, 0, 1}, {0, 0}};
    auto filtered = apply_deletion_mask(lists_column_view{sliced}, lists_column_view{filter});
    auto expected = lists<T>{lists<T>{{X}, null_at(0)},
                             {6, 8},
                             {},
                             lists<T>{{X, 4}, null_at(0)},
                             lists<T>{{X, X}, nulls_at({0, 1})}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
}

TYPED_TEST(ApplyDeletionMaskTypedTest, NullListRowsInTheInputColumn)
{
  using T = TypeParam;
  auto input =
    lists<T>{{{0, 1, 2, 3}, {}, {6, 7, 8, 9}, {}, {2, 3, 4, 5}, {6, 7}}, nulls_at({1, 3})}
      .release();
  auto filter = filter_t{{1, 0, 1, 0}, {}, {1, 0, 1, 0}, {}, {1, 0, 1, 0}, {1, 0}};

  {
    auto filtered = apply_deletion_mask(lists_column_view{*input}, lists_column_view{filter});
    auto expected = lists<T>{{{1, 3}, {}, {7, 9}, {}, {3, 5}, {7}}, nulls_at({1, 3})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
  {
    auto sliced   = cudf::slice(*input, {1, input->size()}).front();
    auto filter   = filter_t{{}, {0, 1, 0, 1}, {}, {0, 1, 0, 1}, {0, 0}};
    auto filtered = apply_deletion_mask(lists_column_view{sliced}, lists_column_view{filter});
    auto expected = lists<T>{{{}, {6, 8}, {}, {2, 4}, {6, 7}}, nulls_at({0, 2})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
}

TYPED_TEST(ApplyDeletionMaskTypedTest, SlicedInput)
{
  using T    = TypeParam;
  auto input = lists<T>{{0, 1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {0, 1}, {2, 3, 4, 5}, {6, 7}}.release();

  {
    auto sliced   = cudf::slice(*input, {1, input->size()}).front();
    auto filter   = filter_t{{1, 0}, {1, 0, 1, 0}, {0, 1}, {1, 0, 1, 0}, {1, 1}};
    auto filtered = apply_deletion_mask(lists_column_view{sliced}, lists_column_view{filter});
    auto expected = lists<T>{{5}, {7, 9}, {0}, {3, 5}, {}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
  {
    auto sliced   = cudf::slice(*input, {2, input->size()}).front();
    auto filter   = filter_t{{0, 1, 0, 1}, {0, 0}, {0, 1, 0, 1}, {0, 0}};
    auto filtered = apply_deletion_mask(lists_column_view{sliced}, lists_column_view{filter});
    auto expected = lists<T>{{6, 8}, {0, 1}, {2, 4}, {6, 7}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
  }
}

TYPED_TEST(ApplyDeletionMaskTypedTest, NullsInDeletionMask)
{
  using T    = TypeParam;
  auto input = lists<T>{{10, 20, 30}, {40, 50}, {60, 70, 80, 90}};
  // mask child (flattened):      {T, null, T,  F, null,  null, F, T, F}
  // mask offsets:                {0,           3,        5,            9}
  // kept (valid AND false):      {},           {40},     {70, 90}
  auto mask_child = fwcw<bool>{{1, X, 1, 0, X, X, 0, 1, 0}, nulls_at({1, 4, 5})};
  auto mask =
    cudf::make_lists_column(3, offsets{0, 3, 5, 9}.release(), mask_child.release(), 0, {});
  auto filtered = apply_deletion_mask(lists_column_view{input}, lists_column_view{*mask});
  auto expected = lists<T>{lists<T>{}, {40}, {70, 90}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*filtered, expected);
}

TEST_F(ApplyDeletionMaskTest, Trivial)
{
  auto const input  = lists<int32_t>{};
  auto const filter = filter_t{};
  auto const result = apply_deletion_mask(lists_column_view{input}, lists_column_view{filter});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, lists<int32_t>{});
}

TEST_F(ApplyDeletionMaskTest, Failure)
{
  {
    auto const input  = lists<int32_t>{{1, 2, 3}, {4, 5, 6}};
    auto const filter = lists<int32_t>{{0, 0, 0}};
    EXPECT_THROW(apply_deletion_mask(lists_column_view{input}, lists_column_view{filter}),
                 cudf::logic_error);
  }
  {
    auto const input  = lists<int32_t>{{1, 2, 3}, {4, 5, 6}};
    auto const filter = filter_t{{0, 0, 0}};
    EXPECT_THROW(apply_deletion_mask(lists_column_view{input}, lists_column_view{filter}),
                 cudf::logic_error);
  }
}

}  // namespace cudf::test
