/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/reduction/unique_count.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <algorithm>
#include <cmath>

using lists_col   = cudf::test::lists_column_wrapper<int32_t>;
using structs_col = cudf::test::structs_column_wrapper;

using cudf::test::iterators::nulls_at;

using cudf::nan_policy;
using cudf::null_equality;
using cudf::null_policy;

constexpr int32_t XXX{70};  // Mark for null elements
constexpr int32_t YYY{3};   // Mark for null elements

template <typename T>
struct TypedUniqueCount : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedUniqueCount, cudf::test::NumericTypes);

TYPED_TEST(TypedUniqueCount, NoNull)
{
  using T = TypeParam;

  auto const input = cudf::test::make_type_param_vector<T>(
    {1, 3, 3, 4, 31, 1, 8, 2, 0, 4, 1, 4, 10, 40, 31, 42, 0, 42, 8, 5, 4});

  cudf::test::fixed_width_column_wrapper<T> input_col(input.begin(), input.end());

  // explicit instantiation to one particular type (`double`) to reduce build time
  std::vector<double> input_data(input.begin(), input.end());
  auto const new_end = std::unique(input_data.begin(), input_data.end());
  auto const gold    = std::distance(input_data.begin(), new_end);
  EXPECT_EQ(gold, cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

TYPED_TEST(TypedUniqueCount, TableNoNull)
{
  using T = TypeParam;

  auto const input1 = cudf::test::make_type_param_vector<T>(
    {1, 3, 3, 3, 4, 31, 1, 8, 2, 0, 4, 1, 4, 10, 40, 31, 42, 0, 42, 8, 5, 4});
  auto const input2 = cudf::test::make_type_param_vector<T>(
    {3, 3, 3, 4, 31, 1, 8, 5, 0, 4, 1, 4, 10, 40, 31, 42, 0, 42, 8, 5, 4, 1});

  std::vector<std::pair<T, T>> pair_input;
  std::transform(
    input1.begin(), input1.end(), input2.begin(), std::back_inserter(pair_input), [](T a, T b) {
      return std::pair(a, b);
    });

  cudf::test::fixed_width_column_wrapper<T> input_col1(input1.begin(), input1.end());
  cudf::test::fixed_width_column_wrapper<T> input_col2(input2.begin(), input2.end());
  cudf::table_view input_table({input_col1, input_col2});

  auto const new_end = std::unique(pair_input.begin(), pair_input.end());
  auto const gold    = std::distance(pair_input.begin(), new_end);
  EXPECT_EQ(gold, cudf::unique_count(input_table, null_equality::EQUAL));
}

struct UniqueCount : public cudf::test::BaseFixture {};

TEST_F(UniqueCount, WithNull)
{
  using T = int32_t;

  std::vector<T> input               = {1,   3,  3,  XXX, 31, 1, 8,  2, 0, XXX, XXX,
                                        XXX, 10, 40, 31,  42, 0, 42, 8, 5, XXX};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  cudf::test::fixed_width_column_wrapper<T> input_col(input.begin(), input.end(), valid.begin());

  auto const new_end = std::unique(input.begin(), input.end());
  auto const gold    = std::distance(input.begin(), new_end) - 3;
  EXPECT_EQ(gold, cudf::unique_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(UniqueCount, IgnoringNull)
{
  using T = int32_t;

  std::vector<T> input               = {1,   YYY, YYY, XXX, 31, 1, 8,  2, 0, XXX, 1,
                                        XXX, 10,  40,  31,  42, 0, 42, 8, 5, XXX};
  std::vector<cudf::size_type> valid = {1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  cudf::test::fixed_width_column_wrapper<T> input_col(input.begin(), input.end(), valid.begin());

  auto const new_end = std::unique(input.begin(), input.end());
  // -1 since `YYY, YYY, XXX` is in the same group of equivalent rows
  auto const gold = std::distance(input.begin(), new_end) - 1;
  EXPECT_EQ(gold, cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(UniqueCount, WithNansAndNull)
{
  using T = float;

  std::vector<T> input               = {1,   3,  NAN, XXX, 31,  1, 8,   2, 0, XXX, 1,
                                        XXX, 10, 40,  31,  NAN, 0, NAN, 8, 5, XXX};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 0};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  auto const new_end = std::unique(input.begin(), input.end());
  auto const gold    = std::distance(input.begin(), new_end);
  EXPECT_EQ(gold, cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));

  input     = {NAN, NAN, XXX};
  valid     = {1, 1, 0};
  input_col = cudf::test::fixed_width_column_wrapper<T>{input.begin(), input.end(), valid.begin()};

  constexpr auto expected_all_nan = 2;
  EXPECT_EQ(expected_all_nan,
            cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(UniqueCount, WithNansOnly)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, 70, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  constexpr auto expected = 5;
  EXPECT_EQ(expected,
            cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));

  input     = {NAN, NAN, NAN};
  valid     = {1, 1, 1};
  input_col = cudf::test::fixed_width_column_wrapper<T>{input.begin(), input.end(), valid.begin()};

  constexpr auto expected_all_nan = 1;
  EXPECT_EQ(expected_all_nan,
            cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID));
}

TEST_F(UniqueCount, NansAsNullWithNoNull)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, 70, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  constexpr auto expected = 5;
  EXPECT_EQ(expected, cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL));

  input     = {NAN, NAN, NAN};
  valid     = {1, 1, 1};
  input_col = cudf::test::fixed_width_column_wrapper<T>{input.begin(), input.end(), valid.begin()};

  constexpr auto expected_all_nan = 1;
  EXPECT_EQ(expected_all_nan,
            cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(UniqueCount, NansAsNullWithNull)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, XXX, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  constexpr auto expected = 4;
  EXPECT_EQ(expected, cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL));

  input     = {NAN, NAN, XXX};
  valid     = {1, 1, 0};
  input_col = cudf::test::fixed_width_column_wrapper<T>{input.begin(), input.end(), valid.begin()};

  constexpr auto expected_all_null = 1;
  EXPECT_EQ(expected_all_null,
            cudf::unique_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(UniqueCount, NansAsNullWithIgnoreNull)
{
  using T = float;

  std::vector<T> input               = {1, 3, NAN, XXX, 31};
  std::vector<cudf::size_type> valid = {1, 1, 1, 0, 1};

  cudf::test::fixed_width_column_wrapper<T> input_col{input.begin(), input.end(), valid.begin()};

  constexpr auto expected = 3;
  EXPECT_EQ(expected, cudf::unique_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL));

  input     = {NAN, NAN, NAN};
  valid     = {1, 1, 1};
  input_col = cudf::test::fixed_width_column_wrapper<T>{input.begin(), input.end(), valid.begin()};

  constexpr auto expected_all_nan = 0;
  EXPECT_EQ(expected_all_nan,
            cudf::unique_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(UniqueCount, EmptyColumn)
{
  using T = float;

  cudf::test::fixed_width_column_wrapper<T> input_col{};

  constexpr auto expected = 0;
  EXPECT_EQ(expected, cudf::unique_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL));
}

TEST_F(UniqueCount, NullableLists)
{
  auto const keys = lists_col{
    {{}, {}, {1, 1}, {1}, {1}, {} /*NULL*/, {} /*NULL*/, {2}, {2}, {2, 1}, {2, 2}, {2, 2}},
    nulls_at({5, 6})};
  auto const input = cudf::table_view{{keys}};

  EXPECT_EQ(7, cudf::unique_count(input, null_equality::EQUAL));
  EXPECT_EQ(8, cudf::unique_count(input, null_equality::UNEQUAL));
}

TEST_F(UniqueCount, NullableStructOfStructs)
{
  //  +-----------------+
  //  |  s1{s2{a,b}, c} |
  //  +-----------------+
  // 0 |  { {1, 1}, 5}  |
  // 1 |  { {1, 1}, 5}  |  // Same as 0
  // 2 |  { {1, 2}, 4}  |
  // 3 |  { Null,   6}  |
  // 4 |  { Null,   4}  |
  // 5 |  { Null,   4}  |  // Same as 4
  // 6 |  Null          |
  // 7 |  Null          |  // Same as 6
  // 8 |  { {2, 1}, 5}  |

  auto const keys = [&] {
    auto a  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 1, 1, XXX, XXX, XXX, 2, 1, 2};
    auto b  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 1, 2, XXX, XXX, XXX, 2, 1, 1};
    auto s2 = structs_col{{a, b}, nulls_at({3, 4, 5})};

    auto c = cudf::test::fixed_width_column_wrapper<int32_t>{5, 5, 4, 6, 4, 4, 3, 3, 5};
    std::vector<std::unique_ptr<cudf::column>> s1_children;
    s1_children.emplace_back(s2.release());
    s1_children.emplace_back(c.release());
    auto const null_it = nulls_at({6, 7});
    return structs_col(std::move(s1_children), std::vector<bool>{null_it, null_it + 9});
  }();

  auto const input = cudf::table_view{{keys}};

  EXPECT_EQ(6, cudf::unique_count(input, null_equality::EQUAL));
  EXPECT_EQ(8, cudf::unique_count(input, null_equality::UNEQUAL));
}
