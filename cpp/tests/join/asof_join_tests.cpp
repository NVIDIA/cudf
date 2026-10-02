/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/asof_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <vector>

namespace {

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

class AsofJoinTest : public cudf::test::BaseFixture {};

void expect_gather_map(rmm::device_uvector<cudf::size_type> const& actual,
                       std::vector<cudf::size_type> const& expected)
{
  auto const stream      = cudf::get_default_stream();
  auto const host_actual = cudf::detail::make_host_vector_async(actual, stream);
  stream.sync();

  ASSERT_EQ(host_actual.size(), expected.size());
  EXPECT_TRUE(std::equal(host_actual.begin(), host_actual.end(), expected.begin()));
}

TEST_F(AsofJoinTest, Ungrouped)
{
  fixed_width_column_wrapper<int32_t> right_on{1, 3, 3, 7};
  fixed_width_column_wrapper<int32_t> left_on{0, 1, 2, 3, 8};

  cudf::asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {cudf::JoinNoMatch, 0, 0, 2, 3});
}

TEST_F(AsofJoinTest, SingleStringGroupingKey)
{
  strings_column_wrapper right_by{"a", "a", "b", "b"};
  fixed_width_column_wrapper<int32_t> right_on{1, 3, 2, 4};
  strings_column_wrapper left_by{"b", "a", "c", "a", "b"};
  fixed_width_column_wrapper<int32_t> left_on{3, 0, 5, 3, 5};

  cudf::asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result =
    join.join(cudf::table_view{{left_by}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {2, cudf::JoinNoMatch, cudf::JoinNoMatch, 1, 3});
}

TEST_F(AsofJoinTest, MultipleGroupingKeys)
{
  strings_column_wrapper right_by_1{"a", "a", "a", "b", "b"};
  fixed_width_column_wrapper<int32_t> right_by_2{1, 1, 2, 1, 1};
  fixed_width_column_wrapper<int32_t> right_on{1, 4, 2, 3, 5};
  strings_column_wrapper left_by_1{"b", "a", "a"};
  fixed_width_column_wrapper<int32_t> left_by_2{1, 2, 1};
  fixed_width_column_wrapper<int32_t> left_on{4, 1, 4};

  cudf::asof_join join(cudf::table_view{{right_by_1, right_by_2}}, right_on);
  auto result = join.join(
    cudf::table_view{{left_by_1, left_by_2}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {3, cudf::JoinNoMatch, 1});
}

TEST_F(AsofJoinTest, NullOrderedKeys)
{
  fixed_width_column_wrapper<int32_t> right_on({0, 1, 3}, cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> left_on({0, 1, 2, 3}, cudf::test::iterators::nulls_at({3}));

  cudf::asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {cudf::JoinNoMatch, 1, 1, cudf::JoinNoMatch});
}

TEST_F(AsofJoinTest, NullGroupingKeysDoNotMatch)
{
  strings_column_wrapper right_by({"", "", "a"}, cudf::test::iterators::nulls_at({0, 1}));
  fixed_width_column_wrapper<int32_t> right_on{1, 2, 1};
  strings_column_wrapper left_by({"", "a"}, cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> left_on{2, 1};

  cudf::asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result =
    join.join(cudf::table_view{{left_by}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {cudf::JoinNoMatch, 2});
}

TEST_F(AsofJoinTest, EmptyInputs)
{
  fixed_width_column_wrapper<int32_t> empty{};
  fixed_width_column_wrapper<int32_t> left_on{1, 2};

  cudf::asof_join empty_right_join(cudf::table_view{}, empty);
  auto no_matches =
    empty_right_join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true);
  expect_gather_map(*no_matches, {cudf::JoinNoMatch, cudf::JoinNoMatch});

  cudf::asof_join join(cudf::table_view{}, left_on);
  auto empty_result =
    join.join(cudf::table_view{}, empty, cudf::asof_join_strategy::BACKWARD, true);
  expect_gather_map(*empty_result, {});
}

TEST_F(AsofJoinTest, TimestampOrderedKey)
{
  fixed_width_column_wrapper<cudf::timestamp_ms, int64_t> right_on{1, 3, 5};
  fixed_width_column_wrapper<cudf::timestamp_ms, int64_t> left_on{2, 5};

  cudf::asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {0, 2});
}

TEST_F(AsofJoinTest, UnsupportedOptions)
{
  fixed_width_column_wrapper<int32_t> right_on{1};
  fixed_width_column_wrapper<int32_t> left_on{1};
  cudf::asof_join join(cudf::table_view{}, right_on);

  EXPECT_THROW(
    (void)join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::FORWARD, true),
    cudf::logic_error);
  EXPECT_THROW(
    (void)join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, false),
    cudf::logic_error);
}

TEST_F(AsofJoinTest, MismatchedOrderedKeyTypes)
{
  fixed_width_column_wrapper<int32_t> right_on{1};
  fixed_width_column_wrapper<int64_t> left_on{1};
  cudf::asof_join join(cudf::table_view{}, right_on);

  EXPECT_THROW(
    (void)join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true),
    cudf::logic_error);
}

TEST_F(AsofJoinTest, NullGroupBeforeNonNullGroup)
{
  strings_column_wrapper right_by({"", "a", "a"}, cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> right_on{1, 1, 2};
  strings_column_wrapper left_by({"", "a"}, cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> left_on{5, 1};

  cudf::asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result =
    join.join(cudf::table_view{{left_by}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {cudf::JoinNoMatch, 1});
}

TEST_F(AsofJoinTest, LeftKeyBetweenGroups)
{
  strings_column_wrapper right_by{"a", "c"};
  fixed_width_column_wrapper<int32_t> right_on{1, 1};
  strings_column_wrapper left_by{"b", "0", "d"};
  fixed_width_column_wrapper<int32_t> left_on{1, 1, 1};

  cudf::asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result =
    join.join(cudf::table_view{{left_by}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {cudf::JoinNoMatch, cudf::JoinNoMatch, cudf::JoinNoMatch});
}

TEST_F(AsofJoinTest, NullOrderedKeyAtGroupFront)
{
  strings_column_wrapper right_by{"a", "a", "b"};
  fixed_width_column_wrapper<int32_t> right_on({0, 5, 3}, cudf::test::iterators::nulls_at({0}));
  strings_column_wrapper left_by{"a", "a"};
  fixed_width_column_wrapper<int32_t> left_on{4, 5};

  cudf::asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result =
    join.join(cudf::table_view{{left_by}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {cudf::JoinNoMatch, 1});
}

TEST_F(AsofJoinTest, SlicedColumns)
{
  fixed_width_column_wrapper<int32_t> right_on_base{9, 1, 3, 3, 7, 9};
  fixed_width_column_wrapper<int32_t> left_on_base{9, 0, 1, 2, 3, 8};
  auto const right_on = cudf::slice(right_on_base, {1, 5})[0];
  auto const left_on  = cudf::slice(left_on_base, {1, 5})[0];

  cudf::asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {cudf::JoinNoMatch, 0, 0, 2});
}

TEST_F(AsofJoinTest, SlicedGroupedColumns)
{
  strings_column_wrapper right_by_base{"z", "a", "a", "b", "b", "z"};
  fixed_width_column_wrapper<int32_t> right_on_base{9, 1, 3, 2, 4, 9};
  auto const right_by = cudf::slice(right_by_base, {1, 5})[0];
  auto const right_on = cudf::slice(right_on_base, {1, 5})[0];
  strings_column_wrapper left_by_base{"z", "b", "a", "z"};
  fixed_width_column_wrapper<int32_t> left_on_base{9, 3, 3, 9};
  auto const left_by = cudf::slice(left_by_base, {1, 3})[0];
  auto const left_on = cudf::slice(left_on_base, {1, 3})[0];

  cudf::asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result =
    join.join(cudf::table_view{{left_by}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  expect_gather_map(*result, {2, 1});
}

TEST_F(AsofJoinTest, ValidationErrors)
{
  strings_column_wrapper right_by{"a", "b"};
  fixed_width_column_wrapper<int32_t> right_on{1, 2};
  cudf::asof_join join(cudf::table_view{{right_by}}, right_on);

  // Left/right number of grouping keys must match
  fixed_width_column_wrapper<int32_t> left_on{1};
  EXPECT_THROW(
    (void)join.join(cudf::table_view{}, left_on, cudf::asof_join_strategy::BACKWARD, true),
    cudf::logic_error);

  // Grouping and ordered keys must have the same number of rows
  strings_column_wrapper short_by{"a"};
  EXPECT_THROW((void)cudf::asof_join(cudf::table_view{{short_by}}, right_on), cudf::logic_error);
  strings_column_wrapper left_by{"a", "b"};
  fixed_width_column_wrapper<int32_t> long_left_on{1, 2, 3};
  EXPECT_THROW(
    (void)join.join(
      cudf::table_view{{left_by}}, long_left_on, cudf::asof_join_strategy::BACKWARD, true),
    cudf::logic_error);

  // Unsupported ordered-key type
  strings_column_wrapper strings_on{"a", "b"};
  EXPECT_THROW((void)cudf::asof_join(cudf::table_view{}, strings_on), cudf::logic_error);

  // Unsupported grouping-key type
  auto lists_by = cudf::test::lists_column_wrapper<int32_t>{{1, 2}, {3}};
  EXPECT_THROW((void)cudf::asof_join(cudf::table_view{{lists_by}}, right_on), cudf::logic_error);
}

TEST_F(AsofJoinTest, LargeMultiBlock)
{
  // 2000 contiguous groups of 1000 rows each, ordered keys ascending within each group.
  // Exercises comparator lifetimes and dispatch across many thread blocks.
  auto constexpr num_rows   = 2'000'000;
  auto constexpr group_size = 1000;
  std::vector<int32_t> right_by_host(num_rows);
  std::vector<int32_t> right_on_host(num_rows);
  for (int32_t i = 0; i < num_rows; ++i) {
    right_by_host[i] = i / group_size;
    right_on_host[i] = i % group_size;
  }
  fixed_width_column_wrapper<int32_t> right_by(right_by_host.begin(), right_by_host.end());
  fixed_width_column_wrapper<int32_t> right_on(right_on_host.begin(), right_on_host.end());

  auto constexpr num_groups = num_rows / group_size;
  std::vector<int32_t> left_by_host(num_groups);
  std::vector<int32_t> left_on_host(num_groups, group_size / 2);
  for (int32_t g = 0; g < num_groups; ++g) {
    left_by_host[g] = g;
  }
  fixed_width_column_wrapper<int32_t> left_by(left_by_host.begin(), left_by_host.end());
  fixed_width_column_wrapper<int32_t> left_on(left_on_host.begin(), left_on_host.end());

  cudf::asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result =
    join.join(cudf::table_view{{left_by}}, left_on, cudf::asof_join_strategy::BACKWARD, true);

  std::vector<cudf::size_type> expected(num_groups);
  for (int32_t g = 0; g < num_groups; ++g) {
    expected[g] = g * group_size + group_size / 2;
  }
  expect_gather_map(*result, expected);
}

}  // namespace
