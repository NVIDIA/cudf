/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_structs_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_structs_test, cudf::test::FixedWidthTypes);

template <typename T>
using fwcw = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using lcw = cudf::test::lists_column_wrapper<T>;

namespace {
static constexpr auto null = -1;  // Signifies null value.

// Checking with a single aggregation, and aggregation column.
// This test is orthogonal to the aggregation type; it focuses on testing the grouping
// with STRUCT keys.

// Set this to true to enable printing, for debugging.
auto constexpr print_enabled = false;

void print_agg_results(cudf::column_view const& keys, cudf::column_view const& vals)
{
  if constexpr (print_enabled) {
    auto requests = std::vector<cudf::groupby::aggregation_request>{};
    requests.push_back(cudf::groupby::aggregation_request{});
    requests.back().values = vals;
    requests.back().aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    requests.back().aggregations.push_back(
      cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));

    auto gby = cudf::groupby::groupby{
      cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {}};
    auto result = gby.aggregate(requests);
    std::cout << "Results: Keys: " << std::endl;
    cudf::test::print(result.first->get_column(0).view());
    std::cout << "Results: Values: " << std::endl;
    cudf::test::print(result.second.front().results[0]->view());
  }
}

}  // namespace

TYPED_TEST(groupby_structs_test, basic)
{
  using V  = int32_t;    // Type of Aggregation Column.
  using M0 = int32_t;    // Type of STRUCT's first (i.e. 0th) member.
  using M1 = TypeParam;  // Type of STRUCT's second (i.e. 1th) member.
  using R  = int64_t;

  // clang-format off
  auto values   = fwcw<V> {  0,    1,    2,    3,    4,    5,    6,    7,    8,    9};
  auto member_0 = fwcw<M0>{  1,    2,    3,    1,    2,    2,    1,    3,    3,    2};
  auto member_1 = fwcw<M1>{ 11,   22,   33,   11,   22,   22,   11,   33,   33,   22};
  auto member_2 = cudf::test::strings_column_wrapper {"11", "22", "33", "11", "22", "22", "11", "33", "33", "22"};
  auto keys     = cudf::test::structs_column_wrapper{member_0, member_1, member_2};

  auto expected_values   = fwcw<R> {  9,   19,   17 };
  auto expected_member_0 = fwcw<M0>{  1,    2,    3 };
  auto expected_member_1 = fwcw<M1>{ 11,   22,   33 };
  auto expected_member_2 = cudf::test::strings_column_wrapper {"11", "22", "33"};
  auto expected_keys     = cudf::test::structs_column_wrapper{expected_member_0, expected_member_1, expected_member_2};
  // clang-format on

  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_structs_test, structs_with_nulls_in_members)
{
  using V  = int32_t;    // Type of Aggregation Column.
  using M0 = int32_t;    // Type of STRUCT's first (i.e. 0th) member.
  using M1 = TypeParam;  // Type of STRUCT's second (i.e. 1th) member.
  using R  = int64_t;

  // clang-format off
  auto values   = fwcw<V> {  0,       1,    2,    3,    4,    5,    6,      7,    8,    9 };
  auto member_0 = fwcw<M0>{{ 1,    null,    3,    1,    2,    2,    1,      3,    3,    2 }, null_at(1)};
  auto member_1 = fwcw<M1>{{ 11,     22,   33,   11,   22,   22,   11,   null,   33,   22 }, null_at(7)};
  auto member_2 = cudf::test::strings_column_wrapper { "11",   "22", "33", "11", "22", "22", "11",   "33", "33", "22"};
  auto keys     = cudf::test::structs_column_wrapper{{member_0, member_1, member_2}};
  // clang-format on

  print_agg_results(keys, values);

  // clang-format off
  auto expected_values   = fwcw<R> {    9,   18,    10,     7,     1  };
  auto expected_member_0 = fwcw<M0>{ {  1,    2,     3,     3,  null  }, null_at(4)};
  auto expected_member_1 = fwcw<M1>{ { 11,   22,    33,  null,    22  }, null_at(3)};
  auto expected_member_2 = cudf::test::strings_column_wrapper {  "11", "22",  "33",  "33",  "22" };
  auto expected_keys     = cudf::test::structs_column_wrapper{expected_member_0, expected_member_1, expected_member_2};
  // clang-format on

  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_structs_test, structs_with_null_rows)
{
  using V  = int32_t;    // Type of Aggregation Column.
  using M0 = int32_t;    // Type of STRUCT's first (i.e. 0th) member.
  using M1 = TypeParam;  // Type of STRUCT's second (i.e. 1th) member.
  using R  = int64_t;

  // clang-format off
  auto values   = fwcw<V> {  0,    1,    2,    3,    4,    5,    6,    7,    8,    9};
  auto member_0 = fwcw<M0>{  1,    2,    3,    1,    2,    2,    1,    3,    3,    2};
  auto member_1 = fwcw<M1>{ 11,   22,   33,   11,   22,   22,   11,   33,   33,   22};
  auto member_2 = cudf::test::strings_column_wrapper {"11", "22", "33", "11", "22", "22", "11", "33", "33", "22"};
  auto keys     = cudf::test::structs_column_wrapper{{member_0, member_1, member_2}, nulls_at({0, 3})};

  auto expected_values   = fwcw<R> {    6,   19,   17,      3  };
  auto expected_member_0 = fwcw<M0>{ {  1,    2,    3,   null  }, null_at(3)};
  auto expected_member_1 = fwcw<M1>{ { 11,   22,   33,   null  }, null_at(3)};
  auto expected_member_2 = cudf::test::strings_column_wrapper { {"11", "22", "33", "" }, null_at(3)};
  auto expected_keys     = cudf::test::structs_column_wrapper{{expected_member_0, expected_member_1, expected_member_2}, null_at(3)};
  // clang-format on

  print_agg_results(keys, values);

  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_structs_test, structs_with_nulls_in_rows_and_members)
{
  using V  = int32_t;    // Type of Aggregation Column.
  using M0 = int32_t;    // Type of STRUCT's first (i.e. 0th) member.
  using M1 = TypeParam;  // Type of STRUCT's second (i.e. 1th) member.
  using R  = int64_t;

  // clang-format off
  auto values   = fwcw<V> {  0,    1,    2,    3,    4,    5,    6,    7,    8,    9  };
  auto member_0 = fwcw<M0>{{ 1,    2,    3,    1,    2,    2,    1,    3,    3,    2  }, null_at(1)};
  auto member_1 = fwcw<M1>{{ 11,   22,   33,   11,   22,   22,   11,   33,   33,   22 }, null_at(7)};
  auto member_2 = cudf::test::strings_column_wrapper { "11", "22", "33", "11", "22", "22", "11", "33", "33", "22"};
  auto keys     = cudf::test::structs_column_wrapper{{member_0, member_1, member_2}, null_at(4)};
  // clang-format on

  print_agg_results(keys, values);

  // clang-format off
  auto expected_values   = fwcw<R> {    9,   14,    10,     7,     1,      4  };
  auto expected_member_0 = fwcw<M0>{{   1,    2,     3,     3,  null,   null  }, nulls_at({4,5})};
  auto expected_member_1 = fwcw<M1>{{  11,   22,    33,  null,    22,   null  }, nulls_at({3,5})};
  auto expected_member_2 = cudf::test::strings_column_wrapper {{ "11", "22",  "33",  "33",  "22", "" }, null_at(5)};
  auto expected_keys     = cudf::test::structs_column_wrapper{{expected_member_0, expected_member_1, expected_member_2}, null_at(5)};
  // clang-format on

  print_agg_results(keys, values);
  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_structs_test, null_members_differ_from_null_structs)
{
  // This test specifically confirms that a non-null STRUCT row `{null, null, null}` is grouped
  // differently from a null STRUCT row (whose members are incidentally null).

  using V  = int32_t;    // Type of Aggregation Column.
  using M0 = int32_t;    // Type of STRUCT's first (i.e. 0th) member.
  using M1 = TypeParam;  // Type of STRUCT's second (i.e. 1th) member.
  using R  = int64_t;

  // clang-format off
  auto values   = fwcw<V> {    0,      1,    2,    3,    4,    5,    6,    7,    8,    9 };
  auto member_0 = fwcw<M0>{{   1,   null,    3,    1,    2,    2,    1,    3,    3,    2 }, null_at(1)};
  auto member_1 = fwcw<M1>{{  11,   null,   33,   11,   22,   22,   11,   33,   33,   22 }, null_at(1)};
  auto member_2 = cudf::test::strings_column_wrapper {{ "11", "null", "33", "11", "22", "22", "11", "33", "33", "22"}, null_at(1)};
  auto keys     = cudf::test::structs_column_wrapper{{member_0, member_1, member_2}, null_at(4)};
  // clang-format on

  print_agg_results(keys, values);

  // Index-3 => Non-null Struct row, with nulls for all members.
  // Index-4 => Null Struct row.

  // clang-format off
  auto expected_values   = fwcw<R> {    9,   14,    17,      1,      4  };
  auto expected_member_0 = fwcw<M0>{ {  1,    2,     3,   null,   null  }, nulls_at({3,4})};
  auto expected_member_1 = fwcw<M1>{ { 11,   22,    33,   null,   null  }, nulls_at({3,4})};
  auto expected_member_2 = cudf::test::strings_column_wrapper { {"11", "22",  "33", "", "" }, nulls_at({3,4})};
  auto expected_keys     = cudf::test::structs_column_wrapper{{expected_member_0, expected_member_1, expected_member_2}, null_at(4)};
  // clang-format on

  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_structs_test, structs_of_structs)
{
  using V  = int32_t;    // Type of Aggregation Column.
  using M0 = int32_t;    // Type of STRUCT's first (i.e. 0th) member.
  using M1 = TypeParam;  // Type of STRUCT's second (i.e. 1th) member.
  using R  = int64_t;

  // clang-format off
  auto values            = fwcw<V> {    0,      1,    2,    3,    4,    5,    6,    7,    8,    9 };
  auto struct_0_member_0 = fwcw<M0>{{   1,   null,    3,    1,    2,    2,    1,    3,    3,    2 }, null_at(1)};
  auto struct_0_member_1 = fwcw<M1>{{  11,   null,   33,   11,   22,   22,   11,   33,   33,   22 }, null_at(1)};
  auto struct_0_member_2 = cudf::test::strings_column_wrapper {{ "11", "", "33", "11", "22", "22", "11", "33", "33", "22"}, null_at(1)};
  // clang-format on

  auto struct_0 = cudf::test::structs_column_wrapper{
    {struct_0_member_0, struct_0_member_1, struct_0_member_2}, null_at(4)};
  auto struct_1_member_1 = fwcw<M1>{8, 9, 6, 8, 0, 7, 8, 6, 6, 7};

  auto keys = cudf::test::structs_column_wrapper{
    {struct_0, struct_1_member_1}};  // Struct of cudf::test::structs_column_wrapper.

  print_agg_results(keys, values);

  // clang-format off
  auto expected_values            = fwcw<R> {    9,   14,    17,      1,      4  };
  auto expected_member_0          = fwcw<M0>{ {  1,    2,     3,   null,   null  }, nulls_at({3,4})};
  auto expected_member_1          = fwcw<M1>{ { 11,   22,    33,   null,   null  }, nulls_at({3,4})};
  auto expected_member_2          = cudf::test::strings_column_wrapper { {"11", "22",  "33", "", "" }, nulls_at({3,4})};
  auto expected_structs           = cudf::test::structs_column_wrapper{{expected_member_0, expected_member_1, expected_member_2}, null_at(4)};
  auto expected_struct_1_member_1 = fwcw<M1>{    8,    7,     6,      9,      0  };
  auto expected_keys              = cudf::test::structs_column_wrapper{{expected_structs, expected_struct_1_member_1}};
  // clang-format on

  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_structs_test, empty_input)
{
  using V  = int32_t;    // Type of Aggregation Column.
  using M0 = int32_t;    // Type of STRUCT's first (i.e. 0th) member.
  using M1 = TypeParam;  // Type of STRUCT's second (i.e. 1th) member.
  using R  = int64_t;

  // clang-format off
  auto values   = fwcw<V> {};
  auto member_0 = fwcw<M0>{};
  auto member_1 = fwcw<M1>{};
  auto member_2 = cudf::test::strings_column_wrapper {};
  auto keys     = cudf::test::structs_column_wrapper{member_0, member_1, member_2};

  auto expected_values   = fwcw<R> {};
  auto expected_member_0 = fwcw<M0>{};
  auto expected_member_1 = fwcw<M1>{};
  auto expected_member_2 = cudf::test::strings_column_wrapper {};
  auto expected_keys     = cudf::test::structs_column_wrapper{expected_member_0, expected_member_1, expected_member_2};
  // clang-format on

  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_structs_test, all_null_input)
{
  using V  = int32_t;    // Type of Aggregation Column.
  using M0 = int32_t;    // Type of STRUCT's first (i.e. 0th) member.
  using M1 = TypeParam;  // Type of STRUCT's second (i.e. 1th) member.
  using R  = int64_t;

  // clang-format off
  auto values   = fwcw<V> {  0,    1,    2,    3,    4,    5,    6,    7,    8,    9};
  auto member_0 = fwcw<M0>{  1,    2,    3,    1,    2,    2,    1,    3,    3,    2};
  auto member_1 = fwcw<M1>{ 11,   22,   33,   11,   22,   22,   11,   33,   33,   22};
  auto member_2 = cudf::test::strings_column_wrapper {"11", "22", "33", "11", "22", "22", "11", "33", "33", "22"};
  auto keys     = cudf::test::structs_column_wrapper{{member_0, member_1, member_2}, all_nulls()};

  auto expected_values   = fwcw<R> {    45 };
  auto expected_member_0 = fwcw<M0>{ null };
  auto expected_member_1 = fwcw<M1>{ null };
  auto expected_member_2 = cudf::test::strings_column_wrapper {""};
  auto expected_keys     = cudf::test::structs_column_wrapper{{expected_member_0, expected_member_1, expected_member_2}, all_nulls()};
  // clang-format on

  test_sum_agg(keys, values, expected_keys, expected_values);
}

TYPED_TEST(groupby_structs_test, lists_as_keys)
{
  using V  = int32_t;    // Type of Aggregation Column.
  using M0 = int32_t;    // Type of STRUCT's first (i.e. 0th) member.
  using M1 = TypeParam;  // Type of STRUCT's second (i.e. 1th) member.
  using R  = int64_t;

  // clang-format off
  auto values   = fwcw<V> {     0,      1,      2,      3,       4  };
  auto member_0 = lcw<M0> { {1,1},  {2,2},  {3,3},   {1,1},   {2,2} };
  auto member_1 = fwcw<M1>{     1,      2,      3,      1,       2  };
  // clang-format on
  auto keys = cudf::test::structs_column_wrapper{{member_0, member_1}};

  // clang-format off
  auto expected_values   = fwcw<R> {     3,      5,      2 };
  auto expected_member_0 = lcw<M0> { {1,1},  {2,2},  {3,3} };
  auto expected_member_1 = fwcw<M1>{     1,      2,      3 };
  // clang-format on
  auto expected_keys = cudf::test::structs_column_wrapper{{expected_member_0, expected_member_1}};

  test_sum_agg(keys, values, expected_keys, expected_values);
}
