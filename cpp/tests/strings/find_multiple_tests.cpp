/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

#include <vector>

struct StringsFindMultipleTest : public cudf::test::BaseFixture {};

TEST_F(StringsFindMultipleTest, FindMultiple)
{
  std::vector<char const*> h_strings{"Héllo", "thesé", nullptr, "lease", "test strings", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<char const*> h_targets{"é", "a", "e", "i", "o", "u", "es"};
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
  auto targets_view = cudf::strings_column_view(targets);

  auto results = cudf::strings::find_multiple(strings_view, targets_view);

  using LCW = cudf::test::lists_column_wrapper<int32_t>;
  LCW expected({LCW{1, -1, -1, -1, 4, -1, -1},
                LCW{4, -1, 2, -1, -1, -1, 2},
                LCW{-1, -1, -1, -1, -1, -1, -1},
                LCW{-1, 2, 1, -1, -1, -1, -1},
                LCW{-1, -1, 1, 8, -1, -1, 1},
                LCW{-1, -1, -1, -1, -1, -1, -1}});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsFindMultipleTest, ZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto strings_view                   = cudf::strings_column_view(zero_size_strings_column);
  std::vector<char const*> h_targets{""};
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());
  auto targets_view = cudf::strings_column_view(targets);

  auto results = cudf::strings::find_multiple(strings_view, targets_view);
  EXPECT_EQ(results->size(), 0);
}

TEST_F(StringsFindMultipleTest, ErrorTest)
{
  cudf::test::strings_column_wrapper strings({""}, {false});
  auto strings_view = cudf::strings_column_view(strings);

  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto empty_view                     = cudf::strings_column_view(zero_size_strings_column);
  // targets must have at least one string
  EXPECT_THROW(cudf::strings::find_multiple(strings_view, empty_view), std::invalid_argument);
  EXPECT_THROW(cudf::strings::contains_multiple(strings_view, empty_view), std::invalid_argument);

  // targets cannot have nulls
  EXPECT_THROW(cudf::strings::find_multiple(strings_view, strings_view), std::invalid_argument);
  EXPECT_THROW(cudf::strings::contains_multiple(strings_view, strings_view), std::invalid_argument);
}

TEST_F(StringsFindMultipleTest, MultiContains)
{
  constexpr int num_rows = 1024 + 1;
  // replicate the following 9 rows:
  std::vector<std::string> s = {
    "Héllo, there world and goodbye",
    "quick brown fox jumped over the lazy brown dog; the fat cats jump in place without moving",
    "the following code snippet demonstrates how to use search for values in an ordered range",
    "it returns the last position where value could be inserted without violating the ordering",
    "algorithms execution is parallelized as determined by an execution policy. t",
    "he this is a continuation of previous row to make sure string boundaries are honored",
    "abcdefghijklmnopqrstuvwxyz 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ !@#$%^&*()~",
    "",
    ""};

  // replicate strings
  auto string_itr =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return s[i % s.size()]; });

  // nulls: 8, 8 + 1 * 9, 8 + 2 * 9 ......
  auto string_v = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return (i + 1) % s.size() != 0; });

  auto const strings =
    cudf::test::strings_column_wrapper(string_itr, string_itr + num_rows, string_v);
  auto strings_view = cudf::strings_column_view(strings);
  std::vector<std::string> match_targets({" the ", "a", "", "é"});
  cudf::test::strings_column_wrapper multi_targets_column(match_targets.begin(),
                                                          match_targets.end());
  auto results =
    cudf::strings::contains_multiple(strings_view, cudf::strings_column_view(multi_targets_column));

  std::vector<bool> ret_0 = {0, 1, 0, 1, 0, 0, 0, 0, 0};
  std::vector<bool> ret_1 = {1, 1, 1, 1, 1, 1, 1, 0, 0};
  std::vector<bool> ret_2 = {1, 1, 1, 1, 1, 1, 1, 1, 0};
  std::vector<bool> ret_3 = {1, 0, 0, 0, 0, 0, 0, 0, 0};

  auto make_bool_col_fn = [&string_v, &num_rows](std::vector<bool> bools) {
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, [&](auto i) { return bools[i % bools.size()]; });
    return cudf::test::fixed_width_column_wrapper<bool>(iter, iter + num_rows, string_v);
  };

  auto expected_0 = make_bool_col_fn(ret_0);
  auto expected_1 = make_bool_col_fn(ret_1);
  auto expected_2 = make_bool_col_fn(ret_2);
  auto expected_3 = make_bool_col_fn(ret_3);

  auto expected = cudf::table_view({expected_0, expected_1, expected_2, expected_3});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsFindMultipleTest, MultiContainsMoreTargets)
{
  auto const strings = cudf::test::strings_column_wrapper{
    "quick brown fox jumped over the lazy brown dog; the fat cats jump in place without moving "
    "quick brown fox jumped",
    "the following code snippet demonstrates how to use search for values in an ordered rangethe "
    "following code snippet",
    "thé it returns the last position where value could be inserted without violating ordering thé "
    "it returns the last position"};
  auto strings_view = cudf::strings_column_view(strings);
  std::vector<std::string> targets({"lazy brown", "non-exist", ""});

  std::vector<cudf::test::fixed_width_column_wrapper<bool>> expects;
  expects.push_back(cudf::test::fixed_width_column_wrapper<bool>({1, 0, 0}));
  expects.push_back(cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0}));
  expects.push_back(cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1}));

  std::vector<std::string> match_targets;
  int max_num_targets = 50;

  for (int num_targets = 1; num_targets < max_num_targets; num_targets++) {
    match_targets.clear();
    for (int i = 0; i < num_targets; i++) {
      match_targets.push_back(targets[i % targets.size()]);
    }

    cudf::test::strings_column_wrapper multi_targets_column(match_targets.begin(),
                                                            match_targets.end());
    auto results = cudf::strings::contains_multiple(
      strings_view, cudf::strings_column_view(multi_targets_column));
    EXPECT_EQ(results->num_columns(), num_targets);
    for (int i = 0; i < num_targets; i++) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->get_column(i), expects[i % expects.size()]);
    }
  }
}

TEST_F(StringsFindMultipleTest, MultiContainsLongStrings)
{
  constexpr int num_rows = 1024 + 1;
  // replicate the following 7 rows:
  std::vector<std::string> s = {
    "quick brown fox jumped over the lazy brown dog; the fat cats jump in place without moving "
    "quick brown fox jumped",
    "the following code snippet demonstrates how to use search for values in an ordered rangethe "
    "following code snippet",
    "thé it returns the last position where value could be inserted without violating ordering thé "
    "it returns the last position",
    "algorithms execution is parallelized as determined by an execution policy. t algorithms "
    "execution is parallelized as ",
    "he this is a continuation of previous row to make sure string boundaries are honored he this "
    "is a continuation of previous row",
    "abcdefghijklmnopqrstuvwxyz 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    "!@#$%^&*()~abcdefghijklmnopqrstuvwxyz 0123456789 ABCDEFGHIJKL",
    ""};

  // replicate strings
  auto string_itr =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return s[i % s.size()]; });

  // nulls: 6, 6 + 1 * 7, 6 + 2 * 7 ......
  auto string_v = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return (i + 1) % s.size() != 0; });

  auto const strings =
    cudf::test::strings_column_wrapper(string_itr, string_itr + num_rows, string_v);

  auto sv      = cudf::strings_column_view(strings);
  auto targets = cudf::test::strings_column_wrapper({" the ", "search", "", "string", "ox", "é "});
  auto results = cudf::strings::contains_multiple(sv, cudf::strings_column_view(targets));

  std::vector<bool> ret_0 = {1, 0, 1, 0, 0, 0, 0};
  std::vector<bool> ret_1 = {0, 1, 0, 0, 0, 0, 0};
  std::vector<bool> ret_2 = {1, 1, 1, 1, 1, 1, 0};
  std::vector<bool> ret_3 = {0, 0, 0, 0, 1, 0, 0};
  std::vector<bool> ret_4 = {1, 0, 0, 0, 0, 0, 0};
  std::vector<bool> ret_5 = {0, 0, 1, 0, 0, 0, 0};

  auto make_bool_col_fn = [&string_v, &num_rows](std::vector<bool> bools) {
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, [&](auto i) { return bools[i % bools.size()]; });
    return cudf::test::fixed_width_column_wrapper<bool>(iter, iter + num_rows, string_v);
  };

  auto expected_0 = make_bool_col_fn(ret_0);
  auto expected_1 = make_bool_col_fn(ret_1);
  auto expected_2 = make_bool_col_fn(ret_2);
  auto expected_3 = make_bool_col_fn(ret_3);
  auto expected_4 = make_bool_col_fn(ret_4);
  auto expected_5 = make_bool_col_fn(ret_5);

  auto expected =
    cudf::table_view({expected_0, expected_1, expected_2, expected_3, expected_4, expected_5});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(results->view(), expected);
}
