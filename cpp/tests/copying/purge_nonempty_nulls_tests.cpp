/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

using cudf::test::iterators::no_nulls;
using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;
using T             = int32_t;  // The actual type of the leaf node isn't really important.
using values_col_t  = cudf::test::fixed_width_column_wrapper<T>;
using offsets_col_t = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
using gather_map_t  = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

template <typename T>
using LCW = cudf::test::lists_column_wrapper<T, int32_t>;

struct HasNonEmptyNullsTest : public cudf::test::BaseFixture {};

TEST_F(HasNonEmptyNullsTest, TrivialTest)
{
  auto const input = LCW<T>{{{{1, 2, 3, 4}, null_at(2)},
                             {5},
                             {6, 7},  // <--- Will be set to NULL. Unsanitized row.
                             {8, 9, 10}},
                            no_nulls()}
                       .release();
  EXPECT_FALSE(cudf::may_have_nonempty_nulls(*input));
  EXPECT_FALSE(cudf::has_nonempty_nulls(*input));

  // Set nullmask, post construction.
  cudf::set_null_mask(input->mutable_view().null_mask(), 2, 3, false);
  input->set_null_count(1);
  EXPECT_TRUE(cudf::may_have_nonempty_nulls(*input));
  EXPECT_TRUE(cudf::has_nonempty_nulls(*input));
}

TEST_F(HasNonEmptyNullsTest, SlicedInputTest)
{
  auto const input = cudf::test::strings_column_wrapper{
    {"" /*NULL*/, "111", "222", "333", "444", "" /*NULL*/, "", "777", "888", "" /*NULL*/, "101010"},
    cudf::test::iterators::nulls_at({0, 5, 9})};

  // Split into 2 columns from rows [0, 2) and [2, 10).
  auto const result = cudf::split(input, {2});
  for (auto const& col : result) {
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(col));
    EXPECT_FALSE(cudf::has_nonempty_nulls(col));
  }
}

struct PurgeNonEmptyNullsTest : public cudf::test::BaseFixture {
  /// Helper to run gather() on a single column, and extract the single column from the result.
  std::unique_ptr<cudf::column> gather(cudf::column_view const& input,
                                       gather_map_t const& gather_map)
  {
    auto gathered =
      cudf::gather(cudf::table_view{{input}}, gather_map, cudf::out_of_bounds_policy::NULLIFY);
    return std::move(gathered->release()[0]);
  }

  /// Verify that the result of `sanitize()` is equivalent to the unsanitized input,
  /// except that the null rows are also empty.
  void test_purge(cudf::column_view const& unpurged)
  {
    auto const purged = cudf::purge_nonempty_nulls(unpurged);
    EXPECT_FALSE(cudf::has_nonempty_nulls(*purged));
  }
};

// List<T>.
TEST_F(PurgeNonEmptyNullsTest, SingleLevelList)
{
  auto const input = LCW<T>{{{{1, 2, 3, 4}, null_at(2)},
                             {5},
                             {6, 7},  // <--- Will be set to NULL. Unsanitized row.
                             {8, 9, 10}},
                            no_nulls()}
                       .release();

  // Set nullmask, post construction.
  cudf::set_null_mask(input->mutable_view().null_mask(), 2, 3, false);
  input->set_null_count(1);

  test_purge(*input);

  {
    // Selecting all rows from input, in different order.
    auto const results           = gather(input->view(), {1, 2, 0, 3});
    auto const results_list_view = cudf::lists_column_view(*results);

    auto const expected = LCW<T>{{{5},
                                  {},  // NULL.
                                  {{1, 2, 3, 4}, null_at(2)},
                                  {8, 9, 10}},
                                 null_at(1)};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_list_view.offsets(), offsets_col_t{0, 1, 1, 5, 8});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_list_view.child(),
                                   values_col_t{{5, 1, 2, 3, 4, 8, 9, 10}, null_at(3)});
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(*results));
    EXPECT_FALSE(cudf::has_nonempty_nulls(*results));
  }
  {
    // Test when gather selects rows preceded by unsanitized rows.
    auto const results  = gather(input->view(), {3, 100, 0});
    auto const expected = LCW<T>{{
                                   {8, 9, 10},
                                   {},  // NULL.
                                   {{1, 2, 3, 4}, null_at(2)},
                                 },
                                 null_at(1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(*results));
    EXPECT_FALSE(cudf::has_nonempty_nulls(*results));
  }
  {
    // Test when gather selects rows followed by unsanitized rows.
    auto const results  = gather(input->view(), {1, 100, 0});
    auto const expected = LCW<T>{{
                                   {5},
                                   {},  // NULL.
                                   {{1, 2, 3, 4}, null_at(2)},
                                 },
                                 null_at(1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(*results));
    EXPECT_FALSE(cudf::has_nonempty_nulls(*results));
  }
  {
    // Test when gather selects unsanitized row specifically.
    auto const results            = gather(input->view(), {2});
    auto const results_lists_view = cudf::lists_column_view(*results);
    auto const expected           = LCW<T>{{
                                             LCW<T>{}  // NULL.
                                           },
                                           null_at(0)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_lists_view.offsets(), offsets_col_t{0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_lists_view.child(), values_col_t{});
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(*results));
    EXPECT_FALSE(cudf::has_nonempty_nulls(*results));
  }
}

// List<List<T>>.
TEST_F(PurgeNonEmptyNullsTest, TwoLevelList)
{
  auto const input =
    LCW<T>{
      {{{1, 2, 3}, {4, 5, 6, 7}, {8}, {9, 1}, {2}},
       {{11, 12}, {13, 14, 15}, {16, 17, 18}, {19}},
       {{21}, {22, 23}, {24, 25, 26}},
       {{31, 32}, {33, 34, 35, 36}, {}, {37, 38}},  //<--- Will be set to NULL. Unsanitized row.
       {{41}, {42, 43}}},
      no_nulls()}
      .release();
  EXPECT_FALSE(cudf::may_have_nonempty_nulls(*input));
  EXPECT_FALSE(cudf::has_nonempty_nulls(*input));

  // Set nullmask, post construction.
  cudf::set_null_mask(input->mutable_view().null_mask(), 3, 4, false);
  input->set_null_count(1);
  EXPECT_TRUE(cudf::may_have_nonempty_nulls(*input));
  EXPECT_TRUE(cudf::has_nonempty_nulls(*input));

  test_purge(*input);

  {
    // Verify that gather() output is sanitized.
    auto const results            = gather(input->view(), {100, 3, 0, 1});
    auto const results_lists_view = cudf::lists_column_view(*results);

    auto const expected = LCW<T>{{
                                   LCW<T>{},  // NULL, because of out of bounds.
                                   LCW<T>{},  // NULL, because input row was null.
                                   {{1, 2, 3}, {4, 5, 6, 7}, {8}, {9, 1}, {2}},  // i.e. input[0]
                                   {{11, 12}, {13, 14, 15}, {16, 17, 18}, {19}}  // i.e. input[1]
                                 },
                                 nulls_at({0, 1})};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_lists_view.offsets(), offsets_col_t{0, 0, 0, 5, 9});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results_lists_view.child(),
      LCW<T>{
        {1, 2, 3}, {4, 5, 6, 7}, {8}, {9, 1}, {2}, {11, 12}, {13, 14, 15}, {16, 17, 18}, {19}});

    auto const child_lists_view = cudf::lists_column_view(results_lists_view.child());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(child_lists_view.offsets(),
                                   offsets_col_t{0, 3, 7, 8, 10, 11, 13, 16, 19, 20});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      child_lists_view.child(),
      values_col_t{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19});
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(*results));
    EXPECT_FALSE(cudf::has_nonempty_nulls(*results));
  }
}

// List<List<List<T>>>.
TEST_F(PurgeNonEmptyNullsTest, ThreeLevelList)
{
  auto const input = LCW<T>{{{{{1, 2}, {3}}, {{4, 5}, {6, 7}}, {{8, 8}, {}}, {{9, 1}}, {{2, 3}}},
                             {{{11, 12}}, {{13}, {14, 15}}, {{16, 17, 18}}, {{19, 19}, {}}},
                             {{{21, 21}}, {{22, 23}, {}}, {{24, 25}, {26}}},
                             {{{31, 32}, {}},
                              {{33, 34, 35}, {36}},
                              {},
                              {{37, 38}}},  //<--- Will be set to NULL. Unsanitized row.
                             {{{41, 41, 41}}, {{42, 43}}}},
                            no_nulls()}
                       .release();
  EXPECT_FALSE(cudf::may_have_nonempty_nulls(*input));
  EXPECT_FALSE(cudf::has_nonempty_nulls(*input));

  // Set nullmask, post construction.
  cudf::set_null_mask(input->mutable_view().null_mask(), 3, 4, false);
  input->set_null_count(1);
  EXPECT_TRUE(cudf::may_have_nonempty_nulls(*input));
  EXPECT_TRUE(cudf::has_nonempty_nulls(*input));

  test_purge(*input);

  {
    auto const results            = gather(input->view(), {100, 3, 0, 1});
    auto const results_lists_view = cudf::lists_column_view(*results);

    auto const expected = LCW<T>{
      {
        LCW<T>{},  // NULL, because of out of bounds.
        LCW<T>{},  // NULL, because input row was null.
        {{{1, 2}, {3}}, {{4, 5}, {6, 7}}, {{8, 8}, {}}, {{9, 1}}, {{2, 3}}},  // i.e. input[0]
        {{{11, 12}}, {{13}, {14, 15}}, {{16, 17, 18}}, {{19, 19}, {}}}        // i.e. input[1]
      },
      nulls_at({0, 1})};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_lists_view.offsets(), offsets_col_t{0, 0, 0, 5, 9});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_lists_view.child(),
                                   LCW<T>{{{1, 2}, {3}},
                                          {{4, 5}, {6, 7}},
                                          {{8, 8}, {}},
                                          {{9, 1}},
                                          {{2, 3}},
                                          {{11, 12}},
                                          {{13}, {14, 15}},
                                          {{16, 17, 18}},
                                          {{19, 19}, {}}});
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(*results));
    EXPECT_FALSE(cudf::has_nonempty_nulls(*results));
  }
}

// List<string>.
TEST_F(PurgeNonEmptyNullsTest, ListOfStrings)
{
  using T = cudf::string_view;

  auto const input = LCW<T>{{{{"1", "22", "", "4444"}, null_at(2)},
                             {"55555"},
                             {"666666", "7777777"},  // <--- Will be set to NULL. Unsanitized row.
                             {"88888888", "999999999", "1010101010"},
                             {"11", "22", "33", "44"},
                             {"55", "66", "77", "88"}},
                            no_nulls()}
                       .release();
  EXPECT_TRUE(cudf::may_have_nonempty_nulls(*input));
  EXPECT_FALSE(cudf::has_nonempty_nulls(*input));

  // Set nullmask, post construction.
  cudf::set_null_mask(input->mutable_view().null_mask(), 2, 3, false);
  input->set_null_count(1);
  EXPECT_TRUE(cudf::may_have_nonempty_nulls(*input));
  EXPECT_TRUE(cudf::has_nonempty_nulls(*input));

  test_purge(*input);

  {
    // Selecting all rows from input, in different order.
    auto const results           = gather(input->view(), {1, 2, 0, 3});
    auto const results_list_view = cudf::lists_column_view(*results);

    auto const expected = LCW<T>{{{"55555"},
                                  {},  // NULL.
                                  {{"1", "22", "", "4444"}, null_at(2)},
                                  {"88888888", "999999999", "1010101010"}},
                                 null_at(1)};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_list_view.offsets(), offsets_col_t{0, 1, 1, 5, 8});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results_list_view.child(),
      cudf::test::strings_column_wrapper{
        {"55555", "1", "22", "", "4444", "88888888", "999999999", "1010101010"}, null_at(3)});
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(*results));
    EXPECT_FALSE(cudf::has_nonempty_nulls(*results));
  }
  {
    // Gathering from a sliced column.
    auto const sliced = cudf::slice({input->view()}, {1, 5})[0];  // Lop off 1 row at each end.
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(sliced));
    EXPECT_TRUE(cudf::has_nonempty_nulls(sliced));

    auto const results           = gather(sliced, {1, 2, 0, 3});
    auto const results_list_view = cudf::lists_column_view(*results);
    auto const expected          = LCW<T>{{
                                            {},
                                            {"88888888", "999999999", "1010101010"},
                                            {"55555"},
                                            {"11", "22", "33", "44"},
                                          },
                                          null_at(0)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_list_view.offsets(), offsets_col_t{0, 0, 3, 4, 8});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      results_list_view.child(),
      cudf::test::strings_column_wrapper{
        "88888888", "999999999", "1010101010", "55555", "11", "22", "33", "44"});
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(*results));
    EXPECT_FALSE(cudf::has_nonempty_nulls(*results));
  }
}

// List<string>.
TEST_F(PurgeNonEmptyNullsTest, UnsanitizedListOfUnsanitizedStrings)
{
  auto strings =
    cudf::test::strings_column_wrapper{
      {"1", "22", "3", "44", "5", "66", "7", "8888", "9", "1010"},  //<--- "8888" will be
                                                                    // unsanitized.
      no_nulls()}
      .release();
  EXPECT_FALSE(cudf::may_have_nonempty_nulls(*strings));
  EXPECT_FALSE(cudf::has_nonempty_nulls(*strings));

  // Set strings nullmask, post construction.
  cudf::set_null_mask(strings->mutable_view().null_mask(), 7, 8, false);
  strings->set_null_count(1);
  EXPECT_TRUE(cudf::may_have_nonempty_nulls(*strings));
  EXPECT_TRUE(cudf::has_nonempty_nulls(*strings));

  test_purge(*strings);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::strings_column_view(*strings).offsets(),
                                 offsets_col_t{0, 1, 3, 4, 6, 7, 9, 10, 14, 15, 19}
                                 // 10-14 indicates that "8888" is unsanitized.
  );

  // Construct a list column from the strings column.
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(no_nulls(), no_nulls() + 4);
  auto const lists             = cudf::make_lists_column(4,
                                                         offsets_col_t{0, 4, 5, 7, 10}.release(),
                                                         std::move(strings),
                                                         null_count,
                                                         std::move(null_mask));
  EXPECT_TRUE(cudf::may_have_nonempty_nulls(*lists));
  EXPECT_TRUE(cudf::has_nonempty_nulls(*lists));

  // Set lists nullmask, post construction.
  cudf::set_null_mask(lists->mutable_view().null_mask(), 2, 3, false);
  lists->set_null_count(1);
  EXPECT_TRUE(cudf::may_have_nonempty_nulls(*lists));
  EXPECT_TRUE(cudf::has_nonempty_nulls(*lists));

  test_purge(*lists);

  // At this point,
  // 1. {"66", "7"} will be unsanitized.
  // 2. {"8888", "9", "1010"} will be actually be {NULL, "9", "1010"}.

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    cudf::lists_column_view(*lists).offsets(),
    offsets_col_t{0, 4, 5, 7, 10});  // 5-7 indicates that list row#2 is unsanitized.

  auto const result   = gather(lists->view(), {1, 2, 0, 3});
  auto const expected = LCW<cudf::string_view>{{{"5"},
                                                {},  // NULL.
                                                {"1", "22", "3", "44"},
                                                {{"", "9", "1010"}, null_at(0)}},
                                               null_at(1)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);

  // Ensure row#2 has been sanitized.
  auto const results_lists_view = cudf::lists_column_view(*result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_lists_view.offsets(), offsets_col_t{0, 1, 1, 5, 8}
                                 // 1-1 indicates that row#2 is sanitized.
  );

  // Ensure that "8888" has been sanitized, and stored as "".
  auto const child_strings_view = cudf::strings_column_view(results_lists_view.child());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(child_strings_view.offsets(),
                                 offsets_col_t{0, 1, 2, 4, 5, 7, 7, 8, 12});
  EXPECT_TRUE(cudf::may_have_nonempty_nulls(*result));
  EXPECT_FALSE(cudf::has_nonempty_nulls(*result));
}

// Struct<List<T>>.
TEST_F(PurgeNonEmptyNullsTest, StructOfList)
{
  auto const structs_input = [] {
    auto child = LCW<T>{{{{1, 2, 3, 4}, null_at(2)},
                         {5},
                         {6, 7},  //<--- Unsanitized row.
                         {8, 9, 10}},
                        no_nulls()};
    EXPECT_FALSE(cudf::has_nonempty_nulls(child));
    return cudf::test::structs_column_wrapper{{child}}.release();
  }();
  auto [null_mask, null_count] = [&] {
    auto const valid_iter = null_at(2);
    return cudf::test::detail::make_null_mask(valid_iter, valid_iter + structs_input->size());
  }();

  // Manually set the null mask for the columns, leaving the null at list index 2 unsanitized.
  structs_input->child(0).set_null_mask(null_mask, null_count, cudf::get_default_stream());
  structs_input->set_null_mask(std::move(null_mask), null_count);

  EXPECT_TRUE(cudf::may_have_nonempty_nulls(*structs_input));
  EXPECT_TRUE(cudf::has_nonempty_nulls(*structs_input));

  test_purge(*structs_input);

  // At this point, even though the structs column has a null at index 2,
  // the child column has a non-empty list row at index 2: {6, 7}.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::lists_column_view(structs_input->child(0)).child(),
                                 values_col_t{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, null_at(2)});

  {
    // Test rearrange.
    auto const gather_map      = gather_map_t{1, 2, 0, 3};
    auto const result          = gather(structs_input->view(), gather_map);
    auto const expected_result = [] {
      auto child = LCW<T>{{{5},
                           LCW<T>{},  //<--- Now, sanitized.
                           {{1, 2, 3, 4}, null_at(2)},
                           {8, 9, 10}},
                          null_at(1)};
      return cudf::test::structs_column_wrapper{{child}, null_at(1)};
    }();

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected_result);
    auto const results_child = cudf::lists_column_view(result->child(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_child.offsets(), offsets_col_t{0, 1, 1, 5, 8});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results_child.child(),
                                   values_col_t{{5, 1, 2, 3, 4, 8, 9, 10}, null_at(3)});
    EXPECT_TRUE(cudf::may_have_nonempty_nulls(*result));
    EXPECT_FALSE(cudf::has_nonempty_nulls(*result));
  }
}
