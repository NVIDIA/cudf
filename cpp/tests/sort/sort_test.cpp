/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <thrust/host_vector.h>

#include <type_traits>
#include <vector>

void run_sort_test(cudf::table_view input,
                   cudf::column_view expected_sorted_indices,
                   std::vector<cudf::order> column_order         = {},
                   std::vector<cudf::null_order> null_precedence = {})
{
  // Sorted table
  auto got_sorted_table      = cudf::sort(input, column_order, null_precedence);
  auto expected_sorted_table = cudf::gather(input, expected_sorted_indices);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sorted_table->view(), got_sorted_table->view());

  // Sorted by key
  auto got_sort_by_key_table      = cudf::sort_by_key(input, input, column_order, null_precedence);
  auto expected_sort_by_key_table = cudf::gather(input, expected_sorted_indices);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sort_by_key_table->view(), got_sort_by_key_table->view());
}

using TestTypes = cudf::test::Concat<cudf::test::NumericTypes,  // include integers, floats and bool
                                     cudf::test::ChronoTypes>;  // include timestamps and durations

template <typename T>
struct Sort : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(Sort, TestTypes);

TYPED_TEST(Sort, WithNullMax)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8, 5}, {1, 1, 0, 1, 1, 1}};
  cudf::test::strings_column_wrapper col2({"d", "e", "", "d", "k", "d"}, {1, 1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2, 10}, {1, 1, 0, 1, 1, 1}};
  cudf::table_view input{{col1, col2, col3}};

  cudf::test::fixed_width_column_wrapper<int32_t> expected{{1, 0, 5, 3, 4, 2}};
  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{
    cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER};

  // Sorted order
  auto got = cudf::sorted_order(input, column_order, null_precedence);

  if (!std::is_same_v<T, bool>) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

    // Run test for sort and sort_by_key
    run_sort_test(input, expected, column_order, null_precedence);
  } else {
    // for bools only validate that the null element landed at the back, since
    // the rest of the values are equivalent and yields random sorted order.
    auto const last_idx = got->size();
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::slice(got->view(), {last_idx - 1, last_idx}).back(),
                                   cudf::slice(expected, {last_idx - 1, last_idx}).back());

    // Run test for sort and sort_by_key
    cudf::test::fixed_width_column_wrapper<int32_t> expected_for_bool{{0, 3, 5, 1, 4, 2}};
    run_sort_test(input, expected_for_bool, column_order, null_precedence);
  }
}

TYPED_TEST(Sort, WithNullMin)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}, {1, 1, 0, 1, 1}};
  cudf::test::strings_column_wrapper col2({"d", "e", "", "d", "k"}, {1, 1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}, {1, 1, 0, 1, 1}};
  cudf::table_view input{{col1, col2, col3}};

  cudf::test::fixed_width_column_wrapper<int32_t> expected{{2, 1, 0, 3, 4}};
  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING};

  auto got = cudf::sorted_order(input, column_order);

  if (!std::is_same_v<T, bool>) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

    // Run test for sort and sort_by_key
    run_sort_test(input, expected, column_order);
  } else {
    // for bools only validate that the null element landed at the front, since
    // the rest of the values are equivalent and yields random sorted order.
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::slice(got->view(), {0, 1}).front(),
                                   cudf::slice(expected, {0, 1}).front());

    // Run test for sort and sort_by_key
    cudf::test::fixed_width_column_wrapper<int32_t> expected_for_bool{{2, 0, 3, 1, 4}};
    run_sort_test(input, expected_for_bool, column_order);
  }
}

TYPED_TEST(Sort, WithMixedNullOrder)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}, {0, 0, 1, 1, 0}};
  cudf::test::strings_column_wrapper col2({"", "e", "", "", "k"}, {0, 1, 0, 0, 1});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}, {1, 0, 1, 0, 1}};
  cudf::table_view input{{col1, col2, col3}};

  cudf::test::fixed_width_column_wrapper<int32_t> expected{{2, 3, 0, 1, 4}};
  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{
    cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};

  auto got = cudf::sorted_order(input, column_order, null_precedence);

  if (!std::is_same_v<T, bool>) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
  } else {
    // for bools only validate that the null element landed at the front, since
    // the rest of the values are equivalent and yields random sorted order.
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::slice(got->view(), {0, 1}).front(),
                                   cudf::slice(expected, {0, 1}).front());
  }

  // Run test for sort and sort_by_key
  run_sort_test(input, expected, column_order, null_precedence);
}

TYPED_TEST(Sort, WithAllValid)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  cudf::table_view input{{col1, col2, col3}};

  cudf::test::fixed_width_column_wrapper<int32_t> expected{{2, 1, 0, 3, 4}};
  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING};

  auto got = cudf::sorted_order(input, column_order);

  // Skip validating bools order. Valid true bools are all
  // equivalent, and yield random order after thrust::sort
  if (!std::is_same_v<T, bool>) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

    // Run test for sort and sort_by_key
    run_sort_test(input, expected, column_order);
  } else {
    // Run test for sort and sort_by_key
    cudf::test::fixed_width_column_wrapper<int32_t> expected_for_bool{{2, 0, 3, 1, 4}};
    run_sort_test(input, expected_for_bool, column_order);
  }
}

TYPED_TEST(Sort, WithStructColumn)
{
  using T = TypeParam;

  std::initializer_list<std::string> names = {"Samuel Vimes",
                                              "Carrot Ironfoundersson",
                                              "Angua von Überwald",
                                              "Cheery Littlebottom",
                                              "Detritus",
                                              "Mr Slant"};
  auto num_rows{std::distance(names.begin(), names.end())};
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};
  auto ages_col  = cudf::test::fixed_width_column_wrapper<T, int32_t>{{48, 27, 25, 31, 351, 351}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_col =
    cudf::test::structs_column_wrapper{{names_col, ages_col, is_human_col}}.release();
  auto struct_col_view{struct_col->view()};
  EXPECT_EQ(num_rows, struct_col->size());

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8, 9}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k", "a"});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2, 20}};
  cudf::table_view input{{col1, col2, col3, struct_col_view}};

  cudf::test::fixed_width_column_wrapper<int32_t> expected{{2, 1, 0, 3, 4, 5}};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING,
                                        cudf::order::ASCENDING,
                                        cudf::order::DESCENDING,
                                        cudf::order::ASCENDING};

  auto got = cudf::sorted_order(input, column_order);

  // Skip validating bools order. Valid true bools are all
  // equivalent, and yield random order after thrust::sort
  if (!std::is_same_v<T, bool>) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

    // Run test for sort and sort_by_key
    run_sort_test(input, expected, column_order);
  } else {
    // Run test for sort and sort_by_key
    cudf::test::fixed_width_column_wrapper<int32_t> expected_for_bool{{2, 5, 3, 0, 1, 4}};
    run_sort_test(input, expected_for_bool, column_order);
  }
}

TYPED_TEST(Sort, WithNestedStructColumn)
{
  using T = TypeParam;

  std::initializer_list<std::string> names = {"Samuel Vimes",
                                              "Carrot Ironfoundersson",
                                              "Angua von Überwald",
                                              "Cheery Littlebottom",
                                              "Detritus",
                                              "Mr Slant"};
  std::vector<bool> v{1, 1, 0, 1, 1, 0};
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};
  auto ages_col  = cudf::test::fixed_width_column_wrapper<T, int32_t>{{48, 27, 25, 31, 351, 351}};
  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};
  auto struct_col1 = cudf::test::structs_column_wrapper{{names_col, ages_col, is_human_col}, v};

  auto ages_col2   = cudf::test::fixed_width_column_wrapper<T, int32_t>{{48, 27, 25, 31, 351, 351}};
  auto struct_col2 = cudf::test::structs_column_wrapper{{ages_col2, struct_col1}}.release();

  auto struct_col_view{struct_col2->view()};

  cudf::test::fixed_width_column_wrapper<T> col1{{6, 6, 6, 6, 6, 6}};
  cudf::test::fixed_width_column_wrapper<T> col2{{1, 1, 1, 2, 2, 2}};
  cudf::table_view input{{col1, col2, struct_col_view}};

  cudf::test::fixed_width_column_wrapper<int32_t> expected{{3, 5, 4, 2, 1, 0}};
  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};

  auto got = cudf::sorted_order(input, column_order);

  // Skip validating bools order. Valid true bools are all
  // equivalent, and yield random order after thrust::sort
  if (!std::is_same_v<T, bool>) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

    // Run test for sort and sort_by_key
    run_sort_test(input, expected, column_order);
  } else {
    // Run test for sort and sort_by_key
    cudf::test::fixed_width_column_wrapper<int32_t> expected_for_bool{{2, 5, 1, 3, 4, 0}};
    run_sort_test(input, expected_for_bool, column_order);
  }
}

TYPED_TEST(Sort, WithNullableStructColumn)
{
  // Test for a struct column that has nulls on struct layer but not pushed down on the child
  using T    = int;
  using fwcw = cudf::test::fixed_width_column_wrapper<T>;
  using mask = std::vector<bool>;

  auto make_struct = [&](std::vector<std::unique_ptr<cudf::column>> child_cols,
                         std::vector<bool> nulls) {
    cudf::test::structs_column_wrapper struct_col(std::move(child_cols));
    auto struct_                 = struct_col.release();
    auto [null_mask, null_count] = cudf::test::detail::make_null_mask(nulls.begin(), nulls.end());
    struct_->set_null_mask(std::move(null_mask), null_count);
    return struct_;
  };

  {
    /*
         /+-------------+
         |s1{s2{a,b}, c}|
         +--------------+
       0 |  { {1, 1}, 5}|
       1 |  { {1, 2}, 4}|
       2 |  {@{2, 1}, 6}|
       3 |  {@{2, 2}, 5}|
       4 | @{ {2, 2}, 3}|
       5 | @{ {1, 1}, 3}|
       6 |  { {1, 2}, 3}|
       7 |  {@{1, 1}, 4}|
       8 |  { {2, 1}, 5}|
         +--------------+

      Intermediate representation:
      s1{s2{a}}, b, c
    */

    auto col_a   = fwcw{1, 1, 2, 2, 2, 1, 1, 1, 2};
    auto col_b   = fwcw{1, 2, 1, 2, 2, 1, 2, 1, 1};
    auto s2_mask = mask{1, 1, 0, 0, 1, 1, 1, 0, 1};
    auto col_c   = fwcw{5, 4, 6, 5, 3, 3, 3, 4, 5};
    auto s1_mask = mask{1, 1, 1, 1, 0, 0, 1, 1, 1};

    std::vector<std::unique_ptr<cudf::column>> s2_children;
    s2_children.push_back(col_a.release());
    s2_children.push_back(col_b.release());
    auto s2 = make_struct(std::move(s2_children), s2_mask);

    std::vector<std::unique_ptr<cudf::column>> s1_children;
    s1_children.push_back(std::move(s2));
    s1_children.push_back(col_c.release());
    auto s1 = make_struct(std::move(s1_children), s1_mask);

    auto expect = fwcw{4, 5, 7, 3, 2, 0, 6, 1, 8};
    run_sort_test(cudf::table_view({s1->view()}), expect);
  }
  { /*
        /+-------------+
        |s1{a,s2{b, c}}|
        +--------------+
      0 |  {1,  {1, 5}}|
      1 |  {1,  {2, 4}}|
      2 |  {2, @{2, 6}}|
      3 |  {2, @{1, 5}}|
      4 | @{2,  {2, 3}}|
      5 | @{1,  {1, 3}}|
      6 |  {1,  {2, 3}}|
      7 |  {1, @{1, 4}}|
      8 |  {2,  {1, 5}}|
        +--------------+

     Intermediate representation:
     s1{a}, s2{b}, c
   */

    auto s1_mask = mask{1, 1, 1, 1, 0, 0, 1, 1, 1};
    auto col_a   = fwcw{1, 1, 2, 2, 2, 1, 1, 1, 2};
    auto s2_mask = mask{1, 1, 0, 0, 1, 1, 1, 0, 1};
    auto col_b   = fwcw{1, 2, 1, 2, 2, 1, 2, 1, 1};
    auto col_c   = fwcw{5, 4, 6, 5, 3, 3, 3, 4, 5};

    std::vector<std::unique_ptr<cudf::column>> s22_children;
    s22_children.push_back(col_b.release());
    s22_children.push_back(col_c.release());
    auto s22 = make_struct(std::move(s22_children), s2_mask);

    std::vector<std::unique_ptr<cudf::column>> s12_children;
    s12_children.push_back(col_a.release());
    s12_children.push_back(std::move(s22));
    auto s12 = make_struct(std::move(s12_children), s1_mask);

    auto expect = fwcw{4, 5, 7, 0, 6, 1, 2, 3, 8};
    run_sort_test(cudf::table_view({s12->view()}), expect);
  }
}

TYPED_TEST(Sort, WithSingleStructColumn)
{
  using T = TypeParam;

  std::initializer_list<std::string> names = {"Samuel Vimes",
                                              "Carrot Ironfoundersson",
                                              "Angua von Überwald",
                                              "Cheery Littlebottom",
                                              "Detritus",
                                              "Mr Slant"};
  std::vector<bool> v{1, 1, 0, 1, 1, 0};
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};
  auto ages_col  = cudf::test::fixed_width_column_wrapper<T, int32_t>{{48, 27, 25, 31, 351, 351}};
  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};
  auto struct_col =
    cudf::test::structs_column_wrapper{{names_col, ages_col, is_human_col}, v}.release();
  auto struct_col_view{struct_col->view()};
  cudf::table_view input{{struct_col_view}};

  cudf::test::fixed_width_column_wrapper<int32_t> expected{{2, 5, 1, 3, 4, 0}};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};

  auto got = cudf::sorted_order(input, column_order);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  // Run test for sort and sort_by_key
  run_sort_test(input, expected, column_order);
}

TYPED_TEST(Sort, WithSlicedStructColumn)
{
  using T = TypeParam;
  /*
       /+-------------+
       |             s|
       +--------------+
     0 | {"bbe", 1, 7}|
     1 | {"bbe", 1, 8}|
     2 | {"aaa", 0, 1}|
     3 | {"abc", 0, 1}|
     4 | {"ab",  0, 9}|
     5 | {"za",  2, 5}|
     6 | {"b",   1, 7}|
     7 | { @,    3, 3}|
       +--------------+
  */
  // clang-format off
  using FWCW = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  std::vector<bool>             string_valids{    1,     1,     1,     1,    1,    1,   1,   0};
  std::initializer_list<std::string> names = {"bbe", "bbe", "aaa", "abc", "ab", "za", "b", ""};
  auto col2 =                           FWCW{{    1,     1,     0,     0,    0,    2,   1,   3}};
  auto col3 =                           FWCW{{    7,     8,     1,     1,    9,    5,   7,   3}};
  auto col1 = cudf::test::strings_column_wrapper{names.begin(), names.end(), string_valids.begin()};
  auto struct_col = cudf::test::structs_column_wrapper{{col1, col2, col3}}.release();
  // clang-format on
  auto struct_col_view{struct_col->view()};
  cudf::table_view input{{struct_col_view}};
  auto sliced_columns = cudf::split(struct_col_view, std::vector<cudf::size_type>{3});
  auto sliced_tables  = cudf::split(input, std::vector<cudf::size_type>{3});
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  /*
        asce_null_first   sliced[3:]
      /+-------------+
      |             s|
      +--------------+
    7 | { @,    3, 3}|   7=4
    2 | {"aaa", 0, 1}|
    4 | {"ab",  0, 9}|   4=1
    3 | {"abc", 0, 1}|   3=0
    6 | {"b",   1, 7}|   6=3
    0 | {"bbe", 1, 7}|
    1 | {"bbe", 1, 8}|
    5 | {"za",  2, 5}|   5=2
      +--------------+
  */

  // normal
  cudf::test::fixed_width_column_wrapper<int32_t> expected{{7, 2, 4, 3, 6, 0, 1, 5}};
  auto got = cudf::sorted_order(input, column_order);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input, expected, column_order);

  // table with sliced column
  cudf::table_view input2{{sliced_columns[1]}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected2{{4, 1, 0, 3, 2}};
  got = cudf::sorted_order(input2, column_order);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input2, expected2, column_order);

  // sliced table[1]
  cudf::test::fixed_width_column_wrapper<int32_t> expected3{{4, 1, 0, 3, 2}};
  got = cudf::sorted_order(sliced_tables[1], column_order);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(sliced_tables[1], expected3, column_order);

  // sliced table[0]
  cudf::test::fixed_width_column_wrapper<int32_t> expected4{{2, 0, 1}};
  got = cudf::sorted_order(sliced_tables[0], column_order);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected4, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(sliced_tables[0], expected4, column_order);
}

TYPED_TEST(Sort, SlicedColumns)
{
  using T    = TypeParam;
  using FWCW = cudf::test::fixed_width_column_wrapper<T, int32_t>;

  // clang-format off
  std::vector<bool>             string_valids{    1,     1,     1,     1,    1,    1,   1,   0};
  std::initializer_list<std::string> names = {"bbe", "bbe", "aaa", "abc", "ab", "za", "b", ""};
  auto col2 =                           FWCW{{    7,     8,     1,     1,    9,    5,   7,   3}};
  auto col1 = cudf::test::strings_column_wrapper{names.begin(), names.end(), string_valids.begin()};
  // clang-format on
  cudf::table_view input{{col1, col2}};
  auto sliced_columns1 = cudf::split(col1, std::vector<cudf::size_type>{3});
  auto sliced_columns2 = cudf::split(col1, std::vector<cudf::size_type>{3});
  auto sliced_tables   = cudf::split(input, std::vector<cudf::size_type>{3});
  std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::ASCENDING};

  // normal
  // cudf::test::fixed_width_column_wrapper<int32_t> expected{{2, 3, 7, 5, 0, 6, 1, 4}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected{{7, 2, 4, 3, 6, 0, 1, 5}};
  auto got = cudf::sorted_order(input, column_order);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input, expected, column_order);

  // table with sliced column
  cudf::table_view input2{{sliced_columns1[1], sliced_columns2[1]}};
  // cudf::test::fixed_width_column_wrapper<int32_t> expected2{{0, 4, 2, 3, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected2{{4, 1, 0, 3, 2}};
  got = cudf::sorted_order(input2, column_order);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input2, expected2, column_order);
}

TYPED_TEST(Sort, WithStructColumnCombinations)
{
  using T    = TypeParam;
  using FWCW = cudf::test::fixed_width_column_wrapper<T, int32_t>;

  // clang-format off
  /*
    +------------+
    |           s|
    +------------+
  0 |   {0, null}|
  1 |   {1, null}|
  2 |        null|
  3 |{null, null}|
  4 |        null|
  5 |{null, null}|
  6 |   {null, 1}|
  7 |   {null, 0}|
    +------------+
  */
  std::vector<bool>                           struct_valids{1, 1, 0, 1, 0, 1, 1, 1};
  auto col1       = FWCW{{ 0,  1,  9, -1,  9, -1, -1, -1}, {1, 1, 1, 0, 1, 0, 0, 0}};
  auto col2       = FWCW{{-1, -1,  9, -1,  9, -1,  1,  0}, {0, 0, 1, 0, 1, 0, 1, 1}};
  auto struct_col = cudf::test::structs_column_wrapper{{col1, col2}, struct_valids}.release();
  /*
    desc_nulls_first     desc_nulls_last     asce_nulls_first     asce_nulls_last
    +------------+       +------------+      +------------+       +------------+
    |           s|       |           s|      |           s|       |           s|
    +------------+       +------------+      +------------+       +------------+
  2 |        null|     1 |   {1, null}|    2 |        null|     0 |   {0, null}|
  4 |        null|     0 |   {0, null}|    4 |        null|     1 |   {1, null}|
  3 |{null, null}|     6 |   {null, 1}|    3 |{null, null}|     7 |   {null, 0}|
  5 |{null, null}|     7 |   {null, 0}|    5 |{null, null}|     6 |   {null, 1}|
  6 |   {null, 1}|     3 |{null, null}|    7 |   {null, 0}|     3 |{null, null}|
  7 |   {null, 0}|     5 |{null, null}|    6 |   {null, 1}|     5 |{null, null}|
  1 |   {1, null}|     2 |        null|    0 |   {0, null}|     2 |        null|
  0 |   {0, null}|     4 |        null|    1 |   {1, null}|     4 |        null|
    +------------+       +------------+      +------------+       +------------+
  */
  // clang-format on
  auto struct_col_view{struct_col->view()};
  cudf::table_view input{{struct_col_view}};
  std::vector<cudf::order> column_order1{cudf::order::DESCENDING};

  // desc_nulls_first
  cudf::test::fixed_width_column_wrapper<int32_t> expected1{{2, 4, 3, 5, 6, 7, 1, 0}};
  auto got = cudf::sorted_order(input, column_order1, {cudf::null_order::AFTER});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input, expected1, column_order1, {cudf::null_order::AFTER});

  // desc_nulls_last
  cudf::test::fixed_width_column_wrapper<int32_t> expected2{{1, 0, 6, 7, 3, 5, 2, 4}};
  got = cudf::sorted_order(input, column_order1, {cudf::null_order::BEFORE});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input, expected2, column_order1, {cudf::null_order::BEFORE});

  // asce_nulls_first
  std::vector<cudf::order> column_order2{cudf::order::ASCENDING};
  cudf::test::fixed_width_column_wrapper<int32_t> expected3{{2, 4, 3, 5, 7, 6, 0, 1}};
  got = cudf::sorted_order(input, column_order2, {cudf::null_order::BEFORE});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input, expected3, column_order2, {cudf::null_order::BEFORE});

  // asce_nulls_last
  cudf::test::fixed_width_column_wrapper<int32_t> expected4{{0, 1, 7, 6, 3, 5, 2, 4}};
  got = cudf::sorted_order(input, column_order2, {cudf::null_order::AFTER});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected4, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input, expected4, column_order2, {cudf::null_order::AFTER});
}

TYPED_TEST(Sort, WithStructColumnCombinationsWithoutNulls)
{
  using T    = TypeParam;
  using FWCW = cudf::test::fixed_width_column_wrapper<T, int32_t>;

  // clang-format off
  /*
    +------------+
    |           s|
    +------------+
  0 |   {0, null}|
  1 |   {1, null}|
  2 |      {9, 9}|
  3 |{null, null}|
  4 |      {9, 9}|
  5 |{null, null}|
  6 |   {null, 1}|
  7 |   {null, 0}|
    +------------+
  */
  auto col1       = FWCW{{ 0,  1,  9, -1,  9, -1, -1, -1}, {1, 1, 1, 0, 1, 0, 0, 0}};
  auto col2       = FWCW{{-1, -1,  9, -1,  9, -1,  1,  0}, {0, 0, 1, 0, 1, 0, 1, 1}};
  auto struct_col = cudf::test::structs_column_wrapper{{col1, col2}}.release();
  /* (nested columns are always nulls_first, spark requirement)
    desc_nulls_*        asce_nulls_*
    +------------+      +------------+
    |           s|      |           s|
    +------------+      +------------+
  3 |{null, null}|    0 |   {0, null}|
  5 |{null, null}|    1 |   {1, null}|
  6 |   {null, 1}|    2 |      {9, 9}|
  7 |   {null, 0}|    4 |      {9, 9}|
  2 |      {9, 9}|    7 |   {null, 0}|
  4 |      {9, 9}|    6 |   {null, 1}|
  1 |   {1, null}|    3 |{null, null}|
  0 |   {0, null}|    5 |{null, null}|
    +------------+      +------------+
  */
  // clang-format on
  auto struct_col_view{struct_col->view()};
  cudf::table_view input{{struct_col_view}};
  std::vector<cudf::order> column_order{cudf::order::DESCENDING};

  // desc_nulls_first
  auto const expected1 = []() {
    if constexpr (std::is_same_v<T, bool>) {
      return cudf::test::fixed_width_column_wrapper<int32_t>{{3, 5, 6, 7, 1, 2, 4, 0}};
    }
    return cudf::test::fixed_width_column_wrapper<int32_t>{{3, 5, 6, 7, 2, 4, 1, 0}};
  }();
  auto got = cudf::sorted_order(input, column_order, {cudf::null_order::AFTER});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input, expected1, column_order, {cudf::null_order::AFTER});

  // desc_nulls_last
  cudf::test::fixed_width_column_wrapper<int32_t> expected2{{2, 4, 1, 0, 6, 7, 3, 5}};
  got = cudf::sorted_order(input, column_order, {cudf::null_order::BEFORE});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input, expected2, column_order, {cudf::null_order::BEFORE});

  // asce_nulls_first
  std::vector<cudf::order> column_order2{cudf::order::ASCENDING};
  cudf::test::fixed_width_column_wrapper<int32_t> expected3{{3, 5, 7, 6, 0, 1, 2, 4}};
  got = cudf::sorted_order(input, column_order2, {cudf::null_order::BEFORE});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input, expected3, column_order2, {cudf::null_order::BEFORE});

  // asce_nulls_last
  auto const expected4 = []() {
    if constexpr (std::is_same_v<T, bool>) {
      return cudf::test::fixed_width_column_wrapper<int32_t>{{0, 2, 4, 1, 7, 6, 3, 5}};
    }
    return cudf::test::fixed_width_column_wrapper<int32_t>{{0, 1, 2, 4, 7, 6, 3, 5}};
  }();
  got = cudf::sorted_order(input, column_order2, {cudf::null_order::AFTER});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected4, got->view());
  // Run test for sort and sort_by_key
  run_sort_test(input, expected4, column_order2, {cudf::null_order::AFTER});
}

TYPED_TEST(Sort, MismatchInColumnOrderSize)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  cudf::table_view input{{col1, col2, col3}};

  std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::DESCENDING};

  EXPECT_THROW(cudf::sorted_order(input, column_order), cudf::logic_error);
  EXPECT_THROW(cudf::sort(input, column_order), cudf::logic_error);
  EXPECT_THROW(cudf::sort_by_key(input, input, column_order), cudf::logic_error);
}

TYPED_TEST(Sort, MismatchInNullPrecedenceSize)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  cudf::table_view input{{col1, col2, col3}};

  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER, cudf::null_order::BEFORE};

  EXPECT_THROW(cudf::sorted_order(input, column_order, null_precedence), cudf::logic_error);
  EXPECT_THROW(cudf::sort(input, column_order, null_precedence), cudf::logic_error);
  EXPECT_THROW(cudf::sort_by_key(input, input, column_order, null_precedence), cudf::logic_error);
}

TYPED_TEST(Sort, ZeroSizedColumns)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col1{};
  cudf::table_view input{{col1}};

  cudf::test::fixed_width_column_wrapper<int32_t> expected{};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};

  auto got = cudf::sorted_order(input, column_order);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  // Run test for sort and sort_by_key
  run_sort_test(input, expected, column_order);
}

TYPED_TEST(Sort, WithListColumn)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) { GTEST_SKIP(); }

  /*
  [
    [[1, 2, 3], [], [4, 5], [], [0, 6, 0]], 0
    [[1, 2, 3], [], [4, 5], [], [0, 6, 0]], 1
    [[1, 2, 3], [], [4, 5], [0, 6, 0]],     2
    [[1, 2], [3], [4, 5], [0, 6, 0]],       3
    [[7, 8], []],                           4
    [[], [], []],                           5
    [[]],                                   6
    [[10]],                                 7
    []                                      8
  ]
  */

  using lcw = cudf::test::lists_column_wrapper<T, int32_t>;
  lcw col{{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
          {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},
          {{1, 2, 3}, {}, {4, 5}, {0, 6, 0}},
          {{1, 2}, {3}, {4, 5}, {0, 6, 0}},
          {{7, 8}, {}},
          lcw{lcw{}, lcw{}, lcw{}},
          lcw{lcw{}},
          {lcw{10}},
          lcw{}};

  auto expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{8, 6, 5, 3, 0, 1, 2, 4, 7};
  auto result = cudf::sorted_order(cudf::table_view({col}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, *result);
}

TYPED_TEST(Sort, WithNullableListColumn)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) { GTEST_SKIP(); }

  using lcw = cudf::test::lists_column_wrapper<T, int32_t>;
  using cudf::test::iterators::nulls_at;

  /*
  [
    [[1, 2, 3], [], [4, 5], [], [0, 6, 0]],   0
    [[1, 2, 3], [], [4, 5], NULL, [0, 6, 0]], 1
    [[1, 2, 3], [], [4, 5], [0, 6, 0]],       2
    [[1, 2], [3], [4, 5], [0, 6, 0]],         3
    [[1, 2], [3], [4, 5], [NULL, 6, 0]],      4
    [[7, 8], []],                             5
    [[], [], []],                             6
    [[]],                                     7
    [[10]],                                   8
    [],                                       9
    [[1, 2], [3], [4, 5], [NULL, 6, NULL]],   10
    [[1, 2], [3], [4, 5], [NULL, 7]]          11
  ]
  */

  lcw col{
    {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},                   // 0
    {{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, nulls_at({3})},  // 1
    {{1, 2, 3}, {}, {4, 5}, {0, 6, 0}},                       // 2
    {{1, 2}, {3}, {4, 5}, {0, 6, 0}},                         // 3
    {{1, 2}, {3}, {4, 5}, {{0, 6, 0}, nulls_at({0})}},        // 4
    {{7, 8}, {}},                                             // 5
    lcw{lcw{}, lcw{}, lcw{}},                                 // 6
    lcw{lcw{}},                                               // 7
    {lcw{10}},                                                // 8
    lcw{},                                                    // 9
    {{1, 2}, {3}, {4, 5}, {{0, 6, 0}, nulls_at({0, 2})}},     // 10
    {{1, 2}, {3}, {4, 5}, {{0, 7}, nulls_at({0})}},           // 11
  };

  auto expect =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{9, 7, 6, 10, 4, 11, 3, 1, 0, 2, 5, 8};
  auto result = cudf::sorted_order(cudf::table_view({col}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, *result);
}

TYPED_TEST(Sort, MoreLists)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) { GTEST_SKIP(); }

  using lcw = cudf::test::lists_column_wrapper<T, int32_t>;
  using cudf::test::iterators::null_at;
  using cudf::test::iterators::nulls_at;

  {
    /*
    [
      [[NULL], [-21827]], 0
      [[NULL, NULL]]      1
    ]
    */
    lcw col{
      lcw{lcw{{0}, nulls_at({0})}, lcw{-21827}},  // 0
      lcw{lcw{{0, 0}, nulls_at({0, 1})}}          // 1
    };
    cudf::test::fixed_width_column_wrapper<int32_t> expected{{0, 1}};
    auto result = cudf::sorted_order(cudf::table_view({col}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  {
    /*
    [
      [[1], [2]],         0
      [[1, NULL], [2, 3]] 1
    ]
    */
    auto const col = lcw{lcw{lcw{1}, lcw{2}}, lcw{lcw{{1, 0}, nulls_at({1})}, lcw{2, 3}}};
    cudf::test::fixed_width_column_wrapper<int32_t> expected{{0, 1}};
    auto result = cudf::sorted_order(cudf::table_view({col}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  {
    /*
    List<List<List<int>
    [
      [[[0, 0, 0]]],                                0
      [[[0], [0], [0]]],                            1
      [[[0]], [[0]], [[0]]],                        2
      [[[0, 0, 0]], [[0]], [[0]]],                  3
      [[[0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0]], [[0]]] 4
    ]
    */
    lcw col{lcw{lcw{lcw{0, 0, 0}}},
            lcw{lcw{lcw{0}, lcw{0}, lcw{0}}},
            lcw{lcw{lcw{0}}, lcw{lcw{0}}, lcw{lcw{0}}},
            lcw{lcw{lcw{0, 0, 0}}, lcw{lcw{0}}, lcw{lcw{0}}},
            lcw{lcw{lcw{0, 0}}, lcw{lcw{0, 0, 0, 0, 0, 0, 0, 0}}, lcw{lcw{0}}}};
    cudf::test::fixed_width_column_wrapper<int32_t> expected{{2, 1, 4, 0, 3}};
    auto result = cudf::sorted_order(cudf::table_view({col}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
  {
    /*
    [
      [],    0
      [1],   1
      [2, 2] 2
      [2, 3] 3
      []     4
      NULL   5
      [2]    6
      NULL   7
      [NULL] 8
    ]
    */
    lcw col{{{}, {1}, {2, 2}, {2, 3}, {}, {} /*NULL*/, {2}, {} /*NULL*/, {{0}, null_at(0)}},
            nulls_at({5, 7})};

    // ASCENDING, null_order::BEFORE
    {
      cudf::test::fixed_width_column_wrapper<int32_t> expected{{5, 7, 0, 4, 8, 1, 6, 2, 3}};
      auto result = cudf::sorted_order(cudf::table_view({col}));
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
    }
    // ASCENDING, null_order::AFTER
    {
      cudf::test::fixed_width_column_wrapper<int32_t> expected{{0, 4, 8, 1, 6, 2, 3, 5, 7}};
      auto result = cudf::sorted_order(cudf::table_view({col}), {}, {cudf::null_order::AFTER});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
    }
    // DESCENDING, null_order::BEFORE
    {
      cudf::test::fixed_width_column_wrapper<int32_t> expected{{3, 2, 6, 1, 8, 0, 4, 5, 7}};
      auto result = cudf::sorted_order(cudf::table_view({col}), {cudf::order::DESCENDING});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
    }
    // DESCENDING, null_order::AFTER
    {
      cudf::test::fixed_width_column_wrapper<int32_t> expected{{5, 7, 3, 2, 6, 1, 8, 0, 4}};
      auto result = cudf::sorted_order(
        cudf::table_view({col}), {cudf::order::DESCENDING}, {cudf::null_order::AFTER});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
    }
  }
  {
    lcw col{lcw{lcw{}, lcw{-729297378, -627961465}},
            lcw{lcw{{0}, null_at(0)}, lcw{881899016, -1415270016}}};

    {
      cudf::test::fixed_width_column_wrapper<int32_t> expected{{0, 1}};
      auto result = cudf::sorted_order(cudf::table_view({col}));
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
    }
  }
}

TYPED_TEST(Sort, WithSlicedListColumn)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) { GTEST_SKIP(); }

  using lcw = cudf::test::lists_column_wrapper<T, int32_t>;
  using cudf::test::iterators::nulls_at;
  lcw col{
    {{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}},                   //
    {{{1, 2, 3}, {}, {4, 5}, {}, {0, 6, 0}}, nulls_at({3})},  // 0
    {{1, 2, 3}, {}, {4, 5}, {0, 6, 0}},                       // 1
    {{1, 2}, {3}, {4, 5}, {0, 6, 0}},                         // 2
    {{1, 2}, {3}, {4, 5}, {{0, 6, 0}, nulls_at({0})}},        // 3
    {{7, 8}, {}},                                             // 4
    lcw{lcw{}, lcw{}, lcw{}},                                 // 5
    lcw{lcw{}},                                               // 6
    {lcw{10}},                                                // 7
    lcw{},                                                    // 8
    {{1, 2}, {3}, {4, 5}, {{0, 6, 0}, nulls_at({0, 2})}},     // 9
    {{1, 2}, {3}, {4, 5}, {{0, 7}, nulls_at({0})}},           //
  };

  auto sliced_col = cudf::slice(col, {1, 10});

  auto expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{8, 6, 5, 3, 2, 0, 1, 4, 7};
  auto result = cudf::sorted_order(cudf::table_view({sliced_col}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, *result);
}

TYPED_TEST(Sort, WithEmptyListColumn)
{
  using T = TypeParam;
  if (std::is_same_v<T, bool>) { GTEST_SKIP(); }

  auto L1 = cudf::make_lists_column(0,
                                    cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)),
                                    cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}),
                                    0,
                                    {});
  auto L0 = cudf::make_lists_column(
    3, cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0}.release(), std::move(L1), 0, {});

  auto expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2};
  auto result = cudf::sorted_order(cudf::table_view({*L0}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, *result);
}

struct SortByKey : public cudf::test::BaseFixture {};

TEST_F(SortByKey, ValueKeysSizeMismatch)
{
  using T = int64_t;

  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8}};
  cudf::test::strings_column_wrapper col2({"d", "e", "a", "d", "k"});
  cudf::test::fixed_width_column_wrapper<T> col3{{10, 40, 70, 5, 2}};
  cudf::table_view values{{col1, col2, col3}};

  cudf::test::fixed_width_column_wrapper<T> key_col{{5, 4, 3, 5}};
  cudf::table_view keys{{key_col}};

  EXPECT_THROW(cudf::sort_by_key(values, keys), cudf::logic_error);
}

template <typename T>
struct SortFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(SortFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(SortFixedPointTest, SortedOrderGather)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const ZERO  = decimalXX{0, scale_type{0}};
  auto const ONE   = decimalXX{1, scale_type{0}};
  auto const TWO   = decimalXX{2, scale_type{0}};
  auto const THREE = decimalXX{3, scale_type{0}};
  auto const FOUR  = decimalXX{4, scale_type{0}};

  auto const input_vec  = std::vector<decimalXX>{TWO, ONE, ZERO, FOUR, THREE};
  auto const index_vec  = std::vector<cudf::size_type>{2, 1, 0, 4, 3};
  auto const sorted_vec = std::vector<decimalXX>{ZERO, ONE, TWO, THREE, FOUR};

  auto const input_col =
    cudf::test::fixed_width_column_wrapper<decimalXX>(input_vec.begin(), input_vec.end());
  auto const index_col =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(index_vec.begin(), index_vec.end());
  auto const sorted_col =
    cudf::test::fixed_width_column_wrapper<decimalXX>(sorted_vec.begin(), sorted_vec.end());

  auto const sorted_table = cudf::table_view{{sorted_col}};
  auto const input_table  = cudf::table_view{{input_col}};

  auto const indices = cudf::sorted_order(input_table);
  auto const sorted  = cudf::gather(input_table, indices->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(index_col, indices->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_table, sorted->view());
}

struct SortCornerTest : public cudf::test::BaseFixture {};

TEST_F(SortCornerTest, WithEmptyStructColumn)
{
  using int_col = cudf::test::fixed_width_column_wrapper<int32_t>;

  // struct{}, int, int
  int_col col_for_mask{{0, 0, 0, 0, 0, 0}, {1, 0, 1, 1, 1, 1}};
  auto null_mask  = cudf::copy_bitmask(col_for_mask);
  auto struct_col = cudf::make_structs_column(
    6, {}, cudf::column_view(col_for_mask).null_count(), std::move(null_mask));

  int_col col1{{1, 2, 3, 1, 2, 3}};
  int_col col2{{1, 1, 1, 2, 2, 2}};
  cudf::table_view input{{struct_col->view(), col1, col2}};

  int_col expected{{1, 0, 3, 4, 2, 5}};
  std::vector<cudf::order> column_order{
    cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING};
  auto got = cudf::sorted_order(input, column_order);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());

  // struct{struct{}, int}
  int_col col3{{0, 1, 2, 3, 4, 5}};
  std::vector<std::unique_ptr<cudf::column>> child_columns;
  child_columns.push_back(std::move(struct_col));
  child_columns.push_back(col3.release());
  auto struct_col2 =
    cudf::make_structs_column(6, std::move(child_columns), 0, rmm::device_buffer{});
  cudf::table_view input2{{struct_col2->view()}};

  int_col expected2{{5, 4, 3, 2, 0, 1}};
  auto got2 = cudf::sorted_order(input2, {cudf::order::DESCENDING});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, got2->view());

  // struct{struct{}, struct{int}}
  int_col col_for_mask2{{0, 0, 0, 0, 0, 0}, {1, 0, 1, 1, 0, 1}};
  auto null_mask2 = cudf::copy_bitmask(col_for_mask2);
  std::vector<std::unique_ptr<cudf::column>> child_columns2;
  auto child_col_1 = cudf::make_structs_column(
    6, {}, cudf::column_view(col_for_mask2).null_count(), std::move(null_mask2));
  child_columns2.push_back(std::move(child_col_1));
  int_col col4{{5, 4, 3, 2, 1, 0}};
  std::vector<std::unique_ptr<cudf::column>> grand_child;
  grand_child.push_back(col4.release());
  auto child_col_2 = cudf::make_structs_column(6, std::move(grand_child), 0, rmm::device_buffer{});
  child_columns2.push_back(std::move(child_col_2));
  auto struct_col3 =
    cudf::make_structs_column(6, std::move(child_columns2), 0, rmm::device_buffer{});
  cudf::table_view input3{{struct_col3->view()}};

  int_col expected3{{4, 1, 5, 3, 2, 0}};
  auto got3 = cudf::sorted_order(input3, {cudf::order::ASCENDING});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, got3->view());
};

using SortDouble = Sort<double>;
TEST_F(SortDouble, InfinityAndNan)
{
  auto constexpr NaN = std::numeric_limits<double>::quiet_NaN();
  auto constexpr Inf = std::numeric_limits<double>::infinity();

  auto input = cudf::test::fixed_width_column_wrapper<double>(
    {-0.0, -NaN, -NaN, NaN, Inf, -Inf, 7.0, 5.0, 6.0, NaN, Inf, -Inf, -NaN, -NaN, -0.0});
  auto expected =  // -inf,-inf,-0,-0,5,6,7,inf,inf,-nan,-nan,nan,nan,-nan,-nan
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(
      {5, 11, 0, 14, 7, 8, 6, 4, 10, 1, 2, 3, 9, 12, 13});
  auto results = cudf::sorted_order(cudf::table_view({input}));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

CUDF_TEST_PROGRAM_MAIN()
