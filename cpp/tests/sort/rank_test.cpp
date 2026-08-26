/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <tuple>
#include <vector>

template <typename T>
using lists_col   = cudf::test::lists_column_wrapper<T, int32_t>;
using structs_col = cudf::test::structs_column_wrapper;

using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

namespace {
void run_rank_test(cudf::table_view input,
                   cudf::table_view expected,
                   cudf::rank_method method,
                   cudf::order column_order,
                   cudf::null_policy null_handling,
                   cudf::null_order null_precedence,
                   bool percentage)
{
  int i = 0;
  for (auto&& input_column : input) {
    // Rank
    auto got_rank_column =
      cudf::rank(input_column, method, column_order, null_handling, null_precedence, percentage);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected.column(i), got_rank_column->view());
    i++;
  }
}

using input_arg_t = std::tuple<cudf::order, cudf::null_policy, cudf::null_order>;
input_arg_t asc_keep{cudf::order::ASCENDING, cudf::null_policy::EXCLUDE, cudf::null_order::AFTER};
input_arg_t asc_top{cudf::order::ASCENDING, cudf::null_policy::INCLUDE, cudf::null_order::BEFORE};
input_arg_t asc_bottom{cudf::order::ASCENDING, cudf::null_policy::INCLUDE, cudf::null_order::AFTER};

input_arg_t desc_keep{
  cudf::order::DESCENDING, cudf::null_policy::EXCLUDE, cudf::null_order::BEFORE};
input_arg_t desc_top{cudf::order::DESCENDING, cudf::null_policy::INCLUDE, cudf::null_order::AFTER};
input_arg_t desc_bottom{
  cudf::order::DESCENDING, cudf::null_policy::INCLUDE, cudf::null_order::BEFORE};
using test_case_t = std::tuple<cudf::table_view, cudf::table_view>;
}  // namespace

template <typename T>
struct Rank : public cudf::test::BaseFixture {
  cudf::test::fixed_width_column_wrapper<T> col1{{5, 4, 3, 5, 8, 5}};
  cudf::test::fixed_width_column_wrapper<T> col2{{5, 4, 3, 5, 8, 5}, {1, 1, 0, 1, 1, 1}};
  cudf::test::strings_column_wrapper col3{{"d", "e", "a", "d", "k", "d"},
                                          {true, true, true, true, true, true}};

  void run_all_tests(cudf::rank_method method,
                     input_arg_t input_arg,
                     cudf::column_view const col1_rank,
                     cudf::column_view const col2_rank,
                     cudf::column_view const col3_rank,
                     bool percentage = false)
  {
    if (std::is_same_v<T, bool>) return;
    for (auto const& test_case : {
           // Non-null column
           test_case_t{cudf::table_view{{col1}}, cudf::table_view{{col1_rank}}},
           // Null column
           test_case_t{cudf::table_view{{col2}}, cudf::table_view{{col2_rank}}},
           // Table
           test_case_t{cudf::table_view{{col1, col2}}, cudf::table_view{{col1_rank, col2_rank}}},
           // Table with String column
           test_case_t{cudf::table_view{{col1, col2, col3}},
                       cudf::table_view{{col1_rank, col2_rank, col3_rank}}},
         }) {
      auto [input, output] = test_case;

      run_rank_test(input,
                    output,
                    method,
                    std::get<0>(input_arg),
                    std::get<1>(input_arg),
                    std::get<2>(input_arg),
                    percentage);
    }
  }
};

TYPED_TEST_SUITE(Rank, cudf::test::NumericTypes);

// fixed_width_column_wrapper<T>   col1{{  5,   4,   3,   5,   8,   5}};
//                                        3,   2,   1,   4,   6,   5
TYPED_TEST(Rank, first_asc_keep)
{
  // ASCENDING
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 4, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {2, 1, -1, 3, 5, 4}, {true, true, false, true, true, true}};  // KEEP
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{
    {2, 5, 1, 3, 6, 4}, {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::FIRST, asc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, first_asc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 4, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {3, 2, 1, 4, 6, 5}};  // BEFORE = TOP
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 5, 1, 3, 6, 4}};
  this->run_all_tests(cudf::rank_method::FIRST, asc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, first_asc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 4, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {2, 1, 6, 3, 5, 4}};  // AFTER  = BOTTOM
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 5, 1, 3, 6, 4}};
  this->run_all_tests(cudf::rank_method::FIRST, asc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, first_desc_keep)
{
  // DESCENDING
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 3, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {2, 5, -1, 3, 1, 4}, {true, true, false, true, true, true}};  // KEEP
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{
    {3, 2, 6, 4, 1, 5}, {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::FIRST, desc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, first_desc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 3, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {3, 6, 1, 4, 2, 5}};  // AFTER  = TOP
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 6, 4, 1, 5}};
  this->run_all_tests(cudf::rank_method::FIRST, desc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, first_desc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 3, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {2, 5, 6, 3, 1, 4}};  // BEFORE = BOTTOM
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 6, 4, 1, 5}};
  this->run_all_tests(cudf::rank_method::FIRST, desc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_asc_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {2, 1, -1, 2, 3, 2}, {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{
    {2, 3, 1, 2, 4, 2}, {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::DENSE, asc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_asc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{3, 2, 1, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 3, 1, 2, 4, 2}};
  this->run_all_tests(cudf::rank_method::DENSE, asc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_asc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 4, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 1, 4, 2, 3, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 3, 1, 2, 4, 2}};
  this->run_all_tests(cudf::rank_method::DENSE, asc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_desc_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 3, 4, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {2, 3, -1, 2, 1, 2}, {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{
    {3, 2, 4, 3, 1, 3}, {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::DENSE, desc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_desc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 3, 4, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{3, 4, 1, 3, 2, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 4, 3, 1, 3}};
  this->run_all_tests(cudf::rank_method::DENSE, desc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, dense_desc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 3, 4, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 3, 4, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 4, 3, 1, 3}};
  this->run_all_tests(cudf::rank_method::DENSE, desc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_asc_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 6, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {2, 1, -1, 2, 5, 2}, {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{
    {2, 5, 1, 2, 6, 2}, {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::MIN, asc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_asc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 6, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{3, 2, 1, 3, 6, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 5, 1, 2, 6, 2}};
  this->run_all_tests(cudf::rank_method::MIN, asc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_asc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{3, 2, 1, 3, 6, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 1, 6, 2, 5, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{2, 5, 1, 2, 6, 2}};
  this->run_all_tests(cudf::rank_method::MIN, asc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_desc_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {2, 5, -1, 2, 1, 2}, {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{
    {3, 2, 6, 3, 1, 3}, {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::MIN, desc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_desc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{3, 6, 1, 3, 2, 3}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 6, 3, 1, 3}};
  this->run_all_tests(cudf::rank_method::MIN, desc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, min_desc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{2, 5, 6, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{2, 5, 6, 2, 1, 2}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{3, 2, 6, 3, 1, 3}};
  this->run_all_tests(cudf::rank_method::MIN, desc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_asc_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{5, 2, 1, 5, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {4, 1, -1, 4, 5, 4}, {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{
    {4, 5, 1, 4, 6, 4}, {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::MAX, asc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_asc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{5, 2, 1, 5, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{5, 2, 1, 5, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{4, 5, 1, 4, 6, 4}};
  this->run_all_tests(cudf::rank_method::MAX, asc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_asc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{5, 2, 1, 5, 6, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{4, 1, 6, 4, 5, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{4, 5, 1, 4, 6, 4}};
  this->run_all_tests(cudf::rank_method::MAX, asc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_desc_keep)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{4, 5, 6, 4, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{
    {4, 5, -1, 4, 1, 4}, {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{
    {5, 2, 6, 5, 1, 5}, {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::MAX, desc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_desc_top)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{4, 5, 6, 4, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{5, 6, 1, 5, 2, 5}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{5, 2, 6, 5, 1, 5}};
  this->run_all_tests(cudf::rank_method::MAX, desc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, max_desc_bottom)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col1_rank{{4, 5, 6, 4, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col2_rank{{4, 5, 6, 4, 1, 4}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col3_rank{{5, 2, 6, 5, 1, 5}};
  this->run_all_tests(cudf::rank_method::MAX, desc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_asc_keep)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{4, 2, 1, 4, 6, 4}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{3, 1, -1, 3, 5, 3},
                                                           {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{3, 5, 1, 3, 6, 3},
                                                           {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::AVERAGE, asc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_asc_top)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{4, 2, 1, 4, 6, 4}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{4, 2, 1, 4, 6, 4}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{3, 5, 1, 3, 6, 3}};
  this->run_all_tests(cudf::rank_method::AVERAGE, asc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_asc_bottom)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{4, 2, 1, 4, 6, 4}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{3, 1, 6, 3, 5, 3}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{3, 5, 1, 3, 6, 3}};
  this->run_all_tests(cudf::rank_method::AVERAGE, asc_bottom, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_desc_keep)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{3, 5, 6, 3, 1, 3}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{3, 5, -1, 3, 1, 3},
                                                           {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{4, 2, 6, 4, 1, 4},
                                                           {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::AVERAGE, desc_keep, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_desc_top)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{3, 5, 6, 3, 1, 3}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{4, 6, 1, 4, 2, 4}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{4, 2, 6, 4, 1, 4}};
  this->run_all_tests(cudf::rank_method::AVERAGE, desc_top, col1_rank, col2_rank, col3_rank);
}

TYPED_TEST(Rank, average_desc_bottom)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{3, 5, 6, 3, 1, 3}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{3, 5, 6, 3, 1, 3}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{4, 2, 6, 4, 1, 4}};
  this->run_all_tests(cudf::rank_method::AVERAGE, desc_bottom, col1_rank, col2_rank, col3_rank);
}

// percentage==true (dense, not-dense)
TYPED_TEST(Rank, dense_asc_keep_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{0.75, 0.5, 0.25, 0.75, 1., 0.75}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{
    {2.0 / 3.0, 1.0 / 3.0, -1., 2.0 / 3.0, 1., 2.0 / 3.0}, {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{0.5, 0.75, 0.25, 0.5, 1., 0.5},
                                                           {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::DENSE, asc_keep, col1_rank, col2_rank, col3_rank, true);
}

TYPED_TEST(Rank, dense_asc_top_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{0.75, 0.5, 0.25, 0.75, 1., 0.75}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{0.75, 0.5, 0.25, 0.75, 1., 0.75}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{0.5, 0.75, 0.25, 0.5, 1., 0.5}};
  this->run_all_tests(cudf::rank_method::DENSE, asc_top, col1_rank, col2_rank, col3_rank, true);
}

TYPED_TEST(Rank, dense_asc_bottom_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{{0.75, 0.5, 0.25, 0.75, 1., 0.75}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{0.5, 0.25, 1., 0.5, 0.75, 0.5}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{{0.5, 0.75, 0.25, 0.5, 1., 0.5}};
  this->run_all_tests(cudf::rank_method::DENSE, asc_bottom, col1_rank, col2_rank, col3_rank, true);
}

TYPED_TEST(Rank, min_desc_keep_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{
    {1.0 / 3.0, 5.0 / 6.0, 1., 1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{{0.4, 1., -1., 0.4, 0.2, 0.4},
                                                           {true, true, false, true, true, true}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{
    {0.5, 1.0 / 3.0, 1., 0.5, 1.0 / 6.0, 0.5}, {true, true, true, true, true, true}};
  this->run_all_tests(cudf::rank_method::MIN, desc_keep, col1_rank, col2_rank, col3_rank, true);
}

TYPED_TEST(Rank, min_desc_top_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{
    {1.0 / 3.0, 5.0 / 6.0, 1., 1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{
    {0.5, 1., 1.0 / 6.0, 0.5, 1.0 / 3.0, 0.5}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{
    {0.5, 1.0 / 3.0, 1., 0.5, 1.0 / 6.0, 0.5}};
  this->run_all_tests(cudf::rank_method::MIN, desc_top, col1_rank, col2_rank, col3_rank, true);
}

TYPED_TEST(Rank, min_desc_bottom_pct)
{
  cudf::test::fixed_width_column_wrapper<double> col1_rank{
    {1.0 / 3.0, 5.0 / 6.0, 1., 1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0}};
  cudf::test::fixed_width_column_wrapper<double> col2_rank{
    {1.0 / 3.0, 5.0 / 6.0, 1., 1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0}};
  cudf::test::fixed_width_column_wrapper<double> col3_rank{
    {0.5, 1.0 / 3.0, 1., 0.5, 1.0 / 6.0, 0.5}};
  this->run_all_tests(cudf::rank_method::MIN, desc_bottom, col1_rank, col2_rank, col3_rank, true);
}

struct RankLarge : public cudf::test::BaseFixture {};

TEST_F(RankLarge, average_large)
{
  // testcase of https://github.com/NVIDIA/cudf/issues/9703
  auto iter = cuda::counting_iterator<int64_t>{0};
  cudf::test::fixed_width_column_wrapper<int64_t> col1(iter, iter + 10558);
  auto result = cudf::rank(col1,
                           cudf::rank_method::AVERAGE,
                           {},
                           cudf::null_policy::EXCLUDE,
                           cudf::null_order::AFTER,
                           false);
  cudf::test::fixed_width_column_wrapper<double, int> expected(iter + 1, iter + 10559);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

template <typename T>
struct RankListAndStruct : public cudf::test::BaseFixture {
  void run_all_tests(cudf::rank_method method,
                     input_arg_t input_arg,
                     cudf::column_view const list_rank,
                     cudf::column_view const struct_rank,
                     bool percentage = false)
  {
    if constexpr (std::is_same_v<T, bool>) { return; }
    /*
    [
      [],
      [1],
      [2, 2],
      [2, 3],
      [2, 2],
      [1],
      [],
      NULL
      [2],
      NULL,
      [1]
    ]
    */
    auto list_col =
      lists_col<T>{{{}, {1}, {2, 2}, {2, 3}, {2, 2}, {1}, {}, {} /*NULL*/, {2}, {} /*NULL*/, {1}},
                   nulls_at({7, 9})};

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
    std::vector<bool>                           struct_valids{true, true, false, true, false, true, true, true};
    auto col1       = cudf::test::fixed_width_column_wrapper<T>{{ 0,  1,  9, -1,  9, -1, -1, -1}, {1, 1, 1, 0, 1, 0, 0, 0}};
    auto col2       = cudf::test::fixed_width_column_wrapper<T>{{-1, -1,  9, -1,  9, -1,  1,  0}, {0, 0, 1, 0, 1, 0, 1, 1}};
    auto struct_col = cudf::test::structs_column_wrapper{{col1, col2}, struct_valids}.release();
    // clang-format on

    for (auto const& test_case : {
           // Non-null column
           test_case_t{cudf::table_view{{list_col}}, cudf::table_view{{list_rank}}},
           // Null column
           test_case_t{cudf::table_view{{struct_col->view()}}, cudf::table_view{{struct_rank}}},
         }) {
      auto [input, output] = test_case;

      run_rank_test(input,
                    output,
                    method,
                    std::get<0>(input_arg),
                    std::get<1>(input_arg),
                    std::get<2>(input_arg),
                    percentage);
    }
  }
};

TYPED_TEST_SUITE(RankListAndStruct, cudf::test::NumericTypes);

TYPED_TEST(RankListAndStruct, first_asc_keep)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> list_rank{
    {1, 3, 7, 9, 8, 4, 2, -1, 6, -1, 5}, nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{{1, 2, -1, 5, -1, 6, 4, 3},
                                                                      nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::FIRST, asc_keep, list_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, first_asc_top)
{
  // ASCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    3, 5, 9, 11, 10, 6, 4, 1, 8, 2, 7};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{7, 8, 1, 3, 2, 4, 6, 5};
  this->run_all_tests(cudf::rank_method::FIRST, asc_top, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, first_asc_bottom)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    1, 3, 7, 9, 8, 4, 2, 10, 6, 11, 5};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{1, 2, 7, 5, 8, 6, 4, 3};
  this->run_all_tests(cudf::rank_method::FIRST, asc_bottom, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, first_desc_keep)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    {8, 5, 2, 1, 3, 6, 9, -1, 4, -1, 7}, nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{{2, 1, -1, 5, -1, 6, 3, 4},
                                                                      nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::FIRST, desc_keep, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, first_desc_top)
{
  // DESCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    10, 7, 4, 3, 5, 8, 11, 1, 6, 2, 9};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{8, 7, 1, 3, 2, 4, 5, 6};
  this->run_all_tests(cudf::rank_method::FIRST, desc_top, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, first_desc_bottom)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    8, 5, 2, 1, 3, 6, 9, 10, 4, 11, 7};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{2, 1, 7, 5, 8, 6, 3, 4};
  this->run_all_tests(cudf::rank_method::FIRST, desc_bottom, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, dense_asc_keep)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    {1, 2, 4, 5, 4, 2, 1, -1, 3, -1, 2}, nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{{1, 2, -1, 5, -1, 5, 4, 3},
                                                                      nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::DENSE, asc_keep, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, dense_asc_top)
{
  // ASCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{2, 3, 5, 6, 5, 3, 2, 1, 4, 1, 3};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{5, 6, 1, 2, 1, 2, 4, 3};
  this->run_all_tests(cudf::rank_method::DENSE, asc_top, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, dense_asc_bottom)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{1, 2, 4, 5, 4, 2, 1, 6, 3, 6, 2};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{1, 2, 6, 5, 6, 5, 4, 3};
  this->run_all_tests(cudf::rank_method::DENSE, asc_bottom, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, dense_desc_keep)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    {5, 4, 2, 1, 2, 4, 5, -1, 3, -1, 4}, nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{{2, 1, -1, 5, -1, 5, 3, 4},
                                                                      nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::DENSE, desc_keep, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, dense_desc_top)
{
  // DESCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{6, 5, 3, 2, 3, 5, 6, 1, 4, 1, 5};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{6, 5, 1, 2, 1, 2, 3, 4};
  this->run_all_tests(cudf::rank_method::DENSE, desc_top, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, dense_desc_bottom)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{5, 4, 2, 1, 2, 4, 5, 6, 3, 6, 4};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{2, 1, 6, 5, 6, 5, 3, 4};
  this->run_all_tests(cudf::rank_method::DENSE, desc_bottom, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, min_asc_keep)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    {1, 3, 7, 9, 7, 3, 1, -1, 6, -1, 3}, nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{{1, 2, -1, 5, -1, 5, 4, 3},
                                                                      nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::MIN, asc_keep, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, min_asc_top)
{
  // ASCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    3, 5, 9, 11, 9, 5, 3, 1, 8, 1, 5};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{7, 8, 1, 3, 1, 3, 6, 5};
  this->run_all_tests(cudf::rank_method::MIN, asc_top, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, min_asc_bottom)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    1, 3, 7, 9, 7, 3, 1, 10, 6, 10, 3};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{1, 2, 7, 5, 7, 5, 4, 3};
  this->run_all_tests(cudf::rank_method::MIN, asc_bottom, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, min_desc_keep)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    {8, 5, 2, 1, 2, 5, 8, -1, 4, -1, 5}, nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{{2, 1, -1, 5, -1, 5, 3, 4},
                                                                      nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::MIN, desc_keep, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, min_desc_top)
{
  // DESCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    10, 7, 4, 3, 4, 7, 10, 1, 6, 1, 7};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{8, 7, 1, 3, 1, 3, 5, 6};
  this->run_all_tests(cudf::rank_method::MIN, desc_top, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, min_desc_bottom)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    8, 5, 2, 1, 2, 5, 8, 10, 4, 10, 5};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{2, 1, 7, 5, 7, 5, 3, 4};
  this->run_all_tests(cudf::rank_method::MIN, desc_bottom, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, max_asc_keep)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    {2, 5, 8, 9, 8, 5, 2, -1, 6, -1, 5}, nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{{1, 2, -1, 6, -1, 6, 4, 3},
                                                                      nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::MAX, asc_keep, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, max_asc_top)
{
  // ASCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    4, 7, 10, 11, 10, 7, 4, 2, 8, 2, 7};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{7, 8, 2, 4, 2, 4, 6, 5};
  this->run_all_tests(cudf::rank_method::MAX, asc_top, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, max_asc_bottom)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    2, 5, 8, 9, 8, 5, 2, 11, 6, 11, 5};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{1, 2, 8, 6, 8, 6, 4, 3};
  this->run_all_tests(cudf::rank_method::MAX, asc_bottom, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, max_desc_keep)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    {9, 7, 3, 1, 3, 7, 9, -1, 4, -1, 7}, nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{{2, 1, -1, 6, -1, 6, 3, 4},
                                                                      nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::MAX, desc_keep, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, max_desc_top)
{
  // DESCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    11, 9, 5, 3, 5, 9, 11, 2, 6, 2, 9};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{8, 7, 2, 4, 2, 4, 5, 6};
  this->run_all_tests(cudf::rank_method::MAX, desc_top, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, max_desc_bottom)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<cudf::size_type> col_rank{
    9, 7, 3, 1, 3, 7, 9, 11, 4, 11, 7};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> struct_rank{2, 1, 8, 6, 8, 6, 3, 4};
  this->run_all_tests(cudf::rank_method::MAX, desc_bottom, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, average_asc_keep)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<double> col_rank{
    {1.5, 4.0, 7.5, 9.0, 7.5, 4.0, 1.5, -1.0, 6.0, -1.0, 4.0}, nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    {1.0, 2.0, -1.0, 5.5, -1.0, 5.5, 4.0, 3.0}, nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::AVERAGE, asc_keep, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, average_asc_top)
{
  // ASCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<double> col_rank{
    3.5, 6.0, 9.5, 11.0, 9.5, 6.0, 3.5, 1.5, 8.0, 1.5, 6.0};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    7.0, 8.0, 1.5, 3.5, 1.5, 3.5, 6.0, 5.0};
  this->run_all_tests(cudf::rank_method::AVERAGE, asc_top, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, average_asc_bottom)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<double> col_rank{
    1.5, 4.0, 7.5, 9.0, 7.5, 4.0, 1.5, 10.5, 6.0, 10.5, 4.0};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    1.0, 2.0, 7.5, 5.5, 7.5, 5.5, 4.0, 3.0};
  this->run_all_tests(cudf::rank_method::AVERAGE, asc_bottom, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, average_desc_keep)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<double> col_rank{
    {8.5, 6.0, 2.5, 1.0, 2.5, 6.0, 8.5, -1.0, 4.0, -1.0, 6.0}, nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    {2.0, 1.0, -1.0, 5.5, -1.0, 5.5, 3.0, 4.0}, nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::AVERAGE, desc_keep, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, average_desc_top)
{
  // DESCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<double> col_rank{
    10.5, 8.0, 4.5, 3.0, 4.5, 8.0, 10.5, 1.5, 6.0, 1.5, 8.0};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    8.0, 7.0, 1.5, 3.5, 1.5, 3.5, 5.0, 6.0};
  this->run_all_tests(cudf::rank_method::AVERAGE, desc_top, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, average_desc_bottom)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<double> col_rank{
    8.5, 6.0, 2.5, 1.0, 2.5, 6.0, 8.5, 10.5, 4.0, 10.5, 6.0};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    2.0, 1.0, 7.5, 5.5, 7.5, 5.5, 3.0, 4.0};
  this->run_all_tests(cudf::rank_method::AVERAGE, desc_bottom, col_rank, struct_rank);
}

TYPED_TEST(RankListAndStruct, dense_asc_keep_pct)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<double> col_rank{{1.0 / 5.0,
                                                           2.0 / 5.0,
                                                           4.0 / 5.0,
                                                           1.0,
                                                           4.0 / 5.0,
                                                           2.0 / 5.0,
                                                           1.0 / 5.0,
                                                           -1.0,
                                                           3.0 / 5.0,
                                                           -1.0,
                                                           2.0 / 5.0},
                                                          nulls_at({7, 9})};

  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    {1.0 / 5.0, 2.0 / 5.0, -1.0, 1.0, -1.0, 1.0, 4.0 / 5.0, 3.0 / 5.0}, nulls_at({2, 4})};

  this->run_all_tests(cudf::rank_method::DENSE, asc_keep, col_rank, struct_rank, true);
}

TYPED_TEST(RankListAndStruct, dense_asc_top_pct)
{
  // ASCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<double> col_rank{1.0 / 3.0,
                                                          1.0 / 2.0,
                                                          5.0 / 6.0,
                                                          1.0,
                                                          5.0 / 6.0,
                                                          1.0 / 2.0,
                                                          1.0 / 3.0,
                                                          1.0 / 6.0,
                                                          2.0 / 3.0,
                                                          1.0 / 6.0,
                                                          1.0 / 2.0};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    5.0 / 6.0, 1.0, 1.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0, 4.0 / 6.0, 3.0 / 6.0};
  this->run_all_tests(cudf::rank_method::DENSE, asc_top, col_rank, struct_rank, true);
}

TYPED_TEST(RankListAndStruct, dense_asc_bottom_pct)
{
  // ASCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<double> col_rank{1.0 / 6.0,
                                                          1.0 / 3.0,
                                                          2.0 / 3.0,
                                                          5.0 / 6.0,
                                                          2.0 / 3.0,
                                                          1.0 / 3.0,
                                                          1.0 / 6.0,
                                                          1.0,
                                                          1.0 / 2.0,
                                                          1.0,
                                                          1.0 / 3.0};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    1.0 / 6.0, 2.0 / 6.0, 1.0, 5.0 / 6.0, 1.0, 5.0 / 6.0, 4.0 / 6.0, 3.0 / 6.0};
  this->run_all_tests(cudf::rank_method::DENSE, asc_bottom, col_rank, struct_rank, true);
}

TYPED_TEST(RankListAndStruct, min_desc_keep_pct)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<double> col_rank{{8.0 / 9.0,
                                                           5.0 / 9.0,
                                                           2.0 / 9.0,
                                                           1.0 / 9.0,
                                                           2.0 / 9.0,
                                                           5.0 / 9.0,
                                                           8.0 / 9.0,
                                                           -1.0,
                                                           4.0 / 9.0,
                                                           -1.0,
                                                           5.0 / 9.0},
                                                          nulls_at({7, 9})};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    {2.0 / 6.0, 1.0 / 6.0, -1.0, 5.0 / 6.0, -1.0, 5.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0},
    nulls_at({2, 4})};
  this->run_all_tests(cudf::rank_method::MIN, desc_keep, col_rank, struct_rank, true);
}

TYPED_TEST(RankListAndStruct, min_desc_top_pct)
{
  // DESCENDING and null_order::AFTER
  cudf::test::fixed_width_column_wrapper<double> col_rank{10.0 / 11.0,
                                                          7.0 / 11.0,
                                                          4.0 / 11.0,
                                                          3.0 / 11.0,
                                                          4.0 / 11.0,
                                                          7.0 / 11.0,
                                                          10.0 / 11.0,
                                                          1.0 / 11.0,
                                                          6.0 / 11.0,
                                                          1.0 / 11.0,
                                                          7.0 / 11.0};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    1.0, 7.0 / 8.0, 1.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0, 6.0 / 8.0};
  this->run_all_tests(cudf::rank_method::MIN, desc_top, col_rank, struct_rank, true);
}

TYPED_TEST(RankListAndStruct, min_desc_bottom_pct)
{
  // DESCENDING and null_order::BEFORE
  cudf::test::fixed_width_column_wrapper<double> col_rank{8.0 / 11.0,
                                                          5.0 / 11.0,
                                                          2.0 / 11.0,
                                                          1.0 / 11.0,
                                                          2.0 / 11.0,
                                                          5.0 / 11.0,
                                                          8.0 / 11.0,
                                                          10.0 / 11.0,
                                                          4.0 / 11.0,
                                                          10.0 / 11.0,
                                                          5.0 / 11.0};
  cudf::test::fixed_width_column_wrapper<double> struct_rank{
    2.0 / 8.0, 1.0 / 8.0, 7.0 / 8.0, 5.0 / 8.0, 7.0 / 8.0, 5.0 / 8.0, 3.0 / 8.0, 4.0 / 8.0};
  this->run_all_tests(cudf::rank_method::MIN, desc_bottom, col_rank, struct_rank, true);
}
