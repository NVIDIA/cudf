/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <vector>

struct RowOperatorTestForNAN : public cudf::test::BaseFixture {};

TEST_F(RowOperatorTestForNAN, NANEquality)
{
  cudf::test::fixed_width_column_wrapper<double> col1{{1., double(NAN), 3., 4.},
                                                      {true, true, false, true}};
  cudf::test::fixed_width_column_wrapper<double> col2{{1., double(NAN), 3., 4.},
                                                      {true, true, false, true}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col1, col2);
}

TEST_F(RowOperatorTestForNAN, NANSorting)
{
  // NULL Before
  cudf::test::fixed_width_column_wrapper<double> input{
    {0.,
     double(NAN),
     -1.,
     7.,
     std::numeric_limits<double>::infinity(),
     1.,
     -1 * std::numeric_limits<double>::infinity()},
    {true, true, true, false, true, true, true, true}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected1{{3, 6, 2, 0, 5, 4, 1}};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence_1{cudf::null_order::BEFORE};
  cudf::table_view input_table{{input}};

  auto got1 = cudf::sorted_order(input_table, column_order, null_precedence_1);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, got1->view());

  // NULL After

  std::vector<cudf::null_order> null_precedence_2{cudf::null_order::AFTER};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected2{{6, 2, 0, 5, 4, 1, 3}};

  auto got2 = cudf::sorted_order(input_table, column_order, null_precedence_2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, got2->view());
}

TEST_F(RowOperatorTestForNAN, NANSortingNonNull)
{
  cudf::test::fixed_width_column_wrapper<double> input{
    {0.,
     double(NAN),
     -1.,
     7.,
     std::numeric_limits<double>::infinity(),
     1.,
     -1 * std::numeric_limits<double>::infinity()}};

  cudf::table_view input_table{{input}};

  auto result = cudf::sorted_order(input_table, {cudf::order::ASCENDING});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_asc{{6, 2, 0, 5, 3, 4, 1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_asc, result->view());
  auto sorted_result = cudf::sort(input_table, {cudf::order::ASCENDING});
  auto gather_result = cudf::gather(input_table, result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_result->view().column(0),
                                      gather_result->view().column(0));

  result = cudf::sorted_order(input_table, {cudf::order::DESCENDING});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_desc{{1, 4, 3, 5, 0, 2, 6}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_desc, result->view());
  sorted_result = cudf::sort(input_table, {cudf::order::DESCENDING});
  gather_result = cudf::gather(input_table, result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_result->view().column(0),
                                      gather_result->view().column(0));
}
