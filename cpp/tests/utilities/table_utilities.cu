/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

namespace cudf::test::detail {
void expect_table_properties_equal(cudf::table_view lhs, cudf::table_view rhs)
{
  EXPECT_EQ(lhs.num_rows(), rhs.num_rows());
  EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
}

void expect_tables_equal(cudf::table_view lhs,
                         cudf::table_view rhs,
                         rmm::cuda_stream_view stream,
                         cudf::memory_resources mr)
{
  expect_table_properties_equal(lhs, rhs);
  for (auto i = 0; i < lhs.num_columns(); ++i) {
    cudf::test::detail::expect_columns_equal(
      lhs.column(i), rhs.column(i), cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }
}

/**
 * @copydoc cudf::test::expect_tables_equivalent
 */
void expect_tables_equivalent(cudf::table_view lhs,
                              cudf::table_view rhs,
                              rmm::cuda_stream_view stream,
                              cudf::memory_resources mr)
{
  auto num_columns = lhs.num_columns();
  for (auto i = 0; i < num_columns; ++i) {
    cudf::test::detail::expect_columns_equivalent(lhs.column(i),
                                                  rhs.column(i),
                                                  cudf::test::debug_output_level::FIRST_ERROR,
                                                  cudf::test::default_ulp,
                                                  stream,
                                                  mr);
  }
}

}  // namespace cudf::test::detail
