/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/table/equality.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

class TableEqualTest : public cudf::test::BaseFixture {};

TEST_F(TableEqualTest, NotEqual)
{
  cudf::test::fixed_width_column_wrapper<int> left(
    {{0, 0, 0, 0, 0}, {false, false, true, true, true}});
  cudf::test::fixed_width_column_wrapper<int> right({1, 1, 1, 1, 1});
  std::ignore = cudf::tables_equal(cudf::table_view{{left}},
                                   cudf::table_view{{right}},
                                   cudf::null_equality::EQUAL,
                                   cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
