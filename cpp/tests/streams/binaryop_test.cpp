/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/udf.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/udf.hpp>

class BinaryopTest : public cudf::test::BaseFixture {};

TEST_F(BinaryopTest, ColumnColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> lhs{10, 20, 30, 40, 50};
  cudf::test::fixed_width_column_wrapper<int32_t> rhs{15, 25, 35, 45, 55};

  cudf::binary_operation(lhs,
                         rhs,
                         cudf::binary_operator::ADD,
                         cudf::data_type(cudf::type_to_id<int32_t>()),
                         cudf::test::get_default_stream());
}

TEST_F(BinaryopTest, ColumnScalar)
{
  cudf::test::fixed_width_column_wrapper<int32_t> lhs{10, 20, 30, 40, 50};
  cudf::numeric_scalar<int32_t> rhs{23, true, cudf::test::get_default_stream()};

  cudf::binary_operation(lhs,
                         rhs,
                         cudf::binary_operator::ADD,
                         cudf::data_type(cudf::type_to_id<int32_t>()),
                         cudf::test::get_default_stream());
}

TEST_F(BinaryopTest, ScalarColumn)
{
  cudf::numeric_scalar<int32_t> lhs{42, true, cudf::test::get_default_stream()};
  cudf::test::fixed_width_column_wrapper<int32_t> rhs{15, 25, 35, 45, 55};

  cudf::binary_operation(lhs,
                         rhs,
                         cudf::binary_operator::ADD,
                         cudf::data_type(cudf::type_to_id<int32_t>()),
                         cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
