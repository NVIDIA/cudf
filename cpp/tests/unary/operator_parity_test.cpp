/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <limits>

namespace {

struct UnaryOperatorParityTest : public cudf::test::BaseFixture {};

static_assert(static_cast<int32_t>(cudf::unary_operator::NEGATE) == 23);

TEST_F(UnaryOperatorParityTest, CheckedUnaryPropagates)
{
  auto min   = std::numeric_limits<int32_t>::min();
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>{min, -2, 3};

  EXPECT_THROW(cudf::unary_operation(input, cudf::unary_operator::NEG_OVERFLOW),
               cudf::evaluation_error);
  EXPECT_THROW(cudf::unary_operation(input, cudf::unary_operator::ABS_OVERFLOW),
               cudf::evaluation_error);
}

TEST_F(UnaryOperatorParityTest, CheckedUnaryNullifies)
{
  auto min   = std::numeric_limits<int32_t>::min();
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>{min, -2, 3};

  auto neg_expected =
    cudf::test::fixed_width_column_wrapper<int32_t>{{0, 2, -3}, {false, true, true}};
  auto neg_result =
    cudf::unary_operation(input, cudf::unary_operator::NEG_OVERFLOW, cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(neg_expected, neg_result->view());

  auto abs_expected =
    cudf::test::fixed_width_column_wrapper<int32_t>{{0, 2, 3}, {false, true, true}};
  auto abs_result =
    cudf::unary_operation(input, cudf::unary_operator::ABS_OVERFLOW, cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(abs_expected, abs_result->view());
}

TEST_F(UnaryOperatorParityTest, CheckedUnarySkipsNullInputs)
{
  auto min      = std::numeric_limits<int32_t>::min();
  auto input    = cudf::test::fixed_width_column_wrapper<int32_t>{{min, 5}, {false, true}};
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>{{0, -5}, {false, true}};

  auto result = cudf::unary_operation(input, cudf::unary_operator::NEG_OVERFLOW);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(UnaryOperatorParityTest, CheckedUnaryDecimal)
{
  auto min   = std::numeric_limits<int32_t>::min();
  auto input = cudf::test::fixed_point_column_wrapper<int32_t>{{min, -25}, numeric::scale_type{-2}};
  auto expected = cudf::test::fixed_point_column_wrapper<int32_t>{
    {0, 25}, {false, true}, numeric::scale_type{-2}};

  auto result =
    cudf::unary_operation(input, cudf::unary_operator::ABS_OVERFLOW, cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}
}  // namespace
