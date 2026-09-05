/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>

#include <limits>

namespace {

struct BinaryOperatorParityTest : public cudf::test::BaseFixture {};

static_assert(static_cast<int32_t>(cudf::binary_operator::INVALID_BINARY) == 34);

TEST_F(BinaryOperatorParityTest, CheckedArithmeticPropagates)
{
  auto max  = std::numeric_limits<int32_t>::max();
  auto min  = std::numeric_limits<int32_t>::min();
  auto lhs  = cudf::test::fixed_width_column_wrapper<int32_t>{max, min, max, 1, min};
  auto rhs  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 1, 2, 0, -1};
  auto type = cudf::data_type{cudf::type_id::INT32};

  EXPECT_THROW(cudf::binary_operation(lhs, rhs, cudf::binary_operator::ADD_OVERFLOW, type),
               cudf::evaluation_error);
  EXPECT_THROW(cudf::binary_operation(lhs, rhs, cudf::binary_operator::SUB_OVERFLOW, type),
               cudf::evaluation_error);
  EXPECT_THROW(cudf::binary_operation(lhs, rhs, cudf::binary_operator::MUL_OVERFLOW, type),
               cudf::evaluation_error);
  EXPECT_THROW(cudf::binary_operation(lhs, rhs, cudf::binary_operator::DIV_OVERFLOW, type),
               cudf::evaluation_error);
  EXPECT_THROW(cudf::binary_operation(lhs, rhs, cudf::binary_operator::MOD_OVERFLOW, type),
               cudf::evaluation_error);
}

TEST_F(BinaryOperatorParityTest, CheckedArithmeticNullifies)
{
  auto max  = std::numeric_limits<int32_t>::max();
  auto min  = std::numeric_limits<int32_t>::min();
  auto type = cudf::data_type{cudf::type_id::INT32};

  auto add_lhs      = cudf::test::fixed_width_column_wrapper<int32_t>{max, 4};
  auto add_rhs      = cudf::test::fixed_width_column_wrapper<int32_t>{1, 5};
  auto add_expected = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 9}, {false, true}};
  auto add_result   = cudf::binary_operation(
    add_lhs, add_rhs, cudf::binary_operator::ADD_OVERFLOW, type, cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(add_expected, add_result->view());

  auto div_lhs = cudf::test::fixed_width_column_wrapper<int32_t>{min, 4, 6};
  auto div_rhs = cudf::test::fixed_width_column_wrapper<int32_t>{-1, 0, 3};
  auto div_expected =
    cudf::test::fixed_width_column_wrapper<int32_t>{{0, 0, 2}, {false, false, true}};
  auto div_result = cudf::binary_operation(
    div_lhs, div_rhs, cudf::binary_operator::DIV_OVERFLOW, type, cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(div_expected, div_result->view());

  auto mod_expected =
    cudf::test::fixed_width_column_wrapper<int32_t>{{0, 0, 0}, {true, false, true}};
  auto mod_result = cudf::binary_operation(
    div_lhs, div_rhs, cudf::binary_operator::MOD_OVERFLOW, type, cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(mod_expected, mod_result->view());
}

TEST_F(BinaryOperatorParityTest, CheckedArithmeticSkipsNullInputs)
{
  auto lhs      = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 8}, {false, true}};
  auto rhs      = cudf::test::fixed_width_column_wrapper<int32_t>{0, 2};
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 4}, {false, true}};

  auto result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::DIV_OVERFLOW, cudf::data_type{cudf::type_id::INT32});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(BinaryOperatorParityTest, CheckedArithmeticScalarOperands)
{
  auto lhs      = cudf::numeric_scalar<int32_t>{20};
  auto rhs      = cudf::test::fixed_width_column_wrapper<int32_t>{2, 4};
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>{{10, 5}, {true, true}};

  auto left_result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::DIV_OVERFLOW, lhs.type(), cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, left_result->view());

  auto divisor      = cudf::numeric_scalar<int32_t>{3};
  auto values       = cudf::test::fixed_width_column_wrapper<int32_t>{7, 8};
  auto mod_expected = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2}, {true, true}};
  auto right_result = cudf::binary_operation(values,
                                             divisor,
                                             cudf::binary_operator::MOD_OVERFLOW,
                                             divisor.type(),
                                             cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(mod_expected, right_result->view());
}

TEST_F(BinaryOperatorParityTest, CheckedDecimalScaleRules)
{
  auto lhs = cudf::test::fixed_point_column_wrapper<int32_t>{{120, 250}, numeric::scale_type{-2}};
  auto rhs = cudf::test::fixed_point_column_wrapper<int32_t>{{3, 5}, numeric::scale_type{-1}};

  auto add_expected = cudf::test::fixed_point_column_wrapper<int32_t>{
    {150, 300}, {true, true}, numeric::scale_type{-2}};
  auto add_type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::ADD_OVERFLOW,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto add_result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::ADD_OVERFLOW, add_type, cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(add_expected, add_result->view());

  auto mul_expected = cudf::test::fixed_point_column_wrapper<int32_t>{
    {360, 1250}, {true, true}, numeric::scale_type{-3}};
  auto mul_type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::MUL_OVERFLOW,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto mul_result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::MUL_OVERFLOW, mul_type, cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(mul_expected, mul_result->view());

  auto div_expected = cudf::test::fixed_point_column_wrapper<int32_t>{
    {40, 50}, {true, true}, numeric::scale_type{-1}};
  auto div_type =
    cudf::binary_operation_fixed_point_output_type(cudf::binary_operator::DIV_OVERFLOW,
                                                   static_cast<cudf::column_view>(lhs).type(),
                                                   static_cast<cudf::column_view>(rhs).type());
  auto div_result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::DIV_OVERFLOW, div_type, cudf::error_policy::NULLIFY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(div_expected, div_result->view());

  EXPECT_THROW(
    cudf::binary_operation(
      lhs, rhs, cudf::binary_operator::ADD_OVERFLOW, cudf::data_type{cudf::type_id::DECIMAL32, -1}),
    cudf::data_type_error);
}
}  // namespace
