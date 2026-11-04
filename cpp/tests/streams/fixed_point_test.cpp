/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/fixed_point/math_ops.hpp>
#include <cudf/scalar/scalar_factories.hpp>

class DivideDecimalTest : public cudf::test::BaseFixture {};

TEST_F(DivideDecimalTest, ColumnColumn)
{
  auto const lhs = cudf::test::fixed_point_column_wrapper<int64_t>{
    {1000, 2000, 3000}, {1, 0, 1}, numeric::scale_type{-2}};
  auto const rhs =
    cudf::test::fixed_point_column_wrapper<int64_t>{{300, 400, 0}, numeric::scale_type{-1}};
  cudf::divide_decimal(lhs,
                       rhs,
                       cudf::data_type{cudf::type_id::DECIMAL128, numeric::scale_type{-4}},
                       cudf::rounding_method::HALF_UP,
                       cudf::decimal_intermediate::STANDARD,
                       cudf::test::get_default_stream());
}

TEST_F(DivideDecimalTest, ColumnScalar)
{
  auto const lhs = cudf::test::fixed_point_column_wrapper<int64_t>{
    {1000, 2000, 3000}, {1, 0, 1}, numeric::scale_type{-2}};
  auto const rhs = cudf::make_fixed_point_scalar<numeric::decimal64>(
    30, numeric::scale_type{-1}, cudf::test::get_default_stream());
  cudf::divide_decimal(lhs,
                       *rhs,
                       cudf::data_type{cudf::type_id::DECIMAL64, numeric::scale_type{-2}},
                       cudf::rounding_method::HALF_EVEN,
                       cudf::decimal_intermediate::STANDARD,
                       cudf::test::get_default_stream());
}

TEST_F(DivideDecimalTest, ScalarColumn)
{
  auto const lhs = cudf::make_fixed_point_scalar<numeric::decimal64>(
    1000, numeric::scale_type{-2}, cudf::test::get_default_stream());
  auto const rhs =
    cudf::test::fixed_point_column_wrapper<int64_t>{{300, 400, 0}, numeric::scale_type{-1}};
  cudf::divide_decimal(*lhs,
                       rhs,
                       cudf::data_type{cudf::type_id::DECIMAL64, numeric::scale_type{-2}},
                       cudf::rounding_method::HALF_UP,
                       cudf::decimal_intermediate::STANDARD,
                       cudf::test::get_default_stream());
}

TEST_F(DivideDecimalTest, WideIntermediate)
{
  // 1.0000000000 / 1.00000000000000000000 at scale -20 rescales the dividend past 128 bits
  auto const ten_10 = __int128_t{10'000'000'000};
  auto const lhs =
    cudf::test::fixed_point_column_wrapper<__int128_t>{{ten_10, -ten_10}, numeric::scale_type{-10}};
  auto const rhs = cudf::test::fixed_point_column_wrapper<__int128_t>{
    {ten_10 * ten_10, ten_10 * ten_10}, numeric::scale_type{-20}};
  cudf::divide_decimal(lhs,
                       rhs,
                       cudf::data_type{cudf::type_id::DECIMAL128, numeric::scale_type{-20}},
                       cudf::rounding_method::HALF_UP,
                       cudf::decimal_intermediate::WIDE,
                       cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
