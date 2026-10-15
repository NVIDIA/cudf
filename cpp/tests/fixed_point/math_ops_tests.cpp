/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/fixed_point/math_ops.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar_factories.hpp>

template <typename T>
struct FixedPointDivideDecimalTest : public cudf::test::BaseFixture {};

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

using DecimalTypes = ::testing::Types<numeric::decimal32, numeric::decimal64>;
TYPED_TEST_SUITE(FixedPointDivideDecimalTest, DecimalTypes);

TYPED_TEST(FixedPointDivideDecimalTest, Basic)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  // 10.00 / 2.00 = 5.00, 20.00 / 4.00 = 5.00, 30.00 / 3.00 = 10.00
  auto const lhs      = fp_wrapper<T>{{1000, 2000, 3000}, scale};
  auto const rhs      = fp_wrapper<T>{{200, 400, 300}, scale};
  auto const expected = fp_wrapper<T>{{500, 500, 1000}, scale};

  auto const result = cudf::divide_decimal(lhs, rhs);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  EXPECT_EQ(result->type().scale(), scale);
}

TYPED_TEST(FixedPointDivideDecimalTest, HalfUpRounding)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  // 10.00 / 3.00 = 3.333... -> 3.33, 20.00 / 3.00 = 6.666... -> 6.67, 5.00 / 2.00 = 2.50
  auto const lhs      = fp_wrapper<T>{{1000, 2000, 500}, scale};
  auto const rhs      = fp_wrapper<T>{{300, 300, 200}, scale};
  auto const expected = fp_wrapper<T>{{333, 667, 250}, scale};

  auto const result = cudf::divide_decimal(lhs, rhs, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  EXPECT_EQ(result->type().scale(), scale);
}

TYPED_TEST(FixedPointDivideDecimalTest, NegativeNumbers)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  // -10.00 / 3.00 = -3.33, 10.00 / -3.00 = -3.33, -10.00 / -3.00 = 3.33
  auto const lhs      = fp_wrapper<T>{{-1000, 1000, -1000}, scale};
  auto const rhs      = fp_wrapper<T>{{300, -300, -300}, scale};
  auto const expected = fp_wrapper<T>{{-333, -333, 333}, scale};

  auto const result = cudf::divide_decimal(lhs, rhs, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(FixedPointDivideDecimalTest, ColumnScalar)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  // 10.00 / 2.00 = 5.00, 20.00 / 2.00 = 10.00, 30.00 / 2.00 = 15.00
  auto const lhs      = fp_wrapper<T>{{1000, 2000, 3000}, scale};
  auto const rhs      = cudf::make_fixed_point_scalar<T>(200, scale);
  auto const expected = fp_wrapper<T>{{500, 1000, 1500}, scale};

  auto const result = cudf::divide_decimal(lhs, *rhs);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(FixedPointDivideDecimalTest, ScalarColumn)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  // 30.00 / 2.00 = 15.00, 30.00 / 3.00 = 10.00, 30.00 / 5.00 = 6.00
  auto const lhs      = cudf::make_fixed_point_scalar<T>(3000, scale);
  auto const rhs      = fp_wrapper<T>{{200, 300, 500}, scale};
  auto const expected = fp_wrapper<T>{{1500, 1000, 600}, scale};

  auto const result = cudf::divide_decimal(*lhs, rhs);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(FixedPointDivideDecimalTest, Nulls)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  // 10.00 / 2.00 = 5.00, null / 3.00 = null, 30.00 / null = null, null / null = null
  auto const lhs      = fp_wrapper<T>{{1000, 2000, 3000, 4000}, {1, 0, 1, 0}, scale};
  auto const rhs      = fp_wrapper<T>{{200, 300, 400, 500}, {1, 1, 0, 0}, scale};
  auto const expected = fp_wrapper<T>{{500, 0, 0, 0}, {1, 0, 0, 0}, scale};

  auto const result = cudf::divide_decimal(lhs, rhs);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(FixedPointDivideDecimalTest, EmptyColumns)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  auto const lhs      = fp_wrapper<T>{{}, scale};
  auto const rhs      = fp_wrapper<T>{{}, scale};
  auto const expected = fp_wrapper<T>{{}, scale};

  auto const result = cudf::divide_decimal(lhs, rhs);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  EXPECT_EQ(result->size(), 0);
}

TYPED_TEST(FixedPointDivideDecimalTest, DifferentScales)
{
  using T              = TypeParam;
  auto const lhs_scale = numeric::scale_type{-2};
  auto const rhs_scale = numeric::scale_type{-1};

  // 1.23 / 2.0 = 0.615 -> 0.62, 4.56 / 3.0 = 1.52, 7.89 / 4.0 = 1.9725 -> 1.97
  auto const lhs      = fp_wrapper<T>{{123, 456, 789}, lhs_scale};
  auto const rhs      = fp_wrapper<T>{{20, 30, 40}, rhs_scale};
  auto const expected = fp_wrapper<T>{{62, 152, 197}, lhs_scale};

  auto const result = cudf::divide_decimal(lhs, rhs, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  EXPECT_EQ(result->type().scale(), lhs_scale);
}
