/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/decimal/decimal_ops.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar_factories.hpp>

using namespace cudf;
using namespace cudf::test;
using namespace numeric;

template <typename DecimalType>
class DecimalOpsTest : public cudf::test::BaseFixture {};

// Use fixed_point_column_wrapper for decimal types
template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

using DecimalTypes = ::testing::Types<numeric::decimal32, numeric::decimal64>;
TYPED_TEST_SUITE(DecimalOpsTest, DecimalTypes);

TYPED_TEST(DecimalOpsTest, DivideDecimalBasic)
{
  using DecimalType = TypeParam;

  // Scale -2 means 2 decimal places (10^-2)
  auto const scale = scale_type{-2};

  // Test basic division with scale preservation
  // 10.00 / 2.00 = 5.00
  // 20.00 / 4.00 = 5.00
  // 30.00 / 3.00 = 10.00
  fp_wrapper<DecimalType> lhs_col{{1000, 2000, 3000}, scale};
  fp_wrapper<DecimalType> rhs_col{{200, 400, 300}, scale};
  fp_wrapper<DecimalType> expected{{500, 500, 1000}, scale};

  auto result = divide_decimal(lhs_col, rhs_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  EXPECT_EQ(result->type().scale(), scale);
}

TYPED_TEST(DecimalOpsTest, DivideDecimalWithRounding)
{
  using DecimalType = TypeParam;

  auto const scale = scale_type{-2};

  // Test rounding HALF_UP
  // 10.00 / 3.00 = 3.33 (rounded from 3.333...)
  // 20.00 / 3.00 = 6.67 (rounded from 6.666...)
  // 5.00 / 2.00 = 2.50
  fp_wrapper<DecimalType> lhs_col{{1000, 2000, 500}, scale};
  fp_wrapper<DecimalType> rhs_col{{300, 300, 200}, scale};
  fp_wrapper<DecimalType> expected_half_up{{333, 667, 250}, scale};

  auto result = divide_decimal(lhs_col, rhs_col, decimal_rounding_mode::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_half_up, *result);
  EXPECT_EQ(result->type().scale(), scale);
}

TYPED_TEST(DecimalOpsTest, DivideDecimalNegativeNumbers)
{
  using DecimalType = TypeParam;

  auto const scale = scale_type{-2};

  // Test with negative numbers
  // -10.00 / 3.00 = -3.33 (rounded from -3.333...)
  // 10.00 / -3.00 = -3.33
  // -10.00 / -3.00 = 3.33
  fp_wrapper<DecimalType> lhs_col{{-1000, 1000, -1000}, scale};
  fp_wrapper<DecimalType> rhs_col{{300, -300, -300}, scale};
  fp_wrapper<DecimalType> expected{{-333, -333, 333}, scale};

  auto result = divide_decimal(lhs_col, rhs_col, decimal_rounding_mode::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(DecimalOpsTest, DivideDecimalColumnScalar)
{
  using DecimalType = TypeParam;

  auto const scale = scale_type{-2};

  // Test column / scalar
  // 10.00 / 2.00 = 5.00
  // 20.00 / 2.00 = 10.00
  // 30.00 / 2.00 = 15.00
  fp_wrapper<DecimalType> lhs_col{{1000, 2000, 3000}, scale};
  auto rhs_scalar = make_fixed_point_scalar<DecimalType>(200, scale);
  fp_wrapper<DecimalType> expected{{500, 1000, 1500}, scale};

  auto result = divide_decimal(lhs_col, *rhs_scalar);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(DecimalOpsTest, DivideDecimalScalarColumn)
{
  using DecimalType = TypeParam;

  auto const scale = scale_type{-2};

  // Test scalar / column
  // 30.00 / 2.00 = 15.00
  // 30.00 / 3.00 = 10.00
  // 30.00 / 5.00 = 6.00
  auto lhs_scalar = make_fixed_point_scalar<DecimalType>(3000, scale);
  fp_wrapper<DecimalType> rhs_col{{200, 300, 500}, scale};
  fp_wrapper<DecimalType> expected{{1500, 1000, 600}, scale};

  auto result = divide_decimal(*lhs_scalar, rhs_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(DecimalOpsTest, DivideDecimalWithNulls)
{
  using DecimalType = TypeParam;

  auto const scale = scale_type{-2};

  // Test with null values
  // 10.00 / 2.00 = 5.00
  // null / 3.00 = null
  // 30.00 / null = null
  // null / null = null
  fp_wrapper<DecimalType> lhs_col{{1000, 2000, 3000, 4000}, {1, 0, 1, 0}, scale};
  fp_wrapper<DecimalType> rhs_col{{200, 300, 400, 500}, {1, 1, 0, 0}, scale};
  fp_wrapper<DecimalType> expected{{500, 0, 0, 0}, {1, 0, 0, 0}, scale};

  auto result = divide_decimal(lhs_col, rhs_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(DecimalOpsTest, DivideDecimalEmptyColumns)
{
  using DecimalType = TypeParam;

  auto const scale = scale_type{-2};

  // Test with empty columns
  fp_wrapper<DecimalType> lhs_col{{}, scale};
  fp_wrapper<DecimalType> rhs_col{{}, scale};
  fp_wrapper<DecimalType> expected{{}, scale};

  auto result = divide_decimal(lhs_col, rhs_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  EXPECT_EQ(result->size(), 0);
}

TYPED_TEST(DecimalOpsTest, DivideDecimalDifferentScales)
{
  using DecimalType = TypeParam;

  // Test with different scales
  // lhs has scale -2 (2 decimal places): 1.23
  // rhs has scale -1 (1 decimal place): 2.0
  // Result should have scale -2: 0.62 (rounded from 0.615)
  auto const lhs_scale = scale_type{-2};
  auto const rhs_scale = scale_type{-1};

  // 1.23 / 2.0 = 0.615 -> 0.62 (HALF_UP)
  // 4.56 / 3.0 = 1.52
  // 7.89 / 4.0 = 1.9725 -> 1.97 (HALF_UP)
  fp_wrapper<DecimalType> lhs_col{{123, 456, 789}, lhs_scale};
  fp_wrapper<DecimalType> rhs_col{{20, 30, 40}, rhs_scale};
  fp_wrapper<DecimalType> expected{{62, 152, 197}, lhs_scale};

  auto result = divide_decimal(lhs_col, rhs_col, decimal_rounding_mode::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  EXPECT_EQ(result->type().scale(), lhs_scale);
}
