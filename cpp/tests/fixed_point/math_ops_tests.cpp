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

#include <cuda/std/limits>

#include <type_traits>

template <typename T>
struct FixedPointDivideDecimalTest : public cudf::test::BaseFixture {};

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

using DecimalTypes = ::testing::Types<numeric::decimal32, numeric::decimal64, numeric::decimal128>;
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

TYPED_TEST(FixedPointDivideDecimalTest, RoundingTies)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{0};

  // 2.5, 3.5, -2.5, -3.5, -0.5, 0.333..., -0.5
  auto const lhs = fp_wrapper<T>{{5, 7, -5, -7, -1, 1, 4}, scale};
  auto const rhs = fp_wrapper<T>{{2, 2, 2, 2, 2, 3, -8}, scale};

  auto const half_up   = fp_wrapper<T>{{3, 4, -3, -4, -1, 0, -1}, scale};
  auto const half_even = fp_wrapper<T>{{2, 4, -2, -4, 0, 0, 0}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(half_up,
                                 *cudf::divide_decimal(lhs, rhs, cudf::rounding_method::HALF_UP));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(half_even,
                                 *cudf::divide_decimal(lhs, rhs, cudf::rounding_method::HALF_EVEN));
}

TYPED_TEST(FixedPointDivideDecimalTest, DivideByZeroIsNull)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{0};

  auto const lhs      = fp_wrapper<T>{{100, 200, 0}, scale};
  auto const rhs      = fp_wrapper<T>{{0, 100, 0}, scale};
  auto const expected = fp_wrapper<T>{{0, 2, 0}, {0, 1, 0}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs));
}

TYPED_TEST(FixedPointDivideDecimalTest, LimitsOfRepresentation)
{
  using T             = TypeParam;
  using Rep           = typename T::rep;
  auto const scale    = numeric::scale_type{0};
  auto constexpr min  = cuda::std::numeric_limits<Rep>::min();
  auto constexpr max  = cuda::std::numeric_limits<Rep>::max();
  Rep constexpr minus = -1;

  // MIN / -1 does not fit, but MAX / -1, MAX / 1 and MIN / 1 do.
  auto const lhs      = fp_wrapper<T>{{min, max, max, min}, scale};
  auto const rhs      = fp_wrapper<T>{{minus, minus, 1, 1}, scale};
  auto const expected = fp_wrapper<T>{{0, -max, max, min}, {0, 1, 1, 1}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs));
}

TYPED_TEST(FixedPointDivideDecimalTest, QuotientTooLargeIsNull)
{
  using T            = TypeParam;
  using Rep          = typename T::rep;
  auto constexpr max = cuda::std::numeric_limits<Rep>::max();

  // MAX / 0.1 = MAX * 10
  auto const lhs      = fp_wrapper<T>{{max, 1}, numeric::scale_type{0}};
  auto const rhs      = fp_wrapper<T>{{1, 1}, numeric::scale_type{-1}};
  auto const expected = fp_wrapper<T>{{0, 10}, {0, 1}, numeric::scale_type{0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs));
}

TYPED_TEST(FixedPointDivideDecimalTest, RescaledDividendWiderThanRep)
{
  using T            = TypeParam;
  using Rep          = typename T::rep;
  auto constexpr max = cuda::std::numeric_limits<Rep>::max();

  // MAX / 2.0 rescales MAX to MAX * 10, which only fits in a type wider than Rep.
  // There is no type wider than 128 bits, so that case is null.
  auto const lhs = fp_wrapper<T>{{max}, numeric::scale_type{0}};
  auto const rhs = fp_wrapper<T>{{20}, numeric::scale_type{-1}};

  auto const result = cudf::divide_decimal(lhs, rhs, cudf::rounding_method::HALF_UP);

  if constexpr (std::is_same_v<T, numeric::decimal128>) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fp_wrapper<T>({0}, {0}, numeric::scale_type{0}), *result);
  } else {
    Rep const half_max_rounded_up = max / 2 + 1;
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fp_wrapper<T>({half_max_rounded_up}, numeric::scale_type{0}),
                                   *result);
  }
}

TYPED_TEST(FixedPointDivideDecimalTest, PositiveDivisorScale)
{
  using T = TypeParam;

  // 10.00 / 50 = 0.20, 10.01 / 50 = 0.2002, -10.25 / 50 = -0.205
  auto const lhs      = fp_wrapper<T>{{1000, 1001, -1025}, numeric::scale_type{-2}};
  auto const rhs      = fp_wrapper<T>{{5, 5, 5}, numeric::scale_type{1}};
  auto const expected = fp_wrapper<T>{{20, 20, -21}, numeric::scale_type{-2}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs));
}

TYPED_TEST(FixedPointDivideDecimalTest, HugeDivisorScaleRoundsToZero)
{
  using T = TypeParam;

  // 100 / 10^40 and -100 / 10^40 both round to zero
  auto const lhs      = fp_wrapper<T>{{100, -100}, numeric::scale_type{0}};
  auto const rhs      = fp_wrapper<T>{{1, 1}, numeric::scale_type{40}};
  auto const expected = fp_wrapper<T>{{0, 0}, numeric::scale_type{0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs));
}

TYPED_TEST(FixedPointDivideDecimalTest, ColumnScalarDifferentScales)
{
  using T = TypeParam;

  // 1.23 / 2.0 = 0.615 -> 0.62, 4.56 / 2.0 = 2.28
  auto const lhs      = fp_wrapper<T>{{123, 456}, numeric::scale_type{-2}};
  auto const rhs      = cudf::make_fixed_point_scalar<T>(20, numeric::scale_type{-1});
  auto const expected = fp_wrapper<T>{{62, 228}, numeric::scale_type{-2}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, *rhs));
}

TYPED_TEST(FixedPointDivideDecimalTest, ScalarColumnKeepsScalarScale)
{
  using T = TypeParam;

  // 1.5 / 3 = 0.5, 1.5 / 4 = 0.375 -> 0.4
  auto const lhs      = cudf::make_fixed_point_scalar<T>(15, numeric::scale_type{-1});
  auto const rhs      = fp_wrapper<T>{{3, 4}, numeric::scale_type{0}};
  auto const expected = fp_wrapper<T>{{5, 4}, numeric::scale_type{-1}};

  auto const result = cudf::divide_decimal(*lhs, rhs);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  EXPECT_EQ(result->type().scale(), -1);
}

TYPED_TEST(FixedPointDivideDecimalTest, InvalidScalarGivesAllNulls)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  auto const column = fp_wrapper<T>{{100, 200}, scale};
  auto scalar       = cudf::make_fixed_point_scalar<T>(0, scale);
  scalar->set_valid_async(false);
  auto const expected = fp_wrapper<T>{{0, 0}, {0, 0}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(column, *scalar));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(*scalar, column));
}

TYPED_TEST(FixedPointDivideDecimalTest, NullRowsAreNotDivided)
{
  using T            = TypeParam;
  using Rep          = typename T::rep;
  auto const scale   = numeric::scale_type{0};
  auto constexpr min = cuda::std::numeric_limits<Rep>::min();

  // The null divisor holds a zero, and the null dividend holds MIN over a divisor of -1.
  auto const lhs      = fp_wrapper<T>{{100, min, 200}, {1, 0, 1}, scale};
  auto const rhs      = fp_wrapper<T>{{0, -1, 100}, {0, 1, 1}, scale};
  auto const expected = fp_wrapper<T>{{0, 0, 2}, {0, 0, 1}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs));
}

TYPED_TEST(FixedPointDivideDecimalTest, AllNullColumnGivesAllNulls)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  auto const valid    = fp_wrapper<T>{{100, 200}, scale};
  auto const all_null = fp_wrapper<T>{{100, 200}, {0, 0}, scale};
  auto const scalar   = cudf::make_fixed_point_scalar<T>(100, scale);
  auto const expected = fp_wrapper<T>{{0, 0}, {0, 0}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(all_null, valid));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(valid, all_null));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(all_null, *scalar));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(*scalar, all_null));
}

TYPED_TEST(FixedPointDivideDecimalTest, RescaledDividendAround64Bits)
{
  using T   = TypeParam;
  using Rep = typename T::rep;
  if constexpr (sizeof(Rep) < sizeof(int64_t)) {
    GTEST_SKIP() << "Rescaled decimal32 dividends always fit in 64 bits";
  } else {
    // Rescaled by 10, the first dividend is 2^64 - 6 and the second is 2^64 + 4.
    auto const lhs = fp_wrapper<T>{{Rep{1'844'674'407'370'955'161}, Rep{1'844'674'407'370'955'162}},
                                   numeric::scale_type{0}};
    auto const rhs = fp_wrapper<T>{{30, 30}, numeric::scale_type{-1}};
    auto const expected = fp_wrapper<T>{
      {Rep{614'891'469'123'651'720}, Rep{614'891'469'123'651'721}}, numeric::scale_type{0}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs));
  }
}

TEST(FixedPointDivideDecimal128Test, DivisorWiderThan64Bits)
{
  using Rep         = __int128_t;
  auto const two_64 = Rep{1} << 64;

  // (2^64 - 1) / (2^64 + 2) rounds up to 1. Truncating the divisor to 64 bits would give 2^63.
  auto const lhs      = fp_wrapper<numeric::decimal128>{{two_64 - 1}, numeric::scale_type{0}};
  auto const rhs      = fp_wrapper<numeric::decimal128>{{two_64 + 2}, numeric::scale_type{0}};
  auto const expected = fp_wrapper<numeric::decimal128>{{1}, numeric::scale_type{0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs));
}

TEST(FixedPointDivideDecimalErrorTest, InvalidInputs)
{
  auto const scale   = numeric::scale_type{0};
  auto const dec32   = cudf::test::fixed_point_column_wrapper<int32_t>{{100, 200}, scale};
  auto const dec64   = cudf::test::fixed_point_column_wrapper<int64_t>{{100, 200}, scale};
  auto const short32 = cudf::test::fixed_point_column_wrapper<int32_t>{{100}, scale};
  auto const ints    = cudf::test::fixed_width_column_wrapper<int32_t>{100, 200};

  EXPECT_THROW(cudf::divide_decimal(dec32, dec64), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(dec32, short32), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(ints, ints), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(dec32, ints), cudf::logic_error);

  auto const dec32_scalar = cudf::make_fixed_point_scalar<numeric::decimal32>(100, scale);
  auto const dec64_scalar = cudf::make_fixed_point_scalar<numeric::decimal64>(100, scale);
  auto const int_scalar   = cudf::numeric_scalar<int32_t>(100);

  EXPECT_THROW(cudf::divide_decimal(dec32, *dec64_scalar), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(*dec64_scalar, dec32), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(ints, *dec32_scalar), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(*dec32_scalar, ints), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(dec32, int_scalar), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(int_scalar, dec32), cudf::logic_error);
}
