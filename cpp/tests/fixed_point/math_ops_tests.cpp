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
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/limits>

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

template <typename T>
struct FixedPointDivideDecimalTest : public cudf::test::BaseFixture {};

template <typename T>
using fp_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

template <typename T>
cudf::data_type decimal_type(numeric::scale_type scale)
{
  return cudf::data_type{cudf::type_to_id<T>(), scale};
}

// The largest representation with no more digits than `T` can always hold
template <typename T>
constexpr typename T::rep max_digits()
{
  return numeric::detail::ipow<typename T::rep, numeric::Radix::BASE_10>(
           cuda::std::numeric_limits<typename T::rep>::digits10) -
         1;
}

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

  auto const result = cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale));

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

  auto const result =
    cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale), cudf::rounding_method::HALF_UP);

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

  auto const result =
    cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale), cudf::rounding_method::HALF_UP);

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

  auto const result = cudf::divide_decimal(lhs, *rhs, decimal_type<T>(scale));

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

  auto const result = cudf::divide_decimal(*lhs, rhs, decimal_type<T>(scale));

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

  auto const result = cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(FixedPointDivideDecimalTest, EmptyColumns)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  auto const lhs = fp_wrapper<T>{{}, scale};
  auto const rhs = fp_wrapper<T>{{}, scale};

  auto const result =
    cudf::divide_decimal(lhs, rhs, decimal_type<numeric::decimal64>(numeric::scale_type{-4}));

  EXPECT_EQ(result->size(), 0);
  EXPECT_EQ(result->type(), decimal_type<numeric::decimal64>(numeric::scale_type{-4}));
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

  auto const result =
    cudf::divide_decimal(lhs, rhs, decimal_type<T>(lhs_scale), cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  EXPECT_EQ(result->type().scale(), lhs_scale);
}

TYPED_TEST(FixedPointDivideDecimalTest, OutputScale)
{
  using T = TypeParam;

  // 1.23 / 2.0 = 0.615
  auto const lhs = fp_wrapper<T>{{123, -123}, numeric::scale_type{-2}};
  auto const rhs = fp_wrapper<T>{{20, 20}, numeric::scale_type{-1}};

  auto const divide = [&](int32_t scale) {
    return cudf::divide_decimal(lhs, rhs, decimal_type<T>(numeric::scale_type{scale}));
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fp_wrapper<T>({6150, -6150}, numeric::scale_type{-4}),
                                 *divide(-4));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fp_wrapper<T>({6, -6}, numeric::scale_type{-1}), *divide(-1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fp_wrapper<T>({1, -1}, numeric::scale_type{0}), *divide(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fp_wrapper<T>({0, 0}, numeric::scale_type{1}), *divide(1));
}

TYPED_TEST(FixedPointDivideDecimalTest, PositiveOutputScale)
{
  using T = TypeParam;

  // 1234.5 / 1 -> 123 tens, 1250.0 / 1 = 125 tens exactly
  auto const lhs  = fp_wrapper<T>{{12345, 12500}, numeric::scale_type{-1}};
  auto const rhs  = fp_wrapper<T>{{1, 1}, numeric::scale_type{0}};
  auto const type = decimal_type<T>(numeric::scale_type{1});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<T>({123, 125}, numeric::scale_type{1}),
    *cudf::divide_decimal(lhs, rhs, type, cudf::rounding_method::HALF_UP));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<T>({123, 125}, numeric::scale_type{1}),
    *cudf::divide_decimal(lhs, rhs, type, cudf::rounding_method::HALF_EVEN));

  // 125 / 1 -> 13 (HALF_UP) or 12 (HALF_EVEN) tens, 135 / 1 -> 14 tens
  auto const ties = fp_wrapper<T>{{125, 135}, numeric::scale_type{0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<T>({13, 14}, numeric::scale_type{1}),
    *cudf::divide_decimal(ties, rhs, type, cudf::rounding_method::HALF_UP));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<T>({12, 14}, numeric::scale_type{1}),
    *cudf::divide_decimal(ties, rhs, type, cudf::rounding_method::HALF_EVEN));
}

TYPED_TEST(FixedPointDivideDecimalTest, SmallerOperandScale)
{
  using T              = TypeParam;
  auto const lhs_scale = numeric::scale_type{-1};
  auto const rhs_scale = numeric::scale_type{-3};

  // 1.5 / 0.700 = 2.142857... -> 2.143, at the scale of the operand with more fractional digits
  auto const lhs      = fp_wrapper<T>{{15}, lhs_scale};
  auto const rhs      = fp_wrapper<T>{{700}, rhs_scale};
  auto const expected = fp_wrapper<T>{{2143}, rhs_scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, *cudf::divide_decimal(lhs, rhs, decimal_type<T>(std::min(lhs_scale, rhs_scale))));
}

TYPED_TEST(FixedPointDivideDecimalTest, RoundingTies)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{0};
  auto const type  = decimal_type<T>(scale);

  // 2.5, 3.5, -2.5, -3.5, -0.5, 0.333..., -0.5
  auto const lhs = fp_wrapper<T>{{5, 7, -5, -7, -1, 1, 4}, scale};
  auto const rhs = fp_wrapper<T>{{2, 2, 2, 2, 2, 3, -8}, scale};

  auto const half_up   = fp_wrapper<T>{{3, 4, -3, -4, -1, 0, -1}, scale};
  auto const half_even = fp_wrapper<T>{{2, 4, -2, -4, 0, 0, 0}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    half_up, *cudf::divide_decimal(lhs, rhs, type, cudf::rounding_method::HALF_UP));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    half_even, *cudf::divide_decimal(lhs, rhs, type, cudf::rounding_method::HALF_EVEN));
}

TYPED_TEST(FixedPointDivideDecimalTest, DivideByZeroIsNull)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{0};

  auto const lhs      = fp_wrapper<T>{{100, 200, 0}, scale};
  auto const rhs      = fp_wrapper<T>{{0, 100, 0}, scale};
  auto const expected = fp_wrapper<T>{{0, 2, 0}, {0, 1, 0}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale)));
}

TYPED_TEST(FixedPointDivideDecimalTest, PrecisionLimit)
{
  using T            = TypeParam;
  using Rep          = typename T::rep;
  auto const scale   = numeric::scale_type{0};
  auto constexpr max = max_digits<T>();
  auto constexpr min = cuda::std::numeric_limits<Rep>::min();

  // Quotients with as many digits as the output can hold fit, and one more digit does not.
  // MIN / -1 does not fit either.
  auto const lhs      = fp_wrapper<T>{{max, -max, max + 1, -max - 1, min}, scale};
  auto const rhs      = fp_wrapper<T>{{1, 1, 1, 1, -1}, scale};
  auto const expected = fp_wrapper<T>{{max, -max, 0, 0, 0}, {1, 1, 0, 0, 0}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale)));
}

TYPED_TEST(FixedPointDivideDecimalTest, RoundingPastPrecisionLimitIsNull)
{
  using T            = TypeParam;
  using Rep          = typename T::rep;
  auto const scale   = numeric::scale_type{0};
  auto constexpr max = max_digits<T>();
  if constexpr (std::is_same_v<T, numeric::decimal128>) {
    GTEST_SKIP() << "2 * 10^38 does not fit in a decimal128 dividend";
  } else {
    // (2 * max + 1) / 2 = max + 0.5, which rounds to a quotient with one digit too many
    auto const lhs      = fp_wrapper<T>{{Rep{2} * max + 1, Rep{2} * max - 1}, scale};
    auto const rhs      = fp_wrapper<T>{{2, 2}, scale};
    auto const expected = fp_wrapper<T>{{0, max}, {0, 1}, scale};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected,
                                   *cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale)));
  }
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

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, *cudf::divide_decimal(lhs, rhs, decimal_type<T>(numeric::scale_type{0})));
}

TYPED_TEST(FixedPointDivideDecimalTest, RescaledDividendWiderThanRep)
{
  using T            = TypeParam;
  auto constexpr max = max_digits<T>();
  auto const scale   = numeric::scale_type{0};

  // max / 2.0 rescales max to max * 10, which only fits in a type wider than the dividend.
  // The standard intermediate has no type wider than 128 bits, so that case is null.
  auto const lhs      = fp_wrapper<T>{{max}, scale};
  auto const rhs      = fp_wrapper<T>{{20}, numeric::scale_type{-1}};
  auto const expected = fp_wrapper<T>{{max / 2 + 1}, scale};

  auto const result =
    cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale), cudf::rounding_method::HALF_UP);

  if constexpr (std::is_same_v<T, numeric::decimal128>) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fp_wrapper<T>({0}, {0}, scale), *result);
  } else {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected,
                                 *cudf::divide_decimal(lhs,
                                                       rhs,
                                                       decimal_type<T>(scale),
                                                       cudf::rounding_method::HALF_UP,
                                                       cudf::decimal_intermediate::WIDE));
}

TYPED_TEST(FixedPointDivideDecimalTest, OutputScaleRescaleOverflow)
{
  using T = TypeParam;

  // Rescaling the dividend by 10^40 overflows every width, so the result is null.
  // Rescaling the divisor by 10^40 makes the quotient round to zero.
  auto const lhs = fp_wrapper<T>{{1, -1}, numeric::scale_type{0}};
  auto const rhs = fp_wrapper<T>{{1, 1}, numeric::scale_type{0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<T>({0, 0}, {0, 0}, numeric::scale_type{-40}),
    *cudf::divide_decimal(lhs, rhs, decimal_type<T>(numeric::scale_type{-40})));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<T>({0, 0}, numeric::scale_type{40}),
    *cudf::divide_decimal(lhs, rhs, decimal_type<T>(numeric::scale_type{40})));
}

TYPED_TEST(FixedPointDivideDecimalTest, PositiveDivisorScale)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  // 10.00 / 50 = 0.20, 10.01 / 50 = 0.2002, -10.25 / 50 = -0.205
  auto const lhs      = fp_wrapper<T>{{1000, 1001, -1025}, scale};
  auto const rhs      = fp_wrapper<T>{{5, 5, 5}, numeric::scale_type{1}};
  auto const expected = fp_wrapper<T>{{20, 20, -21}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale)));
}

TYPED_TEST(FixedPointDivideDecimalTest, HugeDivisorScaleRoundsToZero)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{0};

  // 100 / 10^40 and -100 / 10^40 both round to zero
  auto const lhs      = fp_wrapper<T>{{100, -100}, scale};
  auto const rhs      = fp_wrapper<T>{{1, 1}, numeric::scale_type{40}};
  auto const expected = fp_wrapper<T>{{0, 0}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale)));
}

TYPED_TEST(FixedPointDivideDecimalTest, ColumnScalarDifferentScales)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};

  // 1.23 / 2.0 = 0.615 -> 0.62, 4.56 / 2.0 = 2.28
  auto const lhs      = fp_wrapper<T>{{123, 456}, scale};
  auto const rhs      = cudf::make_fixed_point_scalar<T>(20, numeric::scale_type{-1});
  auto const expected = fp_wrapper<T>{{62, 228}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected,
                                 *cudf::divide_decimal(lhs, *rhs, decimal_type<T>(scale)));
}

TYPED_TEST(FixedPointDivideDecimalTest, ScalarColumnDifferentScales)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-1};

  // 1.5 / 3 = 0.5, 1.5 / 4 = 0.375 -> 0.4
  auto const lhs      = cudf::make_fixed_point_scalar<T>(15, scale);
  auto const rhs      = fp_wrapper<T>{{3, 4}, numeric::scale_type{0}};
  auto const expected = fp_wrapper<T>{{5, 4}, scale};

  auto const result = cudf::divide_decimal(*lhs, rhs, decimal_type<T>(scale));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  EXPECT_EQ(result->type().scale(), -1);
}

TYPED_TEST(FixedPointDivideDecimalTest, InvalidScalarGivesAllNulls)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};
  auto const type  = decimal_type<T>(scale);

  auto const column = fp_wrapper<T>{{100, 200}, scale};
  auto scalar       = cudf::make_fixed_point_scalar<T>(0, scale);
  scalar->set_valid_async(false);
  auto const expected = fp_wrapper<T>{{0, 0}, {0, 0}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(column, *scalar, type));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(*scalar, column, type));
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

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale)));
}

TYPED_TEST(FixedPointDivideDecimalTest, AllNullColumnGivesAllNulls)
{
  using T          = TypeParam;
  auto const scale = numeric::scale_type{-2};
  auto const type  = decimal_type<numeric::decimal128>(numeric::scale_type{-4});

  auto const valid    = fp_wrapper<T>{{100, 200}, scale};
  auto const all_null = fp_wrapper<T>{{100, 200}, {0, 0}, scale};
  auto const scalar   = cudf::make_fixed_point_scalar<T>(100, scale);
  auto const expected = fp_wrapper<numeric::decimal128>{{0, 0}, {0, 0}, numeric::scale_type{-4}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(all_null, valid, type));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(valid, all_null, type));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(all_null, *scalar, type));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *cudf::divide_decimal(*scalar, all_null, type));
}

TYPED_TEST(FixedPointDivideDecimalTest, RescaledDividendAround64Bits)
{
  using T   = TypeParam;
  using Rep = typename T::rep;
  if constexpr (sizeof(Rep) < sizeof(int64_t)) {
    GTEST_SKIP() << "Rescaled decimal32 dividends always fit in 64 bits";
  } else {
    auto const scale = numeric::scale_type{0};
    // Rescaled by 10, the first dividend is 2^64 - 6 and the second is 2^64 + 4.
    auto const lhs =
      fp_wrapper<T>{{Rep{1'844'674'407'370'955'161}, Rep{1'844'674'407'370'955'162}}, scale};
    auto const rhs = fp_wrapper<T>{{30, 30}, numeric::scale_type{-1}};
    auto const expected =
      fp_wrapper<T>{{Rep{614'891'469'123'651'720}, Rep{614'891'469'123'651'721}}, scale};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected,
                                   *cudf::divide_decimal(lhs, rhs, decimal_type<T>(scale)));
  }
}

TEST(FixedPointDivideDecimal128Test, DivisorWiderThan64Bits)
{
  using Rep         = __int128_t;
  auto const two_64 = Rep{1} << 64;
  auto const scale  = numeric::scale_type{0};

  // (2^64 - 1) / (2^64 + 2) rounds up to 1. Truncating the divisor to 64 bits would give 2^63.
  auto const lhs      = fp_wrapper<numeric::decimal128>{{two_64 - 1}, scale};
  auto const rhs      = fp_wrapper<numeric::decimal128>{{two_64 + 2}, scale};
  auto const expected = fp_wrapper<numeric::decimal128>{{1}, scale};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, *cudf::divide_decimal(lhs, rhs, decimal_type<numeric::decimal128>(scale)));
}

TEST(FixedPointDivideDecimalMixedWidthTest, OperandsOfDifferentWidths)
{
  // 1.23 / 2.0 = 0.615
  auto const lhs64  = fp_wrapper<numeric::decimal64>{{123, -123}, numeric::scale_type{-2}};
  auto const rhs128 = fp_wrapper<numeric::decimal128>{{20, 20}, numeric::scale_type{-1}};
  auto const lhs32  = fp_wrapper<numeric::decimal32>{{123, -123}, numeric::scale_type{-2}};
  auto const rhs64  = fp_wrapper<numeric::decimal64>{{20, 20}, numeric::scale_type{-1}};
  auto const lhs128 = fp_wrapper<numeric::decimal128>{{123, -123}, numeric::scale_type{-2}};
  auto const rhs32  = fp_wrapper<numeric::decimal32>{{20, 20}, numeric::scale_type{-1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<numeric::decimal64>({62, -62}, numeric::scale_type{-2}),
    *cudf::divide_decimal(
      lhs64, rhs128, decimal_type<numeric::decimal64>(numeric::scale_type{-2})));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<numeric::decimal128>({6150, -6150}, numeric::scale_type{-4}),
    *cudf::divide_decimal(
      lhs32, rhs64, decimal_type<numeric::decimal128>(numeric::scale_type{-4})));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<numeric::decimal32>({6, -6}, numeric::scale_type{-1}),
    *cudf::divide_decimal(
      lhs128, rhs32, decimal_type<numeric::decimal32>(numeric::scale_type{-1})));
}

TEST(FixedPointDivideDecimalMixedWidthTest, WiderOutputHoldsRescaledDividend)
{
  using Rep = __int128_t;

  // 999999999 / 1 at scale -20 needs 29 digits, which only a decimal128 output can hold.
  auto const max32  = max_digits<numeric::decimal32>();
  auto const lhs    = fp_wrapper<numeric::decimal32>{{max32}, numeric::scale_type{0}};
  auto const rhs    = fp_wrapper<numeric::decimal32>{{1}, numeric::scale_type{0}};
  auto const factor = numeric::detail::ipow<Rep, numeric::Radix::BASE_10>(static_cast<int32_t>(20));
  auto const expected =
    fp_wrapper<numeric::decimal128>{{Rep{max32} * factor}, numeric::scale_type{-20}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected,
    *cudf::divide_decimal(lhs, rhs, decimal_type<numeric::decimal128>(numeric::scale_type{-20})));
}

TEST(FixedPointDivideDecimalMixedWidthTest, NarrowerOutputPrecisionLimit)
{
  using Rep = __int128_t;

  // A decimal128 quotient with more than 9 digits does not fit in a decimal32 output.
  auto const max32 = max_digits<numeric::decimal32>();
  auto const lhs   = fp_wrapper<numeric::decimal128>{{Rep{max32}, Rep{max32} + 1, Rep{1} << 100},
                                                     numeric::scale_type{0}};
  auto const rhs   = fp_wrapper<numeric::decimal128>{{1, 1, 1}, numeric::scale_type{0}};
  auto const expected =
    fp_wrapper<numeric::decimal32>{{max32, 0, 0}, {1, 0, 0}, numeric::scale_type{0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected,
    *cudf::divide_decimal(lhs, rhs, decimal_type<numeric::decimal32>(numeric::scale_type{0})));
}

TEST(FixedPointDivideDecimalMixedWidthTest, ScalarOfDifferentWidth)
{
  // 10.00 / 4 = 2.50, -3.00 / 4 = -0.75
  auto const column = fp_wrapper<numeric::decimal32>{{1000, -300}, numeric::scale_type{-2}};
  auto const scalar = cudf::make_fixed_point_scalar<numeric::decimal128>(4, numeric::scale_type{0});
  auto const type   = decimal_type<numeric::decimal64>(numeric::scale_type{-2});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<numeric::decimal64>({250, -75}, numeric::scale_type{-2}),
    *cudf::divide_decimal(column, *scalar, type));
  // 4 / 10.00 = 0.40, 4 / -3.00 = -1.333... -> -1.33
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    fp_wrapper<numeric::decimal64>({40, -133}, numeric::scale_type{-2}),
    *cudf::divide_decimal(*scalar, column, type));
}

template <typename Lhs, typename Rhs>
std::unique_ptr<cudf::column> divide_wide(
  Lhs const& lhs,
  Rhs const& rhs,
  cudf::data_type output_type,
  cudf::rounding_method method = cudf::rounding_method::HALF_UP)
{
  return cudf::divide_decimal(lhs, rhs, output_type, method, cudf::decimal_intermediate::WIDE);
}

__int128_t pow10_128(int32_t exponent)
{
  return numeric::detail::ipow<__int128_t, numeric::Radix::BASE_10>(exponent);
}

using dec128_wrapper = fp_wrapper<numeric::decimal128>;

TYPED_TEST(FixedPointDivideDecimalTest, WideMatchesStandardWhereStandardIsValid)
{
  using T                = TypeParam;
  using Rep              = typename T::rep;
  auto constexpr max     = max_digits<T>();
  auto constexpr rep_max = cuda::std::numeric_limits<Rep>::max();
  auto constexpr rep_min = cuda::std::numeric_limits<Rep>::min();

  std::vector<Rep> const dividends{0, 1, -7, 123, max / 7, -(max / 3), max, -max, rep_max, rep_min};
  std::vector<Rep> const divisors{1, -3, 7, 20, max / 1000 + 1, -max, rep_min};
  std::vector<Rep> lhs_data;
  std::vector<Rep> rhs_data;
  for (auto const dividend : dividends) {
    for (auto const divisor : divisors) {
      lhs_data.push_back(dividend);
      rhs_data.push_back(divisor);
    }
  }
  auto const lhs = fp_wrapper<T>(lhs_data.begin(), lhs_data.end(), numeric::scale_type{0});
  auto const rhs = fp_wrapper<T>(rhs_data.begin(), rhs_data.end(), numeric::scale_type{0});

  auto const is_valid = [](std::vector<cudf::bitmask_type> const& mask, cudf::size_type i) {
    return mask.empty() || cudf::bit_is_set(mask.data(), i);
  };

  // Exponents from -40 to 45, through the wide path's multi-step rescaling
  auto wide_only = 0;
  for (int32_t scale = 40; scale >= -45; --scale) {
    auto const type     = decimal_type<T>(numeric::scale_type{scale});
    auto const standard = cudf::divide_decimal(lhs, rhs, type, cudf::rounding_method::HALF_EVEN);
    auto const wide     = divide_wide(lhs, rhs, type, cudf::rounding_method::HALF_EVEN);
    auto const [standard_data, standard_mask] = cudf::test::to_host<T>(*standard);
    auto const [wide_data, wide_mask]         = cudf::test::to_host<T>(*wide);
    for (cudf::size_type i = 0; i < standard->size(); ++i) {
      if (is_valid(standard_mask, i)) {
        EXPECT_TRUE(is_valid(wide_mask, i)) << "scale " << scale << ", row " << i;
        EXPECT_EQ(standard_data[i].value(), wide_data[i].value())
          << "scale " << scale << ", row " << i;
      } else if (is_valid(wide_mask, i)) {
        ++wide_only;
      }
    }
  }

  // Only decimal128 has quotients that fit but whose rescaled dividend overflows 128 bits
  if constexpr (std::is_same_v<T, numeric::decimal128>) {
    EXPECT_GT(wide_only, 0);
  } else {
    EXPECT_EQ(wide_only, 0);
  }
}

TEST(FixedPointDivideDecimalWideTest, RescaledDividendWiderThan128Bits)
{
  auto const ten_10 = pow10_128(10);
  auto const ten_20 = pow10_128(20);
  auto const type   = decimal_type<numeric::decimal128>(numeric::scale_type{-20});

  // 1.0000000000 / 1.00000000000000000000 at scale -20 rescales the dividend to 10^40
  auto const lhs = dec128_wrapper{{ten_10, -ten_10, 3 * ten_10}, numeric::scale_type{-10}};
  auto const rhs = dec128_wrapper{{ten_20, ten_20, -ten_20}, numeric::scale_type{-20}};
  auto const lhs_scalar =
    cudf::make_fixed_point_scalar<numeric::decimal128>(ten_10, numeric::scale_type{-10});
  auto const rhs_scalar =
    cudf::make_fixed_point_scalar<numeric::decimal128>(ten_20, numeric::scale_type{-20});
  auto const all_null = dec128_wrapper{{0, 0, 0}, {0, 0, 0}, numeric::scale_type{-20}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(all_null, *cudf::divide_decimal(lhs, rhs, type));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(all_null, *cudf::divide_decimal(lhs, *rhs_scalar, type));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(all_null, *cudf::divide_decimal(*lhs_scalar, rhs, type));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    dec128_wrapper({ten_20, -ten_20, -3 * ten_20}, numeric::scale_type{-20}),
    *divide_wide(lhs, rhs, type));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    dec128_wrapper({ten_20, -ten_20, 3 * ten_20}, numeric::scale_type{-20}),
    *divide_wide(lhs, *rhs_scalar, type));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    dec128_wrapper({ten_20, ten_20, -ten_20}, numeric::scale_type{-20}),
    *divide_wide(*lhs_scalar, rhs, type));
}

TEST(FixedPointDivideDecimalWideTest, ThirtyEightDigitDividend)
{
  auto constexpr max = max_digits<numeric::decimal128>();
  auto const type    = decimal_type<numeric::decimal128>(numeric::scale_type{-37});

  // max * 10^37 / (3 * 10^37) = max / 3 exactly
  auto const lhs = dec128_wrapper{{max, -max}, numeric::scale_type{0}};
  auto const rhs = dec128_wrapper{{3 * pow10_128(37), 3 * pow10_128(37)}, numeric::scale_type{0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dec128_wrapper({0, 0}, {0, 0}, numeric::scale_type{-37}),
                                 *cudf::divide_decimal(lhs, rhs, type));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dec128_wrapper({max / 3, -max / 3}, numeric::scale_type{-37}),
                                 *divide_wide(lhs, rhs, type));
}

TEST(FixedPointDivideDecimalWideTest, LargestExponents)
{
  using Rep              = __int128_t;
  auto constexpr max     = max_digits<numeric::decimal128>();
  auto constexpr rep_max = cuda::std::numeric_limits<Rep>::max();

  // 10^76 / (2^127 - 1) = 58774717541114375398436826861112283890.6... has 38 digits
  auto const lhs = dec128_wrapper{{1, -1}, numeric::scale_type{0}};
  auto const rhs = dec128_wrapper{{rep_max, rep_max}, numeric::scale_type{0}};
  auto const quotient =
    Rep{5'877'471'754'111'437'539} * pow10_128(19) + Rep{8'436'826'861'112'283'891};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    dec128_wrapper({quotient, -quotient}, numeric::scale_type{-76}),
    *divide_wide(lhs, rhs, decimal_type<numeric::decimal128>(numeric::scale_type{-76})));

  // From an exponent of 77, every quotient has more than 38 digits. Exponents beyond 80 are
  // clamped.
  auto const ones = dec128_wrapper{{1, max}, numeric::scale_type{0}};
  auto const big  = dec128_wrapper{{rep_max, 1}, numeric::scale_type{0}};
  for (int32_t scale : {-77, -80, -81, -200}) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      dec128_wrapper({0, 0}, {0, 0}, numeric::scale_type{scale}),
      *divide_wide(ones, big, decimal_type<numeric::decimal128>(numeric::scale_type{scale})));
  }
}

TEST(FixedPointDivideDecimalWideTest, QuotientLimit)
{
  auto constexpr max = max_digits<numeric::decimal128>();

  // max * 10 / 10 = max fits, and max * 10 / 9 has 39 digits
  auto const lhs = dec128_wrapper{{max, max}, numeric::scale_type{0}};
  auto const rhs = dec128_wrapper{{10, 9}, numeric::scale_type{0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    dec128_wrapper({max, 0}, {1, 0}, numeric::scale_type{-1}),
    *divide_wide(lhs, rhs, decimal_type<numeric::decimal128>(numeric::scale_type{-1})));

  // max * 10^38 / 1 is at least 2^128, so the quotient doesn't fit
  auto const one = dec128_wrapper{{1}, numeric::scale_type{0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    dec128_wrapper({0}, {0}, numeric::scale_type{-38}),
    *divide_wide(dec128_wrapper{{max}, numeric::scale_type{0}},
                 one,
                 decimal_type<numeric::decimal128>(numeric::scale_type{-38})));
}

TEST(FixedPointDivideDecimalWideTest, RoundingTies)
{
  using Rep          = __int128_t;
  auto constexpr max = max_digits<numeric::decimal128>();
  auto const half    = 5 * pow10_128(37);

  // max / 2 = half - 0.5 and (max - 2) / 2 = half - 1.5, for each pair of signs. The rescaled
  // dividend overflows 128 bits, with divisors both below and above 2^64.
  auto const lhs = dec128_wrapper{{max, -max, max - 2, -(max - 2)}, numeric::scale_type{0}};
  for (auto const [divisor, exponent] : {std::pair{Rep{20}, 1}, std::pair{2 * pow10_128(20), 20}}) {
    auto const scale = numeric::scale_type{-exponent};
    auto const type  = decimal_type<numeric::decimal128>(scale);
    auto const rhs = dec128_wrapper{{divisor, divisor, -divisor, -divisor}, numeric::scale_type{0}};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(dec128_wrapper({0, 0, 0, 0}, {0, 0, 0, 0}, scale),
                                   *cudf::divide_decimal(lhs, rhs, type));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(dec128_wrapper({half, -half, -(half - 1), half - 1}, scale),
                                   *divide_wide(lhs, rhs, type, cudf::rounding_method::HALF_UP));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(dec128_wrapper({half, -half, -(half - 2), half - 2}, scale),
                                   *divide_wide(lhs, rhs, type, cudf::rounding_method::HALF_EVEN));
  }
}

TEST(FixedPointDivideDecimalWideTest, NarrowerOperands)
{
  using Rep            = __int128_t;
  auto constexpr max32 = max_digits<numeric::decimal32>();
  auto constexpr max64 = max_digits<numeric::decimal64>();

  // decimal64 operands into a decimal128 output: max64 * 10^21 / 30 = (max64 / 3) * 10^20
  auto const lhs64  = fp_wrapper<numeric::decimal64>{{max64, -max64}, numeric::scale_type{0}};
  auto const rhs64  = fp_wrapper<numeric::decimal64>{{30, 30}, numeric::scale_type{0}};
  auto const type21 = decimal_type<numeric::decimal128>(numeric::scale_type{-21});
  auto const q64    = Rep{max64 / 3} * pow10_128(20);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dec128_wrapper({0, 0}, {0, 0}, numeric::scale_type{-21}),
                                 *cudf::divide_decimal(lhs64, rhs64, type21));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dec128_wrapper({q64, -q64}, numeric::scale_type{-21}),
                                 *divide_wide(lhs64, rhs64, type21));

  // decimal32 operands into a decimal128 output: max32 * 10^37 / max32 = 10^37
  auto const lhs32  = fp_wrapper<numeric::decimal32>{{max32}, numeric::scale_type{0}};
  auto const rhs32  = fp_wrapper<numeric::decimal32>{{max32}, numeric::scale_type{0}};
  auto const type37 = decimal_type<numeric::decimal128>(numeric::scale_type{-37});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dec128_wrapper({0}, {0}, numeric::scale_type{-37}),
                                 *cudf::divide_decimal(lhs32, rhs32, type37));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(dec128_wrapper({pow10_128(37)}, numeric::scale_type{-37}),
                                 *divide_wide(lhs32, rhs32, type37));

  // With only decimal32 types, a dividend that overflows 64 bits never gives a quotient that fits
  auto const type11 = decimal_type<numeric::decimal32>(numeric::scale_type{-11});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fp_wrapper<numeric::decimal32>({0}, {0}, numeric::scale_type{-11}),
                                 *divide_wide(lhs32, rhs32, type11));
}

TEST(FixedPointDivideDecimalErrorTest, InvalidInputs)
{
  auto const scale    = numeric::scale_type{0};
  auto const dec32    = cudf::test::fixed_point_column_wrapper<int32_t>{{100, 200}, scale};
  auto const short32  = cudf::test::fixed_point_column_wrapper<int32_t>{{100}, scale};
  auto const ints     = cudf::test::fixed_width_column_wrapper<int32_t>{100, 200};
  auto const type     = decimal_type<numeric::decimal32>(scale);
  auto const int_type = cudf::data_type{cudf::type_id::INT32};

  EXPECT_THROW(cudf::divide_decimal(dec32, short32, type), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(ints, ints, type), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(dec32, ints, type), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(dec32, dec32, int_type), cudf::logic_error);

  auto const dec32_scalar = cudf::make_fixed_point_scalar<numeric::decimal32>(100, scale);
  auto const int_scalar   = cudf::numeric_scalar<int32_t>(100);

  EXPECT_THROW(cudf::divide_decimal(ints, *dec32_scalar, type), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(*dec32_scalar, ints, type), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(dec32, int_scalar, type), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(int_scalar, dec32, type), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(dec32, *dec32_scalar, int_type), cudf::logic_error);
  EXPECT_THROW(cudf::divide_decimal(*dec32_scalar, dec32, int_type), cudf::logic_error);
}
