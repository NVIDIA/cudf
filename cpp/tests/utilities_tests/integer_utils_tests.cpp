/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/integer_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

template <typename T>
struct IntegerUtilsTest : public testing::Test {};

using SignedIntegerTypes = testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(IntegerUtilsTest, SignedIntegerTypes);

TYPED_TEST(IntegerUtilsTest, DivRoundingUpInexact)
{
  using I = TypeParam;
  using cudf::util::div_rounding_up_safe;

  EXPECT_EQ(I{-3}, div_rounding_up_safe<I>(-7, 2));
  EXPECT_EQ(I{-3}, div_rounding_up_safe<I>(7, -2));
  EXPECT_EQ(I{4}, div_rounding_up_safe<I>(-7, -2));
  EXPECT_EQ(I{4}, div_rounding_up_safe<I>(7, 2));
  EXPECT_EQ(I{0}, div_rounding_up_safe<I>(-1, 2));
  EXPECT_EQ(I{0}, div_rounding_up_safe<I>(1, -2));
}

TYPED_TEST(IntegerUtilsTest, DivRoundingUpExact)
{
  using I = TypeParam;
  using cudf::util::div_rounding_up_safe;

  EXPECT_EQ(I{-3}, div_rounding_up_safe<I>(-6, 2));
  EXPECT_EQ(I{-3}, div_rounding_up_safe<I>(6, -2));
  EXPECT_EQ(I{3}, div_rounding_up_safe<I>(-6, -2));
  EXPECT_EQ(I{3}, div_rounding_up_safe<I>(6, 2));
  EXPECT_EQ(I{0}, div_rounding_up_safe<I>(0, -2));
  EXPECT_EQ(I{0}, div_rounding_up_safe<I>(0, 2));
}

TYPED_TEST(IntegerUtilsTest, DivRoundingUpLimits)
{
  using I = TypeParam;
  using cudf::util::div_rounding_up_safe;
  auto constexpr min = std::numeric_limits<I>::min();
  auto constexpr max = std::numeric_limits<I>::max();

  EXPECT_EQ(min, div_rounding_up_safe<I>(min, 1));
  EXPECT_EQ(max, div_rounding_up_safe<I>(min + 1, -1));
  EXPECT_EQ(-max, div_rounding_up_safe<I>(max, -1));
  EXPECT_EQ(max / 2 + 1, div_rounding_up_safe<I>(max, 2));
  EXPECT_EQ(min / 2 + 1, div_rounding_up_safe<I>(min + 1, 2));
  EXPECT_EQ(-(max / 2), div_rounding_up_safe<I>(max, -2));
  EXPECT_EQ(-(min / 2), div_rounding_up_safe<I>(min + 1, -2));
  EXPECT_EQ(I{-1}, div_rounding_up_safe<I>(min, max));
  EXPECT_EQ(I{0}, div_rounding_up_safe<I>(max, min));
  EXPECT_EQ(I{1}, div_rounding_up_safe<I>(min, min));
}
