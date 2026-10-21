/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cuda/std/type_traits>

#include <src/rolling/detail/range_utils.cuh>

#include <limits>
#include <type_traits>

struct RangeComparatorTest : cudf::test::BaseFixture {};

template <typename T>
struct RangeComparatorTypedTest : RangeComparatorTest {};

using TestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(RangeComparatorTypedTest, TestTypes);

TYPED_TEST(RangeComparatorTypedTest, TestLessComparator)
{
  auto const less     = cudf::detail::rolling::less<TypeParam>{};
  auto constexpr nine = TypeParam{9};
  auto constexpr ten  = TypeParam{10};

  EXPECT_TRUE(less(nine, ten));
  EXPECT_FALSE(less(ten, nine));
  EXPECT_FALSE(less(ten, ten));

  if constexpr (std::is_floating_point_v<TypeParam>) {
    auto constexpr NaN = std::numeric_limits<TypeParam>::quiet_NaN();
    auto constexpr Inf = std::numeric_limits<TypeParam>::infinity();
    // NaN.
    EXPECT_FALSE(less(NaN, ten));
    EXPECT_FALSE(less(NaN, NaN));
    EXPECT_FALSE(less(NaN, Inf));
    EXPECT_FALSE(less(NaN, -Inf));
    // Infinity.
    EXPECT_TRUE(less(Inf, NaN));
    EXPECT_FALSE(less(Inf, Inf));
    EXPECT_FALSE(less(Inf, ten));
    EXPECT_FALSE(less(Inf, -Inf));
    // -Infinity.
    EXPECT_TRUE(less(-Inf, NaN));
    EXPECT_TRUE(less(-Inf, Inf));
    EXPECT_TRUE(less(-Inf, ten));
    EXPECT_FALSE(less(-Inf, -Inf));
    // Finite.
    EXPECT_TRUE(less(ten, NaN));
    EXPECT_TRUE(less(ten, Inf));
    EXPECT_FALSE(less(ten, -Inf));
  }
}

TYPED_TEST(RangeComparatorTypedTest, TestGreaterComparator)
{
  auto const greater  = cudf::detail::rolling::greater<TypeParam>{};
  auto constexpr nine = TypeParam{9};
  auto constexpr ten  = TypeParam{10};

  EXPECT_FALSE(greater(nine, ten));
  EXPECT_TRUE(greater(ten, nine));
  EXPECT_FALSE(greater(ten, ten));

  if constexpr (std::is_floating_point_v<TypeParam>) {
    auto constexpr NaN = std::numeric_limits<TypeParam>::quiet_NaN();
    auto constexpr Inf = std::numeric_limits<TypeParam>::infinity();
    // NaN.
    EXPECT_TRUE(greater(NaN, ten));
    EXPECT_FALSE(greater(NaN, NaN));
    EXPECT_TRUE(greater(NaN, Inf));
    EXPECT_TRUE(greater(NaN, -Inf));
    // Infinity.
    EXPECT_FALSE(greater(Inf, NaN));
    EXPECT_FALSE(greater(Inf, Inf));
    EXPECT_TRUE(greater(Inf, ten));
    EXPECT_TRUE(greater(Inf, -Inf));
    // -Infinity.
    EXPECT_FALSE(greater(-Inf, NaN));
    EXPECT_FALSE(greater(-Inf, Inf));
    EXPECT_FALSE(greater(-Inf, ten));
    EXPECT_FALSE(greater(-Inf, -Inf));
    // Finite.
    EXPECT_FALSE(greater(ten, NaN));
    EXPECT_FALSE(greater(ten, Inf));
    EXPECT_TRUE(greater(ten, -Inf));
  }
}

TYPED_TEST(RangeComparatorTypedTest, TestAddSafe)
{
  using T        = TypeParam;
  using result_t = decltype(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(T{1}, T{2}));
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(T{3}, T{4}),
            (result_t{T{7}, false}));

  if constexpr (std::numeric_limits<T>::is_signed) {
    EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(T{-3}, T{4}),
              (result_t{T{1}, false}));
  }

  auto constexpr max = std::numeric_limits<T>::max();
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(T{max - 5}, T{4}),
            (result_t{max - 1, false}));
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(T{max - 4}, T{4}),
            (result_t{max, false}));
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(T{max - 3}, T{4}),
            (result_t{max, !std::is_floating_point_v<T>}));
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(max, T{4}),
            (result_t{max, !std::is_floating_point_v<T>}));
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(max, T{max - 10}),
            (result_t{max, true}));
  if constexpr (cuda::std::is_signed_v<T>) {
    auto constexpr min = std::numeric_limits<T>::lowest();
    EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(T{-10}, min),
              (result_t{min, !std::is_floating_point_v<T>}));
    EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(T{min + 10}, min),
              (result_t{min, true}));
  }
  if constexpr (std::is_floating_point_v<T>) {
    auto const NaN           = std::numeric_limits<T>::quiet_NaN();
    auto const Inf           = std::numeric_limits<T>::infinity();
    auto [val, did_overflow] = cudf::detail::rolling::saturating<cuda::std::plus<>>{}(NaN, T{4});
    EXPECT_TRUE(std::isnan(val));
    EXPECT_FALSE(did_overflow);
    EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::plus<>>{}(Inf, T{4}),
              (result_t{Inf, false}));
  }
}

TYPED_TEST(RangeComparatorTypedTest, TestSubtractSafe)
{
  using T        = TypeParam;
  using result_t = decltype(cudf::detail::rolling::saturating<cuda::std::minus<>>{}(T{1}, T{2}));
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::minus<>>{}(T{4}, T{3}),
            (result_t{T{1}, false}));

  if constexpr (std::numeric_limits<T>::is_signed) {
    EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::minus<>>{}(T{3}, T{4}),
              (result_t{T{-1}, false}));
  }

  auto constexpr min = std::numeric_limits<T>::lowest();
  auto constexpr max = std::numeric_limits<T>::max();
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::minus<>>{}(T{min + 5}, T{4}),
            (result_t{min + 1, false}));
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::minus<>>{}(T{min + 4}, T{4}),
            (result_t{min, false}));
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::minus<>>{}(T{min + 3}, T{4}),
            (result_t{min, !std::is_floating_point_v<T>}));
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::minus<>>{}(min, T{4}),
            (result_t{min, !std::is_floating_point_v<T>}));
  EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::minus<>>{}(min, max),
            (result_t{min, true}));

  if constexpr (cuda::std::is_signed_v<T>) {
    EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::minus<>>{}(T{max - 1}, min),
              (result_t{max, true}));
  }

  if constexpr (std::is_floating_point_v<T>) {
    auto const NaN           = std::numeric_limits<T>::quiet_NaN();
    auto const Inf           = std::numeric_limits<T>::infinity();
    auto [val, did_overflow] = cudf::detail::rolling::saturating<cuda::std::minus<>>{}(NaN, T{4});
    EXPECT_TRUE(std::isnan(val));
    EXPECT_FALSE(did_overflow);
    EXPECT_EQ(cudf::detail::rolling::saturating<cuda::std::minus<>>{}(-Inf, T{4}),
              (result_t{-Inf, false}));
  }
}
