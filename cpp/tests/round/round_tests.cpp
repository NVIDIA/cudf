/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/round.hpp>

#include <limits>

struct RoundTests : public cudf::test::BaseFixture {};

template <typename T>
struct RoundTestsIntegerTypes : public cudf::test::BaseFixture {};

template <typename T>
struct RoundTestsFixedPointTypes : public cudf::test::BaseFixture {};

using IntegerTypes = cudf::test::Types<int16_t, int32_t, int64_t>;

TYPED_TEST_SUITE(RoundTestsIntegerTypes, IntegerTypes);
TYPED_TEST_SUITE(RoundTestsFixedPointTypes, cudf::test::FixedPointTypes);

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUpZero)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input = fp_wrapper{
    {1140, 1150, 1160, 1240, 1250, 1260, -1140, -1150, -1160, -1240, -1250, -1260}, scale_type{-2}};
  auto const expected =
    fp_wrapper{{11, 12, 12, 12, 13, 13, -11, -12, -12, -12, -13, -13}, scale_type{0}};
  auto const result = cudf::round_decimal(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUpZeroNoOp)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input = fp_wrapper{
    {1125, 1150, 1160, 1240, 1250, 1260, -1125, -1150, -1160, -1240, -1250, -1260}, scale_type{0}};
  auto const result = cudf::round_decimal(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUpZeroNullMask)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const null_mask = std::vector<int>{1, 1, 0, 1};
  auto const input     = fp_wrapper{{1150, 1160, 1240, 1250}, null_mask.cbegin(), scale_type{-2}};
  auto const expected  = fp_wrapper{{12, 12, 1240, 13}, null_mask.cbegin(), scale_type{0}};
  auto const result    = cudf::round_decimal(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfEvenZero)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input = fp_wrapper{
    {1140, 1150, 1160, 1240, 1250, 1260, -1140, -1150, -1160, -1240, -1250, -1260}, scale_type{-2}};
  auto const expected =
    fp_wrapper{{11, 12, 12, 12, 12, 13, -11, -12, -12, -12, -12, -13}, scale_type{0}};
  auto const result = cudf::round_decimal(input, 0, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUp)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1140, 1150, 1160, -1140, -1150, -1160}, scale_type{-3}};
  auto const expected = fp_wrapper{{11, 12, 12, -11, -12, -12}, scale_type{-1}};
  auto const result   = cudf::round_decimal(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUp2)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{114, 115, 116, -114, -115, -116}, scale_type{-2}};
  auto const expected = fp_wrapper{{11, 12, 12, -11, -12, -12}, scale_type{-1}};
  auto const result   = cudf::round_decimal(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUp3)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1, 2, 3, -1, -2, -3}, scale_type{1}};
  auto const expected = fp_wrapper{{100, 200, 300, -100, -200, -300}, scale_type{-1}};
  auto const result   = cudf::round_decimal(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfEven)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input = fp_wrapper{
    {1140, 1150, 1160, 1240, 1250, 1260, -1140, -1150, -1160, -1240, -1250, -1260}, scale_type{-3}};
  auto const expected =
    fp_wrapper{{11, 12, 12, 12, 12, 13, -11, -12, -12, -12, -12, -13}, scale_type{-1}};
  auto const result = cudf::round_decimal(input, 1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfEven2)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input =
    fp_wrapper{{114, 115, 116, 124, 125, 126, -114, -115, -116, -124, -125, -126}, scale_type{-2}};
  auto const expected =
    fp_wrapper{{11, 12, 12, 12, 12, 13, -11, -12, -12, -12, -12, -13}, scale_type{-1}};
  auto const result = cudf::round_decimal(input, 1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfEven3)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1, 2, 3, -1, -2, -3}, scale_type{1}};
  auto const expected = fp_wrapper{{100, 200, 300, -100, -200, -300}, scale_type{-1}};
  auto const result   = cudf::round_decimal(input, 1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, EmptyFixedPointTypeTest)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input         = fp_wrapper{{}, scale_type{1}};
  auto const expected      = fp_wrapper{{}, scale_type{-1}};
  auto const expected_type = cudf::data_type{cudf::type_to_id<decimalXX>(), scale_type{-1}};
  auto const result        = cudf::round_decimal(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  EXPECT_EQ(result->view().type(), expected_type);
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestNegHalfUp)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input =
    fp_wrapper{{14, 15, 16, 24, 25, 26, -14, -15, -16, -24, -25, -26}, scale_type{2}};
  auto const expected = fp_wrapper{{1, 2, 2, 2, 3, 3, -1, -2, -2, -2, -3, -3}, scale_type{3}};
  auto const result   = cudf::round_decimal(input, -3, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestNegHalfUp2)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input =
    fp_wrapper{{14, 15, 16, 24, 25, 26, -14, -15, -16, -24, -25, -26}, scale_type{3}};
  auto const expected = fp_wrapper{{1, 2, 2, 2, 3, 3, -1, -2, -2, -2, -3, -3}, scale_type{4}};
  auto const result   = cudf::round_decimal(input, -4, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfNegUp3)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1, 2, 3, -1, -2, -3}, scale_type{2}};
  auto const expected = fp_wrapper{{10, 20, 30, -10, -20, -30}, scale_type{1}};
  auto const result   = cudf::round_decimal(input, -1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestNegHalfEven)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input =
    fp_wrapper{{14, 15, 16, 24, 25, 26, -14, -15, -16, -24, -25, -26}, scale_type{2}};
  auto const expected = fp_wrapper{{1, 2, 2, 2, 2, 3, -1, -2, -2, -2, -2, -3}, scale_type{3}};
  auto const result   = cudf::round_decimal(input, -3, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestNegHalfEven2)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input =
    fp_wrapper{{14, 15, 16, 24, 25, 26, -14, -15, -16, -24, -25, -26}, scale_type{3}};
  auto const expected = fp_wrapper{{1, 2, 2, 2, 2, 3, -1, -2, -2, -2, -2, -3}, scale_type{4}};
  auto const result   = cudf::round_decimal(input, -4, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfNegEven3)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1, 2, 3, -1, -2, -3}, scale_type{2}};
  auto const expected = fp_wrapper{{10, 20, 30, -10, -20, -30}, scale_type{1}};
  auto const result   = cudf::round_decimal(input, -1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, TestForBlog)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{25649999}, scale_type{-5}};
  auto const expected = fp_wrapper{{256}, scale_type{0}};
  auto const result   = cudf::round_decimal(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, TestScaleMovementExceedingMaxPrecision)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  // max precision of int32 = 9
  // scale movement = -(-11) -1 = 10 > 9
  // max precision of int64 = 18
  // scale movement = -(-20) -1 = 19 > 18
  // max precision of int128 = 38
  // scale movement = -(-40) -1 = 39 > 38
  auto const target_scale = cuda::std::numeric_limits<RepType>::digits10 + 1 + 1;

  auto const input =
    fp_wrapper{{14, 15, 16, 24, 25, 26, -14, -15, -16, -24, -25, -26}, scale_type{1}};
  auto const expected = fp_wrapper{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, scale_type{target_scale}};
  auto const result   = cudf::round_decimal(input, -target_scale, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

  auto const input_even =
    fp_wrapper{{14, 15, 16, 24, 25, 26, -14, -15, -16, -24, -25, -26}, scale_type{1}};
  auto const expected_even =
    fp_wrapper{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, scale_type{target_scale}};
  auto const result_even =
    cudf::round_decimal(input, -target_scale, cudf::rounding_method::HALF_EVEN);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_even, result_even->view());

  std::initializer_list<bool> const validity = {
    true, false, true, true, true, false, false, true, true, true, true, false};
  auto const input_null =
    fp_wrapper{{14, 15, 16, 24, 25, 26, -14, -15, -16, -24, -25, -26}, validity, scale_type{1}};
  auto const expected_null =
    fp_wrapper{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, validity, scale_type{target_scale}};
  auto const result_null =
    cudf::round_decimal(input_null, -target_scale, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_null, result_null->view());
}

TYPED_TEST(RoundTestsIntegerTypes, LargeIntegerHalfEven)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto transform   = [](int i) -> float { return 10 * i + 5; };
  auto begin       = cudf::detail::make_counting_transform_iterator(1, transform);
  auto const input = fw_wrapper(begin, begin + 2000);

  auto transform2     = [](int i) { return i % 2 == 0 ? 10 * i : 10 + 10 * i; };
  auto begin2         = cudf::detail::make_counting_transform_iterator(1, transform2);
  auto const expected = fw_wrapper(begin2, begin2 + 2000);

  auto const result = cudf::round_decimal(input, -1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimpleIntegerTestNegHalfUp2)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{12, 135, 1454, 1455, 1500};
  auto const expected = fw_wrapper{0, 100, 1500, 1500, 1500};
  auto const result   = cudf::round_decimal(input, -2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimpleIntegerTestNegHalfEven)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{12, 135, 1450, 1550, 1650};
  auto const expected = fw_wrapper{0, 100, 1400, 1600, 1600};
  auto const result   = cudf::round_decimal(input, -2, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimpleNegativeIntegerHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{-12, -135, -1454, -1455, -1500, -140, -150, -160};
  auto const expected = fw_wrapper{0, -100, -1500, -1500, -1500, -100, -200, -200};
  auto const result   = cudf::round_decimal(input, -2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimpleNegativeIntegerHalfEven)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{-12, -135, -145, -146, -1454, -1455, -1500};
  auto const expected = fw_wrapper{-10, -140, -140, -150, -1450, -1460, -1500};
  auto const result   = cudf::round_decimal(input, -1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, SimpleNegativeIntegerWithUnsignedNumbersHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<uint32_t>;

  auto const input    = fw_wrapper{12, 135, 1454, 1455, 1500, 140, 150, 160};
  auto const expected = fw_wrapper{0, 100, 1500, 1500, 1500, 100, 200, 200};
  auto const result   = cudf::round_decimal(input, -2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, SimpleNegativeInt8HalfEven)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int8_t>;

  auto const input    = fw_wrapper{12, 35, 36, 15, 16, 24, 25, 26};
  auto const expected = fw_wrapper{10, 40, 40, 20, 20, 20, 20, 30};
  auto const result   = cudf::round_decimal(input, -1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, SimpleNegativeInt8HalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int8_t>;

  auto const input    = fw_wrapper{12, 35, 36, 15, 16, 24, 25, 26};
  auto const expected = fw_wrapper{10, 40, 40, 20, 20, 20, 30, 30};
  auto const result   = cudf::round_decimal(input, -1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimplePositiveIntegerHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{-12, -135, -1454, -1455, -1500};
  auto const expected = fw_wrapper{-12, -135, -1454, -1455, -1500};
  auto const result   = cudf::round_decimal(input, 2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, Int64AtBoundaryHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

  auto const m     = std::numeric_limits<int64_t>::max();  // 9223372036854775807
  auto const input = fw_wrapper{m};

  auto const expected = fw_wrapper{9223372036854775800};
  auto const result   = cudf::round_decimal(input, -2, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

  auto const expected2 = fw_wrapper{9223372036850000000};
  auto const result2   = cudf::round_decimal(input, -7, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result2->view());

  auto const expected3 = fw_wrapper{9223372000000000000};
  auto const result3   = cudf::round_decimal(input, -11, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, result3->view());

  auto const result4 = cudf::round_decimal(input, -12, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, result4->view());

  auto const expected5 = fw_wrapper{9000000000000000000};
  auto const result5   = cudf::round_decimal(input, -18, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected5, result5->view());
}

TEST_F(RoundTests, FixedPoint128HalfUp)
{
  using namespace numeric;
  using RepType    = cudf::device_storage_type_t<decimal128>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  {
    auto const input    = fp_wrapper{{-160714515306}, scale_type{-13}};
    auto const expected = fp_wrapper{{-16071451531}, scale_type{-12}};
    auto const result   = cudf::round_decimal(input, 12, cudf::rounding_method::HALF_UP);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(RoundTests, FixedPointAtBoundaryTestHalfUp)
{
  using namespace numeric;
  using RepType    = cudf::device_storage_type_t<decimal128>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const m = std::numeric_limits<RepType>::max();  // 170141183460469231731687303715884105727

  {
    auto const input    = fp_wrapper{{m}, scale_type{0}};
    auto const expected = fp_wrapper{{m / 100000}, scale_type{5}};
    auto const result   = cudf::round_decimal(input, -5, cudf::rounding_method::HALF_UP);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const input    = fp_wrapper{{m}, scale_type{0}};
    auto const expected = fp_wrapper{{m / 100000000000}, scale_type{11}};
    auto const result   = cudf::round_decimal(input, -11, cudf::rounding_method::HALF_UP);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto const input    = fp_wrapper{{m}, scale_type{0}};
    auto const expected = fp_wrapper{{m / 1000000000000000}, scale_type{15}};
    auto const result   = cudf::round_decimal(input, -15, cudf::rounding_method::HALF_UP);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(RoundTests, Errors)
{
  auto const bools = cudf::test::fixed_width_column_wrapper<bool>{0, 1, 0};
  EXPECT_THROW(cudf::round_decimal(bools, -2, cudf::rounding_method::HALF_UP),
               std::invalid_argument);
  auto const floats = cudf::test::fixed_width_column_wrapper<double>{0.0, 1.0, 10.0};
  EXPECT_THROW(cudf::round_decimal(floats, -2, cudf::rounding_method::HALF_UP),
               std::invalid_argument);
}

// Use __uint128_t for demonstration.
constexpr __uint128_t operator""_uint128_t(char const* s)
{
  __uint128_t ret = 0;
  for (int i = 0; s[i] != '\0'; ++i) {
    ret *= 10;
    if ('0' <= s[i] && s[i] <= '9') { ret += s[i] - '0'; }
  }
  return ret;
}

TEST_F(RoundTests, HalfEvenErrorsA)
{
  using namespace numeric;
  using RepType    = cudf::device_storage_type_t<decimal128>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  {
    // 0.5 at scale -37 should round HALF_EVEN to 0, because 0 is an even number
    auto const input =
      fp_wrapper{{5000000000000000000000000000000000000_uint128_t}, scale_type{-37}};
    auto const expected = fp_wrapper{{0}, scale_type{0}};
    auto const result   = cudf::round_decimal(input, 0, cudf::rounding_method::HALF_EVEN);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(RoundTests, HalfEvenErrorsB)
{
  using namespace numeric;
  using RepType    = cudf::device_storage_type_t<decimal128>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  {
    // 0.125 at scale -37 should round HALF_EVEN to 0.12, because 2 is an even number
    auto const input =
      fp_wrapper{{1250000000000000000000000000000000000_uint128_t}, scale_type{-37}};
    auto const expected = fp_wrapper{{12}, scale_type{-2}};
    auto const result   = cudf::round_decimal(input, 2, cudf::rounding_method::HALF_EVEN);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(RoundTests, HalfEvenErrorsC)
{
  using namespace numeric;
  using RepType    = cudf::device_storage_type_t<decimal128>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  {
    // 0.0625 at scale -37 should round HALF_EVEN to 0.062, because 2 is an even number
    auto const input =
      fp_wrapper{{0625000000000000000000000000000000000_uint128_t}, scale_type{-37}};
    auto const expected = fp_wrapper{{62}, scale_type{-3}};
    auto const result   = cudf::round_decimal(input, 3, cudf::rounding_method::HALF_EVEN);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(RoundTests, HalfUpErrorsA)
{
  using namespace numeric;
  using RepType    = cudf::device_storage_type_t<decimal128>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  {
    // 0.25 at scale -37 should round HALF_UP to 0.3
    auto const input =
      fp_wrapper{{2500000000000000000000000000000000000_uint128_t}, scale_type{-37}};
    auto const expected = fp_wrapper{{3}, scale_type{-1}};
    auto const result   = cudf::round_decimal(input, 1, cudf::rounding_method::HALF_UP);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

CUDF_TEST_PROGRAM_MAIN()
