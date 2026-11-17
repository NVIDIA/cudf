/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/unary.hpp>

#include <cuda/iterator>

template <typename T>
cudf::test::fixed_width_column_wrapper<T> create_fixed_columns(cudf::size_type start,
                                                               cudf::size_type size,
                                                               bool nullable)
{
  auto iter = cudf::detail::make_counting_transform_iterator(start, [](auto i) { return T(i); });

  if (not nullable) {
    return cudf::test::fixed_width_column_wrapper<T>(iter, iter + size);
  } else {
    auto valids = cudf::test::iterators::valids_at_multiples_of(2);
    return cudf::test::fixed_width_column_wrapper<T>(iter, iter + size, valids);
  }
}

template <typename T>
cudf::test::fixed_width_column_wrapper<T> create_expected_columns(cudf::size_type size,
                                                                  bool nullable,
                                                                  bool nulls_to_be)
{
  if (not nullable) {
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, [nulls_to_be](auto i) { return not nulls_to_be; });
    return cudf::test::fixed_width_column_wrapper<T>(iter, iter + size);
  } else {
    auto iter = cudf::detail::make_counting_transform_iterator(
      0, [nulls_to_be](auto i) { return i % 2 == 0 ? not nulls_to_be : nulls_to_be; });
    return cudf::test::fixed_width_column_wrapper<T>(iter, iter + size);
  }
}

template <typename T>
struct IsNull : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(IsNull, cudf::test::NumericTypes);

TYPED_TEST(IsNull, AllValid)
{
  using T = TypeParam;

  cudf::size_type start                         = 0;
  cudf::size_type size                          = 10;
  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, false);
  cudf::test::fixed_width_column_wrapper<bool> expected =
    create_expected_columns<bool>(size, false, true);

  std::unique_ptr<cudf::column> got = cudf::is_null(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(IsNull, WithInvalids)
{
  using T = TypeParam;

  cudf::size_type start                         = 0;
  cudf::size_type size                          = 10;
  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
  cudf::test::fixed_width_column_wrapper<bool> expected =
    create_expected_columns<bool>(size, true, true);

  std::unique_ptr<cudf::column> got = cudf::is_null(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(IsNull, EmptyColumns)
{
  using T = TypeParam;

  cudf::size_type start                         = 0;
  cudf::size_type size                          = 0;
  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
  cudf::test::fixed_width_column_wrapper<bool> expected =
    create_expected_columns<bool>(size, true, true);

  std::unique_ptr<cudf::column> got = cudf::is_null(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

template <typename T>
struct IsNotNull : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(IsNotNull, cudf::test::NumericTypes);

TYPED_TEST(IsNotNull, AllValid)
{
  using T = TypeParam;

  cudf::size_type start                         = 0;
  cudf::size_type size                          = 10;
  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, false);
  cudf::test::fixed_width_column_wrapper<bool> expected =
    create_expected_columns<bool>(size, false, false);

  std::unique_ptr<cudf::column> got = cudf::is_valid(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(IsNotNull, WithInvalids)
{
  using T = TypeParam;

  cudf::size_type start                         = 0;
  cudf::size_type size                          = 10;
  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
  cudf::test::fixed_width_column_wrapper<bool> expected =
    create_expected_columns<bool>(size, true, false);

  std::unique_ptr<cudf::column> got = cudf::is_valid(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(IsNotNull, EmptyColumns)
{
  using T = TypeParam;

  cudf::size_type start                         = 0;
  cudf::size_type size                          = 0;
  cudf::test::fixed_width_column_wrapper<T> col = create_fixed_columns<T>(start, size, true);
  cudf::test::fixed_width_column_wrapper<bool> expected =
    create_expected_columns<bool>(size, true, false);

  std::unique_ptr<cudf::column> got = cudf::is_valid(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

template <typename T>
struct IsNAN : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(IsNAN, cudf::test::FloatingPointTypes);

TYPED_TEST(IsNAN, AllValid)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col{{T(1), T(2), T(NAN), T(4), T(NAN), T(6), T(7)}};
  cudf::test::fixed_width_column_wrapper<bool> expected = {
    false, false, true, false, true, false, false};

  std::unique_ptr<cudf::column> got = cudf::is_nan(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(IsNAN, WithNull)
{
  using T = TypeParam;

  // The last NAN is null
  cudf::test::fixed_width_column_wrapper<T> col{{T(1), T(2), T(NAN), T(4), T(NAN), T(6), T(7)},
                                                {1, 0, 1, 1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<bool> expected = {
    false, false, true, false, false, false, false};

  std::unique_ptr<cudf::column> got = cudf::is_nan(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(IsNAN, EmptyColumn)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col{};
  cudf::test::fixed_width_column_wrapper<bool> expected = {};

  std::unique_ptr<cudf::column> got = cudf::is_nan(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(IsNAN, NonFloatingColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col{{1, 2, 5, 3, 5, 6, 7}, {1, 0, 1, 1, 0, 1, 1}};

  EXPECT_THROW(std::unique_ptr<cudf::column> got = cudf::is_nan(col), cudf::logic_error);
}

template <typename T>
struct IsNotNAN : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(IsNotNAN, cudf::test::FloatingPointTypes);

TYPED_TEST(IsNotNAN, AllValid)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col{{T(1), T(2), T(NAN), T(4), T(NAN), T(6), T(7)}};
  cudf::test::fixed_width_column_wrapper<bool> expected = {
    true, true, false, true, false, true, true};

  std::unique_ptr<cudf::column> got = cudf::is_not_nan(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(IsNotNAN, WithNull)
{
  using T = TypeParam;

  // The last NAN is null
  cudf::test::fixed_width_column_wrapper<T> col{{T(1), T(2), T(NAN), T(4), T(NAN), T(6), T(7)},
                                                {1, 0, 1, 1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<bool> expected = {
    true, true, false, true, true, true, true};

  std::unique_ptr<cudf::column> got = cudf::is_not_nan(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(IsNotNAN, EmptyColumn)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> col{};
  cudf::test::fixed_width_column_wrapper<bool> expected = {};

  std::unique_ptr<cudf::column> got = cudf::is_not_nan(col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(IsNotNAN, NonFloatingColumn)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col{{1, 2, 5, 3, 5, 6, 7}, {1, 0, 1, 1, 0, 1, 1}};

  EXPECT_THROW(std::unique_ptr<cudf::column> got = cudf::is_not_nan(col), cudf::logic_error);
}

template <typename T>
struct FixedPointUnaryTests : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointUnaryTests, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointUnaryTests, FixedPointUnaryNegate)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{0, -1234, -3456, -6789, 1234, 3456, 6789}, scale_type{-3}};
  auto const expected = fp_wrapper{{0, 1234, 3456, 6789, -1234, -3456, -6789}, scale_type{-3}};
  auto const result   = cudf::unary_operation(input, cudf::unary_operator::NEGATE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointUnaryTests, FixedPointUnaryAbs)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{-1234, -3456, -6789, 1234, 3456, 6789}, scale_type{-3}};
  auto const expected = fp_wrapper{{1234, 3456, 6789, 1234, 3456, 6789}, scale_type{-3}};
  auto const result   = cudf::unary_operation(input, cudf::unary_operator::ABS);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointUnaryTests, FixedPointUnaryAbsPositiveScale)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{-1234, -3456, -6789, 1234, 3456, 6789}, scale_type{1}};
  auto const expected = fp_wrapper{{1234, 3456, 6789, 1234, 3456, 6789}, scale_type{1}};
  auto const result   = cudf::unary_operation(input, cudf::unary_operator::ABS);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointUnaryTests, FixedPointUnaryAbsLarge)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto a = cuda::counting_iterator{-2000};
  auto b =
    cudf::detail::make_counting_transform_iterator(-2000, [](auto e) { return std::abs(e); });
  auto const input    = fp_wrapper{a, a + 4000, scale_type{-1}};
  auto const expected = fp_wrapper{b, b + 4000, scale_type{-1}};
  auto const result   = cudf::unary_operation(input, cudf::unary_operator::ABS);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointUnaryTests, FixedPointUnaryCeil)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{-1234, -3456, -6789, 1234, 3456, 7000, 0}, scale_type{-3}};
  auto const expected = fp_wrapper{{-1000, -3000, -6000, 2000, 4000, 7000, 0}, scale_type{-3}};
  auto const result   = cudf::unary_operation(input, cudf::unary_operator::CEIL);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointUnaryTests, FixedPointUnaryCeilLarge)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto a = cuda::counting_iterator{-5000};
  auto b =
    cudf::detail::make_counting_transform_iterator(-5000, [](int e) { return (e / 10) * 10; });
  auto const input    = fp_wrapper{a, a + 4000, scale_type{-1}};
  auto const expected = fp_wrapper{b, b + 4000, scale_type{-1}};
  auto const result   = cudf::unary_operation(input, cudf::unary_operator::CEIL);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointUnaryTests, FixedPointUnaryFloor)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{-1234, -3456, -6789, 1234, 3456, 6000, 0}, scale_type{-3}};
  auto const expected = fp_wrapper{{-2000, -4000, -7000, 1000, 3000, 6000, 0}, scale_type{-3}};
  auto const result   = cudf::unary_operation(input, cudf::unary_operator::FLOOR);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointUnaryTests, FixedPointUnaryFloorLarge)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto a = cudf::detail::make_counting_transform_iterator(100, [](int i) { return RepType{i}; });
  auto b =
    cudf::detail::make_counting_transform_iterator(100, [](auto e) { return (e / 10) * 10; });
  auto const input    = fp_wrapper{a, a + 4000, scale_type{-1}};
  auto const expected = fp_wrapper{b, b + 4000, scale_type{-1}};
  auto const result   = cudf::unary_operation(input, cudf::unary_operator::FLOOR);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointUnaryTests, ValidateCeilFloorPrecision)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  // This test is designed to protect against floating point conversion
  // introducing errors in fixed-point arithmetic. The rounding that occurs
  // during ceil/floor should only use fixed-precision math. Realistically,
  // we are only able to show precision failures due to floating conversion in
  // a few very specific circumstances where dividing by specific powers of 10
  // works against us.  Some examples: 10^23, 10^25, 10^26, 10^27, 10^30,
  // 10^32, 10^36. See https://godbolt.org/z/cP1MddP8P for a derivation. For
  // completeness and to ensure that we are not missing any other cases, we
  // test all scales representable in the range of each decimal type.
  constexpr auto min_scale = -cuda::std::numeric_limits<RepType>::digits10;
  for (int input_scale = 0; input_scale >= min_scale; --input_scale) {
    RepType input_value = 1;
    for (int k = 0; k > input_scale; --k) {
      input_value *= 10;
    }
    auto const input       = fp_wrapper{{input_value}, scale_type{input_scale}};
    auto const expected    = fp_wrapper{{input_value}, scale_type{input_scale}};
    auto const ceil_result = cudf::unary_operation(input, cudf::unary_operator::CEIL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, ceil_result->view());
    auto const floor_result = cudf::unary_operation(input, cudf::unary_operator::FLOOR);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, floor_result->view());
  }
}

CUDF_TEST_PROGRAM_MAIN()
