/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <gtest/gtest.h>

struct ClampErrorTest : public cudf::test::BaseFixture {};

TEST_F(ClampErrorTest, MisMatchingScalarTypes)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid_async(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT64));
  hi->set_valid_async(true);

  cudf::test::fixed_width_column_wrapper<int32_t> input({1, 2, 3, 4, 5, 6});

  EXPECT_THROW(cudf::clamp(input, *lo, *hi), cudf::data_type_error);
}

TEST_F(ClampErrorTest, MisMatchingInputAndScalarTypes)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid_async(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid_async(true);

  cudf::test::fixed_width_column_wrapper<int64_t> input({1, 2, 3, 4, 5, 6});

  EXPECT_THROW(cudf::clamp(input, *lo, *hi), cudf::data_type_error);
}

TEST_F(ClampErrorTest, MisMatchingReplaceScalarTypes)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid_async(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid_async(true);
  auto lo_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT64));
  lo_replace->set_valid_async(true);
  auto hi_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi_replace->set_valid_async(true);

  cudf::test::fixed_width_column_wrapper<int64_t> input({1, 2, 3, 4, 5, 6});

  EXPECT_THROW(cudf::clamp(input, *lo, *lo_replace, *hi, *hi_replace), cudf::data_type_error);
}

TEST_F(ClampErrorTest, InValidCase1)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid_async(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid_async(true);
  auto lo_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo_replace->set_valid_async(false);
  auto hi_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi_replace->set_valid_async(true);

  cudf::test::fixed_width_column_wrapper<int64_t> input({1, 2, 3, 4, 5, 6});

  EXPECT_THROW(cudf::clamp(input, *lo, *lo_replace, *hi, *hi_replace), cudf::logic_error);
}

TEST_F(ClampErrorTest, InValidCase2)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid_async(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid_async(true);
  auto lo_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo_replace->set_valid_async(true);
  auto hi_replace = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi_replace->set_valid_async(false);

  cudf::test::fixed_width_column_wrapper<int64_t> input({1, 2, 3, 4, 5, 6});

  EXPECT_THROW(cudf::clamp(input, *lo, *lo_replace, *hi, *hi_replace), cudf::logic_error);
}

struct ClampEmptyCaseTest : public cudf::test::BaseFixture {};

TEST_F(ClampEmptyCaseTest, BothScalarEmptyInvalid)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid_async(false);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid_async(false);

  cudf::test::fixed_width_column_wrapper<int32_t> input({1, 2, 3, 4, 5, 6});

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, got->view());
}

TEST_F(ClampEmptyCaseTest, EmptyInput)
{
  auto lo = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  lo->set_valid_async(true);
  auto hi = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  hi->set_valid_async(true);

  cudf::test::fixed_width_column_wrapper<int32_t> input({});

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, got->view());
}

template <class T>
struct ClampTestNumeric : public cudf::test::BaseFixture {
  std::unique_ptr<cudf::column> run_clamp(cudf::host_span<T const> input,
                                          cudf::host_span<cudf::size_type const> input_validity,
                                          T lo,
                                          bool lo_validity,
                                          T hi,
                                          bool hi_validity,
                                          T lo_replace,
                                          bool lo_replace_validity,
                                          T hi_replace,
                                          bool hi_replace_validity)
  {
    using ScalarType = cudf::scalar_type_t<T>;
    std::unique_ptr<cudf::scalar> lo_scalar{nullptr};
    std::unique_ptr<cudf::scalar> hi_scalar{nullptr};
    std::unique_ptr<cudf::scalar> lo_replace_scalar{nullptr};
    std::unique_ptr<cudf::scalar> hi_replace_scalar{nullptr};
    if (cudf::is_numeric<T>()) {
      lo_scalar =
        cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_scalar =
        cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      lo_replace_scalar =
        cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_replace_scalar =
        cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
    } else if (cudf::is_timestamp<T>()) {
      lo_scalar =
        cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_scalar =
        cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      lo_replace_scalar =
        cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_replace_scalar =
        cudf::make_timestamp_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
    } else if (cudf::is_duration<T>()) {
      lo_scalar =
        cudf::make_duration_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_scalar =
        cudf::make_duration_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      lo_replace_scalar =
        cudf::make_duration_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
      hi_replace_scalar =
        cudf::make_duration_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
    }

    static_cast<ScalarType*>(lo_scalar.get())->set_value(lo);
    static_cast<ScalarType*>(lo_scalar.get())->set_valid_async(lo_validity);
    static_cast<ScalarType*>(lo_replace_scalar.get())->set_value(lo_replace);
    static_cast<ScalarType*>(lo_replace_scalar.get())->set_valid_async(lo_replace_validity);
    static_cast<ScalarType*>(hi_scalar.get())->set_value(hi);
    static_cast<ScalarType*>(hi_scalar.get())->set_valid_async(hi_validity);
    static_cast<ScalarType*>(hi_replace_scalar.get())->set_value(hi_replace);
    static_cast<ScalarType*>(hi_replace_scalar.get())->set_valid_async(hi_replace_validity);

    if (input.size() == input_validity.size()) {
      cudf::test::fixed_width_column_wrapper<T> input_column(
        input.begin(), input.end(), input_validity.begin());

      return cudf::clamp(
        input_column, *lo_scalar, *lo_replace_scalar, *hi_scalar, *hi_replace_scalar);
    } else {
      cudf::test::fixed_width_column_wrapper<T> input_column(input.begin(), input.end());
      return cudf::clamp(
        input_column, *lo_scalar, *lo_replace_scalar, *hi_scalar, *hi_replace_scalar);
    }
  }
};
using Types = cudf::test::FixedWidthTypesWithoutFixedPoint;

TYPED_TEST_SUITE(ClampTestNumeric, Types);

TYPED_TEST(ClampTestNumeric, WithNoNull)
{
  using T = TypeParam;

  T lo(cudf::test::make_type_param_scalar<T>(2));
  T hi(cudf::test::make_type_param_scalar<T>(8));
  auto input = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  auto got = this->run_clamp(input, {}, lo, true, hi, true, lo, true, hi, true);

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampTestNumeric, LowerNull)
{
  using T = TypeParam;

  T lo(cudf::test::make_type_param_scalar<T>(2));
  T hi(cudf::test::make_type_param_scalar<T>(8));
  auto input = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  auto got = this->run_clamp(input, {}, lo, false, hi, true, lo, false, hi, true);

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampTestNumeric, UpperNull)
{
  using T = TypeParam;

  T lo(cudf::test::make_type_param_scalar<T>(2));
  T hi(cudf::test::make_type_param_scalar<T>(8));
  auto input = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  auto got = this->run_clamp(input, {}, lo, true, hi, false, lo, true, hi, false);

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampTestNumeric, InputNull)
{
  using T = TypeParam;

  T lo(cudf::test::make_type_param_scalar<T>(2));
  T hi(cudf::test::make_type_param_scalar<T>(8));
  auto input = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  std::vector<cudf::size_type> input_validity({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});

  auto got = this->run_clamp(input, input_validity, lo, true, hi, true, lo, true, hi, true);

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8},
                                                              {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampTestNumeric, InputNulliWithReplace)
{
  using T = TypeParam;

  T lo(cudf::test::make_type_param_scalar<T>(2));
  T hi(cudf::test::make_type_param_scalar<T>(8));
  T lo_replace(cudf::test::make_type_param_scalar<T>(16));
  T hi_replace(cudf::test::make_type_param_scalar<T>(32));
  auto input = cudf::test::make_type_param_vector<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  std::vector<cudf::size_type> input_validity({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});

  auto got =
    this->run_clamp(input, input_validity, lo, true, hi, true, lo_replace, true, hi_replace, true);

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected({16, 16, 2, 3, 4, 5, 6, 7, 8, 32, 32},
                                                              {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

template <typename T>
struct ClampFloatTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ClampFloatTest, cudf::test::FloatingPointTypes);

TYPED_TEST(ClampFloatTest, WithNANandNoNull)
{
  using T          = TypeParam;
  using ScalarType = cudf::scalar_type_t<T>;

  cudf::test::fixed_width_column_wrapper<T> input(
    {T(8.0), T(6.0), T(NAN), T(3.0), T(4.0), T(5.0), T(1.0), T(NAN), T(2.0), T(9.0)});
  auto lo_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
  auto hi_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));

  static_cast<ScalarType*>(lo_scalar.get())->set_value(2.0);
  static_cast<ScalarType*>(lo_scalar.get())->set_valid_async(true);
  static_cast<ScalarType*>(hi_scalar.get())->set_value(6.0);
  static_cast<ScalarType*>(hi_scalar.get())->set_valid_async(true);

  auto got = cudf::clamp(input, *lo_scalar, *hi_scalar);
  cudf::test::fixed_width_column_wrapper<T> expected(
    {T(6.0), T(6.0), T(NAN), T(3.0), T(4.0), T(5.0), T(2.0), T(NAN), T(2.0), T(6.0)});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampFloatTest, WithNANandNull)
{
  using T          = TypeParam;
  using ScalarType = cudf::scalar_type_t<T>;

  cudf::test::fixed_width_column_wrapper<T> input(
    {T(8.0), T(6.0), T(NAN), T(3.0), T(4.0), T(5.0), T(1.0), T(NAN), T(2.0), T(9.0)},
    {1, 1, 1, 0, 1, 1, 1, 0, 1, 1});
  auto lo_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
  auto hi_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));

  static_cast<ScalarType*>(lo_scalar.get())->set_value(2.0);
  static_cast<ScalarType*>(lo_scalar.get())->set_valid_async(true);
  static_cast<ScalarType*>(hi_scalar.get())->set_value(6.0);
  static_cast<ScalarType*>(hi_scalar.get())->set_valid_async(true);

  auto got = cudf::clamp(input, *lo_scalar, *hi_scalar);
  cudf::test::fixed_width_column_wrapper<T> expected(
    {T(6.0), T(6.0), T(NAN), T(3.0), T(4.0), T(5.0), T(2.0), T(NAN), T(2.0), T(6.0)},
    {1, 1, 1, 0, 1, 1, 1, 0, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TYPED_TEST(ClampFloatTest, SignOfAFloat)
{
  using T          = TypeParam;
  using ScalarType = cudf::scalar_type_t<T>;

  cudf::test::fixed_width_column_wrapper<T> input(
    {T(2.0), T(0.0), T(NAN), T(4.0), T(-0.5), T(-1.0), T(1.0), T(NAN), T(0.5), T(9.0)},
    {1, 1, 1, 0, 1, 1, 1, 0, 1, 1});
  auto lo_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
  auto lo_replace_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
  auto hi_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));
  auto hi_replace_scalar =
    cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<T>()}));

  static_cast<ScalarType*>(lo_scalar.get())->set_value(0.0);
  static_cast<ScalarType*>(lo_scalar.get())->set_valid_async(true);
  static_cast<ScalarType*>(hi_scalar.get())->set_value(0.0);
  static_cast<ScalarType*>(hi_scalar.get())->set_valid_async(true);
  static_cast<ScalarType*>(lo_replace_scalar.get())->set_value(-1.0);
  static_cast<ScalarType*>(lo_replace_scalar.get())->set_valid_async(true);
  static_cast<ScalarType*>(hi_replace_scalar.get())->set_value(1.0);
  static_cast<ScalarType*>(hi_replace_scalar.get())->set_valid_async(true);

  auto got = cudf::clamp(input, *lo_scalar, *lo_replace_scalar, *hi_scalar, *hi_replace_scalar);
  cudf::test::fixed_width_column_wrapper<T> expected(
    {T(1.0), T(0.0), T(NAN), T(4.0), T(-1.0), T(-1.0), T(1.0), T(NAN), T(1.0), T(1.0)},
    {1, 1, 1, 0, 1, 1, 1, 0, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

struct ClampStringTest : public cudf::test::BaseFixture {};

TEST_F(ClampStringTest, WithNullableColumn)
{
  std::vector<std::string> strings{"A", "b", "c", "", "e", "F", "G", "H", "", "j", "B"};
  std::vector<bool> valids{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

  auto lo = cudf::make_string_scalar("B");
  auto hi = cudf::make_string_scalar("e");
  lo->set_valid_async(true);
  hi->set_valid_async(true);

  std::vector<std::string> expected_strings{"B", "b", "c", "", "e", "F", "G", "H", "", "e", "B"};

  cudf::test::strings_column_wrapper expected(
    expected_strings.begin(), expected_strings.end(), valids.begin());

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TEST_F(ClampStringTest, WithNonNullableColumn)
{
  std::vector<std::string> strings{"A", "b", "c", "D", "e", "F", "G", "H", "i", "j", "B"};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end());

  auto lo = cudf::make_string_scalar("B");
  auto hi = cudf::make_string_scalar("e");
  lo->set_valid_async(true);
  hi->set_valid_async(true);

  std::vector<std::string> expected_strings{"B", "b", "c", "D", "e", "F", "G", "H", "e", "e", "B"};

  cudf::test::strings_column_wrapper expected(expected_strings.begin(), expected_strings.end());

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TEST_F(ClampStringTest, WithNullableColumnNullLow)
{
  std::vector<std::string> strings{"A", "b", "c", "", "e", "F", "G", "H", "", "j", "B"};
  std::vector<bool> valids{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

  auto lo = cudf::make_string_scalar("B");
  auto hi = cudf::make_string_scalar("e");
  lo->set_valid_async(false);
  hi->set_valid_async(true);

  std::vector<std::string> expected_strings{"A", "b", "c", "", "e", "F", "G", "H", "", "e", "B"};

  cudf::test::strings_column_wrapper expected(
    expected_strings.begin(), expected_strings.end(), valids.begin());

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TEST_F(ClampStringTest, WithNullableColumnNullHigh)
{
  std::vector<std::string> strings{"A", "b", "c", "", "e", "F", "G", "H", "", "j", "B"};
  std::vector<bool> valids{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

  auto lo = cudf::make_string_scalar("B");
  auto hi = cudf::make_string_scalar("e");
  lo->set_valid_async(true);
  hi->set_valid_async(false);

  std::vector<std::string> expected_strings{"B", "b", "c", "", "e", "F", "G", "H", "", "j", "B"};

  cudf::test::strings_column_wrapper expected(
    expected_strings.begin(), expected_strings.end(), valids.begin());

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

TEST_F(ClampStringTest, WithNullableColumnBothLoAndHiNull)
{
  std::vector<std::string> strings{"A", "b", "c", "", "e", "F", "G", "H", "", "j", "B"};
  std::vector<bool> valids{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

  auto lo = cudf::make_string_scalar("B");
  auto hi = cudf::make_string_scalar("e");
  lo->set_valid_async(false);
  hi->set_valid_async(false);

  auto got = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, got->view());
}

TEST_F(ClampStringTest, WithReplaceString)
{
  std::vector<std::string> strings{"A", "b", "c", "", "e", "F", "G", "H", "", "j", "B"};
  std::vector<bool> valids{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};

  cudf::test::strings_column_wrapper input(strings.begin(), strings.end(), valids.begin());

  auto lo         = cudf::make_string_scalar("B");
  auto lo_replace = cudf::make_string_scalar("Z");
  auto hi         = cudf::make_string_scalar("e");
  auto hi_replace = cudf::make_string_scalar("z");
  lo->set_valid_async(true);
  lo_replace->set_valid_async(true);
  hi->set_valid_async(true);
  hi_replace->set_valid_async(true);

  std::vector<std::string> expected_strings{"Z", "b", "c", "", "e", "F", "G", "H", "", "z", "B"};

  cudf::test::strings_column_wrapper expected(
    expected_strings.begin(), expected_strings.end(), valids.begin());

  auto got = cudf::clamp(input, *lo, *lo_replace, *hi, *hi_replace);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
}

struct ClampDictionaryTest : public cudf::test::BaseFixture {};

TEST_F(ClampDictionaryTest, WithNullableColumn)
{
  cudf::test::strings_column_wrapper input_s({"a", "b", "c", "d", "", "d", "c", "b", "a"},
                                             {1, 1, 1, 1, 0, 1, 1, 1, 1});
  auto input = cudf::dictionary::encode(input_s);

  auto results = cudf::clamp(input->view(), cudf::string_scalar("b"), cudf::string_scalar("c"));
  auto decoded = cudf::dictionary::decode(results->view());

  cudf::test::strings_column_wrapper expected({"b", "b", "c", "c", "", "c", "c", "b", "b"},
                                              {1, 1, 1, 1, 0, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
}

TEST_F(ClampDictionaryTest, WithNonNullableColumn)
{
  cudf::test::fixed_width_column_wrapper<int8_t> input_s({3, 3, 1, 1, 2, 2, 4, 4});
  auto input = cudf::dictionary::encode(input_s);

  auto results =
    cudf::clamp(input->view(), cudf::numeric_scalar<int8_t>(2), cudf::numeric_scalar<int8_t>(3));
  auto decoded = cudf::dictionary::decode(results->view());

  cudf::test::fixed_width_column_wrapper<int8_t> expected({3, 3, 2, 2, 2, 2, 3, 3});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
}

TEST_F(ClampDictionaryTest, NullLowHi)
{
  cudf::test::fixed_width_column_wrapper<int16_t> input_s({200, 100, 0, 300, 300, 400, 100, 200, 0},
                                                          {1, 1, 0, 1, 1, 1, 1, 1, 0});
  auto input = cudf::dictionary::encode(input_s);
  {
    auto results = cudf::clamp(
      input->view(), cudf::numeric_scalar<int16_t>(0, false), cudf::numeric_scalar<int16_t>(300));
    auto decoded = cudf::dictionary::decode(results->view());
    cudf::test::fixed_width_column_wrapper<int16_t> expected(
      {200, 100, 0, 300, 300, 300, 100, 200, 0}, {1, 1, 0, 1, 1, 1, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
  }
  {
    auto results = cudf::clamp(
      input->view(), cudf::numeric_scalar<int16_t>(200), cudf::numeric_scalar<int16_t>(0, false));
    auto decoded = cudf::dictionary::decode(results->view());
    cudf::test::fixed_width_column_wrapper<int16_t> expected(
      {200, 200, 0, 300, 300, 400, 200, 200, 0}, {1, 1, 0, 1, 1, 1, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
  }
  {
    auto results = cudf::clamp(input->view(),
                               cudf::numeric_scalar<int16_t>(0, false),
                               cudf::numeric_scalar<int16_t>(0, false));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input->view(), results->view());
  }
}

TEST_F(ClampDictionaryTest, WithReplace)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input_s({1, 2, 3, 4, 0, 4, 3, 2, 1},
                                                          {1, 1, 1, 1, 0, 1, 1, 1, 1});
  auto input = cudf::dictionary::encode(input_s);

  auto results = cudf::clamp(input->view(),
                             cudf::numeric_scalar<int64_t>(2),
                             cudf::numeric_scalar<int64_t>(2000),
                             cudf::numeric_scalar<int64_t>(3),
                             cudf::numeric_scalar<int64_t>(3000));
  auto decoded = cudf::dictionary::decode(results->view());

  cudf::test::fixed_width_column_wrapper<int64_t> expected({2000, 2, 3, 3000, 0, 3000, 3, 2, 2000},
                                                           {1, 1, 1, 1, 0, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, decoded->view());
}

template <typename T>
struct FixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTest, ZeroScale)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const scale    = scale_type{0};
  auto const lo       = cudf::make_fixed_point_scalar<decimalXX>(2, scale);
  auto const hi       = cudf::make_fixed_point_scalar<decimalXX>(8, scale);
  auto const input    = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, scale};
  auto const expected = fp_wrapper{{2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8}, scale};
  auto const result   = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTest, LargeTest)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const scale = scale_type{-3};
  auto const lo    = cudf::make_fixed_point_scalar<decimalXX>(1000, scale);
  auto const hi    = cudf::make_fixed_point_scalar<decimalXX>(2000, scale);

  auto begin          = cuda::counting_iterator<int>{-1000};
  auto clamp          = [](int e) { return e < 1000 ? 1000 : e > 2000 ? 2000 : e; };
  auto begin2         = cudf::detail::make_counting_transform_iterator(-1000, clamp);
  auto const input    = fp_wrapper{begin, begin + 5000, scale};
  auto const expected = fp_wrapper{begin2, begin2 + 5000, scale};
  auto const result   = cudf::clamp(input, *lo, *hi);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTest, MismatchedScalarScales)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const scale = scale_type{0};
  auto const lo    = cudf::make_fixed_point_scalar<decimalXX>(2, scale_type{-1});
  auto const hi    = cudf::make_fixed_point_scalar<decimalXX>(8, scale);
  auto const input = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, scale};

  EXPECT_THROW(cudf::clamp(input, *lo, *hi), cudf::data_type_error);
}

TYPED_TEST(FixedPointTest, MismatchedColumnScalarScale)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const scale = scale_type{0};
  auto const lo    = cudf::make_fixed_point_scalar<decimalXX>(2, scale);
  auto const hi    = cudf::make_fixed_point_scalar<decimalXX>(8, scale);
  auto const input = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, scale_type{-4}};

  EXPECT_THROW(cudf::clamp(input, *lo, *hi), cudf::data_type_error);
}

CUDF_TEST_PROGRAM_MAIN()
