/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/quantiles.hpp>
#include <cudf/types.hpp>

#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace {
struct q_res {
  q_res(double value, bool is_valid = true) : is_valid(is_valid), value(value) {}

  bool is_valid;
  double value;
};

// ----- test data -------------------------------------------------------------

namespace testdata {
struct q_expect {
  q_expect(double quantile)
    : quantile(quantile),
      higher(0, false),
      lower(0, false),
      linear(0, false),
      midpoint(0, false),
      nearest(0, false)
  {
  }

  q_expect(
    double quantile, double higher, double lower, double linear, double midpoint, double nearest)
    : quantile(quantile),
      higher(higher),
      lower(lower),
      linear(linear),
      midpoint(midpoint),
      nearest(nearest)
  {
  }

  double quantile;
  q_res higher;
  q_res lower;
  q_res linear;
  q_res midpoint;
  q_res nearest;
};

template <typename T>
struct test_case {
  cudf::test::fixed_width_column_wrapper<T> column;
  std::vector<q_expect> expectations;
  cudf::test::fixed_width_column_wrapper<cudf::size_type> ordered_indices;
};

// interpolate_center

template <typename T>
test_case<T> interpolate_center()
{
  auto low     = std::numeric_limits<T>::lowest();
  auto max     = std::numeric_limits<T>::max();
  double mid_d = [] {
    if (std::is_floating_point_v<T>) return 0.0;
    if (std::is_signed_v<T>) return -0.5;
    return static_cast<double>(std::numeric_limits<T>::max()) / 2.0;
  }();

  // int64_t is internally casted to a double, meaning the lerp center point
  // is float-like.
  double lin_d = [] {
    if (std::is_floating_point_v<T> || std::is_same_v<T, int64_t>) return 0.0;
    if (std::is_signed_v<T>) return -0.5;
    return static_cast<double>(std::numeric_limits<T>::max()) / 2.0;
  }();
  auto max_d = static_cast<double>(max);
  auto low_d = static_cast<double>(low);
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>({low, max}),
                      {q_expect{0.50, max_d, low_d, lin_d, mid_d, low_d}}};
}

template <>
test_case<bool> interpolate_center()
{
  auto low   = std::numeric_limits<bool>::lowest();
  auto max   = std::numeric_limits<bool>::max();
  auto mid_d = 0.5;
  auto low_d = static_cast<double>(low);
  auto max_d = static_cast<double>(max);
  return test_case<bool>{cudf::test::fixed_width_column_wrapper<bool>({low, max}),
                         {q_expect{0.5, max_d, low_d, mid_d, mid_d, low_d}}};
}

// interpolate_extrema_high

template <typename T>
test_case<T> interpolate_extrema_high()
{
  T max        = std::numeric_limits<T>::max();
  T low        = max - 2;
  auto low_d   = static_cast<double>(low);
  auto max_d   = static_cast<double>(max);
  auto exact_d = static_cast<double>(max - 1);
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>({low, max}),
                      {q_expect{0.50, max_d, low_d, exact_d, exact_d, low_d}}};
}

template <>
test_case<bool> interpolate_extrema_high<bool>()
{
  return interpolate_center<bool>();
}

// interpolate_extrema_low

template <typename T>
test_case<T> interpolate_extrema_low()
{
  T lowest     = std::numeric_limits<T>::lowest();
  T a          = lowest;
  T b          = lowest + 2;
  auto a_d     = static_cast<double>(a);
  auto b_d     = static_cast<double>(b);
  auto exact_d = static_cast<double>(a + 1);
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>({a, b}),
                      {q_expect{0.50, b_d, a_d, exact_d, exact_d, a_d}}};
}

template <>
test_case<bool> interpolate_extrema_low<bool>()
{
  return interpolate_center<bool>();
}

// single

template <typename T>
test_case<T> single()
  requires(std::is_floating_point_v<T>)
{
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>({7.309999942779541}),
                      {
                        q_expect{
                          -1.0,
                          7.309999942779541,
                          7.309999942779541,
                          7.309999942779541,
                          7.309999942779541,
                          7.309999942779541,
                        },
                        q_expect{
                          0.0,
                          7.309999942779541,
                          7.309999942779541,
                          7.309999942779541,
                          7.309999942779541,
                          7.309999942779541,
                        },
                        q_expect{
                          1.0,
                          7.309999942779541,
                          7.309999942779541,
                          7.309999942779541,
                          7.309999942779541,
                          7.309999942779541,
                        },
                      }};
}

template <typename T>
test_case<T> single()
  requires(std::is_integral_v<T> and not cudf::is_boolean<T>())
{
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>({1}),
                      {q_expect{0.7, 1, 1, 1, 1, 1}}};
}

template <typename T>
test_case<T> single()
  requires(cudf::is_boolean<T>())
{
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>({1}),
                      {q_expect{0.7, 1.0, 1.0, 1.0, 1.0, 1.0}}};
}

// all_invalid

template <typename T>
test_case<T> all_invalid()
  requires(std::is_floating_point_v<T>)
{
  return test_case<T>{
    cudf::test::fixed_width_column_wrapper<T>({6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.01, 0.8, 5.7},
                                              {0, 0, 0, 0, 0, 0, 0, 0, 0}),
    {q_expect{-1.0}, q_expect{0.0}, q_expect{0.5}, q_expect{1.0}, q_expect{2.0}}};
}

template <typename T>
test_case<T> all_invalid()
  requires(std::is_integral_v<T> and not cudf::is_boolean<T>())
{
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>({6, 0, 3, 4, 2, 1, -1, 1, 6},
                                                                {0, 0, 0, 0, 0, 0, 0, 0, 0}),
                      {q_expect{0.7}}};
}

template <typename T>
test_case<T> all_invalid()
  requires(cudf::is_boolean<T>())
{
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>({1, 0, 1, 1, 0, 1, 0, 1, 1},
                                                                {0, 0, 0, 0, 0, 0, 0, 0, 0}),
                      {q_expect{0.7}}};
}

// some invalid

template <typename T>
test_case<T> some_invalid()
  requires(std::is_same_v<T, double>)
{
  T high = 0.16;
  T low  = -1.024;
  T mid  = -0.432;
  T lin  = -0.432;
  return test_case<T>{
    cudf::test::fixed_width_column_wrapper<T>({6.8, high, 3.4, 4.17, 2.13, 1.11, low, 0.8, 5.7},
                                              {0, 1, 0, 0, 0, 0, 1, 0, 0}),
    {q_expect{-1.0, low, low, low, low, low},
     q_expect{0.0, low, low, low, low, low},
     q_expect{0.5, high, low, lin, mid, low},
     q_expect{1.0, high, high, high, high, high},
     q_expect{2.0, high, high, high, high, high}},
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({6, 1})};
}

template <typename T>
test_case<T> some_invalid()
  requires(std::is_same_v<T, float>)
{
  T high     = 0.16;
  T low      = -1.024;
  double mid = -0.43200002610683441;
  double lin = -0.43200002610683441;
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>(
                        {T(6.8), high, T(3.4), T(4.17), T(2.13), T(1.11), low, T(0.8), T(5.7)},
                        {0, 1, 0, 0, 0, 0, 1, 0, 0}),
                      {q_expect{-1.0, low, low, low, low, low},
                       q_expect{0.0, low, low, low, low, low},
                       q_expect{0.5, high, low, lin, mid, low},
                       q_expect{1.0, high, high, high, high, high},
                       q_expect{2.0, high, high, high, high, high}},
                      cudf::test::fixed_width_column_wrapper<cudf::size_type>({6, 1})};
}

template <typename T>
test_case<T> some_invalid()
  requires(std::is_integral_v<T> and not cudf::is_boolean<T>())
{
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>({6, 0, 3, 4, 2, 1, -1, 1, 6},
                                                                {0, 0, 1, 0, 0, 0, 0, 0, 1}),
                      {q_expect{0.0, 3.0, 3.0, 3.0, 3.0, 3.0},
                       q_expect{0.5, 6.0, 3.0, 4.5, 4.5, 3.0},
                       q_expect{1.0, 6.0, 6.0, 6.0, 6.0, 6.0}},
                      cudf::test::fixed_width_column_wrapper<cudf::size_type>({2, 8})};
}

template <typename T>
test_case<T> some_invalid()
  requires(cudf::is_boolean<T>())
{
  return test_case<T>{cudf::test::fixed_width_column_wrapper<T>({1, 0, 1, 1, 0, 1, 0, 1, 1},
                                                                {0, 0, 1, 0, 1, 0, 0, 0, 0}),
                      {q_expect{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                       q_expect{0.5, 1.0, 0.0, 0.5, 0.5, 0.0},
                       q_expect{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
                      cudf::test::fixed_width_column_wrapper<cudf::size_type>({4, 2})};
}

// unsorted

template <typename T>
test_case<T> unsorted()
  requires(std::is_floating_point_v<T>)
{
  return test_case<T>{
    cudf::test::fixed_width_column_wrapper<T>({6.8, 0.15, 3.4, 4.17, 2.13, 1.11, -1.00, 0.8, 5.7}),
    {
      q_expect{0.0, -1.00, -1.00, -1.00, -1.00, -1.00},
    },
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({6, 1, 7, 5, 4, 2, 3, 8, 0})};
}

template <typename T>
test_case<T> unsorted()
  requires(std::is_integral_v<T> and not cudf::is_boolean<T>())
{
  return std::is_signed<T>()
           ? test_case<T>{cudf::test::fixed_width_column_wrapper<T>({6, 0, 3, 4, 2, 1, -1, 1, 6}),
                          {q_expect{0.0, -1, -1, -1, -1, -1}},
                          cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                            {6, 1, 7, 5, 4, 2, 3, 8, 0})}
           : test_case<T>{cudf::test::fixed_width_column_wrapper<T>({6, 0, 3, 4, 2, 1, 1, 1, 6}),
                          {q_expect{0.0, 1, 1, 1, 1, 1}},
                          cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                            {6, 1, 7, 5, 4, 2, 3, 8, 0})};
}

template <typename T>
test_case<T> unsorted()
  requires(cudf::is_boolean<T>())
{
  return test_case<T>{
    cudf::test::fixed_width_column_wrapper<T>({0, 0, 1, 1, 0, 1, 1, 0, 1}),
    {q_expect{
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    }},
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 1, 4, 7, 2, 3, 5, 6, 9})};
}

}  // namespace testdata

// =============================================================================
// ----- helper functions ------------------------------------------------------

template <typename T>
void test(testdata::test_case<T> test_case)
{
  for (auto& expected : test_case.expectations) {
    auto q = std::vector<double>{expected.quantile};

    auto nullable = static_cast<cudf::column_view>(test_case.column).nullable();

    auto make_expected_column = [nullable](q_res expected) {
      return nullable ? cudf::test::fixed_width_column_wrapper<double>({expected.value},
                                                                       {expected.is_valid})
                      : cudf::test::fixed_width_column_wrapper<double>({expected.value});
    };

    auto actual_higher =
      cudf::quantile(test_case.column, q, cudf::interpolation::HIGHER, test_case.ordered_indices);
    auto expected_higher_col = make_expected_column(expected.higher);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_higher_col, actual_higher->view());

    auto actual_lower =
      cudf::quantile(test_case.column, q, cudf::interpolation::LOWER, test_case.ordered_indices);
    auto expected_lower_col = make_expected_column(expected.lower);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lower_col, actual_lower->view());

    auto actual_linear =
      cudf::quantile(test_case.column, q, cudf::interpolation::LINEAR, test_case.ordered_indices);
    auto expected_linear_col = make_expected_column(expected.linear);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_linear_col, actual_linear->view());

    auto actual_midpoint =
      cudf::quantile(test_case.column, q, cudf::interpolation::MIDPOINT, test_case.ordered_indices);
    auto expected_midpoint_col = make_expected_column(expected.midpoint);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_midpoint_col, actual_midpoint->view());

    auto actual_nearest =
      cudf::quantile(test_case.column, q, cudf::interpolation::NEAREST, test_case.ordered_indices);
    auto expected_nearest_col = make_expected_column(expected.nearest);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_nearest_col, actual_nearest->view());
  }
}

// =============================================================================
// ----- tests -----------------------------------------------------------------

template <typename T>
struct QuantileTest : public cudf::test::BaseFixture {};

using TestTypes = cudf::test::NumericTypes;
TYPED_TEST_SUITE(QuantileTest, TestTypes);

TYPED_TEST(QuantileTest, TestSingle) { test(testdata::single<TypeParam>()); }

TYPED_TEST(QuantileTest, TestSomeElementsInvalid) { test(testdata::some_invalid<TypeParam>()); }

TYPED_TEST(QuantileTest, TestAllElementsInvalid) { test(testdata::all_invalid<TypeParam>()); }

TYPED_TEST(QuantileTest, TestUnsorted) { test(testdata::unsorted<TypeParam>()); }

TYPED_TEST(QuantileTest, TestInterpolateCenter) { test(testdata::interpolate_center<TypeParam>()); }

TYPED_TEST(QuantileTest, TestInterpolateExtremaHigh)
{
  test(testdata::interpolate_extrema_high<TypeParam>());
}

TYPED_TEST(QuantileTest, TestInterpolateExtremaLow)
{
  test(testdata::interpolate_extrema_low<TypeParam>());
}

TYPED_TEST(QuantileTest, TestEmpty)
{
  auto input    = cudf::test::fixed_width_column_wrapper<TypeParam>({});
  auto expected = cudf::test::fixed_width_column_wrapper<double>({0, 0}, {false, false});
  auto actual   = cudf::quantile(input, {0.5, 0.25});
}

template <typename T>
struct QuantileUnsupportedTypesTest : public cudf::test::BaseFixture {};

// TODO add tests for FixedPointTypes
using UnsupportedTestTypes = cudf::test::RemoveIf<
  cudf::test::ContainedIn<cudf::test::Concat<TestTypes, cudf::test::FixedPointTypes>>,
  cudf::test::AllTypes>;
TYPED_TEST_SUITE(QuantileUnsupportedTypesTest, UnsupportedTestTypes);

TYPED_TEST(QuantileUnsupportedTypesTest, TestZeroElements)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> input({});

  EXPECT_THROW(cudf::quantile(input, {0}), cudf::logic_error);
}

TYPED_TEST(QuantileUnsupportedTypesTest, TestOneElements)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> input({0});

  EXPECT_THROW(cudf::quantile(input, {0}), cudf::logic_error);
}

TYPED_TEST(QuantileUnsupportedTypesTest, TestMultipleElements)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> input({0, 1, 2});

  EXPECT_THROW(cudf::quantile(input, {0}), cudf::logic_error);
}

struct QuantileDictionaryTest : public cudf::test::BaseFixture {};

TEST_F(QuantileDictionaryTest, TestValid)
{
  cudf::test::dictionary_column_wrapper<int32_t> col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  cudf::test::fixed_width_column_wrapper<int32_t> indices{0, 2, 4, 6, 8, 1, 3, 5, 7, 9};

  auto result = cudf::quantile(col, {0.5}, cudf::interpolation::LINEAR);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(),
                                      cudf::test::fixed_width_column_wrapper<double>{5.5});

  result = cudf::quantile(col, {0.5}, cudf::interpolation::LINEAR, indices);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(),
                                      cudf::test::fixed_width_column_wrapper<double>{5.5});

  result = cudf::quantile(col, {0.1, 0.2}, cudf::interpolation::HIGHER);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(),
                                      cudf::test::fixed_width_column_wrapper<double>{2.0, 3.0});

  result = cudf::quantile(col, {0.25, 0.5, 0.75}, cudf::interpolation::MIDPOINT);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    result->view(), cudf::test::fixed_width_column_wrapper<double>{3.5, 5.5, 7.5});
};

}  // anonymous namespace

CUDF_TEST_PROGRAM_MAIN()
