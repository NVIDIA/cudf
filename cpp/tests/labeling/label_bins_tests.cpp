/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/labeling/label_bins.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace {

template <typename T>
using fwc_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using fpc_wrapper = cudf::test::fixed_point_column_wrapper<T>;

// TODO: Should we move these into type_lists? They seem generally useful.
using cudf::test::FixedPointTypes;
using cudf::test::FloatingPointTypes;
using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, FloatingPointTypes>;
using SignedNumericTypesNotBool =
  cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

struct BinTestFixture : public cudf::test::BaseFixture {};

void expect_label_bins_uses_explicit_memory_resources(
  cudf::test::memory_resource_test_harness& harness,
  cudf::column_view input,
  cudf::column_view left_edges,
  cudf::column_view right_edges,
  cudf::column_view expected,
  bool expect_device_allocations)
{
  auto const stream = cudf::get_default_stream();
  auto expectations = cudf::test::memory_resource_expectations{};
  if (expect_device_allocations) {
    expectations.temporary = cudf::test::temporary_allocation_expectation::SOME;
  }

  auto const comparison_resources = cudf::memory_resources{harness.setup_mr(), harness.setup_mr()};
  cudf::test::expect_api_uses_memory_resources(
    harness,
    [&](cudf::memory_resources resources) {
      return cudf::label_bins(input,
                              left_edges,
                              cudf::inclusive::YES,
                              right_edges,
                              cudf::inclusive::NO,
                              stream,
                              resources);
    },
    [](auto const& result) { return result->alloc_size(); },
    [&](auto const& result) {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected,
                                     result->view(),
                                     cudf::test::debug_output_level::FIRST_ERROR,
                                     stream,
                                     comparison_resources);
    },
    expectations,
    stream);
}

/*
 * Test error cases.
 *
 * Most of these are not parameterized by type to avoid unnecessary test overhead.
 */

// Left edges type check.
TEST(BinColumnErrorTests, TestInvalidLeft)
{
  fwc_wrapper<double> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fwc_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(
    cudf::label_bins(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
    cudf::data_type_error);
};

// Right edges type check.
TEST(BinColumnErrorTests, TestInvalidRight)
{
  fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<double> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fwc_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(
    cudf::label_bins(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
    cudf::data_type_error);
};

// Input type check.
TEST(BinColumnErrorTests, TestInvalidInput)
{
  fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  fwc_wrapper<double> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(
    cudf::label_bins(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
    cudf::data_type_error);
};

// Number of left and right edges must match.
TEST(BinColumnErrorTests, TestMismatchedEdges)
{
  fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(
    cudf::label_bins(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO),
    cudf::logic_error);
};

// Left edges with nulls.
TEST(BinColumnErrorTests, TestLeftEdgesWithNullsBefore)
{
  fwc_wrapper<float> left_edges{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  fwc_wrapper<float> right_edges{1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(
    cudf::label_bins(input, left_edges, cudf::inclusive::NO, right_edges, cudf::inclusive::NO),
    cudf::logic_error);
};

// Right edges with nulls.
TEST(BinColumnErrorTests, TestRightEdgesWithNullsBefore)
{
  fwc_wrapper<float> left_edges{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fwc_wrapper<float> right_edges{{1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  fwc_wrapper<float> input{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

  EXPECT_THROW(
    cudf::label_bins(input, left_edges, cudf::inclusive::NO, right_edges, cudf::inclusive::NO),
    cudf::logic_error);
};

/*
 * Memory-resource validation.
 */

TEST_F(BinTestFixture, ExplicitMemoryResourcesNumeric)
{
  auto harness   = cudf::test::memory_resource_test_harness{this->mr()};
  auto resources = cudf::memory_resources{harness.setup_mr(), harness.setup_mr()};
  fwc_wrapper<int32_t> input({0, 1, 2, 3, 4, 5, 6}, cudf::get_default_stream(), resources);
  fwc_wrapper<int32_t> left_edges({0, 2, 4}, cudf::get_default_stream(), resources);
  fwc_wrapper<int32_t> right_edges({2, 4, 6}, cudf::get_default_stream(), resources);
  fwc_wrapper<cudf::size_type> expected({0, 0, 1, 1, 2, 2, 0},
                                        {true, true, true, true, true, true, false},
                                        cudf::get_default_stream(),
                                        resources);

  expect_label_bins_uses_explicit_memory_resources(
    harness, input, left_edges, right_edges, expected, true);
}

TEST_F(BinTestFixture, ExplicitMemoryResourcesNullableNumeric)
{
  auto harness   = cudf::test::memory_resource_test_harness{this->mr()};
  auto resources = cudf::memory_resources{harness.setup_mr(), harness.setup_mr()};
  fwc_wrapper<int32_t> input(
    {1, 3, 5, 7}, {true, false, true, true}, cudf::get_default_stream(), resources);
  fwc_wrapper<int32_t> left_edges({0, 2, 4, 6}, cudf::get_default_stream(), resources);
  fwc_wrapper<int32_t> right_edges({2, 4, 6, 8}, cudf::get_default_stream(), resources);
  fwc_wrapper<cudf::size_type> expected(
    {0, 0, 2, 3}, {true, false, true, true}, cudf::get_default_stream(), resources);

  expect_label_bins_uses_explicit_memory_resources(
    harness, input, left_edges, right_edges, expected, true);
}

TEST_F(BinTestFixture, ExplicitMemoryResourcesStrings)
{
  auto harness   = cudf::test::memory_resource_test_harness{this->mr()};
  auto resources = cudf::memory_resources{harness.setup_mr(), harness.setup_mr()};
  cudf::test::strings_column_wrapper input(
    {"abc", "bcd", "cde", "def", "efg"}, cudf::get_default_stream(), resources);
  cudf::test::strings_column_wrapper left_edges(
    {"a", "b", "c", "d", "e"}, cudf::get_default_stream(), resources);
  cudf::test::strings_column_wrapper right_edges(
    {"b", "c", "d", "e", "f"}, cudf::get_default_stream(), resources);
  fwc_wrapper<cudf::size_type> expected(
    {0, 1, 2, 3, 4}, {true, true, true, true, true}, cudf::get_default_stream(), resources);

  expect_label_bins_uses_explicit_memory_resources(
    harness, input, left_edges, right_edges, expected, true);
}

TEST_F(BinTestFixture, ExplicitMemoryResourcesEmpty)
{
  auto harness   = cudf::test::memory_resource_test_harness{this->mr()};
  auto resources = cudf::memory_resources{harness.setup_mr(), harness.setup_mr()};
  fwc_wrapper<int32_t> input;
  fwc_wrapper<int32_t> left_edges({0, 2, 4}, cudf::get_default_stream(), resources);
  fwc_wrapper<int32_t> right_edges({2, 4, 6}, cudf::get_default_stream(), resources);
  fwc_wrapper<cudf::size_type> expected;

  expect_label_bins_uses_explicit_memory_resources(
    harness, input, left_edges, right_edges, expected, false);
}

/*
 * Valid exceptional cases.
 */

template <typename T>
struct GenericExceptionCasesBinTestFixture : public BinTestFixture {
  void test(fwc_wrapper<T> input,
            fwc_wrapper<cudf::size_type> expected,
            fwc_wrapper<T> left_edges,
            fwc_wrapper<T> right_edges)
  {
    auto result =
      cudf::label_bins(input, left_edges, cudf::inclusive::NO, right_edges, cudf::inclusive::NO);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
};

template <typename T>
struct ExceptionCasesBinTestFixture : public GenericExceptionCasesBinTestFixture<T> {};

TYPED_TEST_SUITE(ExceptionCasesBinTestFixture, NumericTypesNotBool);

// Empty input must return an empty output.
TYPED_TEST(ExceptionCasesBinTestFixture, TestEmptyInput)
{
  this->test({}, {}, {0, 2, 4, 6, 8}, {2, 4, 6, 8, 10});
};

// If no edges are provided, the bin for all inputs is null.
TYPED_TEST(ExceptionCasesBinTestFixture, TestEmptyEdges)
{
  this->test({1, 1}, {{0, 0}, {0, 0}}, {}, {});
};

// Values outside the bounds should be labeled NULL.
TYPED_TEST(ExceptionCasesBinTestFixture, TestOutOfBoundsInput)
{
  this->test({7, 9, 11, 13}, {{3, 4, 0, 0}, {1, 1, 0, 0}}, {0, 2, 4, 6, 8}, {2, 4, 6, 8, 10});
};

// Null inputs must map to nulls.
TYPED_TEST(ExceptionCasesBinTestFixture, TestInputWithNulls)
{
  this->test(
    {{1, 3, 5, 7}, {0, 1, 0, 1}}, {{0, 1, 0, 3}, {0, 1, 0, 1}}, {0, 2, 4, 6, 8}, {2, 4, 6, 8, 10});
};

// Test that nan values are assigned the NULL label.
template <typename T>
struct NaNBinTestFixture : public GenericExceptionCasesBinTestFixture<T> {};

TYPED_TEST_SUITE(NaNBinTestFixture, FloatingPointTypes);

TYPED_TEST(NaNBinTestFixture, TestNaN)
{
  if (std::numeric_limits<TypeParam>::has_quiet_NaN) {
    this->test(
      {std::numeric_limits<TypeParam>::quiet_NaN()}, {{0}, {0}}, {0, 2, 4, 6, 8}, {2, 4, 6, 8, 10});
  }
}

/*
 * Test inclusion options.
 */

template <typename T>
struct BoundaryExclusionBinTestFixture : public BinTestFixture {
  void test(cudf::inclusive left_inc,
            cudf::inclusive right_inc,
            fwc_wrapper<cudf::size_type> expected)
  {
    fwc_wrapper<T> left_edges{0, 2, 4, 6, 8};
    fwc_wrapper<T> right_edges{2, 4, 6, 8, 10};
    fwc_wrapper<T> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto result = cudf::label_bins(input, left_edges, left_inc, right_edges, right_inc);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
};

TYPED_TEST_SUITE(BoundaryExclusionBinTestFixture, NumericTypesNotBool);

// Boundary points when both bounds are excluded should be labeled null.
TYPED_TEST(BoundaryExclusionBinTestFixture, TestNoIncludes)
{
  this->test(cudf::inclusive::NO,
             cudf::inclusive::NO,
             {{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5}, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}});
};

// Boundary point 1 should be in bin 1 [1, 2).
TYPED_TEST(BoundaryExclusionBinTestFixture, TestIncludeLeft)
{
  this->test(cudf::inclusive::YES,
             cudf::inclusive::NO,
             {{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}});
};

// Boundary point 1 should be in bin 0 (0, 1].
TYPED_TEST(BoundaryExclusionBinTestFixture, TestIncludeRight)
{
  this->test(cudf::inclusive::NO,
             cudf::inclusive::YES,
             {{0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4}, {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}});
};

/*
 * Test real data.
 */

// Test numeric data of reasonable size with noncontiguous bins.
template <typename T>
struct RealDataBinTestFixture : public BinTestFixture {
  void test(unsigned int num_elements   = 512,
            unsigned int inputs_per_bin = 4,
            T left_edge_start_val       = 0)
  {
    // Avoid testing numbers that are larger than the current type supports.
    T const largest_value = (num_elements / inputs_per_bin) * 4;
    num_elements          = std::min(std::numeric_limits<T>::max(), largest_value);

    unsigned int num_edges = num_elements / inputs_per_bin;

    std::vector<T> left_edge_vector(num_edges);
    std::vector<T> right_edge_vector(num_edges);
    std::vector<T> partial_input_vector(num_edges);
    std::vector<T> input_vector;
    std::vector<cudf::size_type> partial_expected_vector(num_edges);
    std::vector<cudf::size_type> expected_vector;
    std::vector<unsigned int> expected_validity(num_elements, 1);

    std::iota(left_edge_vector.begin(), left_edge_vector.end(), left_edge_start_val);

    // Create noncontiguous bins of width 2 separate by 2, and place inputs in the middle of each
    // bin.
    std::transform(
      left_edge_vector.begin(), left_edge_vector.end(), left_edge_vector.begin(), [](T val) {
        return val * 4;
      });
    std::transform(
      left_edge_vector.begin(), left_edge_vector.end(), right_edge_vector.begin(), [](T val) {
        return val + 2;
      });
    std::transform(
      left_edge_vector.begin(), left_edge_vector.end(), partial_input_vector.begin(), [](T val) {
        return val + 1;
      });
    std::iota(partial_expected_vector.begin(), partial_expected_vector.end(), 0);

    // Create vector containing duplicates of all the inputs.
    input_vector.reserve(num_elements);
    expected_vector.reserve(num_elements);
    for (unsigned int i = 0; i < inputs_per_bin; ++i) {
      input_vector.insert(
        input_vector.end(), partial_input_vector.begin(), partial_input_vector.end());
      expected_vector.insert(
        expected_vector.end(), partial_expected_vector.begin(), partial_expected_vector.end());
    }

    // Column wrappers are necessary inputs for the function.
    fwc_wrapper<T> left_edges(left_edge_vector.begin(), left_edge_vector.end());
    fwc_wrapper<T> right_edges(right_edge_vector.begin(), right_edge_vector.end());
    fwc_wrapper<T> input(input_vector.begin(), input_vector.end());
    fwc_wrapper<cudf::size_type> expected(
      expected_vector.begin(), expected_vector.end(), expected_validity.begin());

    auto result =
      cudf::label_bins(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
};

TYPED_TEST_SUITE(RealDataBinTestFixture, NumericTypesNotBool);

TYPED_TEST(RealDataBinTestFixture, TestRealData256) { this->test(256); };
TYPED_TEST(RealDataBinTestFixture, TestRealData512) { this->test(512); };
TYPED_TEST(RealDataBinTestFixture, TestRealData1024) { this->test(1024); };

// Test negative numbers for signed types.
template <typename T>
struct NegativeNumbersBinTestFixture : public RealDataBinTestFixture<T> {
  void test(unsigned int num_elements = 512, unsigned int inputs_per_bin = 4)
  {
    RealDataBinTestFixture<T>::test(
      num_elements, inputs_per_bin, -static_cast<T>(num_elements / 2));
  }
};

TYPED_TEST_SUITE(NegativeNumbersBinTestFixture, SignedNumericTypesNotBool);

TYPED_TEST(NegativeNumbersBinTestFixture, TestNegativeNumbers256) { this->test(256); };
TYPED_TEST(NegativeNumbersBinTestFixture, TestNegativeNumbers512) { this->test(512); };
TYPED_TEST(NegativeNumbersBinTestFixture, TestNegativeNumbers1024) { this->test(1024); };

/*
 * Test fixed point types.
 */

template <typename T>
struct FixedPointBinTestFixture : public BinTestFixture {};

TYPED_TEST_SUITE(FixedPointBinTestFixture, FixedPointTypes);

TYPED_TEST(FixedPointBinTestFixture, TestFixedPointData)
{
  using fpc_type_wrapper = fpc_wrapper<cudf::device_storage_type_t<TypeParam>>;

  fpc_type_wrapper left_edges{{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}, numeric::scale_type{0}};
  fpc_type_wrapper right_edges{{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}, numeric::scale_type{0}};
  fpc_type_wrapper input{{25, 25, 25, 25, 25, 25, 25, 25, 25, 25}, numeric::scale_type{0}};

  auto result =
    cudf::label_bins(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO);

  // Check that every element is placed in bin 2.
  fwc_wrapper<cudf::size_type> expected{{2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};

/*
 * Test strings.
 */

// Basic test of strings of lowercase alphanumerics.
TEST(TestStringData, SimpleStringTest)
{
  cudf::test::strings_column_wrapper left_edges{"a", "b", "c", "d", "e"};
  cudf::test::strings_column_wrapper right_edges{"b", "c", "d", "e", "f"};
  cudf::test::strings_column_wrapper input{"abc", "bcd", "cde", "def", "efg"};

  auto result =
    cudf::label_bins(input, left_edges, cudf::inclusive::YES, right_edges, cudf::inclusive::NO);

  fwc_wrapper<cudf::size_type> expected{{0, 1, 2, 3, 4}, {1, 1, 1, 1, 1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
};

// Test non-ASCII characters.
TEST(TestStringData, NonAsciiStringTest)
{
  cudf::test::strings_column_wrapper left_edges{"A"};
  cudf::test::strings_column_wrapper right_edges{"z"};
  cudf::test::strings_column_wrapper input{"Héllo",
                                           "thesé",
                                           "HERE",
                                           "tést strings",
                                           "",
                                           "1.75",
                                           "-34",
                                           "+9.8",
                                           "17¼",
                                           "x³",
                                           "2³",
                                           " 12⅝",
                                           "1234567890",
                                           "de",
                                           "\t\r\n\f "};

  auto result =
    cudf::label_bins(input, left_edges, cudf::inclusive::NO, right_edges, cudf::inclusive::NO);

  fwc_wrapper<cudf::size_type> expected{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                        {1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

// Test sliced non-ASCII characters.
TEST(TestStringData, SlicedNonAsciiStringTest)
{
  cudf::test::strings_column_wrapper left_edges{"A"};
  cudf::test::strings_column_wrapper right_edges{"z"};
  cudf::test::strings_column_wrapper input{"Héllo",
                                           "thesé",
                                           "HERE",
                                           "tést strings",
                                           "",
                                           "1.75",
                                           "-34",
                                           "+9.8",
                                           "17¼",
                                           "x³",
                                           "2³",
                                           " 12⅝",
                                           "1234567890",
                                           "de",
                                           "\t\r\n\f "};

  auto sliced_inputs = cudf::slice(input, {1, 5, 5, 11});

  {
    auto result = cudf::label_bins(
      sliced_inputs[0], left_edges, cudf::inclusive::NO, right_edges, cudf::inclusive::NO);
    fwc_wrapper<cudf::size_type> expected{{0, 0, 0, 0}, {1, 1, 1, 0}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }

  {
    auto result = cudf::label_bins(
      sliced_inputs[1], left_edges, cudf::inclusive::NO, right_edges, cudf::inclusive::NO);
    fwc_wrapper<cudf::size_type> expected{{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 1, 0}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

}  // anonymous namespace

CUDF_TEST_PROGRAM_MAIN()
