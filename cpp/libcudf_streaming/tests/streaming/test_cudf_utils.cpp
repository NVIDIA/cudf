/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf_streaming/utils.hpp>

using namespace cudf_streaming;

class BaseEstimatedMemoryUsageTest : public ::testing::Test {
 protected:
  void SetUp() override { stream = cudf::get_default_stream(); }

  rmm::cuda_stream_view stream;
};

/**
 * @brief Templated test suite for testing estimated_memory_usage with different column
 * types
 */
template <typename T>
class EstimatedMemoryUsageTest : public BaseEstimatedMemoryUsageTest {};

// Define the types to test
using ColumnTypes =
  ::testing::Types<std::int8_t, std::int16_t, std::int32_t, std::int64_t, float, double, bool>;

TYPED_TEST_SUITE(EstimatedMemoryUsageTest, ColumnTypes);

TYPED_TEST(EstimatedMemoryUsageTest, FixedWidthColumnMemoryUsage)
{
  using T = TypeParam;
  // Test with different sizes
  std::vector<std::size_t> test_sizes = {0, 1, 10, 100, 1000, 1000000};

  for (auto size : test_sizes) {
    SCOPED_TRACE("test size: " + std::to_string(size));
    std::vector<T> data(size);

    cudf::test::fixed_width_column_wrapper<T> wrapper(data.begin(), data.end());
    auto column = wrapper.release();

    std::size_t exp = column->alloc_size();
    std::size_t est = cudf_streaming::estimated_memory_usage(column->view(), this->stream);

    EXPECT_EQ(exp, est);
  }
}

/**
 * @brief Test suite for string column memory usage estimation
 */
TEST_F(BaseEstimatedMemoryUsageTest, StringType)
{
  // Test with different string data
  std::vector<std::vector<std::string>> test_cases = {
    {},                                                                   // Empty column
    {"hello"},                                                            // Single string
    {"hello", "world", "test"},                                           // Multiple strings
    {"café", "こんにちは", "gpu"},                                        // Multi-byte UTF-8
    {"", "a", "very long string that should take more memory", "short"},  // Mixed lengths
    std::vector<std::string>(100, "repeated string")                      // Many repeated strings
  };

  for (auto const& data : test_cases) {
    // Create a string column
    cudf::test::strings_column_wrapper wrapper(data.begin(), data.end());
    auto column = wrapper.release();

    std::size_t exp = column->alloc_size();
    std::size_t est = cudf_streaming::estimated_memory_usage(column->view(), stream);

    EXPECT_EQ(exp, est);
  }
}

/**
 * @brief Test suite for list column memory usage estimation
 */
TEST_F(BaseEstimatedMemoryUsageTest, ListType)
{
  // Test with different list data
  std::vector<std::vector<std::int32_t>> test_cases = {
    {},                                 // Empty column
    {1, 2, 3},                          // Single list
    {1, 2, 3, 4, 5, 6},                 // Multiple values
    {0, 1, 2, 3, 4, 5},                 // Mixed values
    std::vector<std::int32_t>(100, 42)  // Many repeated values
  };

  for (auto const& data : test_cases) {
    // Create a list column
    cudf::test::lists_column_wrapper<std::int32_t> wrapper(data.begin(), data.end());
    auto column = wrapper.release();

    std::size_t exp = column->alloc_size();
    std::size_t est = cudf_streaming::estimated_memory_usage(column->view(), stream);

    EXPECT_EQ(exp, est);
  }
}

/**
 * @brief Test suite for struct column memory usage estimation
 */
TEST_F(BaseEstimatedMemoryUsageTest, StructType)
{
  // Test with different struct data configurations
  std::vector<std::vector<std::pair<std::int32_t, std::string>>> test_cases = {
    {},                                                          // Empty struct column
    {{std::make_pair(1, "hello")}},                              // Single struct
    {{std::make_pair(1, "hello"), std::make_pair(2, "world")}},  // Two structs
    {{std::make_pair(0, ""),
      std::make_pair(100, "very long string"),
      std::make_pair(42, "short")}},  // Mixed data
    std::vector<std::pair<std::int32_t, std::string>>(
      50, std::make_pair(42, "repeated"))  // Many repeated structs
  };

  for (auto const& data : test_cases) {
    // Create struct columns for each field
    std::vector<std::int32_t> int_data;
    std::vector<std::string> string_data;

    for (auto const& item : data) {
      int_data.push_back(item.first);
      string_data.push_back(item.second);
    }

    cudf::test::fixed_width_column_wrapper<std::int32_t> int_wrapper(int_data.begin(),
                                                                     int_data.end());
    cudf::test::strings_column_wrapper string_wrapper(string_data.begin(), string_data.end());

    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(int_wrapper.release());
    children.push_back(string_wrapper.release());

    cudf::test::structs_column_wrapper wrapper(std::move(children));
    auto column = wrapper.release();

    std::size_t exp = column->alloc_size();
    std::size_t est = cudf_streaming::estimated_memory_usage(column->view(), stream);

    EXPECT_EQ(exp, est);
  }
}

/**
 * @brief Test suite for dictionary column memory usage estimation
 */
TEST_F(BaseEstimatedMemoryUsageTest, DictionaryType)
{
  // Test with different dictionary data
  std::vector<std::vector<std::string>> test_cases = {
    {},                                                               // Empty column
    {"hello"},                                                        // Single value
    {"hello", "world", "test", "hello", "world"},                     // Repeated values
    {"", "a", "very long string", "short", "a", "very long string"},  // Mixed with repetition
    std::vector<std::string>(100, "repeated")                         // Many repeated values
  };

  for (auto const& data : test_cases) {
    // Create a dictionary column
    cudf::test::dictionary_column_wrapper<std::string> wrapper(data.begin(), data.end());
    auto column = wrapper.release();

    std::size_t exp = column->alloc_size();
    std::size_t est = cudf_streaming::estimated_memory_usage(column->view(), stream);

    EXPECT_EQ(exp, est);
  }
}
