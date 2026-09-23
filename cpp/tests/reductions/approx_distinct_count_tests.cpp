/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <thrust/host_vector.h>

#include <cstddef>

using cudf::nan_policy;
using cudf::null_policy;

constexpr int32_t XXX{70};

bool is_reasonable_approximation(std::size_t approx_count,
                                 std::size_t exact_count,
                                 int precision           = 12,
                                 double tolerance_factor = 2.0)
{
  if (exact_count == 0) { return approx_count == 0; }
  if (exact_count == 1) { return approx_count <= 2; }

  double const relative_standard_error = 1.04 / std::sqrt(1ull << precision);
  double const tolerance               = tolerance_factor * relative_standard_error;
  double const relative_error =
    std::abs((static_cast<double>(approx_count) / static_cast<double>(exact_count)) - 1.0);

  return relative_error < tolerance;
}

template <typename T>
std::vector<T> generate_data(int size, int num_distinct)
{
  std::vector<T> data;
  data.reserve(size);
  for (int i = 0; i < size; ++i) {
    data.push_back(static_cast<T>(i % num_distinct));
  }
  return data;
}

thrust::host_vector<bool> generate_validity(int size, int null_frequency = 10)
{
  thrust::host_vector<bool> validity(size);
  for (int i = 0; i < size; ++i) {
    validity[i] = (i % null_frequency != 0);
  }
  return validity;
}

std::vector<float> generate_float_with_nans(int size, int num_distinct, int nan_frequency = 15)
{
  std::vector<float> data;
  data.reserve(size);
  for (int i = 0; i < size; ++i) {
    if (i % nan_frequency == 0) {
      data.push_back(std::numeric_limits<float>::quiet_NaN());
    } else {
      data.push_back(static_cast<float>(i % num_distinct));
    }
  }
  return data;
}

struct ApproxDistinctCount : public cudf::test::BaseFixture {};

TEST_F(ApproxDistinctCount, BasicFunctionality)
{
  auto data = generate_data<int32_t>(2000, 100);
  cudf::test::fixed_width_column_wrapper<int32_t> input_col(data.begin(), data.end());
  cudf::table_view input_table({input_col});

  auto adc          = cudf::approx_distinct_count(input_table);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, TableBasic)
{
  auto data1 = generate_data<int32_t>(1000, 50);
  auto data2 = generate_data<int32_t>(1000, 30);
  cudf::test::fixed_width_column_wrapper<int32_t> col1(data1.begin(), data1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col2(data2.begin(), data2.end());
  cudf::table_view input_table({col1, col2});

  auto adc               = cudf::approx_distinct_count(input_table);
  auto approx_count      = adc.estimate();
  auto const exact_count = cudf::distinct_count(input_table, cudf::null_equality::EQUAL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, WithNull)
{
  auto data     = generate_data<int32_t>(2000, 100);
  auto validity = generate_validity(2000, 20);
  cudf::test::fixed_width_column_wrapper<int32_t> input_col(
    data.begin(), data.end(), validity.begin());
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, IgnoreNull)
{
  auto data     = generate_data<int32_t>(2000, 100);
  auto validity = generate_validity(2000, 20);
  cudf::test::fixed_width_column_wrapper<int32_t> input_col(
    data.begin(), data.end(), validity.begin());
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, EmptyColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{};
  cudf::table_view input_table({input_col});

  auto adc    = cudf::approx_distinct_count(input_table);
  auto result = adc.estimate();
  EXPECT_EQ(0, result);
}

TEST_F(ApproxDistinctCount, StringColumn)
{
  std::vector<std::string> strings;
  strings.reserve(2000);
  for (int i = 0; i < 2000; ++i) {
    strings.push_back("str_" + std::to_string(i % 100));
  }
  cudf::test::strings_column_wrapper input_col(strings.begin(), strings.end());
  cudf::table_view input_table({input_col});

  auto adc          = cudf::approx_distinct_count(input_table);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, DifferentPrecisions)
{
  auto data = generate_data<int32_t>(2000, 150);
  cudf::test::fixed_width_column_wrapper<int32_t> input_col(data.begin(), data.end());
  cudf::table_view input_table({input_col});

  auto adc_low  = cudf::approx_distinct_count(input_table, 10);
  auto adc_mid  = cudf::approx_distinct_count(input_table, 12);
  auto adc_high = cudf::approx_distinct_count(input_table, 14);

  auto result_low  = adc_low.estimate();
  auto result_mid  = adc_mid.estimate();
  auto result_high = adc_high.estimate();

  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(result_low, exact_count, 10));
  EXPECT_TRUE(is_reasonable_approximation(result_mid, exact_count, 12));
  EXPECT_TRUE(is_reasonable_approximation(result_high, exact_count, 14));
}

TEST_F(ApproxDistinctCount, NullEquality)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_col{{1, XXX, 3, XXX, 1}, {1, 0, 1, 0, 1}};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NaNHandling)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{1.0f,
                                                          std::numeric_limits<float>::quiet_NaN(),
                                                          3.0f,
                                                          std::numeric_limits<float>::quiet_NaN(),
                                                          1.0f};
  cudf::table_view input_table({input_col});

  auto adc_exclude =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_exclude = adc_exclude.estimate();
  auto const exact_exclude =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_exclude, exact_exclude))
    << "Exclude - Exact: " << exact_exclude << ", Approx: " << approx_exclude;

  auto adc_include =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_include = adc_include.estimate();
  auto const exact_include =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_include, exact_include))
    << "Include - Exact: " << exact_include << ", Approx: " << approx_include;
}

TEST_F(ApproxDistinctCount, NaNEqualityUnequal)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{1.0f,
                                                          std::numeric_limits<float>::quiet_NaN(),
                                                          3.0f,
                                                          std::numeric_limits<float>::quiet_NaN(),
                                                          1.0f};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, TableNullHandling)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{{1, XXX, 3, 1}, {1, 0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> col2{{1, 2, XXX, 1}, {1, 1, 0, 1}};
  cudf::table_view input_table({col1, col2});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count      = adc.estimate();
  auto const exact_count = cudf::distinct_count(input_table, cudf::null_equality::EQUAL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, CombinedNullNaNHandling)
{
  cudf::test::fixed_width_column_wrapper<float> input_col{{1.0f,
                                                           std::numeric_limits<float>::quiet_NaN(),
                                                           0.0f,
                                                           3.0f,
                                                           std::numeric_limits<float>::quiet_NaN()},
                                                          {1, 1, 0, 1, 1}};
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, AddToExistingSketch)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{1, 2, 3};
  cudf::table_view table1({col1});
  auto adc = cudf::approx_distinct_count(table1);

  cudf::test::fixed_width_column_wrapper<int32_t> col2{3, 4, 5};
  cudf::table_view table2({col2});
  adc.add(table2);

  auto approx_count = adc.estimate();

  cudf::test::fixed_width_column_wrapper<int32_t> combined{1, 2, 3, 3, 4, 5};
  auto exact_count = cudf::distinct_count(combined, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NullInclude_NaNValid)
{
  auto data     = generate_float_with_nans(2000, 100, 15);
  auto validity = generate_validity(2000, 20);
  cudf::test::fixed_width_column_wrapper<float> input_col(
    data.begin(), data.end(), validity.begin());
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NullInclude_NaNNull)
{
  auto data     = generate_float_with_nans(2000, 100, 15);
  auto validity = generate_validity(2000, 20);
  cudf::test::fixed_width_column_wrapper<float> input_col(
    data.begin(), data.end(), validity.begin());
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NullExclude_NaNValid)
{
  auto data     = generate_float_with_nans(2000, 100, 15);
  auto validity = generate_validity(2000, 20);
  cudf::test::fixed_width_column_wrapper<float> input_col(
    data.begin(), data.end(), validity.begin());
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, NullExclude_NaNNull)
{
  auto data     = generate_float_with_nans(2000, 100, 15);
  auto validity = generate_validity(2000, 20);
  cudf::test::fixed_width_column_wrapper<float> input_col(
    data.begin(), data.end(), validity.begin());
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, TableWithNoNulls_ExcludePolicy)
{
  auto data1 = generate_data<int32_t>(1000, 50);
  auto data2 = generate_data<int32_t>(1000, 30);
  cudf::test::fixed_width_column_wrapper<int32_t> col1(data1.begin(), data1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col2(data2.begin(), data2.end());
  cudf::table_view input_table({col1, col2});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);
  auto approx_count      = adc.estimate();
  auto const exact_count = cudf::distinct_count(input_table, cudf::null_equality::EQUAL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, FloatColumnWithNoNaNs_NaNNullPolicy)
{
  auto data = generate_data<float>(2000, 100);
  cudf::test::fixed_width_column_wrapper<float> input_col(data.begin(), data.end());
  cudf::table_view input_table({input_col});

  auto adc =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);
  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, MixedTypes_NullAndNaNHandling)
{
  auto int_data       = generate_data<int32_t>(1000, 50);
  auto int_validity   = generate_validity(1000, 20);
  auto float_data     = generate_float_with_nans(1000, 50, 15);
  auto float_validity = generate_validity(1000, 25);

  cudf::test::fixed_width_column_wrapper<int32_t> int_col(
    int_data.begin(), int_data.end(), int_validity.begin());
  cudf::test::fixed_width_column_wrapper<float> float_col(
    float_data.begin(), float_data.end(), float_validity.begin());

  cudf::table_view input_table({int_col, float_col});

  auto adc_in_v =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto adc_in_n =
    cudf::approx_distinct_count(input_table, 12, null_policy::INCLUDE, nan_policy::NAN_IS_NULL);
  auto adc_ex_v =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);
  auto adc_ex_n =
    cudf::approx_distinct_count(input_table, 12, null_policy::EXCLUDE, nan_policy::NAN_IS_NULL);

  auto count_in_v = adc_in_v.estimate();
  auto count_in_n = adc_in_n.estimate();
  auto count_ex_v = adc_ex_v.estimate();
  auto count_ex_n = adc_ex_n.estimate();

  EXPECT_LE(count_ex_v, count_in_v);
  EXPECT_LE(count_ex_n, count_in_n);
  EXPECT_LE(count_ex_n, count_ex_v);
  EXPECT_LE(count_in_n, count_in_v);

  EXPECT_GT(count_in_v, 0);
  EXPECT_GT(count_in_n, 0);
  EXPECT_GT(count_ex_v, 0);
  EXPECT_GT(count_ex_n, 0);
}

TEST_F(ApproxDistinctCount, MergeObjects)
{
  auto data1 = generate_data<int32_t>(1000, 80);
  auto data2 = generate_data<int32_t>(1000, 120);
  cudf::test::fixed_width_column_wrapper<int32_t> col1(data1.begin(), data1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col2(data2.begin(), data2.end());
  cudf::table_view table1({col1});
  cudf::table_view table2({col2});

  auto adc1 = cudf::approx_distinct_count(table1);
  auto adc2 = cudf::approx_distinct_count(table2);

  adc1.merge(adc2);
  auto merged_count = adc1.estimate();

  std::vector<int32_t> combined_data;
  combined_data.insert(combined_data.end(), data1.begin(), data1.end());
  combined_data.insert(combined_data.end(), data2.begin(), data2.end());
  cudf::test::fixed_width_column_wrapper<int32_t> combined(combined_data.begin(),
                                                           combined_data.end());
  auto const exact_count =
    cudf::distinct_count(combined, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(merged_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << merged_count;
}

TEST_F(ApproxDistinctCount, MergeSpan)
{
  auto data1 = generate_data<int32_t>(1000, 80);
  auto data2 = generate_data<int32_t>(1000, 120);
  cudf::test::fixed_width_column_wrapper<int32_t> col1(data1.begin(), data1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col2(data2.begin(), data2.end());
  cudf::table_view table1({col1});
  cudf::table_view table2({col2});

  auto adc1 = cudf::approx_distinct_count(table1);
  auto adc2 = cudf::approx_distinct_count(table2);

  auto sketch_span = adc2.sketch();
  adc1.merge(sketch_span);
  auto merged_count = adc1.estimate();

  std::vector<int32_t> combined_data;
  combined_data.insert(combined_data.end(), data1.begin(), data1.end());
  combined_data.insert(combined_data.end(), data2.begin(), data2.end());
  cudf::test::fixed_width_column_wrapper<int32_t> combined(combined_data.begin(),
                                                           combined_data.end());
  auto const exact_count =
    cudf::distinct_count(combined, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(merged_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << merged_count;
}

TEST_F(ApproxDistinctCount, MergeWithNullHandling)
{
  auto data1     = generate_data<int32_t>(1000, 80);
  auto validity1 = generate_validity(1000, 20);
  auto data2     = generate_data<int32_t>(1000, 120);
  auto validity2 = generate_validity(1000, 25);

  cudf::test::fixed_width_column_wrapper<int32_t> col1(
    data1.begin(), data1.end(), validity1.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> col2(
    data2.begin(), data2.end(), validity2.begin());
  cudf::table_view table1({col1});
  cudf::table_view table2({col2});

  auto adc1 =
    cudf::approx_distinct_count(table1, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);
  auto adc2 =
    cudf::approx_distinct_count(table2, 12, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  adc1.merge(adc2);
  auto merged_count = adc1.estimate();

  std::vector<int32_t> combined_data;
  std::vector<bool> combined_validity;
  combined_data.insert(combined_data.end(), data1.begin(), data1.end());
  combined_data.insert(combined_data.end(), data2.begin(), data2.end());
  combined_validity.insert(combined_validity.end(), validity1.begin(), validity1.end());
  combined_validity.insert(combined_validity.end(), validity2.begin(), validity2.end());
  cudf::test::fixed_width_column_wrapper<int32_t> combined(
    combined_data.begin(), combined_data.end(), combined_validity.begin());
  auto const exact_count =
    cudf::distinct_count(combined, null_policy::INCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(merged_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << merged_count;
}

TEST_F(ApproxDistinctCount, MergeMultipleSketches)
{
  auto data1 = generate_data<int32_t>(1000, 60);
  auto data2 = generate_data<int32_t>(1000, 80);
  auto data3 = generate_data<int32_t>(1000, 100);
  cudf::test::fixed_width_column_wrapper<int32_t> col1(data1.begin(), data1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col2(data2.begin(), data2.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col3(data3.begin(), data3.end());
  cudf::table_view table1({col1});
  cudf::table_view table2({col2});
  cudf::table_view table3({col3});

  auto adc1 = cudf::approx_distinct_count(table1);
  auto adc2 = cudf::approx_distinct_count(table2);
  auto adc3 = cudf::approx_distinct_count(table3);

  adc1.merge(adc2);
  adc1.merge(adc3);
  auto merged_count = adc1.estimate();

  std::vector<int32_t> combined_data;
  combined_data.insert(combined_data.end(), data1.begin(), data1.end());
  combined_data.insert(combined_data.end(), data2.begin(), data2.end());
  combined_data.insert(combined_data.end(), data3.begin(), data3.end());
  cudf::test::fixed_width_column_wrapper<int32_t> combined(combined_data.begin(),
                                                           combined_data.end());
  auto const exact_count =
    cudf::distinct_count(combined, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(merged_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << merged_count;
}

TEST_F(ApproxDistinctCount, SketchRoundtrip)
{
  auto data = generate_data<int32_t>(2000, 100);
  cudf::test::fixed_width_column_wrapper<int32_t> col(data.begin(), data.end());
  cudf::table_view table({col});

  auto adc1        = cudf::approx_distinct_count(table);
  auto count1      = adc1.estimate();
  auto sketch_span = adc1.sketch();

  cudf::test::fixed_width_column_wrapper<int32_t> empty_col{};
  cudf::table_view empty_table({empty_col});
  auto adc2 = cudf::approx_distinct_count(empty_table);
  adc2.merge(sketch_span);
  auto count2 = adc2.estimate();

  EXPECT_EQ(count1, count2);
}

TEST_F(ApproxDistinctCount, SpanConstructor)
{
  auto data = generate_data<int32_t>(2000, 100);
  cudf::test::fixed_width_column_wrapper<int32_t> col(data.begin(), data.end());
  cudf::table_view table({col});

  auto adc       = cudf::approx_distinct_count(table);
  auto from_span = cudf::approx_distinct_count(adc.sketch(), 12);

  EXPECT_EQ(adc.estimate(), from_span.estimate());
}

TEST_F(ApproxDistinctCount, SpanConstructorMergeTwoSpans)
{
  auto data1 = generate_data<int32_t>(1000, 80);
  auto data2 = generate_data<int32_t>(1000, 120);
  cudf::test::fixed_width_column_wrapper<int32_t> col1(data1.begin(), data1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col2(data2.begin(), data2.end());

  auto adc1 = cudf::approx_distinct_count(cudf::table_view({col1}));
  auto adc2 = cudf::approx_distinct_count(cudf::table_view({col2}));

  auto merger = cudf::approx_distinct_count(adc1.sketch(), 12);
  merger.merge(adc2.sketch());

  std::vector<int32_t> combined;
  combined.insert(combined.end(), data1.begin(), data1.end());
  combined.insert(combined.end(), data2.begin(), data2.end());
  cudf::test::fixed_width_column_wrapper<int32_t> combined_col(combined.begin(), combined.end());
  auto const exact =
    cudf::distinct_count(combined_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(merger.estimate(), exact));
}

TEST_F(ApproxDistinctCount, SpanConstructorMergeMultiple)
{
  auto data1 = generate_data<int32_t>(1000, 60);
  auto data2 = generate_data<int32_t>(1000, 80);
  auto data3 = generate_data<int32_t>(1000, 100);
  cudf::test::fixed_width_column_wrapper<int32_t> col1(data1.begin(), data1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col2(data2.begin(), data2.end());
  cudf::test::fixed_width_column_wrapper<int32_t> col3(data3.begin(), data3.end());

  auto adc1 = cudf::approx_distinct_count(cudf::table_view({col1}));
  auto adc2 = cudf::approx_distinct_count(cudf::table_view({col2}));
  auto adc3 = cudf::approx_distinct_count(cudf::table_view({col3}));

  auto merger = cudf::approx_distinct_count(adc1.sketch(), 12);
  merger.merge(adc2.sketch());
  merger.merge(adc3.sketch());

  std::vector<int32_t> combined;
  combined.insert(combined.end(), data1.begin(), data1.end());
  combined.insert(combined.end(), data2.begin(), data2.end());
  combined.insert(combined.end(), data3.begin(), data3.end());
  cudf::test::fixed_width_column_wrapper<int32_t> combined_col(combined.begin(), combined.end());
  auto const exact =
    cudf::distinct_count(combined_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(merger.estimate(), exact));
}

TEST_F(ApproxDistinctCount, SpanConstructorPrecisions)
{
  auto data = generate_data<int32_t>(2000, 150);
  cudf::test::fixed_width_column_wrapper<int32_t> col(data.begin(), data.end());
  cudf::table_view table({col});

  for (auto precision : {10, 12, 14}) {
    auto adc       = cudf::approx_distinct_count(table, precision);
    auto from_span = cudf::approx_distinct_count(adc.sketch(), precision);
    EXPECT_EQ(adc.estimate(), from_span.estimate());
  }
}

TEST_F(ApproxDistinctCount, ListColumn)
{
  // Generate 2000 lists with 100 distinct patterns (i % 100)
  // Use offsets and flat values to build the list column
  constexpr int num_rows     = 2000;
  constexpr int num_distinct = 100;

  // Build flat values: each list [val, val+1] where val = i % num_distinct
  std::vector<int32_t> values;
  std::vector<int32_t> offsets;
  values.reserve(num_rows * 2);
  offsets.reserve(num_rows + 1);
  offsets.push_back(0);
  for (int i = 0; i < num_rows; ++i) {
    int const val = i % num_distinct;
    values.push_back(val);
    values.push_back(val + 1);
    offsets.push_back(static_cast<int32_t>(values.size()));
  }

  cudf::test::fixed_width_column_wrapper<int32_t> values_col(values.begin(), values.end());
  cudf::test::fixed_width_column_wrapper<int32_t> offsets_col(offsets.begin(), offsets.end());
  auto input_col = cudf::make_lists_column(
    num_rows, offsets_col.release(), values_col.release(), 0, rmm::device_buffer{});
  cudf::table_view input_table({*input_col});

  auto adc          = cudf::approx_distinct_count(input_table);
  auto approx_count = adc.estimate();

  EXPECT_TRUE(is_reasonable_approximation(approx_count, num_distinct))
    << "Exact: " << num_distinct << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, StructColumn)
{
  // Generate 2000 structs with 100 distinct patterns
  // Each struct has fields {i % 100, "str_" + (i % 100)}
  constexpr int num_rows     = 2000;
  constexpr int num_distinct = 100;

  auto int_data = generate_data<int32_t>(num_rows, num_distinct);
  cudf::test::fixed_width_column_wrapper<int32_t> field1(int_data.begin(), int_data.end());

  std::vector<std::string> str_data;
  str_data.reserve(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    str_data.push_back("str_" + std::to_string(i % num_distinct));
  }
  cudf::test::strings_column_wrapper field2(str_data.begin(), str_data.end());

  cudf::test::structs_column_wrapper input_col({field1, field2});
  cudf::table_view input_table({input_col});

  auto adc          = cudf::approx_distinct_count(input_table);
  auto approx_count = adc.estimate();

  EXPECT_TRUE(is_reasonable_approximation(approx_count, num_distinct))
    << "Exact: " << num_distinct << ", Approx: " << approx_count;
}

TEST_F(ApproxDistinctCount, PrecisionGetter)
{
  auto data = generate_data<int32_t>(1000, 50);
  cudf::test::fixed_width_column_wrapper<int32_t> col(data.begin(), data.end());
  cudf::table_view table({col});

  auto adc10 = cudf::approx_distinct_count(table, 10);
  auto adc12 = cudf::approx_distinct_count(table, 12);
  auto adc14 = cudf::approx_distinct_count(table, 14);

  EXPECT_EQ(10, adc10.precision());
  EXPECT_EQ(12, adc12.precision());
  EXPECT_EQ(14, adc14.precision());
}

TEST_F(ApproxDistinctCount, MergeDifferentPrecisionsFails)
{
  auto data = generate_data<int32_t>(1000, 50);
  cudf::test::fixed_width_column_wrapper<int32_t> col(data.begin(), data.end());
  cudf::table_view table({col});

  auto adc10 = cudf::approx_distinct_count(table, 10);
  auto adc12 = cudf::approx_distinct_count(table, 12);

  EXPECT_THROW(adc10.merge(adc12), std::invalid_argument);
}

TEST_F(ApproxDistinctCount, MergeSpanWrongSizeFails)
{
  auto data = generate_data<int32_t>(1000, 50);
  cudf::test::fixed_width_column_wrapper<int32_t> col(data.begin(), data.end());
  cudf::table_view table({col});

  auto adc10 = cudf::approx_distinct_count(table, 10);
  auto adc12 = cudf::approx_distinct_count(table, 12);

  auto sketch12 = adc12.sketch();
  EXPECT_THROW(adc10.merge(sketch12), std::invalid_argument);
}

TEST_F(ApproxDistinctCount, SpanConstructorWrongSizeFails)
{
  auto data = generate_data<int32_t>(1000, 50);
  cudf::test::fixed_width_column_wrapper<int32_t> col(data.begin(), data.end());
  cudf::table_view table({col});

  auto adc10    = cudf::approx_distinct_count(table, 10);
  auto sketch10 = adc10.sketch();

  EXPECT_THROW(cudf::approx_distinct_count(sketch10, 12), std::invalid_argument);
}

TEST_F(ApproxDistinctCount, StandardErrorConstructor)
{
  auto data = generate_data<int32_t>(10000, 500);
  cudf::test::fixed_width_column_wrapper<int32_t> col(data.begin(), data.end());
  cudf::table_view table({col});

  // ~1% standard error should give precision 14 (1.04/sqrt(2^14) ≈ 0.0081)
  auto adc_stderr =
    cudf::approx_distinct_count(table, cudf::approx_distinct_count::desired_standard_error{0.01});
  EXPECT_EQ(14, adc_stderr.precision());
  EXPECT_TRUE(is_reasonable_approximation(adc_stderr.estimate(), 500, 14));

  // 3.25% standard error (exact boundary) gives precision 10 (1.04/sqrt(2^10) = 0.0325)
  auto adc_stderr2 =
    cudf::approx_distinct_count(table, cudf::approx_distinct_count::desired_standard_error{0.0325});
  EXPECT_EQ(10, adc_stderr2.precision());

  // Slightly smaller than boundary gets higher precision
  auto adc_stderr3 =
    cudf::approx_distinct_count(table, cudf::approx_distinct_count::desired_standard_error{0.032});
  EXPECT_EQ(11, adc_stderr3.precision());
}

TEST_F(ApproxDistinctCount, StandardErrorGetter)
{
  cudf::table_view empty_table{std::vector<cudf::column_view>{}};

  auto adc10 = cudf::approx_distinct_count(empty_table, 10);
  auto adc12 = cudf::approx_distinct_count(empty_table, 12);
  auto adc14 = cudf::approx_distinct_count(empty_table, 14);

  EXPECT_GT(adc10.standard_error(), adc12.standard_error());
  EXPECT_GT(adc12.standard_error(), adc14.standard_error());

  auto adc_rt = cudf::approx_distinct_count(
    empty_table, cudf::approx_distinct_count::desired_standard_error{adc12.standard_error()});
  EXPECT_EQ(adc12.precision(), adc_rt.precision());
}

TEST_F(ApproxDistinctCount, InvalidPrecisionThrows)
{
  auto data = generate_data<int32_t>(100, 10);
  cudf::test::fixed_width_column_wrapper<int32_t> col(data.begin(), data.end());
  cudf::table_view table({col});

  EXPECT_THROW(cudf::approx_distinct_count(table, 3), std::invalid_argument);
  EXPECT_THROW(cudf::approx_distinct_count(table, 19), std::invalid_argument);
  EXPECT_THROW(cudf::approx_distinct_count(table, -1), std::invalid_argument);

  EXPECT_NO_THROW(cudf::approx_distinct_count(table, 4));
  EXPECT_NO_THROW(cudf::approx_distinct_count(table, 18));
}

TEST_F(ApproxDistinctCount, InvalidStandardErrorThrows)
{
  auto data = generate_data<int32_t>(100, 10);
  cudf::test::fixed_width_column_wrapper<int32_t> col(data.begin(), data.end());
  cudf::table_view table({col});

  EXPECT_THROW(
    cudf::approx_distinct_count(table, cudf::approx_distinct_count::desired_standard_error{0.0}),
    std::invalid_argument);
  EXPECT_THROW(
    cudf::approx_distinct_count(table, cudf::approx_distinct_count::desired_standard_error{-0.01}),
    std::invalid_argument);
}

TEST_F(ApproxDistinctCount, SketchBytesAndAlignment)
{
  EXPECT_EQ(cudf::approx_distinct_count::sketch_bytes(10), (1 << 10) * sizeof(std::int32_t));
  EXPECT_EQ(cudf::approx_distinct_count::sketch_bytes(12), (1 << 12) * sizeof(std::int32_t));
  EXPECT_GT(cudf::approx_distinct_count::sketch_bytes(14),
            cudf::approx_distinct_count::sketch_bytes(12));
  EXPECT_GE(cudf::approx_distinct_count::sketch_alignment(), alignof(std::int32_t));
}

TEST_F(ApproxDistinctCount, MemoryResource)
{
  auto data = generate_data<int32_t>(2000, 100);
  cudf::test::fixed_width_column_wrapper<int32_t> input_col(data.begin(), data.end());
  cudf::table_view input_table({input_col});

  auto mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());

  constexpr std::int32_t precision = 12;
  auto adc                         = cudf::approx_distinct_count(input_table,
                                         precision,
                                         null_policy::EXCLUDE,
                                         nan_policy::NAN_IS_NULL,
                                         cudf::get_default_stream(),
                                         mr);

  EXPECT_GE(mr.get_bytes_counter().peak, cudf::approx_distinct_count::sketch_bytes(precision));

  auto approx_count = adc.estimate();
  auto const exact_count =
    cudf::distinct_count(input_col, null_policy::EXCLUDE, nan_policy::NAN_IS_VALID);

  EXPECT_TRUE(is_reasonable_approximation(approx_count, exact_count))
    << "Exact: " << exact_count << ", Approx: " << approx_count;
}
