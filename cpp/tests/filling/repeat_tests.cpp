/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <numeric>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

class RepeatMemoryResourcesTest : public cudf::test::BaseFixture {};

TEST_F(RepeatMemoryResourcesTest, ScalarCount)
{
  auto harness = cudf::test::memory_resource_test_harness{};
  auto stream  = cudf::test::get_default_stream();
  auto setup   = cudf::memory_resources{harness.setup_mr(), harness.setup_mr()};
  cudf::test::fixed_width_column_wrapper<int32_t> values{
    {1, 2, 3}, {true, false, true}, stream, setup};
  cudf::test::strings_column_wrapper strings{
    {"one", "two", "three"}, {true, false, true}, stream, setup};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_values{
    {1, 1, 2, 2, 3, 3}, {true, true, false, false, true, true}, stream, setup};
  cudf::test::strings_column_wrapper expected_strings{{"one", "one", "", "", "three", "three"},
                                                      {true, true, false, false, true, true},
                                                      stream,
                                                      setup};
  cudf::table_view input{{values, strings}};
  cudf::table_view expected{{expected_values, expected_strings}};

  cudf::test::expect_api_uses_memory_resources(
    harness,
    [&](cudf::memory_resources mr) { return cudf::repeat(input, 2, stream, mr); },
    [](auto const& result) { return result->alloc_size(); },
    [&](auto const& result) { CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view()); },
    {cudf::test::output_allocation_expectation::EXACT,
     cudf::test::temporary_allocation_expectation::SOME},
    stream);
}

TEST_F(RepeatMemoryResourcesTest, ColumnCount)
{
  auto harness = cudf::test::memory_resource_test_harness{};
  auto stream  = cudf::test::get_default_stream();
  auto setup   = cudf::memory_resources{harness.setup_mr(), harness.setup_mr()};
  cudf::test::fixed_width_column_wrapper<int32_t> values{
    {1, 2, 3, 4}, {true, false, true, true}, stream, setup};
  cudf::test::strings_column_wrapper strings{
    {"one", "two", "three", "four"}, {true, false, true, true}, stream, setup};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> counts{{2, 1, 0, 1}, stream, setup};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_values{
    {1, 1, 2, 4}, {true, true, false, true}, stream, setup};
  cudf::test::strings_column_wrapper expected_strings{
    {"one", "one", "", "four"}, {true, true, false, true}, stream, setup};
  cudf::table_view input{{values, strings}};
  cudf::table_view expected{{expected_values, expected_strings}};

  cudf::test::expect_api_uses_memory_resources(
    harness,
    [&](cudf::memory_resources mr) { return cudf::repeat(input, counts, stream, mr); },
    [](auto const& result) { return result->alloc_size(); },
    [&](auto const& result) { CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view()); },
    {cudf::test::output_allocation_expectation::EXACT,
     cudf::test::temporary_allocation_expectation::SOME},
    stream);
}

template <typename T>
class RepeatTypedTestFixture : public cudf::test::BaseFixture,
                               cudf::test::UniformRandomGenerator<cudf::size_type> {
 public:
  RepeatTypedTestFixture() : cudf::test::UniformRandomGenerator<cudf::size_type>{0, 10} {}

  cudf::size_type repeat_count() { return this->generate(); }
};

TYPED_TEST_SUITE(RepeatTypedTestFixture, cudf::test::FixedWidthTypes);

TYPED_TEST(RepeatTypedTestFixture, RepeatScalarCount)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>(), "this code assumes fixed-width types.");

  constexpr cudf::size_type num_values{10};
  constexpr cudf::size_type repeat_count{10};

  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + num_values);

  static_assert(repeat_count > 0, "repeat_count should be larger than 0.");
  auto expected_elements = cudf::detail::make_counting_transform_iterator(
    0, [repeat_count](auto i) { return i / repeat_count; });
  auto expected =
    cudf::test::fixed_width_column_wrapper<T, typename decltype(expected_elements)::value_type>(
      expected_elements, expected_elements + num_values * repeat_count);

  auto input_table = cudf::table_view{{input}};
  auto const p_ret = cudf::repeat(input_table, repeat_count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected, verbosity);
}

TYPED_TEST(RepeatTypedTestFixture, RepeatColumnCount)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>(), "this code assumes fixed-width types.");

  constexpr cudf::size_type num_values{10};

  std::vector<int64_t> inputs(num_values);
  std::iota(inputs.begin(), inputs.end(), 0);

  std::vector<cudf::size_type> counts(num_values);
  std::transform(counts.begin(), counts.end(), counts.begin(), [&](cudf::size_type count) {
    return this->repeat_count();
  });

  std::vector<T> expected_values;
  for (size_t i{0}; i < counts.size(); i++) {
    for (cudf::size_type j{0}; j < counts[i]; j++) {
      expected_values.push_back(cudf::test::make_type_param_scalar<T>(inputs[i]));
    }
  }

  cudf::test::fixed_width_column_wrapper<T, int64_t> input(inputs.begin(), inputs.end());

  auto count =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(counts.begin(), counts.end());

  cudf::test::fixed_width_column_wrapper<T> expected(expected_values.begin(),
                                                     expected_values.end());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

TYPED_TEST(RepeatTypedTestFixture, RepeatNullable)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>(), "this code assumes fixed-width types.");

  constexpr cudf::size_type num_values{10};

  std::vector<int64_t> input_values(num_values);
  std::iota(input_values.begin(), input_values.end(), 0);
  std::vector<bool> input_valids(num_values);
  for (size_t i{0}; i < input_valids.size(); i++) {
    input_valids[i] = (i % 2) == 0;
  }

  std::vector<cudf::size_type> counts(num_values);
  std::transform(counts.begin(), counts.end(), counts.begin(), [&](cudf::size_type count) {
    return this->repeat_count();
  });

  std::vector<T> expected_values;
  std::vector<bool> expected_valids;
  for (size_t i{0}; i < counts.size(); i++) {
    for (cudf::size_type j{0}; j < counts[i]; j++) {
      expected_values.push_back(cudf::test::make_type_param_scalar<T>(input_values[i]));
      expected_valids.push_back(input_valids[i]);
    }
  }

  cudf::test::fixed_width_column_wrapper<T, int64_t> input(
    input_values.begin(), input_values.end(), input_valids.begin());

  auto count =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(counts.begin(), counts.end());

  cudf::test::fixed_width_column_wrapper<T> expected(
    expected_values.begin(), expected_values.end(), expected_valids.begin());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

TYPED_TEST(RepeatTypedTestFixture, ZeroSizeInput)
{
  using T = TypeParam;
  static_assert(cudf::is_fixed_width<T>(), "this code assumes fixed-width types.");

  cudf::test::fixed_width_column_wrapper<T, int32_t> input(cuda::counting_iterator<int32_t>{0},
                                                           cuda::counting_iterator<int32_t>{0});

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    cuda::counting_iterator<cudf::size_type>{0}, cuda::counting_iterator<cudf::size_type>{0});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected(cuda::counting_iterator<int32_t>{0},
                                                              cuda::counting_iterator<int32_t>{0});

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

TYPED_TEST(RepeatTypedTestFixture, ZeroCount)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T, int32_t> input(cuda::counting_iterator<int32_t>{0},
                                                           cuda::counting_iterator<int32_t>{10});

  auto expected = cudf::make_empty_column(cudf::type_to_id<T>());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, 0);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected->view());
}

class RepeatStringTestFixture : public cudf::test::BaseFixture,
                                cudf::test::UniformRandomGenerator<cudf::size_type> {
 public:
  RepeatStringTestFixture() : cudf::test::UniformRandomGenerator<cudf::size_type>{0, 10} {}

  cudf::size_type repeat_count() { return this->generate(); }
};

TEST_F(RepeatStringTestFixture, RepeatNullable)
{
  constexpr cudf::size_type num_values{10};

  std::vector<std::string> input_values(num_values);
  std::vector<bool> input_valids(num_values);
  for (size_t i{0}; i < num_values; i++) {
    input_values[i] = "#" + std::to_string(i);
    input_valids[i] = (i % 2) == 0;
  }

  std::vector<cudf::size_type> counts(num_values);
  std::transform(counts.begin(), counts.end(), counts.begin(), [&](cudf::size_type count) {
    return this->repeat_count();
  });

  std::vector<std::string> expected_values;
  std::vector<bool> expected_valids;
  for (size_t i{0}; i < counts.size(); i++) {
    for (cudf::size_type j{0}; j < counts[i]; j++) {
      expected_values.push_back(input_values[i]);
      expected_valids.push_back(input_valids[i]);
    }
  }

  auto input = cudf::test::strings_column_wrapper(
    input_values.begin(), input_values.end(), input_valids.begin());

  auto count =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(counts.begin(), counts.end());

  auto expected = cudf::test::strings_column_wrapper(
    expected_values.begin(), expected_values.end(), expected_valids.begin());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

TEST_F(RepeatStringTestFixture, ZeroSizeInput)
{
  std::vector<std::string> input_values{};
  auto input = cudf::test::strings_column_wrapper(input_values.begin(), input_values.end());

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    cuda::counting_iterator<cudf::size_type>{0}, cuda::counting_iterator<cudf::size_type>{0});

  auto expected = cudf::test::strings_column_wrapper(input_values.begin(), input_values.end());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::repeat(input_table, count);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

class RepeatErrorTestFixture : public cudf::test::BaseFixture {};

TEST_F(RepeatErrorTestFixture, LengthMismatch)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + 100);

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    cuda::counting_iterator<cudf::size_type>{0}, cuda::counting_iterator<cudf::size_type>{0} + 200);

  cudf::table_view input_table{{input}};

  // input_table.num_rows() != count.size()
  EXPECT_THROW(auto p_ret = cudf::repeat(input_table, count), cudf::logic_error);
}

TEST_F(RepeatErrorTestFixture, CountHasNulls)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + 100);

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator<cudf::size_type>{0} + 100,
    cuda::make_constant_iterator(false));

  cudf::table_view input_table{{input}};

  // input_table.has_nulls() == true
  EXPECT_THROW(auto ret = cudf::repeat(input_table, count), cudf::logic_error);
}

TEST_F(RepeatErrorTestFixture, Overflow)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + 100);
  cudf::table_view input_table{{input}};
  // set the count such that (count * num_rows) > max(size_type);
  // the extra divide by 2 ensures the max is exceeded despite truncation in integer division
  auto count = std::numeric_limits<cudf::size_type>::max() / (input_table.num_rows() / 2);
  EXPECT_THROW(cudf::repeat(input_table, count), std::overflow_error);
}

TEST_F(RepeatErrorTestFixture, NegativeCount)
{
  auto input = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + 100);
  cudf::table_view input_table{{input}};
  EXPECT_THROW(cudf::repeat(input_table, -1), cudf::logic_error);
}
