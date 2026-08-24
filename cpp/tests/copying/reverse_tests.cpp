/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>

#include <numeric>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
class ReverseTypedTestFixture : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ReverseTypedTestFixture, cudf::test::AllTypes);

struct ReverseZeroColumnTest : public cudf::test::BaseFixture {};

TEST_F(ReverseZeroColumnTest, PreservesRowCount)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 6};
  auto result = cudf::reverse(input);
  EXPECT_EQ(result->num_columns(), 0);
  EXPECT_EQ(result->num_rows(), 6);
}
TYPED_TEST(ReverseTypedTestFixture, ReverseTable)
{
  using T = TypeParam;
  constexpr cudf::size_type num_values{10};

  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + num_values);

  auto expected_elements = cudf::detail::make_counting_transform_iterator(
    0, [num_values] __device__(auto i) { return num_values - i - 1; });

  auto expected =
    cudf::test::fixed_width_column_wrapper<T, typename decltype(expected_elements)::value_type>(
      expected_elements, expected_elements + num_values);

  auto input_table = cudf::table_view{{input}};
  auto const p_ret = cudf::reverse(input_table);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected, verbosity);
}

TYPED_TEST(ReverseTypedTestFixture, ReverseColumn)
{
  using T = TypeParam;
  constexpr cudf::size_type num_values{10};

  auto input = cudf::test::fixed_width_column_wrapper<T, int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + num_values);

  auto expected_elements = cudf::detail::make_counting_transform_iterator(
    0, [num_values] __device__(auto i) { return num_values - i - 1; });

  auto expected =
    cudf::test::fixed_width_column_wrapper<T, typename decltype(expected_elements)::value_type>(
      expected_elements, expected_elements + num_values);

  auto const column_ret = cudf::reverse(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column_ret->view(), expected, verbosity);
}

TYPED_TEST(ReverseTypedTestFixture, ReverseNullable)
{
  using T = TypeParam;
  constexpr cudf::size_type num_values{20};

  std::vector<int64_t> input_values(num_values);
  std::iota(input_values.begin(), input_values.end(), 1);

  thrust::host_vector<bool> input_valids(num_values);
  thrust::tabulate(
    thrust::seq, input_valids.begin(), input_valids.end(), [](auto i) { return not(i % 2); });

  std::vector<T> expected_values(input_values.size());
  thrust::host_vector<bool> expected_valids(input_valids.size());

  std::transform(std::make_reverse_iterator(input_values.end()),
                 std::make_reverse_iterator(input_values.begin()),
                 expected_values.begin(),
                 [](auto i) { return cudf::test::make_type_param_scalar<T>(i); });
  std::reverse_copy(input_valids.begin(), input_valids.end(), expected_valids.begin());

  cudf::test::fixed_width_column_wrapper<T, int64_t> input(
    input_values.begin(), input_values.end(), input_valids.begin());

  cudf::test::fixed_width_column_wrapper<T> expected(
    expected_values.begin(), expected_values.end(), expected_valids.begin());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::reverse(input_table);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

TYPED_TEST(ReverseTypedTestFixture, ZeroSizeInput)
{
  using T = TypeParam;
  cudf::test::fixed_width_column_wrapper<T, int32_t> input(cuda::counting_iterator<int32_t>{0},
                                                           cuda::counting_iterator<int32_t>{0});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected(cuda::counting_iterator<int32_t>{0},
                                                              cuda::counting_iterator<int32_t>{0});

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::reverse(input_table);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

class ReverseStringTestFixture : public cudf::test::BaseFixture {};

TEST_F(ReverseStringTestFixture, ReverseNullable)
{
  constexpr cudf::size_type num_values{20};

  std::vector<std::string> input_values(num_values);
  thrust::host_vector<bool> input_valids(num_values);

  thrust::tabulate(thrust::seq, input_values.begin(), input_values.end(), [](auto i) {
    return "#" + std::to_string(i);
  });
  thrust::tabulate(
    thrust::seq, input_valids.begin(), input_valids.end(), [](auto i) { return not(i % 2); });

  std::vector<std::string> expected_values(input_values.size());
  thrust::host_vector<bool> expected_valids(input_valids.size());

  std::reverse_copy(input_values.begin(), input_values.end(), expected_values.begin());
  std::reverse_copy(input_valids.begin(), input_valids.end(), expected_valids.begin());

  auto input = cudf::test::strings_column_wrapper(
    input_values.begin(), input_values.end(), input_valids.begin());

  auto expected = cudf::test::strings_column_wrapper(
    expected_values.begin(), expected_values.end(), expected_valids.begin());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::reverse(input_table);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}

TEST_F(ReverseStringTestFixture, ZeroSizeInput)
{
  std::vector<std::string> input_values{};
  auto input = cudf::test::strings_column_wrapper(input_values.begin(), input_values.end());

  auto count = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    cuda::counting_iterator<cudf::size_type>{0}, cuda::counting_iterator<cudf::size_type>{0});

  auto expected = cudf::test::strings_column_wrapper(input_values.begin(), input_values.end());

  cudf::table_view input_table{{input}};
  auto p_ret = cudf::reverse(input_table);

  EXPECT_EQ(p_ret->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(p_ret->view().column(0), expected);
}
