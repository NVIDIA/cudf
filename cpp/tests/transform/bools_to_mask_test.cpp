/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column.hpp>
#include <cudf/transform.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>

struct MaskToNullTest : public cudf::test::BaseFixture {
  void run_test(std::vector<bool> input, std::vector<bool> val)
  {
    cudf::test::fixed_width_column_wrapper<bool> input_column(
      input.begin(), input.end(), val.begin());
    std::transform(val.begin(), val.end(), input.begin(), input.begin(), std::logical_and<>());

    auto sample = cuda::counting_iterator{0};

    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      sample, sample + input.size(), input.begin());

    auto [null_mask, null_count] = cudf::bools_to_mask(input_column);
    cudf::column got_column(expected);
    got_column.set_null_mask(std::move(*null_mask), null_count);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got_column.view());
  }

  void run_test(thrust::host_vector<bool> input)
  {
    cudf::test::fixed_width_column_wrapper<bool> input_column(input.begin(), input.end());

    auto sample = cuda::counting_iterator{0};
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      sample, sample + input.size(), input.begin());

    auto [null_mask, null_count] = cudf::bools_to_mask(input_column);
    cudf::column got_column(expected);
    got_column.set_null_mask(std::move(*null_mask), null_count);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got_column.view());
  }
};

TEST_F(MaskToNullTest, WithNoNull)
{
  std::vector<bool> input({1, 0, 1, 0, 1, 0, 1, 0});

  run_test(input);
}

TEST_F(MaskToNullTest, WithNull)
{
  std::vector<bool> input({1, 0, 1, 0, 1, 0, 1, 0});
  std::vector<bool> val({1, 1, 1, 1, 1, 1, 0, 1});

  run_test(input, val);
}

TEST_F(MaskToNullTest, ZeroSize)
{
  std::vector<bool> input({});
  run_test(input);
}

TEST_F(MaskToNullTest, NonBoolTypeColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_column({1, 2, 3, 4, 5});

  EXPECT_THROW(cudf::bools_to_mask(input_column), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
