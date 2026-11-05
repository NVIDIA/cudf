/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

class ReductionTest : public cudf::test::BaseFixture {};

TEST_F(ReductionTest, ReductionSum)
{
  cudf::test::fixed_width_column_wrapper<int> input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  cudf::reduce(input,
               *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
               cudf::data_type(cudf::type_id::INT32),
               cudf::test::get_default_stream());
}

TEST_F(ReductionTest, ReductionSumScalarInit)
{
  cudf::test::fixed_width_column_wrapper<int> input({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto const init_scalar = cudf::make_fixed_width_scalar<int>(3, cudf::test::get_default_stream());
  cudf::reduce(input,
               *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
               cudf::data_type(cudf::type_id::INT32),
               *init_scalar,
               cudf::test::get_default_stream());
}

TEST_F(ReductionTest, SegmentedReductionSum)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int>{
    {1, 2, 3, 1, 0, 3, 1, 0, 0, 0},
    {true, true, true, true, false, true, true, false, false, false}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_buffer_async(
    offsets, cudf::test::get_default_stream(), cudf::get_current_device_resource_ref());

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type(cudf::type_id::INT32),
                           cudf::null_policy::EXCLUDE,
                           cudf::test::get_default_stream());
}

TEST_F(ReductionTest, SegmentedReductionSumScalarInit)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int>{
    {1, 2, 3, 1, 0, 3, 1, 0, 0, 0},
    {true, true, true, true, false, true, true, false, false, false}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_buffer_async(
    offsets, cudf::test::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const init_scalar = cudf::make_fixed_width_scalar<int>(3, cudf::test::get_default_stream());
  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type(cudf::type_id::INT32),
                           cudf::null_policy::EXCLUDE,
                           *init_scalar,
                           cudf::test::get_default_stream());
}

TEST_F(ReductionTest, ScanMin)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int>{
    {123, 64, 63, 99, -5, 123, -16, -120, -111},
    {true, false, true, true, true, true, false, false, true}};

  cudf::scan(input,
             *cudf::make_min_aggregation<cudf::scan_aggregation>(),
             cudf::scan_type::INCLUSIVE,
             cudf::null_policy::EXCLUDE,
             cudf::test::get_default_stream());
}

TEST_F(ReductionTest, MinMax)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int>{
    {123, 64, 63, 99, -5, 123, -16, -120, -111},
    {true, false, true, true, true, true, false, false, true}};

  cudf::minmax(input, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
