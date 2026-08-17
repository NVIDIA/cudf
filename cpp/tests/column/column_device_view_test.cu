/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/copy.h>

struct ColumnDeviceViewTest : public cudf::test::BaseFixture {};

TEST_F(ColumnDeviceViewTest, Sample)
{
  using T = int32_t;
  cuda::stream_ref stream{cudf::get_default_stream()};
  cudf::test::fixed_width_column_wrapper<T> input({1, 2, 3, 4, 5, 6});
  auto output            = cudf::allocate_like(input);
  auto input_device_view = cudf::column_device_view::create(input, stream);
  auto output_device_view =
    cudf::mutable_column_device_view::create(output->mutable_view(), stream);

  EXPECT_NO_THROW(thrust::copy(rmm::exec_policy_nosync(stream),
                               input_device_view->begin<T>(),
                               input_device_view->end<T>(),
                               output_device_view->begin<T>()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, output->view());
}

TEST_F(ColumnDeviceViewTest, MismatchingType)
{
  using T = int32_t;
  cuda::stream_ref stream{cudf::get_default_stream()};
  cudf::test::fixed_width_column_wrapper<T> input({1, 2, 3, 4, 5, 6});
  auto output            = cudf::allocate_like(input);
  auto input_device_view = cudf::column_device_view::create(input, stream);
  auto output_device_view =
    cudf::mutable_column_device_view::create(output->mutable_view(), stream);

  EXPECT_THROW(thrust::copy(rmm::exec_policy_nosync(stream),
                            input_device_view->begin<T>(),
                            input_device_view->end<T>(),
                            output_device_view->begin<int64_t>()),
               cudf::logic_error);
}

TEST_F(ColumnDeviceViewTest, ExplicitMemoryResourceControl)
{
  auto harness = cudf::test::memory_resource_test_harness{this->mr()};
  auto stream  = cudf::get_default_stream();
  auto input   = cudf::test::strings_column_wrapper({"one", "two"}).release();

  auto immutable_view = [&] {
    auto current_scope = harness.fail_on_current_device_resource_use();
    auto result = cudf::column_device_view::create(input->view(), stream, harness.output_mr());
    harness.synchronize(stream);
    return result;
  }();
  auto mutable_view = [&] {
    auto current_scope = harness.fail_on_current_device_resource_use();
    auto result =
      cudf::mutable_column_device_view::create(input->mutable_view(), stream, harness.output_mr());
    harness.synchronize(stream);
    return result;
  }();

  harness.expect_output_allocations_live(stream);
  immutable_view.reset();
  mutable_view.reset();
  harness.expect_no_live_allocations(stream);
}
