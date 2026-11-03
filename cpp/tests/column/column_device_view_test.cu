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

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/transform.h>

struct ColumnDeviceViewTest : public cudf::test::BaseFixture {
  void test_noncanonical_bool();
  void test_nullable_noncanonical_bool();
};

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

void ColumnDeviceViewTest::test_noncanonical_bool()
{
  auto const stream = cudf::get_default_stream();
  // Construct byte storage directly so the bool wrapper does not normalize the input.
  cudf::test::fixed_width_column_wrapper<uint8_t> const storage{255, 0, 1, 2, 255};
  cudf::column_view const bytes{storage};
  cudf::column_view const input{
    cudf::data_type{cudf::type_id::BOOL8}, bytes.size(), bytes.head<uint8_t>(), nullptr, 0};
  auto const sliced      = cudf::slice(input, {1, 5}).front();
  auto const device_view = cudf::column_device_view::create(sliced, stream);

  // Store results as bytes so invalid bool representations cannot hide in bool comparisons.
  cudf::test::fixed_width_column_wrapper<uint8_t> const expected{0, 1, 1, 1};
  auto output        = cudf::allocate_like(expected);
  auto const indices = cuda::counting_iterator<cudf::size_type>{0};
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    indices,
    indices + sliced.size(),
    output->mutable_view().data<uint8_t>(),
    [col = *device_view] __device__(auto i) { return static_cast<uint8_t>(col.element<bool>(i)); });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());

  thrust::transform(rmm::exec_policy_nosync(stream),
                    indices,
                    indices + sliced.size(),
                    output->mutable_view().data<uint8_t>(),
                    [col = *device_view] __device__(auto i) {
                      return static_cast<uint8_t>(col.element<bool const>(i));
                    });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());

  thrust::copy(rmm::exec_policy_nosync(stream),
               device_view->begin<bool>(),
               device_view->end<bool>(),
               output->mutable_view().data<uint8_t>());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

void ColumnDeviceViewTest::test_nullable_noncanonical_bool()
{
  auto const stream = cudf::get_default_stream();
  cudf::test::fixed_width_column_wrapper<uint8_t> const storage{
    {255, 0, 1, 2, 255, 255}, {false, true, true, true, true, false}};
  cudf::column_view const bytes{storage};
  cudf::column_view const input{cudf::data_type{cudf::type_id::BOOL8},
                                bytes.size(),
                                bytes.head<uint8_t>(),
                                bytes.null_mask(),
                                bytes.null_count()};
  auto const sliced      = cudf::slice(input, {1, 6}).front();
  auto const device_view = cudf::column_device_view::create(sliced, stream);

  // Use 2 for nulls, distinct from the canonical bool bytes 0 and 1.
  cudf::test::fixed_width_column_wrapper<uint8_t> const expected{0, 1, 1, 1, 2};
  auto output        = cudf::allocate_like(expected);
  auto const indices = cuda::counting_iterator<cudf::size_type>{0};
  thrust::transform(rmm::exec_policy_nosync(stream),
                    indices,
                    indices + sliced.size(),
                    output->mutable_view().data<uint8_t>(),
                    [col = *device_view] __device__(auto i) {
                      auto const value = col.nullable_element<bool>(i);
                      return value.has_value() ? static_cast<uint8_t>(*value) : uint8_t{2};
                    });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());

  auto const begin = device_view->optional_begin<bool>(cudf::nullate::YES{});
  thrust::transform(rmm::exec_policy_nosync(stream),
                    begin,
                    begin + sliced.size(),
                    output->mutable_view().data<uint8_t>(),
                    [] __device__(auto value) {
                      return value.has_value() ? static_cast<uint8_t>(*value) : uint8_t{2};
                    });
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
}

TEST_F(ColumnDeviceViewTest, NonCanonicalBool) { test_noncanonical_bool(); }

TEST_F(ColumnDeviceViewTest, NullableNonCanonicalBool) { test_nullable_noncanonical_bool(); }

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
