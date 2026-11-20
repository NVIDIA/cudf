/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <algorithm>

using TestingTypes = cudf::test::IntegralTypesNotBool;

template <typename T>
struct SizesToOffsetsIteratorTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(SizesToOffsetsIteratorTestTyped, TestingTypes);

TYPED_TEST(SizesToOffsetsIteratorTestTyped, ExclusiveScan)
{
  using T        = TypeParam;
  using LastType = int64_t;

  auto stream = cudf::get_default_stream();

  auto sizes = std::vector<T>({0, 6, 0, 14, 13, 64, 10, 20, 41});

  auto d_col  = cudf::test::fixed_width_column_wrapper<T>(sizes.begin(), sizes.end());
  auto d_view = cudf::column_view(d_col);

  auto last   = cudf::detail::device_scalar<LastType>(0, stream);
  auto result = cuda::device_buffer<T>(
    stream, cudf::get_current_device_resource_ref(), d_view.size(), cuda::no_init);
  auto output_itr =
    cudf::detail::make_sizes_to_offsets_iterator(result.begin(), result.end(), last.data());

  thrust::exclusive_scan(
    rmm::exec_policy_nosync(stream), d_view.begin<T>(), d_view.end<T>(), output_itr, LastType{0});

  auto expected_values = std::vector<T>(sizes.size());
  std::exclusive_scan(sizes.begin(), sizes.end(), expected_values.begin(), T{0});
  auto expected_reduce =
    static_cast<LastType>(std::reduce(sizes.begin(), sizes.begin() + sizes.size() - 1, T{0}));

  auto expected =
    cudf::test::fixed_width_column_wrapper<T>(expected_values.begin(), expected_values.end());
  auto result_col = cudf::column_view(
    cudf::data_type(cudf::type_to_id<T>()), d_view.size(), result.data(), nullptr, 0);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_col, expected);
  EXPECT_EQ(last.value(stream), expected_reduce);
}

struct SizesToOffsetsIteratorTest : public cudf::test::BaseFixture {};

TEST_F(SizesToOffsetsIteratorTest, ScanWithOverflow)
{
  auto stream = cudf::get_default_stream();

  std::vector<int32_t> values(30000, 100000);
  auto d_col  = cudf::test::fixed_width_column_wrapper<int32_t>(values.begin(), values.end());
  auto d_view = cudf::column_view(d_col);

  auto last   = cudf::detail::device_scalar<int64_t>(0, stream);
  auto result = cuda::device_buffer<int32_t>(
    stream, cudf::get_current_device_resource_ref(), d_view.size(), cuda::no_init);
  auto output_itr =
    cudf::detail::make_sizes_to_offsets_iterator(result.begin(), result.end(), last.data());

  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         d_view.begin<int32_t>(),
                         d_view.end<int32_t>(),
                         output_itr,
                         int64_t{0});

  auto expected = static_cast<int64_t>(
    std::reduce(values.begin(), values.begin() + values.size() - 1, int64_t{0}));
  EXPECT_EQ(last.value(stream), expected);
}
