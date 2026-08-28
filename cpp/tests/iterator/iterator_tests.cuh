/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/transform_unary_functions.cuh>  // for meanvar
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

#include <bitset>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

// Base Typed test fixture for iterator test
template <typename T>
struct IteratorTest : public cudf::test::BaseFixture {
  // iterator test case which uses cub
  template <typename InputIterator, typename T_output>
  void iterator_test_cub(T_output expected, InputIterator d_in, int num_items)
  {
    T_output init = cudf::test::make_type_param_scalar<T_output>(0);
    rmm::device_uvector<T_output> dev_result(1, cudf::get_default_stream());

    // Get temporary storage size
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(nullptr,
                              temp_storage_bytes,
                              d_in,
                              dev_result.begin(),
                              num_items,
                              cuda::minimum{},
                              init,
                              cudf::get_default_stream().get());

    // Allocate temporary storage
    rmm::device_buffer d_temp_storage(temp_storage_bytes, cudf::get_default_stream());

    // Run reduction
    cub::DeviceReduce::Reduce(d_temp_storage.data(),
                              temp_storage_bytes,
                              d_in,
                              dev_result.begin(),
                              num_items,
                              cuda::minimum{},
                              init,
                              cudf::get_default_stream().get());

    evaluate(expected, dev_result, "cub test");
  }

  // iterator test case which uses thrust
  template <typename InputIterator, typename T_output>
  void iterator_test_thrust(thrust::host_vector<T_output> const& expected,
                            InputIterator d_in,
                            int num_items)
  {
    InputIterator d_in_last = d_in + num_items;
    EXPECT_EQ(cuda::std::distance(d_in, d_in_last), num_items);
    auto dev_expected = cudf::detail::make_device_uvector(
      expected, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

    // using a temporary vector and calling transform and all_of separately is
    // equivalent to thrust::equal but compiles ~3x faster
    auto dev_results = rmm::device_uvector<bool>(num_items, cudf::get_default_stream());
    thrust::transform(rmm::exec_policy_nosync(cudf::get_default_stream()),
                      d_in,
                      d_in_last,
                      dev_expected.begin(),
                      dev_results.begin(),
                      cuda::std::equal_to{});
    auto result = thrust::all_of(rmm::exec_policy_nosync(cudf::get_default_stream()),
                                 dev_results.begin(),
                                 dev_results.end(),
                                 cuda::std::identity{});
    EXPECT_TRUE(result) << "thrust test";
  }

  template <typename T_output>
  void evaluate(T_output expected,
                rmm::device_uvector<T_output> const& dev_result,
                char const* msg = nullptr)
  {
    auto host_result = cudf::detail::make_host_vector(dev_result, cudf::get_default_stream());

    EXPECT_EQ(expected, host_result[0]) << msg;
  }

  template <typename T_output>
  void values_equal_test(thrust::host_vector<T_output> const& expected,
                         cudf::column_device_view const& col)
  {
    if (col.nullable()) {
      auto it_dev = cudf::detail::make_null_replacement_iterator(
        col, cudf::test::make_type_param_scalar<T_output>(0));
      iterator_test_thrust(expected, it_dev, col.size());
    } else {
      auto it_dev = col.begin<T_output>();
      iterator_test_thrust(expected, it_dev, col.size());
    }
  }
};
