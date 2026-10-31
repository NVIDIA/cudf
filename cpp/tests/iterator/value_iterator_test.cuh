/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <tests/iterator/iterator_tests.cuh>

#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/host_vector.h>

// tests for non-null iterator (pointer of device array)
template <typename T>
void non_null_iterator(IteratorTest<T>& testFixture)
{
  auto host_array = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});
  auto dev_array  = cudf::detail::make_device_buffer(
    host_array, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  // calculate the expected value by CPU.
  thrust::host_vector<T> replaced_array(host_array);

  // driven by iterator as a pointer of device array.
  auto it_dev      = dev_array.begin();
  T expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  testFixture.iterator_test_thrust(replaced_array, it_dev, dev_array.size());
  testFixture.iterator_test_cub(expected_value, it_dev, dev_array.size());

  // test column input
  cudf::test::fixed_width_column_wrapper<T> w_col(host_array.begin(), host_array.end());
  testFixture.values_equal_test(replaced_array, *cudf::column_device_view::create(w_col));
}

// Tests for null input iterator (column with null bitmap)
// Actually, we can use cub for reduction with nulls without creating custom
// kernel or multiple steps. We may accelerate the reduction for a column using
// cub
template <typename T>
void null_iterator(IteratorTest<T>& testFixture)
{
  T init = cudf::test::make_type_param_scalar<T>(0);
  // data and valid arrays
  auto host_values = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});
  std::vector<bool> host_bools({1, 1, 0, 1, 1, 1, 0, 1, 1});

  // create a column with bool vector
  cudf::test::fixed_width_column_wrapper<T> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  thrust::host_vector<T> replaced_array(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x : init; });
  T expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  // TODO uncomment after time_point ostream operator<<
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev =
    cudf::detail::make_null_replacement_iterator(*d_col, cudf::test::make_type_param_scalar<T>(0));
  testFixture.iterator_test_cub(expected_value, it_dev, d_col->size());
  testFixture.values_equal_test(replaced_array, *d_col);
}
