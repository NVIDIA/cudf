/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/iterator_tests.cuh>

#include <cudf_test/random.hpp>

#include <cuda/std/utility>
#include <thrust/host_vector.h>

using TestingTypes = cudf::test::FixedWidthTypesWithoutFixedPoint;

TYPED_TEST_SUITE(IteratorTest, TestingTypes);

TYPED_TEST(IteratorTest, scalar_iterator)
{
  using T = TypeParam;
  T init  = cudf::test::make_type_param_scalar<T>(
    cudf::test::UniformRandomGenerator<int>(-128, 128).generate());
  // data and valid arrays
  thrust::host_vector<T> host_values(100, init);
  std::vector<bool> host_bools(100, true);

  // create a scalar
  using ScalarType = cudf::scalar_type_t<T>;
  std::unique_ptr<cudf::scalar> s(new ScalarType{init, true});

  // calculate the expected value by CPU.
  thrust::host_vector<cuda::std::pair<T, bool>> value_and_validity(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 value_and_validity.begin(),
                 [](auto v, auto b) { return cuda::std::pair<T, bool>{v, b}; });

  // GPU test
  auto const scalar_view = s->as_column_view();
  auto const d_scalar =
    cudf::column_device_view::create(scalar_view.as_column_view(), cudf::get_default_stream());
  auto it_dev = cuda::make_permutation_iterator(d_scalar->begin<T>(),
                                                cuda::make_constant_iterator<cudf::size_type>(0));
  this->iterator_test_thrust(host_values, it_dev, host_values.size());

  auto it_pair_dev =
    cuda::make_permutation_iterator(cudf::detail::make_pair_iterator<T, true>(*d_scalar),
                                    cuda::make_constant_iterator<cudf::size_type>(0));
  this->iterator_test_thrust(value_and_validity, it_pair_dev, host_values.size());
}

TYPED_TEST(IteratorTest, null_scalar_iterator)
{
  using T = TypeParam;
  T init  = cudf::test::make_type_param_scalar<T>(
    cudf::test::UniformRandomGenerator<int>(-128, 128).generate());
  // data and valid arrays
  std::vector<T> host_values(100, init);
  std::vector<bool> host_bools(100, false);

  // create a scalar
  using ScalarType = cudf::scalar_type_t<T>;
  std::unique_ptr<cudf::scalar> s(new ScalarType{init, false});

  // calculate the expected value by CPU.
  thrust::host_vector<cuda::std::pair<T, bool>> value_and_validity(host_values.size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 value_and_validity.begin(),
                 [](auto v, auto b) { return cuda::std::pair<T, bool>{v, b}; });

  // GPU test
  auto const scalar_view = s->as_column_view();
  auto const d_scalar =
    cudf::column_device_view::create(scalar_view.as_column_view(), cudf::get_default_stream());
  auto it_pair_dev =
    cuda::make_permutation_iterator(cudf::detail::make_pair_iterator<T, true>(*d_scalar),
                                    cuda::make_constant_iterator<cudf::size_type>(0));
  this->iterator_test_thrust(value_and_validity, it_pair_dev, host_values.size());
}
