/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/iterator_tests.cuh>

#include <cudf_test/random.hpp>

#include <cuda/functional>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>

struct TransformedIteratorTest : public IteratorTest<int8_t> {};

template <typename T>
struct cast_fn {
  template <typename U>
  __device__ T operator()(U const& val) const
  {
    return static_cast<T>(val);
  }
};

// Tests up cast reduction with null iterator.
// The up cast iterator will be created by transform_iterator and
// cudf::detail::make_null_replacement_iterator(col, T{0})
TEST_F(TransformedIteratorTest, null_iterator_upcast)
{
  int const column_size{1000};
  using T        = int8_t;
  using T_upcast = int64_t;
  T init{0};

  // data and valid arrays
  std::vector<T> host_values(column_size);
  std::vector<bool> host_bools(column_size);

  cudf::test::UniformRandomGenerator<T> rng(-128, 127);
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(host_values.begin(), host_values.end(), [&rng]() { return rng.generate(); });
  std::generate(host_bools.begin(), host_bools.end(), [&rbg]() { return rbg.generate(); });

  cudf::test::fixed_width_column_wrapper<T> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  thrust::host_vector<T> replaced_array(d_col->size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x : init; });
  T_upcast expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev        = cudf::detail::make_null_replacement_iterator(*d_col, T{0});
  auto it_dev_upcast = thrust::make_transform_iterator(it_dev, cast_fn<T_upcast>{});
  this->iterator_test_thrust(replaced_array, it_dev_upcast, d_col->size());
  this->iterator_test_cub(expected_value, it_dev, d_col->size());
}

// Tests for square input iterator using helper strcut
// `cudf::transformer_squared<T, T_upcast>` The up cast iterator will be created
// by make_transform_iterator(
//        cudf::detail::make_null_replacement_iterator(col, T{0}),
//        cudf::detail::transformer_squared<T_upcast>)
TEST_F(TransformedIteratorTest, null_iterator_square)
{
  int const column_size{1000};
  using T        = int8_t;
  using T_upcast = int64_t;
  T init{0};
  cudf::transformer_squared<T_upcast> transformer{};

  // data and valid arrays
  std::vector<T> host_values(column_size);
  std::vector<bool> host_bools(column_size);

  cudf::test::UniformRandomGenerator<T> rng(-128, 127);
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(host_values.begin(), host_values.end(), [&rng]() { return rng.generate(); });
  std::generate(host_bools.begin(), host_bools.end(), [&rbg]() { return rbg.generate(); });

  cudf::test::fixed_width_column_wrapper<T> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate the expected value by CPU.
  thrust::host_vector<T_upcast> replaced_array(d_col->size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x * x : init; });
  T_upcast expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev         = cudf::detail::make_null_replacement_iterator(*d_col, T{0});
  auto it_dev_upcast  = thrust::make_transform_iterator(it_dev, cast_fn<T_upcast>{});
  auto it_dev_squared = thrust::make_transform_iterator(it_dev_upcast, transformer);
  this->iterator_test_thrust(replaced_array, it_dev_squared, d_col->size());
  this->iterator_test_cub(expected_value, it_dev_squared, d_col->size());
}

// TODO only few types
TEST_F(TransformedIteratorTest, large_size_reduction)
{
  using T = int64_t;

  int const column_size{1000000};
  T const init{0};

  // data and valid arrays
  std::vector<T> host_values(column_size);
  std::vector<bool> host_bools(column_size);

  cudf::test::UniformRandomGenerator<T> rng(-128, 128);
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(host_values.begin(), host_values.end(), [&rng]() { return rng.generate(); });
  std::generate(host_bools.begin(), host_bools.end(), [&rbg]() { return rbg.generate(); });

  cudf::test::fixed_width_column_wrapper<T> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate by cudf::reduce
  thrust::host_vector<T> replaced_array(d_col->size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? x : init; });
  T expected_value = *std::min_element(replaced_array.begin(), replaced_array.end());
  // std::cout << "expected <null_iterator> = " << expected_value << std::endl;

  // GPU test
  auto it_dev = cudf::detail::make_null_replacement_iterator(*d_col, init);
  this->iterator_test_thrust(replaced_array, it_dev, d_col->size());
  this->iterator_test_cub(expected_value, it_dev, d_col->size());
}
