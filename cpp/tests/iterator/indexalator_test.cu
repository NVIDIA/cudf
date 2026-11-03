/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/iterator/iterator_tests.cuh>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/indexalator.cuh>

#include <cuda/std/optional>
#include <cuda/std/utility>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

using TestingTypes = cudf::test::IntegralTypesNotBool;

template <typename T>
struct IndexalatorTest : public IteratorTest<T> {};

TYPED_TEST_SUITE(IndexalatorTest, TestingTypes);

TYPED_TEST(IndexalatorTest, input_iterator)
{
  using T = TypeParam;

  auto host_values = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});

  auto d_col = cudf::test::fixed_width_column_wrapper<T>(host_values.begin(), host_values.end());

  auto expected_values = thrust::host_vector<cudf::size_type>(host_values.size());
  std::transform(host_values.begin(), host_values.end(), expected_values.begin(), [](auto v) {
    return static_cast<cudf::size_type>(v);
  });

  auto it_dev = cudf::detail::indexalator_factory::make_input_iterator(d_col);
  this->iterator_test_thrust(expected_values, it_dev, host_values.size());
}

TYPED_TEST(IndexalatorTest, optional_iterator)
{
  using T = TypeParam;

  auto host_values = cudf::test::make_type_param_vector<T>({0, 6, 0, -104, 103, 64, -13, -20, 45});
  auto validity    = std::vector<bool>({0, 1, 1, 1, 1, 1, 0, 1, 1});

  auto d_col = cudf::test::fixed_width_column_wrapper<T>(
    host_values.begin(), host_values.end(), validity.begin());

  auto expected_values =
    thrust::host_vector<cuda::std::optional<cudf::size_type>>(host_values.size());

  std::transform(host_values.begin(),
                 host_values.end(),
                 validity.begin(),
                 expected_values.begin(),
                 [](T v, bool b) {
                   return (b) ? cuda::std::make_optional(static_cast<cudf::size_type>(v))
                              : cuda::std::nullopt;
                 });

  auto it_dev = cudf::detail::indexalator_factory::make_input_optional_iterator(d_col);
  this->iterator_test_thrust(expected_values, it_dev, host_values.size());
}

template <typename Integer>
struct transform_fn {
  __device__ cudf::size_type operator()(Integer v)
  {
    return static_cast<cudf::size_type>(v) + static_cast<cudf::size_type>(v);
  }
};

TYPED_TEST(IndexalatorTest, output_iterator)
{
  using T = TypeParam;

  auto d_col1 =
    cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 6, 7, 14, 23, 33, 43, 45, 63});
  auto d_col2 =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto itr    = cudf::detail::indexalator_factory::make_output_iterator(d_col2);
  auto input  = cudf::column_view(d_col1);
  auto stream = cudf::get_default_stream();

  auto map   = cudf::test::fixed_width_column_wrapper<int>({0, 2, 4, 6, 8, 1, 3, 5, 7});
  auto d_map = cudf::column_view(map);
  thrust::gather(
    rmm::exec_policy_nosync(stream), d_map.begin<int>(), d_map.end<int>(), input.begin<T>(), itr);
  auto expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 7, 23, 43, 63, 6, 14, 33, 45});
  thrust::scatter(
    rmm::exec_policy_nosync(stream), input.begin<T>(), input.end<T>(), d_map.begin<int>(), itr);
  expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 33, 6, 43, 7, 45, 14, 63, 23});

  thrust::transform(
    rmm::exec_policy_nosync(stream), input.begin<T>(), input.end<T>(), itr, transform_fn<T>{});
  expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 12, 14, 28, 46, 66, 86, 90, 126});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(d_col2, expected);

  thrust::fill(rmm::exec_policy_nosync(stream), itr, itr + input.size(), 77);
  expected =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>({77, 77, 77, 77, 77, 77, 77, 77, 77});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(d_col2, expected);

  thrust::sequence(rmm::exec_policy_nosync(stream), itr, itr + input.size());
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 1, 2, 3, 4, 5, 6, 7, 8});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(d_col2, expected);

  auto indices =
    cudf::test::fixed_width_column_wrapper<T, int32_t>({0, 10, 20, 30, 40, 50, 60, 70, 80});
  auto d_indices = cudf::column_view(indices);
  thrust::lower_bound(rmm::exec_policy_nosync(stream),
                      d_indices.begin<T>(),
                      d_indices.end<T>(),
                      input.begin<T>(),
                      input.end<T>(),
                      itr);
  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 1, 1, 2, 3, 4, 5, 5, 7});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(d_col2, expected);
}
