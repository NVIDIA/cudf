/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/iterator/iterator_tests.cuh>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/offsets_iterator_factory.cuh>

#include <cuda/iterator>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

using TestingTypes = cudf::test::Types<int32_t, int64_t>;

template <typename T>
struct OffsetalatorTest : public IteratorTest<T> {};

TYPED_TEST_SUITE(OffsetalatorTest, TestingTypes);

TYPED_TEST(OffsetalatorTest, input_iterator)
{
  using T = TypeParam;

  auto host_values = cudf::test::make_type_param_vector<T>({0, 6, 0, -14, 13, 64, -13, -20, 45});

  auto d_col = cudf::test::fixed_width_column_wrapper<T>(host_values.begin(), host_values.end());

  auto expected_values = thrust::host_vector<cudf::size_type>(host_values.size());
  std::transform(host_values.begin(), host_values.end(), expected_values.begin(), [](auto v) {
    return static_cast<cudf::size_type>(v);
  });

  auto it_dev = cudf::detail::offsetalator_factory::make_input_iterator(d_col);
  this->iterator_test_thrust(expected_values, it_dev, host_values.size());
}

TYPED_TEST(OffsetalatorTest, output_iterator)
{
  using T = TypeParam;

  auto d_col1 = cudf::test::fixed_width_column_wrapper<int64_t>({0, 6, 7, 14, 23, 33, 43, 45, 63});
  auto d_col2 = cudf::test::fixed_width_column_wrapper<T>({0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto itr    = cudf::detail::offsetalator_factory::make_output_iterator(d_col2);
  auto input  = cudf::column_view(d_col1);
  auto stream = cudf::get_default_stream();

  auto map   = cudf::test::fixed_width_column_wrapper<int>({0, 2, 4, 6, 8, 1, 3, 5, 7});
  auto d_map = cudf::column_view(map);
  thrust::gather(rmm::exec_policy_nosync(stream),
                 d_map.begin<int>(),
                 d_map.end<int>(),
                 input.begin<int64_t>(),
                 itr);
  auto expected = cudf::test::fixed_width_column_wrapper<T>({0, 7, 23, 43, 63, 6, 14, 33, 45});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(d_col2, expected);

  thrust::scatter(rmm::exec_policy_nosync(stream),
                  input.begin<int64_t>(),
                  input.end<int64_t>(),
                  d_map.begin<int>(),
                  itr);
  expected = cudf::test::fixed_width_column_wrapper<T>({0, 33, 6, 43, 7, 45, 14, 63, 23});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(d_col2, expected);

  thrust::fill(rmm::exec_policy_nosync(stream), itr, itr + input.size(), 77);
  expected = cudf::test::fixed_width_column_wrapper<T>({77, 77, 77, 77, 77, 77, 77, 77, 77});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(d_col2, expected);

  thrust::sequence(rmm::exec_policy_nosync(stream), itr, itr + input.size());
  expected = cudf::test::fixed_width_column_wrapper<T>({0, 1, 2, 3, 4, 5, 6, 7, 8});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(d_col2, expected);

  auto offsets =
    cudf::test::fixed_width_column_wrapper<int64_t>({0, 10, 20, 30, 40, 50, 60, 70, 80});
  auto d_offsets = cudf::column_view(offsets);
  thrust::lower_bound(rmm::exec_policy_nosync(stream),
                      d_offsets.begin<int64_t>(),
                      d_offsets.end<int64_t>(),
                      input.begin<int64_t>(),
                      input.end<int64_t>(),
                      itr);
  expected = cudf::test::fixed_width_column_wrapper<T>({0, 1, 1, 2, 3, 4, 5, 5, 7});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(d_col2, expected);
}

namespace {
/**
 * For testing creating and using the offsetalator in device code.
 */
struct device_functor_fn {
  cudf::column_device_view const d_col;
  __device__ int32_t operator()(int idx)
  {
    auto const itr = cudf::detail::input_offsetalator(d_col.head(), d_col.type());
    return static_cast<int32_t>(itr[idx] * 3);
  }
};
}  // namespace

TYPED_TEST(OffsetalatorTest, device_offsetalator)
{
  using T = TypeParam;

  auto d_col1 = cudf::test::fixed_width_column_wrapper<T>({0, 6, 7, 14, 23, 33, 43, 45, 63});
  auto d_col2 = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto input  = cudf::column_view(d_col1);
  auto output = cudf::mutable_column_view(d_col2);
  auto stream = cudf::get_default_stream();

  auto d_input = cudf::column_device_view::create(input, stream);

  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<int>{0},
                    cuda::counting_iterator<int>{input.size()},
                    output.begin<int32_t>(),
                    device_functor_fn{*d_input});

  auto expected =
    cudf::test::fixed_width_column_wrapper<int32_t>({0, 18, 21, 42, 69, 99, 129, 135, 189});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(d_col2, expected);
}
