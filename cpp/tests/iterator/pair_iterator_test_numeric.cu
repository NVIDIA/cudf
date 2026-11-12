/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/pair_iterator_test.cuh>

#include <cudf_test/random.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/utility>
#include <thrust/reduce.h>

using TestingTypes = cudf::test::NumericTypes;

template <typename T>
struct NumericPairIteratorTest : public IteratorTest<T> {};

TYPED_TEST_SUITE(NumericPairIteratorTest, TestingTypes);
TYPED_TEST(NumericPairIteratorTest, nonull_pair_iterator) { nonull_pair_iterator(*this); }
TYPED_TEST(NumericPairIteratorTest, null_pair_iterator) { null_pair_iterator(*this); }

// Transformers and Operators for pair_iterator test
template <typename ElementType>
struct transformer_pair_meanvar {
  using ResultType = cuda::std::pair<cudf::meanvar<ElementType>, bool>;

  CUDF_HOST_DEVICE inline ResultType operator()(cuda::std::pair<ElementType, bool> const& pair)
  {
    ElementType v = pair.first;
    return {{v, static_cast<ElementType>(v * v), (pair.second) ? 1 : 0}, pair.second};
  };
};

namespace {
struct sum_if_not_null {
  template <typename T>
  CUDF_HOST_DEVICE inline cuda::std::pair<T, bool> operator()(cuda::std::pair<T, bool> const& lhs,
                                                              cuda::std::pair<T, bool> const& rhs)
  {
    if (lhs.second & rhs.second)
      return {lhs.first + rhs.first, true};
    else if (lhs.second)
      return {lhs};
    else
      return {rhs};
  }
};
}  // namespace

// TODO: enable this test also at __CUDACC_DEBUG__
// This test causes fatal compilation error only at device debug mode.
// Workaround: exclude this test only at device debug mode.
#if !defined(__CUDACC_DEBUG__)
// This test computes `count`, `sum`, `sum_of_squares` at a single reduction call.
// It would be useful for `var`, `std` operation
TYPED_TEST(NumericPairIteratorTest, mean_var_output)
{
  using T        = TypeParam;
  using T_output = cudf::meanvar<T>;
  transformer_pair_meanvar<T> transformer{};

  int const column_size{5000};
  T const init{0};

  // data and valid arrays
  std::vector<T> host_values(column_size);
  std::vector<bool> host_bools(column_size);

  if constexpr (std::is_floating_point<T>()) {
    cudf::test::UniformRandomGenerator<int32_t> rng;
    std::generate(host_values.begin(), host_values.end(), [&rng]() {
      return static_cast<T>(rng.generate() % 10);  // reduces float-op errors
    });
  } else {
    cudf::test::UniformRandomGenerator<T> rng;
    std::generate(host_values.begin(), host_values.end(), [&rng]() { return rng.generate(); });
  }

  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(host_bools.begin(), host_bools.end(), [&rbg]() { return rbg.generate(); });

  cudf::test::fixed_width_column_wrapper<TypeParam> w_col(
    host_values.begin(), host_values.end(), host_bools.begin());
  auto d_col = cudf::column_device_view::create(w_col);

  // calculate expected values by CPU
  T_output expected_value;

  expected_value.count = d_col->size() - static_cast<cudf::column_view>(w_col).null_count();

  std::vector<T> replaced_array(d_col->size());
  std::transform(host_values.begin(),
                 host_values.end(),
                 host_bools.begin(),
                 replaced_array.begin(),
                 [&](T x, bool b) { return (b) ? static_cast<T>(x) : init; });

  expected_value.count = d_col->size() - static_cast<cudf::column_view>(w_col).null_count();
  expected_value.value = std::accumulate(replaced_array.begin(), replaced_array.end(), T{0});
  expected_value.value_squared = std::accumulate(
    replaced_array.begin(), replaced_array.end(), T{0}, [](T acc, T i) { return acc + i * i; });

  // GPU test
  auto it_dev         = d_col->template pair_begin<T, true>();
  auto it_dev_squared = cuda::transform_iterator(it_dev, transformer);
  auto result         = thrust::reduce(rmm::exec_policy_nosync(cudf::get_default_stream()),
                               it_dev_squared,
                               it_dev_squared + d_col->size(),
                               cuda::std::make_pair(T_output{}, true),
                               sum_if_not_null{});
  if constexpr (not std::is_floating_point<T>()) {
    EXPECT_EQ(expected_value, result.first) << "pair iterator reduction sum";
  } else {
    EXPECT_NEAR(expected_value.value, result.first.value, 1e-3) << "pair iterator reduction sum";
    EXPECT_NEAR(expected_value.value_squared, result.first.value_squared, 1e-3)
      << "pair iterator reduction sum squared";
    EXPECT_EQ(expected_value.count, result.first.count) << "pair iterator reduction count";
  }
}
#endif
