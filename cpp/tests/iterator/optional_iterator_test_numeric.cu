/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/optional_iterator_test.cuh>

#include <cudf_test/random.hpp>

#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>
#include <cuda/std/optional>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

using TestingTypes = cudf::test::NumericTypes;

template <typename T>
struct NumericOptionalIteratorTest : public IteratorTest<T> {};

TYPED_TEST_SUITE(NumericOptionalIteratorTest, TestingTypes);
TYPED_TEST(NumericOptionalIteratorTest, nonull_optional_iterator)
{
  nonull_optional_iterator(*this);
}
TYPED_TEST(NumericOptionalIteratorTest, null_optional_iterator) { null_optional_iterator(*this); }

namespace {
// Transformers and Operators for optional_iterator test
template <typename ElementType>
struct transformer_optional_meanvar {
  using ResultType = cuda::std::optional<cudf::meanvar<ElementType>>;

  CUDF_HOST_DEVICE inline ResultType operator()(cuda::std::optional<ElementType> const& optional)
  {
    if (optional.has_value()) {
      auto v = *optional;
      return cudf::meanvar<ElementType>{v, static_cast<ElementType>(v * v), 1};
    }
    return cuda::std::nullopt;
  }
};

template <typename T>
struct optional_to_meanvar {
  CUDF_HOST_DEVICE inline T operator()(cuda::std::optional<T> const& v) { return v.value_or(T{0}); }
};
}  // namespace

// TODO: enable this test also at __CUDACC_DEBUG__
// This test causes fatal compilation error only at device debug mode.
// Workaround: exclude this test only at device debug mode.
#if !defined(__CUDACC_DEBUG__)
TYPED_TEST(NumericOptionalIteratorTest, mean_var_output)
{
  using T        = TypeParam;
  using T_output = cudf::meanvar<T>;
  transformer_optional_meanvar<T> transformer{};

  int const column_size{50};
  T const init{0};

  // data and valid arrays
  std::vector<T> host_values(column_size);
  std::vector<bool> host_bools(column_size);

  cudf::test::UniformRandomGenerator<T> rng;
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(host_values.begin(), host_values.end(), [&rng]() { return rng.generate(); });
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
  auto it_dev         = d_col->template optional_begin<T>(cudf::nullate::YES{});
  auto it_dev_squared = cuda::transform_iterator(it_dev, transformer);

  // this can be computed with a single reduce and without a temporary output vector
  // but the approach increases the compile time by ~2x
  auto results = rmm::device_uvector<T_output>(d_col->size(), cudf::get_default_stream());
  thrust::transform(rmm::exec_policy_nosync(cudf::get_default_stream()),
                    it_dev_squared,
                    it_dev_squared + d_col->size(),
                    results.begin(),
                    optional_to_meanvar<T_output>{});
  auto result = thrust::reduce(rmm::exec_policy_nosync(cudf::get_default_stream()),
                               results.begin(),
                               results.end(),
                               T_output{});

  if (not std::is_floating_point<T>()) {
    EXPECT_EQ(expected_value, result) << "optional iterator reduction sum";
  } else {
    EXPECT_NEAR(expected_value.value, result.value, 1e-3) << "optional iterator reduction sum";
    EXPECT_NEAR(expected_value.value_squared, result.value_squared, 1e-3)
      << "optional iterator reduction sum squared";
    EXPECT_EQ(expected_value.count, result.count) << "optional iterator reduction count";
  }
}
#endif
