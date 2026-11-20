/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tests/iterator/iterator_tests.cuh>

#include <cudf_test/random.hpp>

#include <cuda/std/optional>
#include <cuda/std/utility>
#include <thrust/host_vector.h>

using TestingTypes = cudf::test::FixedWidthTypes;

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
  thrust::host_vector<cuda::std::optional<T>> optional_values(host_values.size());
  std::transform(host_values.begin(), host_values.end(), optional_values.begin(), [](auto v) {
    return cuda::std::optional<T>{v};
  });

  // GPU test
  auto it_dev = cudf::detail::make_scalar_iterator<T>(*s);
  this->iterator_test_thrust(host_values, it_dev, host_values.size());

  auto it_pair_dev = cudf::detail::make_pair_iterator<T>(*s);
  this->iterator_test_thrust(value_and_validity, it_pair_dev, host_values.size());

  auto it_optional_dev = cudf::detail::make_optional_iterator<T>(*s, cudf::nullate::DYNAMIC{true});
  this->iterator_test_thrust(optional_values, it_optional_dev, host_values.size());
}

TYPED_TEST(IteratorTest, null_scalar_iterator)
{
  using T = TypeParam;
  T init  = cudf::test::make_type_param_scalar<T>(
    cudf::test::UniformRandomGenerator<int>(-128, 128).generate());
  // data and valid arrays
  std::vector<T> host_values(100, init);
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
  auto it_pair_dev = cudf::detail::make_pair_iterator<T>(*s);
  this->iterator_test_thrust(value_and_validity, it_pair_dev, host_values.size());
}

template <typename T>
struct FixedPointScalarIteratorTest : IteratorTest<T> {};

TYPED_TEST_SUITE(FixedPointScalarIteratorTest, cudf::test::FixedPointTypes);

struct fixed_point_rep_and_scale {
  template <typename T>
  CUDF_HOST_DEVICE auto operator()(T value) const
  {
    return cuda::std::pair<typename T::rep, int32_t>{value.value(),
                                                     static_cast<int32_t>(value.scale())};
  }
};

struct scalar_pair_validity {
  template <typename T>
  CUDF_HOST_DEVICE bool operator()(cuda::std::pair<T, bool> value) const
  {
    return value.second;
  }
};

TYPED_TEST(FixedPointScalarIteratorTest, ValuesPreserveScale)
{
  using T             = TypeParam;
  using rep_type      = typename T::rep;
  auto constexpr size = 100;

  for (auto const scale : {-2, 0, 3}) {
    SCOPED_TRACE(scale);
    for (auto const rep : {rep_type{12'345}, rep_type{-6'789}}) {
      auto const expected = T{numeric::scaled_integer<rep_type>{rep, numeric::scale_type{scale}}};
      cudf::fixed_point_scalar<T> scalar{rep, numeric::scale_type{scale}};

      thrust::host_vector<T> values(size, expected);
      auto const value_it = cudf::detail::make_scalar_iterator<T>(scalar);
      this->iterator_test_thrust(values, value_it, size);

      // Decimal equality allows different representations of the same value. Check both
      // representation and scale explicitly to ensure the iterator preserves them.
      auto const rep_and_scale_it = cuda::transform_iterator(value_it, fixed_point_rep_and_scale{});
      thrust::host_vector<cuda::std::pair<rep_type, int32_t>> reps_and_scales(
        size, cuda::std::pair<rep_type, int32_t>{rep, scale});
      this->iterator_test_thrust(reps_and_scales, rep_and_scale_it, size);

      thrust::host_vector<cuda::std::pair<T, bool>> pairs(size,
                                                          cuda::std::pair<T, bool>{expected, true});
      this->iterator_test_thrust(pairs, cudf::detail::make_pair_iterator<T>(scalar), size);

      thrust::host_vector<cuda::std::optional<T>> optionals(size, cuda::std::optional<T>{expected});
      auto const check_optional = [&](auto nullate) {
        this->iterator_test_thrust(
          optionals, cudf::detail::make_optional_iterator<T>(scalar, nullate), size);
      };
      check_optional(cudf::nullate::NO{});
      check_optional(cudf::nullate::YES{});
      check_optional(cudf::nullate::DYNAMIC{false});
      check_optional(cudf::nullate::DYNAMIC{true});
    }
  }
}

TYPED_TEST(FixedPointScalarIteratorTest, NullsPreserveValidity)
{
  using T             = TypeParam;
  using rep_type      = typename T::rep;
  auto constexpr size = 100;

  for (auto const scale : {-2, 0, 3}) {
    SCOPED_TRACE(scale);
    cudf::fixed_point_scalar<T> scalar{rep_type{12'345}, numeric::scale_type{scale}, false};
    EXPECT_THROW(cudf::detail::make_scalar_iterator<T>(scalar), cudf::logic_error);

    auto const validity_it =
      cuda::transform_iterator(cudf::detail::make_pair_iterator<T>(scalar), scalar_pair_validity{});
    thrust::host_vector<bool> validities(size, false);
    this->iterator_test_thrust(validities, validity_it, size);

    thrust::host_vector<cuda::std::optional<T>> optionals(size, cuda::std::nullopt);
    this->iterator_test_thrust(
      optionals, cudf::detail::make_optional_iterator<T>(scalar, cudf::nullate::YES{}), size);
    this->iterator_test_thrust(
      optionals,
      cudf::detail::make_optional_iterator<T>(scalar, cudf::nullate::DYNAMIC{true}),
      size);
  }
}

TYPED_TEST(FixedPointScalarIteratorTest, RejectMismatchedType)
{
  using T        = TypeParam;
  using rep_type = typename T::rep;

  for (auto const scale : {-2, 0, 3}) {
    SCOPED_TRACE(scale);
    cudf::fixed_point_scalar<T> scalar{rep_type{12'345}, numeric::scale_type{scale}};
    EXPECT_THROW(cudf::detail::make_scalar_iterator<rep_type>(scalar), cudf::logic_error);
  }
}
