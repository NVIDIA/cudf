/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "generate_input.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/random>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <type_traits>

/**
 * @brief Real Type that has at least number of bits of integral type in its mantissa.
 *  number of bits of integrals < 23 bits of mantissa in float
 * to allow full range of integer bits to be generated.
 * @tparam T integral type
 */
template <typename T>
using integral_to_realType =
  std::conditional_t<cuda::std::is_floating_point_v<T>,
                     T,
                     std::conditional_t<sizeof(T) * 8 <= 23, float, double>>;

// standard deviation such that most samples fall within the given range
template <typename T>
constexpr double std_dev_from_range(T lower_bound, T upper_bound)
{
  // 99.7% samples are within 3 standard deviations of the mean
  constexpr double k    = 6.0;
  auto const range_size = std::abs(static_cast<double>(upper_bound) - lower_bound);
  return range_size / k;
}

/**
 * @brief Generates a normal distribution between zero and upper_bound.
 */
template <typename T>
auto make_normal_dist(T lower_bound, T upper_bound)
{
  using realT        = integral_to_realType<T>;
  realT const mean   = lower_bound / 2. + upper_bound / 2.;
  realT const stddev = std_dev_from_range(lower_bound, upper_bound);
  return cuda::std::normal_distribution<realT>(mean, stddev);
}

template <typename T, std::enable_if_t<cuda::std::is_integral_v<T>, T>* = nullptr>
auto make_uniform_dist(T range_start, T range_end)
{
  return cuda::std::uniform_int_distribution<T>(range_start, range_end);
}

template <typename T, std::enable_if_t<cudf::is_floating_point<T>()>* = nullptr>
auto make_uniform_dist(T range_start, T range_end)
{
  return cuda::std::uniform_real_distribution<T>(range_start, range_end);
}

/**
 * @brief Generates a geometric distribution between lower_bound and upper_bound.
 * This distribution is an approximation generated using normal distribution.
 *
 * @tparam T Result type of the number to produce.
 */
template <typename T>
class geometric_distribution : public cuda::std::normal_distribution<integral_to_realType<T>> {
  using realType = integral_to_realType<T>;
  using super_t  = cuda::std::normal_distribution<realType>;
  T _lower_bound;
  T _upper_bound;

  super_t make_approx_normal_dist(T lower_bound, T upper_bound) const
  {
    auto const abs_range_size = std::abs(static_cast<realType>(upper_bound) - lower_bound);
    // Generate normal distribution around zero; output will be shifted by lower_bound
    return make_normal_dist(-abs_range_size, abs_range_size);
  }

 public:
  using result_type = T;
  explicit geometric_distribution(T lower_bound, T upper_bound)
    : super_t(make_approx_normal_dist(lower_bound, upper_bound)),
      _lower_bound(lower_bound),
      _upper_bound(upper_bound)
  {
  }

  template <typename UniformRandomNumberGenerator>
  __host__ __device__ result_type operator()(UniformRandomNumberGenerator& urng)
  {
    // Distribution always biases towards lower_bound
    realType const result = _lower_bound < _upper_bound
                              ? cuda::std::abs(super_t::operator()(urng)) + _lower_bound
                              : _lower_bound - cuda::std::abs(super_t::operator()(urng));
    return cuda::std::round(result);
  }
};

template <typename T, typename Generator>
struct value_generator {
  using result_type = T;

  value_generator(T lower_bound, T upper_bound, cuda::std::philox4x32& engine, Generator gen)
    : lower_bound(std::min(lower_bound, upper_bound)),
      upper_bound(std::max(lower_bound, upper_bound)),
      engine(engine),
      dist(gen)
  {
  }

  __device__ T operator()(size_t n)
  {
    engine.discard(n);
    if constexpr (cuda::std::is_integral_v<T> &&
                  cuda::std::is_floating_point_v<decltype(dist(engine))>) {
      return cuda::std::clamp(
        static_cast<T>(cuda::std::round(dist(engine))), lower_bound, upper_bound);
    } else {
      return cuda::std::clamp(dist(engine), lower_bound, upper_bound);
    }
    // Note: uniform does not need clamp, because already range is guaranteed to be within bounds.
  }

  T lower_bound;
  T upper_bound;
  cuda::std::philox4x32 engine;
  Generator dist;
};

template <typename T>
using distribution_fn = std::function<rmm::device_uvector<T>(cuda::std::philox4x32&, size_t)>;

template <
  typename T,
  std::enable_if_t<cuda::std::is_integral_v<T> or cuda::std::is_floating_point_v<T>, T>* = nullptr>
distribution_fn<T> make_distribution(distribution_id dist_id, T lower_bound, T upper_bound)
{
  switch (dist_id) {
    case distribution_id::NORMAL:
      return [lower_bound, upper_bound, dist = make_normal_dist(lower_bound, upper_bound)](
               cuda::std::philox4x32& engine, size_t size) -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, cudf::get_default_stream());
        thrust::tabulate(thrust::device,
                         result.begin(),
                         result.end(),
                         value_generator{lower_bound, upper_bound, engine, dist});
        return result;
      };
    case distribution_id::UNIFORM:
      return [lower_bound, upper_bound, dist = make_uniform_dist(lower_bound, upper_bound)](
               cuda::std::philox4x32& engine, size_t size) -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, cudf::get_default_stream());
        thrust::tabulate(thrust::device,
                         result.begin(),
                         result.end(),
                         value_generator{lower_bound, upper_bound, engine, dist});
        return result;
      };
    case distribution_id::GEOMETRIC:
      // kind of exponential distribution from lower_bound to upper_bound.
      return [lower_bound, upper_bound, dist = geometric_distribution<T>(lower_bound, upper_bound)](
               cuda::std::philox4x32& engine, size_t size) -> rmm::device_uvector<T> {
        rmm::device_uvector<T> result(size, cudf::get_default_stream());
        thrust::tabulate(thrust::device,
                         result.begin(),
                         result.end(),
                         value_generator{lower_bound, upper_bound, engine, dist});
        return result;
      };
    default: CUDF_FAIL("Unsupported probability distribution");
  }
}
