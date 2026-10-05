/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/traits.hpp>

#include <random>

namespace CUDF_EXPORT cudf {
namespace test {

template <typename T, typename Enable = void>
struct uniform_distribution_impl {};
template <typename T>
struct uniform_distribution_impl<T, std::enable_if_t<std::is_integral_v<T>>> {
  using type = std::uniform_int_distribution<T>;
};

template <>
struct uniform_distribution_impl<bool> {
  using type = std::bernoulli_distribution;
};

template <typename T>
struct uniform_distribution_impl<T, std::enable_if_t<std::is_floating_point_v<T>>> {
  using type = std::uniform_real_distribution<T>;
};

template <typename T>
struct uniform_distribution_impl<
  T,
  std::enable_if_t<cudf::is_chrono<T>() or cudf::is_fixed_point<T>()>> {
  using type = std::uniform_int_distribution<typename T::rep>;
};

template <typename T>
using uniform_distribution_t = typename uniform_distribution_impl<T>::type;

namespace detail {

/**
 * @brief Returns an incrementing seed value for use with UniformRandomGenerator.
 *
 *  The intent behind this is to handle the following case:
 *
 * auto lhs = make_random_wrapped_column<TypeLhs>(10000);
 * auto rhs = make_random_wrapped_column<TypeRhs>(10000);
 *
 * Previously, the binops test framework had a persistent UniformRandomGenerator
 * that would produce unique values across two calls to make_random_wrapped_column()
 * like this.  However that code has been changed and each call to make_random_wrapped_column()
 * now uses a local UniformRandomGenerator object.  If we didn't generate an incrementing seed
 * for each one, every call to make_random_wrapped_column() would return the same values. This
 * fixes that case and also leaves results across multiple test runs deterministic.
 */
uint64_t random_generator_incrementing_seed();

}  // namespace detail

/**
 * @brief Provides uniform random number generation.
 *
 * It is often useful in testing to have a convenient source of random numbers.
 * This class is intended to serve as a base class for test fixtures to provide
 * random number generation. `UniformRandomGenerator::generate()` will generate
 * the next random number in the sequence.
 *
 * Example:
 * ```c++
 * UniformRandomGenerator g(0,100);
 * g.generate(); // Returns a random number in the range [0,100]
 * ```
 *
 * @tparam T The type of values that will be generated.
 */
template <typename T = cudf::size_type, typename Engine = std::default_random_engine>
class UniformRandomGenerator {
 public:
  using uniform_distribution = uniform_distribution_t<T>;  ///< The uniform distribution type for T.

  UniformRandomGenerator() : rng{std::mt19937_64{detail::random_generator_incrementing_seed()}()} {}

  /**
   * @brief Construct a new Uniform Random Generator to generate uniformly
   * random numbers in the range `[upper,lower]`
   *
   * @param lower Lower bound of the range
   * @param upper Upper bound of the desired range
   * @param seed  seed to initialize generator with
   */
  template <typename TL                                                          = T,
            std::enable_if_t<cudf::is_numeric<TL>() && !cudf::is_boolean<TL>()>* = nullptr>
  UniformRandomGenerator(T lower,
                         T upper,
                         uint64_t seed = detail::random_generator_incrementing_seed())
    : dist{lower, upper}, rng{std::mt19937_64{seed}()}
  {
  }

  /**
   * @brief Construct a new Uniform Random Generator to generate uniformly random booleans
   *
   * @param lower ignored
   * @param upper ignored
   * @param seed  seed to initialize generator with
   */
  template <typename TL = T, std::enable_if_t<cudf::is_boolean<TL>()>* = nullptr>
  UniformRandomGenerator(T lower,
                         T upper,
                         uint64_t seed = detail::random_generator_incrementing_seed())
    : dist{0.5}, rng{std::mt19937_64{seed}()}
  {
  }

  /**
   * @brief Construct a new Uniform Random Generator to generate uniformly
   * random numbers in the range `[upper,lower]`
   *
   * @param lower Lower bound of the range
   * @param upper Upper bound of the desired range
   * @param seed  seed to initialize generator with
   */
  template <typename TL                                                            = T,
            std::enable_if_t<cudf::is_chrono<TL>() or cudf::is_fixed_point<TL>()>* = nullptr>
  UniformRandomGenerator(typename TL::rep lower,
                         typename TL::rep upper,
                         uint64_t seed = detail::random_generator_incrementing_seed())
    : dist{lower, upper}, rng{std::mt19937_64{seed}()}
  {
  }

  /**
   * @brief Returns the next random number.
   *
   * @return generated random number
   */
  template <typename TL = T, std::enable_if_t<!cudf::is_timestamp<TL>()>* = nullptr>
  T generate()
  { return T{dist(rng)}; }

  /**
   * @brief Returns the next random number.
   * @return generated random number
   */
  template <typename TL = T, std::enable_if_t<cudf::is_timestamp<TL>()>* = nullptr>
  T generate()
  { return T{typename T::duration{dist(rng)}}; }

 private:
  uniform_distribution dist{};  ///< Distribution
  Engine rng;                   ///< Random generator
};

}  // namespace test
}  // namespace CUDF_EXPORT cudf
