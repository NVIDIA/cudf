/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cuda/std/limits>

#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#endif

/**
 * @file
 * @brief Concrete type definition for dictionary columns.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup dictionary_classes
 * @{
 */

/**
 * @brief A strongly typed wrapper for indices in a DICTIONARY type column.
 *
 * IndexType will be integer types like int32_t.
 *
 * For example, `dictionary32` is a strongly typed wrapper around an `int32_t`
 * value that holds the offset into the dictionary keys for a specific element.
 *
 * This wrapper provides common conversion and comparison operations for
 * the IndexType.
 */
template <typename IndexType>
struct dictionary_wrapper {
  using value_type = IndexType;  ///< The underlying type of the dictionary

  dictionary_wrapper()                          = default;
  ~dictionary_wrapper()                         = default;
  dictionary_wrapper(dictionary_wrapper&&)      = default;  ///< Move constructor
  dictionary_wrapper(dictionary_wrapper const&) = default;  ///< Copy constructor

  /**
   * @brief Move assignment operator
   *
   * @return The reference to this dictionary wrapper object
   */
  dictionary_wrapper& operator=(dictionary_wrapper&&) = default;

  /**
   * @brief Copy assignment operator
   *
   * @return The reference to this dictionary wrapper object
   */
  dictionary_wrapper& operator=(dictionary_wrapper const&) = default;

  /**
   * @brief Construct dictionary_wrapper from a value
   *
   * @param v The value to construct the dictionary_wrapper from
   */
  CUDF_HOST_DEVICE inline constexpr explicit dictionary_wrapper(value_type v) : _value{v} {}

  /**
   * @brief Conversion operator
   *
   * @return The value of this dictionary wrapper
   */
  CUDF_HOST_DEVICE inline explicit operator value_type() const { return _value; }

  /**
   * @brief Simple accessor
   *
   * @return The value of this dictionary wrapper
   */
  CUDF_HOST_DEVICE [[nodiscard]] inline value_type value() const { return _value; }

  /**
   * @brief Returns the maximum value of the value type.
   *
   * @return The maximum value of the value type
   */
  static CUDF_HOST_DEVICE inline constexpr value_type max_value()
  {
    return cuda::std::numeric_limits<value_type>::max();
  }

  /**
   * @brief Returns the minimum value of the value type.
   *
   * @return The minimum value of the value type
   */
  static CUDF_HOST_DEVICE inline constexpr value_type min_value()
  {
    return cuda::std::numeric_limits<value_type>::min();
  }

  /**
   * @brief Returns the lowest value of the value type.
   *
   * @return The lowest value of the value type
   */
  static CUDF_HOST_DEVICE inline constexpr value_type lowest_value()
  {
    return cuda::std::numeric_limits<value_type>::lowest();
  }

 private:
  value_type _value;
};

// comparison operators
/**
 * @brief Wqual to operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is equal to rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline bool operator==(dictionary_wrapper<Integer> const& lhs,
                                        dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() == rhs.value();
}

/**
 * @brief Not equal to operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is not equal to rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline bool operator!=(dictionary_wrapper<Integer> const& lhs,
                                        dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() != rhs.value();
}

/**
 * @brief Less than or equal to operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is less than or equal to rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline bool operator<=(dictionary_wrapper<Integer> const& lhs,
                                        dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() <= rhs.value();
}

/**
 * @brief Greater than or equal to operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is greater than or equal to rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline bool operator>=(dictionary_wrapper<Integer> const& lhs,
                                        dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() >= rhs.value();
}

/**
 * @brief Less than operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is less than rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline constexpr bool operator<(dictionary_wrapper<Integer> const& lhs,
                                                 dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() < rhs.value();
}

/**
 * @brief Greater than operator for dictionary_wrapper
 *
 * @tparam Integer Index type
 * @param lhs Left hand side of comparison
 * @param rhs Right hand side of comparison
 * @return Returns true if lhs is greater than rhs, false otherwise
 */
template <typename Integer>
CUDF_HOST_DEVICE inline bool operator>(dictionary_wrapper<Integer> const& lhs,
                                       dictionary_wrapper<Integer> const& rhs)
{
  return lhs.value() > rhs.value();
}

using dictionary32 = dictionary_wrapper<int32_t>;  ///< 32-bit integer indexed dictionary wrapper

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
