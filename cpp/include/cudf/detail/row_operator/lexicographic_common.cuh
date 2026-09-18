/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/utilities/export.hpp>

#include <cuda/std/type_traits>

#include <cmath>

namespace CUDF_EXPORT cudf {
namespace detail::row::lexicographic {

/**
 * @brief Computes a weak ordering of two values with special sorting behavior.
 *
 * This relational comparator functor compares physical values rather than logical
 * elements like lists, strings, or structs. It evaluates `NaN` as not less than, equal to, or
 * greater than other values and is IEEE-754 compliant.
 */
struct physical_element_comparator {
  /**
   * @brief Operator for relational comparisons.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return Relation between elements
   */
  template <typename Element>
  __device__ constexpr cudf::detail::weak_ordering operator()(Element const lhs,
                                                              Element const rhs) const noexcept
  {
    return cudf::detail::compare_elements(lhs, rhs);
  }
};

/**
 * @brief Relational comparator functor that compares physical values rather than logical
 * elements like lists, strings, or structs. It evaluates `NaN` as equivalent to other `NaN`s and
 * greater than all other values.
 */
struct sorting_physical_element_comparator {
  /**
   * @brief Operator for relational comparison of non-floating point values.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return Relation between elements
   */
  template <typename Element>
  __device__ constexpr cudf::detail::weak_ordering operator()(Element const lhs,
                                                              Element const rhs) const noexcept
    requires(not cuda::std::is_floating_point_v<Element>)
  {
    return cudf::detail::compare_elements(lhs, rhs);
  }

  /**
   * @brief Operator for relational comparison of floating point values.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return Relation between elements
   */
  template <typename Element>
  __device__ constexpr cudf::detail::weak_ordering operator()(Element const lhs,
                                                              Element const rhs) const noexcept
    requires(cuda::std::is_floating_point_v<Element>)
  {
    if (isnan(lhs)) {
      return isnan(rhs) ? cudf::detail::weak_ordering::EQUIVALENT
                        : cudf::detail::weak_ordering::GREATER;
    } else if (isnan(rhs)) {
      return cudf::detail::weak_ordering::LESS;
    }

    return cudf::detail::compare_elements(lhs, rhs);
  }
};

/**
 * @brief Wraps and interprets the result of templated Comparator that returns a
 * cudf::detail::weak_ordering. Returns true if the cudf::detail::weak_ordering matches any of the
 * templated values.
 *
 * Note that this should never be used with only `cudf::detail::weak_ordering::EQUIVALENT`.
 * An equality comparator should be used instead for optimal performance.
 *
 * @tparam Comparator generic comparator that returns a cudf::detail::weak_ordering.
 * @tparam values cudf::detail::weak_ordering parameter pack of orderings to interpret as true
 */
template <typename Comparator, cudf::detail::weak_ordering... values>
struct weak_ordering_comparator_impl {
  static_assert(
    not((cudf::detail::weak_ordering::EQUIVALENT == values) && ...),
    "cudf::detail::weak_ordering_comparator should not be used for pure equality comparisons. The "
    "`row_equality_comparator` should be used instead");

  template <typename LhsType, typename RhsType>
  __device__ constexpr bool operator()(LhsType const lhs_index,
                                       RhsType const rhs_index) const noexcept
  {
    cudf::detail::weak_ordering const result = comparator(lhs_index, rhs_index);
    return ((result == values) || ...);
  }
  Comparator comparator;
};

/**
 * @brief Wraps and interprets the result of device_row_comparator, true if the result is
 * cudf::detail::weak_ordering::LESS meaning one row is lexicographically *less* than another row.
 *
 * @tparam Comparator generic comparator that returns a cudf::detail::weak_ordering
 */
template <typename Comparator>
struct less_comparator
  : weak_ordering_comparator_impl<Comparator, cudf::detail::weak_ordering::LESS> {
  /**
   * @brief Constructs a less_comparator
   *
   * @param comparator The comparator to wrap
   */
  less_comparator(Comparator const& comparator)
    : weak_ordering_comparator_impl<Comparator, cudf::detail::weak_ordering::LESS>{comparator}
  {
  }
};

/**
 * @brief Wraps and interprets the result of device_row_comparator, true if the result is
 * cudf::detail::weak_ordering::LESS or cudf::detail::weak_ordering::EQUIVALENT meaning one row is
 * lexicographically *less* than or *equivalent* to another row.
 *
 * @tparam Comparator generic comparator that returns a cudf::detail::weak_ordering
 */
template <typename Comparator>
struct less_equivalent_comparator
  : weak_ordering_comparator_impl<Comparator,
                                  cudf::detail::weak_ordering::LESS,
                                  cudf::detail::weak_ordering::EQUIVALENT> {
  /**
   * @brief Constructs a less_equivalent_comparator
   *
   * @param comparator The comparator to wrap
   */
  less_equivalent_comparator(Comparator const& comparator)
    : weak_ordering_comparator_impl<Comparator,
                                    cudf::detail::weak_ordering::LESS,
                                    cudf::detail::weak_ordering::EQUIVALENT>{comparator}
  {
  }
};

}  // namespace detail::row::lexicographic
}  // namespace CUDF_EXPORT cudf
