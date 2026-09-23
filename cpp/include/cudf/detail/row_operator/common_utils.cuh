/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/type_traits>
#include <thrust/detail/use_default.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_facade.h>

namespace cudf::detail {

/**
 * @brief A map from cudf::type_id to cudf type that excludes LIST and STRUCT types.
 *
 * To be used with type_dispatcher in place of the default map, when it is required that STRUCT and
 * LIST map to void. This is useful when we want to avoid recursion in a functor. For example, in
 * element_comparator, we have a specialization for STRUCT but the type_dispatcher in it is only
 * used to dispatch to the same functor for non-nested types. Even when we're guaranteed to not have
 * non-nested types at that point, the compiler doesn't know this and would try to create recursive
 * code which is very slow.
 *
 * Usage:
 * @code
 * type_dispatcher<dispatch_nested_to_void>(data_type(), functor{});
 * @endcode
 */
template <cudf::type_id t>
struct dispatch_void_if_nested {
  using type =
    cuda::std::conditional_t<t == type_id::STRUCT or t == type_id::LIST, void, id_to_type<t>>;
};

namespace row {

enum class lhs_index_type : size_type {};
enum class rhs_index_type : size_type {};

/**
 * @brief A counting iterator that uses strongly typed indices bound to tables.
 *
 * Performing lexicographic or equality comparisons between values in two
 * tables requires the use of strongly typed indices. The strong index types
 * `lhs_index_type` and `rhs_index_type` ensure that index values are bound to
 * the correct table, regardless of the order in which these indices are
 * provided to the call operator. This struct and its type aliases
 * `lhs_iterator` and `rhs_iterator` provide an interface similar to a counting
 * iterator, with strongly typed values to represent the table indices.
 *
 * @tparam Index The strong index type
 */
template <typename Index, typename Underlying = cuda::std::underlying_type_t<Index>>
struct strong_index_iterator : public thrust::iterator_facade<strong_index_iterator<Index>,
                                                              Index,
                                                              thrust::use_default,
                                                              thrust::random_access_traversal_tag,
                                                              Index,
                                                              Underlying> {
  using super_t =
    thrust::iterator_adaptor<strong_index_iterator<Index>, Index>;  ///< The base class

  /**
   * @brief Constructs a strong index iterator
   *
   * @param n The beginning index
   */
  explicit constexpr strong_index_iterator(Underlying n) : begin{n} {}

  friend class thrust::iterator_core_access;  ///< Allow access to the base class

 private:
  __device__ constexpr void increment() { ++begin; }
  __device__ constexpr void decrement() { --begin; }

  __device__ constexpr void advance(Underlying n) { begin += n; }

  __device__ constexpr bool equal(strong_index_iterator<Index> const& other) const noexcept
  {
    return begin == other.begin;
  }

  __device__ constexpr Index dereference() const noexcept { return static_cast<Index>(begin); }

  __device__ constexpr Underlying distance_to(
    strong_index_iterator<Index> const& other) const noexcept
  {
    return other.begin - begin;
  }

  Underlying begin{};
};

/**
 * @brief Iterator representing indices into a left-side table.
 */
using lhs_iterator = strong_index_iterator<lhs_index_type>;

/**
 * @brief Iterator representing indices into a right-side table.
 */
using rhs_iterator = strong_index_iterator<rhs_index_type>;

}  // namespace row

/**
 * @brief Result type of comparison operations.
 *
 * Indicates how two elements `a` and `b` compare with one and another.
 *
 * Equivalence is defined as `not (a<b) and not (b<a)`. Elements that are
 * EQUIVALENT may not necessarily be *equal*.
 */
enum class weak_ordering {
  LESS,        ///< Indicates `a` is less than (ordered before) `b`
  EQUIVALENT,  ///< Indicates `a` is ordered neither before nor after `b`
  GREATER      ///< Indicates `a` is greater than (ordered after) `b`
};

/**
 * @brief Compare the elements ordering with respect to `lhs`.
 *
 * @param lhs first element
 * @param rhs second element
 * @return Indicates the relationship between the elements in
 * the `lhs` and `rhs` columns.
 */
template <typename Element>
__device__ weak_ordering compare_elements(Element lhs, Element rhs)
{
  if (lhs < rhs) {
    return weak_ordering::LESS;
  } else if (rhs < lhs) {
    return weak_ordering::GREATER;
  }
  return weak_ordering::EQUIVALENT;
}

/**
 * @brief A specialization for `string_view` to compare the elements ordering with respect to `lhs`.
 *
 * The generic implementation resolves the ordering using two `<` comparisons and each one is a
 * full character-by-character comparison for strings. Here `string_view::compare` already
 * produces the 3-way result so only a single comparison pass is needed.
 *
 * @param lhs first element
 * @param rhs second element
 * @return Indicates the relationship between the elements in
 * the `lhs` and `rhs` columns.
 */
__device__ inline weak_ordering compare_elements(string_view lhs, string_view rhs)
{
  auto const result = lhs.compare(rhs);
  if (result < 0) { return weak_ordering::LESS; }
  if (result > 0) { return weak_ordering::GREATER; }
  return weak_ordering::EQUIVALENT;
}

/**
 * @brief A specialization for floating-point `Element` type relational comparison
 * to derive the order of the elements with respect to `lhs`.
 *
 * This specialization handles `nan` in the following order:
 * `[-Inf, -ve, 0, -0, +ve, +Inf, NaN, NaN, null] (for null_order::AFTER)`
 * `[null, -Inf, -ve, 0, -0, +ve, +Inf, NaN, NaN] (for null_order::BEFORE)`
 *
 * @param lhs The first element
 * @param rhs The second element
 * @return Indicates the relationship between the elements
 */
template <typename Element>
__device__ weak_ordering relational_compare(Element lhs, Element rhs)
  requires(cuda::std::is_floating_point_v<Element>)
{
  if (isnan(lhs) and isnan(rhs)) {
    return weak_ordering::EQUIVALENT;
  } else if (isnan(rhs)) {
    return weak_ordering::LESS;
  } else if (isnan(lhs)) {
    return weak_ordering::GREATER;
  }

  return detail::compare_elements(lhs, rhs);
}

/**
 * @brief Compare the nulls according to null order.
 *
 * @param lhs_is_null boolean representing if lhs is null
 * @param rhs_is_null boolean representing if rhs is null
 * @param null_precedence null order
 * @return Indicates the relationship between null in lhs and rhs columns.
 */
inline __device__ auto null_compare(bool lhs_is_null, bool rhs_is_null, null_order null_precedence)
{
  if (lhs_is_null and rhs_is_null) {  // null <? null
    return weak_ordering::EQUIVALENT;
  } else if (lhs_is_null) {  // null <? x
    return (null_precedence == null_order::BEFORE) ? weak_ordering::LESS : weak_ordering::GREATER;
  } else if (rhs_is_null) {  // x <? null
    return (null_precedence == null_order::AFTER) ? weak_ordering::LESS : weak_ordering::GREATER;
  }
  return weak_ordering::EQUIVALENT;
}

/**
 * @brief A specialization for non-floating-point `Element` type relational
 * comparison to derive the order of the elements with respect to `lhs`.
 *
 * @param lhs The first element
 * @param rhs The second element
 * @return Indicates the relationship between the elements in the `lhs` and `rhs` columns
 */
template <typename Element>
__device__ weak_ordering relational_compare(Element lhs, Element rhs)
  requires(not cuda::std::is_floating_point_v<Element>)
{
  return detail::compare_elements(lhs, rhs);
}

/**
 * @brief A specialization for floating-point `Element` type to check if
 * `lhs` is equivalent to `rhs`. `nan == nan`.
 *
 * @param lhs first element
 * @param rhs second element
 * @return `true` if `lhs` == `rhs` else `false`.
 */
template <typename Element>
__device__ bool equality_compare(Element lhs, Element rhs)
  requires(cuda::std::is_floating_point_v<Element>)
{
  if (isnan(lhs) and isnan(rhs)) { return true; }
  return lhs == rhs;
}

/**
 * @brief A specialization for non-floating-point `Element` type to check if
 * `lhs` is equivalent to `rhs`.
 *
 * @param lhs first element
 * @param rhs second element
 * @return `true` if `lhs` == `rhs` else `false`.
 */
template <typename Element>
__device__ bool equality_compare(Element const lhs, Element const rhs)
  requires(not cuda::std::is_floating_point_v<Element>)
{
  return lhs == rhs;
}

}  // namespace cudf::detail
