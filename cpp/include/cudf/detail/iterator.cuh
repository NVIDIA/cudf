/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief provides column input iterator with nulls replaced with a specified value
 * @file iterator.cuh
 *
 * The column input iterator is designed to be used as an input
 * iterator for thrust and cub.
 *
 * Usage:
 * auto iter = make_null_replacement_iterator(column, null_value);
 *
 * The column input iterator returns only a scalar value of data at [id] or
 * the null_replacement value passed while creating the iterator.
 * For non-null column, use
 * auto iter = column.begin<Element>();
 *
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>

#include <cuda/iterator>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace detail {
/**
 * @brief Convenience wrapper for creating a `thrust::transform_iterator` over a
 * `cuda::counting_iterator` within the range [0, INT_MAX].
 *
 *
 * Example:
 * @code{.cpp}
 * // Returns square of the value of the counting iterator
 * auto iter = make_counting_transform_iterator(0, [](auto i){ return (i * i);});
 * iter[0] == 0
 * iter[1] == 1
 * iter[2] == 4
 * ...
 * iter[n] == n * n
 * @endcode
 *
 * @param start The starting value of the counting iterator (must be size_type or smaller type).
 * @param f The unary function to apply to the counting iterator.
 * @return A transform iterator that applies `f` to a counting iterator
 */
template <typename CountingIterType, typename UnaryFunction>
CUDF_HOST_DEVICE inline auto make_counting_transform_iterator(CountingIterType start,
                                                              UnaryFunction f)
{
  // Check if the `start` for counting_iterator is of size_type or a smaller integral type
  static_assert(
    cuda::std::is_integral_v<CountingIterType> and
      cuda::std::numeric_limits<CountingIterType>::digits <=
        cuda::std::numeric_limits<cudf::size_type>::digits,
    "The `start` for the counting_transform_iterator must be size_type or smaller type");

  return thrust::make_transform_iterator(cuda::counting_iterator{start}, f);
}

/**
 * @brief Value accessor of column that may have a null bitmask.
 *
 * This unary functor returns scalar value at `id`.
 * The `operator()(cudf::size_type id)` computes the `element` and valid flag at `id`.
 *
 * The return value for element `i` will return `column[i]`
 * if it is valid, or `null_replacement` if it is null.
 *
 * @tparam Element The type of elements in the column
 */
template <typename Element>
struct null_replaced_value_accessor {
  column_device_view const col;      ///< column view of column in device
  Element const null_replacement{};  ///< value returned when element is null
  bool const has_nulls;              ///< true if col has null elements

  /**
   * @brief Creates an accessor for a null-replacement iterator.
   *
   * @throws cudf::logic_error if `col` type does not match Element type.
   * @throws cudf::logic_error if `has_nulls` is true but `col` does not have a validity mask.
   *
   * @param[in] col column device view of cudf column
   * @param[in] null_val The value to return for null elements
   * @param[in] has_nulls Must be set to true if `col` has nulls.
   */
  null_replaced_value_accessor(column_device_view const& col,
                               Element null_val,
                               bool has_nulls = true)
    : col{col}, null_replacement{null_val}, has_nulls{has_nulls}
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(col.type().id()),
                 "the data type mismatch");
    if (has_nulls) CUDF_EXPECTS(col.nullable(), "column with nulls must have a validity bitmask");
  }

  __device__ inline Element const operator()(cudf::size_type i) const
  {
    return has_nulls && col.is_null_nocheck(i) ? null_replacement : col.element<Element>(i);
  }
};

/**
 * @brief validity accessor of column with null bitmask
 * A unary functor that returns validity at index `i`.
 *
 * @tparam safe If false, the accessor will throw a logic_error if the column is not nullable. If
 * true, the accessor checks for nullability and if col is not nullable, returns true.
 */
template <bool safe = false>
struct validity_accessor {
  column_device_view const col;

  /**
   * @brief constructor
   *
   * @throws cudf::logic_error if not safe and `col` does not have a validity bitmask
   *
   * @param[in] _col column device view of cudf column
   */
  CUDF_HOST_DEVICE validity_accessor(column_device_view const& _col) : col{_col}
  {
    if constexpr (not safe) {
      // verify col is nullable, otherwise, is_valid_nocheck() will crash
#if defined(__CUDA_ARCH__)
      cudf_assert(_col.nullable() && "Unexpected non-nullable column.");
#else
      CUDF_EXPECTS(_col.nullable(), "Unexpected non-nullable column.");
#endif
    }
  }

  __device__ inline bool operator()(cudf::size_type i) const
  {
    if constexpr (safe) {
      return col.is_valid(i);
    } else {
      return col.is_valid_nocheck(i);
    }
  }
};

/**
 * @brief Constructs an iterator over a column's values that replaces null
 * elements with a specified value.
 *
 * Dereferencing the returned iterator for element `i` will return `column[i]`
 * if it is valid, or `null_replacement` if it is null.
 * This iterator is only allowed for both nullable and non-nullable columns.
 *
 * @throws cudf::logic_error if the column is not nullable.
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @param column The column to iterate
 * @param null_replacement The value to return for null elements
 * @param has_nulls Must be set to true if `column` has nulls.
 * @return Iterator that returns valid column elements, or a null
 * replacement value for null elements.
 */
template <typename Element>
auto make_null_replacement_iterator(column_device_view const& column,
                                    Element const null_replacement = Element{0},
                                    bool has_nulls                 = true)
{
  return make_counting_transform_iterator(
    0, null_replaced_value_accessor<Element>{column, null_replacement, has_nulls});
}

/**
 * @brief Constructs an optional iterator over a column's values and its validity.
 *
 * Dereferencing the returned iterator returns a `cuda::std::optional<Element>`.
 *
 * The element of this iterator contextually converts to bool. The conversion returns true
 * if the object contains a value and false if it does not contain a value.
 *
 * Calling this function with `nullate::DYNAMIC` defers the assumption
 * of nullability to runtime with the caller indicating if the column has nulls.
 * This is useful when an algorithm is going to execute on multiple iterators and all
 * the combinations of iterator types are not required at compile time.
 *
 * @code{.cpp}
 * template<typename T>
 * void some_function(cudf::column_view<T> const& col_view){
 *    auto d_col = cudf::column_device_view::create(col_view);
 *    // Create a `DYNAMIC` optional iterator
 *    auto optional_iterator =
 *      cudf::detail::make_optional_iterator<T>(
 *        d_col, cudf::nullate::DYNAMIC{col_view.has_nulls()});
 * }
 * @endcode
 *
 * Calling this function with `nullate::YES` means that the column supports
 * nulls and the optional returned might not contain a value.
 * Calling this function with `nullate::NO` means that the column has no
 * null values and the optional returned will always contain a value.
 *
 * @code{.cpp}
 * template<typename T, bool has_nulls>
 * void some_function(cudf::column_view<T> const& col_view){
 *    auto d_col = cudf::column_device_view::create(col_view);
 *    if constexpr(has_nulls) {
 *      auto optional_iterator =
 *        cudf::detail::make_optional_iterator<T>(d_col, cudf::nullate::YES{});
 *      //use optional_iterator
 *    } else {
 *      auto optional_iterator =
 *        cudf::detail::make_optional_iterator<T>(d_col, cudf::nullate::NO{});
 *      //use optional_iterator
 *    }
 * }
 * @endcode
 *
 * @throws cudf::logic_error if the column is not nullable and `has_nulls` is true.
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column.
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 *
 * @param column The column to iterate
 * @param has_nulls Indicates whether `column` is checked for nulls.
 * @return Iterator that returns valid column elements and the validity of the
 * element in a `cuda::std::optional`
 */
template <typename Element, typename Nullate>
auto make_optional_iterator(column_device_view const& column, Nullate has_nulls)
{
  return column.optional_begin<Element, Nullate>(has_nulls);
}

/**
 * @brief Constructs a pair iterator over a column's values and its validity.
 *
 * Dereferencing the returned iterator returns a `cuda::std::pair<Element, bool>`.
 *
 * If an element at position `i` is valid (or `has_nulls == false`), then for `p = *(iter + i)`,
 * `p.first` contains the value of the element at `i` and `p.second == true`.
 *
 * Else, if the element at `i` is null, then the value of `p.first` is undefined and `p.second ==
 * false`. `pair(column[i], validity)`. `validity` is `true` if `has_nulls=false`. `validity` is
 * validity of the element at `i` if `has_nulls=true` and the column is nullable.
 *
 * @throws cudf::logic_error if the column is nullable.
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @tparam has_nulls boolean indicating to treat the column is nullable
 * @param column The column to iterate
 * @return auto Iterator that returns valid column elements, and validity of the
 * element in a pair
 */
template <typename Element, bool has_nulls = false>
auto make_pair_iterator(column_device_view const& column)
{
  return column.pair_begin<Element, has_nulls>();
}

/**
 * @brief Constructs a pair rep iterator over a column's representative values and its validity.
 *
 * Dereferencing the returned iterator returns a `cuda::std::pair<rep_type, bool>`,
 * where `rep_type` is `device_storage_type<T>`, the type used to store
 * the value on the device.
 *
 * If an element at position `i` is valid (or `has_nulls == false`), then for `p = *(iter + i)`,
 * `p.first` contains the value of the element at `i` and `p.second == true`.
 *
 * Else, if the element at `i` is null, then the value of `p.first` is undefined and `p.second ==
 * false`. `pair(column[i], validity)`. `validity` is `true` if `has_nulls=false`. `validity` is
 * validity of the element at `i` if `has_nulls=true` and the column is nullable.
 *
 * @throws cudf::logic_error if the column is nullable.
 * @throws cudf::logic_error if column datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the column
 * @tparam has_nulls boolean indicating to treat the column is nullable
 * @param column The column to iterate
 * @return auto Iterator that returns valid column elements, and validity of the
 * element in a pair
 */
template <typename Element, bool has_nulls = false>
auto make_pair_rep_iterator(column_device_view const& column)
{
  return column.pair_rep_begin<Element, has_nulls>();
}

/**
 * @brief Constructs an iterator over a column's validities.
 *
 * Dereferencing the returned iterator for element `i` will return the validity
 * of `column[i]`
 * If `safe` = false, the column must be nullable.
 * When safe = true, if the column is not nullable then the validity is always true.
 *
 * @throws cudf::logic_error if the column is not nullable and safe = false
 *
 * @tparam safe If false, the accessor will throw a logic_error if the column is not nullable. If
 * true, the accessor checks for nullability and if col is not nullable, returns true.
 * @param column The column to iterate
 * @return auto Iterator that returns validities of column elements.
 */
template <bool safe = false>
CUDF_HOST_DEVICE auto inline make_validity_iterator(column_device_view const& column)
{
  return make_counting_transform_iterator(cudf::size_type{0}, validity_accessor<safe>{column});
}

/**
 * @brief Constructs a constant device iterator over a scalar's validity.
 *
 * Dereferencing the returned iterator returns a `bool`.
 *
 * For `p = *(iter + i)`, `p` is the validity of the scalar.
 *
 * @tparam safe unused. This template parameter exists to enforce the same
 *         template interface as @ref make_validity_iterator(column_device_view const&).
 * @param scalar_value The scalar to iterate
 * @return auto Iterator that returns scalar validity
 */
template <bool safe = false>
auto inline make_validity_iterator(scalar const& scalar_value)
{
  return cuda::make_constant_iterator(scalar_value.is_valid());
}

/**
 * @brief value accessor for scalar with valid data.
 * The unary functor returns data of Element type of the scalar.
 *
 * @throws `cudf::logic_error` if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of return type of functor
 */
template <typename Element>
struct scalar_value_accessor {
  using ScalarType       = scalar_type_t<Element>;
  using ScalarDeviceType = scalar_device_type_t<Element>;
  ScalarDeviceType const dscalar;  ///< scalar device view

  scalar_value_accessor(scalar const& scalar_value)
    : dscalar(get_scalar_device_view(static_cast<ScalarType&>(const_cast<scalar&>(scalar_value))))
  {
    CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
                 "the data type mismatch");
  }

  __device__ inline Element const operator()(size_type) const { return dscalar.value(); }
};

/**
 * @brief Constructs a constant device iterator over a scalar's value.
 *
 * Dereferencing the returned iterator returns a `Element`.
 *
 * For `p = *(iter + i)`, `p` is the value stored in the scalar.
 *
 * The behavior is undefined if the scalar is destroyed before iterator dereferencing.
 *
 * @throws cudf::logic_error if scalar datatype and Element type mismatch.
 * @throws cudf::logic_error if scalar is null.
 * @throws cudf::logic_error if the returned iterator is dereferenced in host
 *
 * @tparam Element The type of element in the scalar
 * @param scalar_value The scalar to iterate
 * @return auto Iterator that returns scalar value
 */
template <typename Element>
auto inline make_scalar_iterator(scalar const& scalar_value)
{
  CUDF_EXPECTS(data_type(type_to_id<Element>()) == scalar_value.type(), "the data type mismatch");
  CUDF_EXPECTS(scalar_value.is_valid(), "the scalar value must be valid");
  return thrust::make_transform_iterator(cuda::make_constant_iterator<size_type>(0),
                                         scalar_value_accessor<Element>{scalar_value});
}

/**
 * @brief Optional accessor for a scalar
 *
 * The `scalar_optional_accessor` always returns a `cuda::std::optional` of the scalar.
 * The validity of the optional is determined by the `Nullate` parameter which may
 * be one of the following:
 *
 * - `nullate::YES` means that the scalar may be valid or invalid and the optional returned
 *    will contain a value only if the scalar is valid.
 *
 * - `nullate::NO` means the caller attests that the scalar will always be valid,
 *    no checks will occur and `cuda::std::optional{column[i]}` will return a value
 *    for each `i`.
 *
 * - `nullate::DYNAMIC` defers the assumption of nullability to runtime and the caller
 *    specifies if the scalar may be valid or invalid.
 *    For `DYNAMIC{true}` the return value will be a `cuda::std::optional{scalar}` when the
 *      scalar is valid and a `cuda::std::optional{}` when the scalar is invalid.
 *    For `DYNAMIC{false}` the return value will always be a `cuda::std::optional{scalar}`.
 *
 * @throws `cudf::logic_error` if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of return type of functor
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename Element, typename Nullate>
struct scalar_optional_accessor : public scalar_value_accessor<Element> {
  using super_t    = scalar_value_accessor<Element>;
  using value_type = cuda::std::optional<Element>;

  scalar_optional_accessor(scalar const& scalar_value, Nullate with_nulls)
    : scalar_value_accessor<Element>(scalar_value), has_nulls{with_nulls}
  {
  }

  __device__ inline value_type const operator()(size_type) const
  {
    if (has_nulls && !super_t::dscalar.is_valid()) { return value_type{cuda::std::nullopt}; }

    if constexpr (cudf::is_fixed_point<Element>()) {
      using namespace numeric;
      using rep        = typename Element::rep;
      auto const value = super_t::dscalar.rep();
      auto const scale = scale_type{super_t::dscalar.type().scale()};
      return Element{scaled_integer<rep>{value, scale}};
    } else {
      return Element{super_t::dscalar.value()};
    }
  }

  Nullate has_nulls{};
};

/**
 * @brief pair accessor for scalar.
 * The unary functor returns a pair of data of Element type and bool validity of the scalar.
 *
 * @throws `cudf::logic_error` if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of return type of functor
 */
template <typename Element>
struct scalar_pair_accessor : public scalar_value_accessor<Element> {
  using super_t    = scalar_value_accessor<Element>;
  using value_type = cuda::std::pair<Element, bool>;
  scalar_pair_accessor(scalar const& scalar_value) : scalar_value_accessor<Element>(scalar_value) {}

  __device__ inline value_type const operator()(size_type) const
  {
    return {Element(super_t::dscalar.value()), super_t::dscalar.is_valid()};
  }
};

/**
 * @brief Utility to discard template type arguments.
 *
 * Substitute for std::void_t.
 *
 * @tparam T Ignored template parameter
 */
template <typename... T>
using void_t = void;

/**
 * @brief Compile-time reflection to check if `Element` type has a `rep()` member.
 */
template <typename Element, typename = void>
struct has_rep_member : std::false_type {};

template <typename Element>
struct has_rep_member<Element, void_t<decltype(std::declval<Element>().rep())>> : std::true_type {};

/**
 * @brief Pair accessor for scalar's representation value and validity.
 *
 * @tparam Element The type of element in the scalar.
 */
template <typename Element>
struct scalar_representation_pair_accessor : public scalar_value_accessor<Element> {
  using base       = scalar_value_accessor<Element>;
  using rep_type   = device_storage_type_t<Element>;
  using value_type = cuda::std::pair<rep_type, bool>;

  scalar_representation_pair_accessor(scalar const& scalar_value) : base(scalar_value) {}

  __device__ inline value_type const operator()(size_type) const
  {
    return {get_rep(base::dscalar), base::dscalar.is_valid()};
  }

 private:
  template <typename DeviceScalar>
  __device__ inline rep_type get_rep(DeviceScalar const& dscalar) const
    requires(!has_rep_member<DeviceScalar>::value)
  {
    return dscalar.value();
  }

  template <typename DeviceScalar>
  __device__ inline rep_type get_rep(DeviceScalar const& dscalar) const
    requires(has_rep_member<DeviceScalar>::value)
  {
    return dscalar.rep();
  }
};

/**
 * @brief Constructs an optional iterator over a scalar's values and its validity.
 *
 * Dereferencing the returned iterator returns a `cuda::std::optional<Element>`.
 *
 * The element of this iterator contextually converts to bool. The conversion returns true
 * if the object contains a value and false if it does not contain a value.
 *
 * The iterator behavior is undefined if the scalar is destroyed before iterator dereferencing.
 *
 * Calling this function with `nullate::DYNAMIC` defers the assumption
 * of nullability to runtime with the caller indicating if the scalar is valid.
 *
 * @code{.cpp}
 * template<typename T>
 * void some_function(cudf::column_view<T> const& col_view,
 *                    scalar const& scalar_value,
 *                    bool col_has_nulls){
 *    auto d_col = cudf::column_device_view::create(col_view);
 *    auto column_iterator = cudf::detail::make_optional_iterator<T>(
 *      d_col, cudf::nullate::DYNAMIC{col_has_nulls});
 *    auto scalar_iterator = cudf::detail::make_optional_iterator<T>(
 *      scalar_value, cudf::nullate::DYNAMIC{scalar_value.is_valid()});
 *    //use iterators
 * }
 * @endcode
 *
 * Calling this function with `nullate::YES` means that the scalar maybe invalid
 * and the optional return might not contain a value.
 * Calling this function with `nullate::NO` means that the scalar is valid
 * and the optional returned will always contain a value.
 *
 * @code{.cpp}
 * template<typename T, bool any_nulls>
 * void some_function(cudf::column_view<T> const& col_view, scalar const& scalar_value){
 *    auto d_col = cudf::column_device_view::create(col_view);
 *    if constexpr(any_nulls) {
 *      auto column_iterator =
 *        cudf::detail::make_optional_iterator<T>(d_col, cudf::nullate::YES{});
 *      auto scalar_iterator =
 *        cudf::detail::make_optional_iterator<T>(scalar_value, cudf::nullate::YES{});
 *      //use iterators
 *    } else {
 *      auto column_iterator =
 *        cudf::detail::make_optional_iterator<T>(d_col, cudf::nullate::NO{});
 *      auto scalar_iterator =
 *        cudf::detail::make_optional_iterator<T>(scalar_value, cudf::nullate::NO{});
 *      //use iterators
 *    }
 * }
 * @endcode
 *
 * @throws cudf::logic_error if scalar datatype and Element type mismatch.
 *
 * @tparam Element The type of elements in the scalar
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 *
 * @param scalar_value The scalar to be returned by the iterator.
 * @param has_nulls Indicates if the scalar value may be invalid.
 * @return Iterator that returns scalar and the validity of the scalar in a cuda::std::optional
 */
template <typename Element, typename Nullate>
auto inline make_optional_iterator(scalar const& scalar_value, Nullate has_nulls)
{
  CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
               "the data type mismatch");
  return thrust::make_transform_iterator(
    cuda::make_constant_iterator<size_type>(0),
    scalar_optional_accessor<Element, Nullate>{scalar_value, has_nulls});
}

/**
 * @brief Constructs a constant device pair iterator over a scalar's value and its validity.
 *
 * Dereferencing the returned iterator returns a `cuda::std::pair<Element, bool>`.
 *
 * If scalar is valid, then for `p = *(iter + i)`, `p.first` contains
 * the value of the scalar and `p.second == true`.
 *
 * Else, if the scalar is null, then the value of `p.first` is undefined and `p.second == false`.
 *
 * The behavior is undefined if the scalar is destroyed before iterator dereferencing.
 *
 * @throws cudf::logic_error if scalar datatype and Element type mismatch.
 * @throws cudf::logic_error if the returned iterator is dereferenced in host
 *
 * @tparam Element The type of elements in the scalar
 * @tparam bool unused. This template parameter exists to enforce same
 * template interface as @ref make_pair_iterator(column_device_view const&).
 * @param scalar_value The scalar to iterate
 * @return auto Iterator that returns scalar, and validity of the scalar in a pair
 */
template <typename Element, bool = false>
auto inline make_pair_iterator(scalar const& scalar_value)
{
  CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
               "the data type mismatch");
  return thrust::make_transform_iterator(cuda::make_constant_iterator<size_type>(0),
                                         scalar_pair_accessor<Element>{scalar_value});
}

/**
 * @brief Constructs a constant device pair iterator over a scalar's representative value
 *        and its validity.
 *
 * Dereferencing the returned iterator returns a `cuda::std::pair<Element::rep, bool>`.
 * E.g. For a valid `decimal32` row, a `cuda::std::pair<int32_t, bool>` is returned,
 * with the value set to the `int32_t` representative value of the decimal,
 * and validity `true`, indicating that the row is valid.
 *
 * If scalar is valid, then for `p = *(iter + i)`, `p.first` contains
 * the representative value of the scalar and `p.second == true`.
 *
 * Else, if the scalar is null, then the value of `p.first` is undefined and `p.second == false`.
 *
 * The behavior is undefined if the scalar is destroyed before iterator dereferencing.
 *
 * @throws cudf::logic_error if scalar datatype and Element type mismatch.
 * @throws cudf::logic_error if the returned iterator is dereferenced in host
 *
 * @tparam Element The type of elements in the scalar
 * @tparam bool unused. This template parameter exists to enforce same
 * template interface as @ref make_pair_iterator(column_device_view const&).
 * @param scalar_value The scalar to iterate
 * @return auto Iterator that returns scalar's representative value,
 *         and validity of the scalar in a pair
 */
template <typename Element, bool = false>
auto make_pair_rep_iterator(scalar const& scalar_value)
{
  CUDF_EXPECTS(type_id_matches_device_storage_type<Element>(scalar_value.type().id()),
               "the data type mismatch");
  return make_counting_transform_iterator(
    0, scalar_representation_pair_accessor<Element>{scalar_value});
}

}  // namespace detail
}  // namespace cudf
