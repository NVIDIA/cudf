/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/detail/dremel.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/sorting.hpp>
#include <cudf/structs/structs_column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <thrust/detail/use_default.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_facade.h>

#include <memory>
#include <type_traits>
#include <vector>

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

using optional_dremel_view = cuda::std::optional<detail::dremel_device_view const>;

// The has_nested_columns template parameter of the device_row_comparator is
// necessary to help the compiler optimize our code. Without it, the list and
// struct view specializations are present in the code paths used for primitive
// types, and the compiler fails to inline this nearly as well resulting in a
// significant performance drop.  As a result, there is some minor tension in
// the current design between the presence of this parameter and the way that
// the Dremel data is passed around, first as a
// std::optional<device_span<dremel_device_view>> in the
// preprocessed_table/device_row_comparator (which is always valid when
// has_nested_columns and is otherwise invalid) that is then unpacked to a
// cuda::std::optional<dremel_device_view> at the element_comparator level (which
// is always valid for a list column and otherwise invalid).  We cannot use an
// additional template parameter for the element_comparator on a per-column
// basis because we cannot conditionally define dremel_device_view member
// variables without jumping through extra hoops with inheritance, so the
// cuda::std::optional<dremel_device_view> member must be an optional rather than
// a raw dremel_device_view.
/**
 * @brief Computes the lexicographic comparison between 2 rows.
 *
 * Lexicographic ordering is determined by:
 * - Two rows are compared element by element.
 * - The first mismatching element defines which row is lexicographically less
 * or greater than the other.
 * - If the rows are compared without mismatched elements, the rows are equivalent
 *
 *
 * Lexicographic ordering is exactly equivalent to doing an alphabetical sort of
 * two words, for example, `aac` would be *less* than (or precede) `abb`. The
 * second letter in both words is the first non-equal letter, and `a < b`, thus
 * `aac < abb`.
 *
 * @note The operator overloads in sub-class `element_comparator` are templated via the
 *        `type_dispatcher` to help select an overload instance for each column in a table.
 *        So, `cudf::is_nested<Element>` will return `true` if the table has nested-type columns,
 *        but it will be a runtime error if template parameter `has_nested_columns != true`.
 *
 * @tparam has_nested_columns compile-time optimization for primitive types.
 *         This template parameter is to be used by the developer by querying
 *         `cudf::has_nested_columns(input)`. `true` compiles operator
 *         overloads for nested types, while `false` only compiles operator
 *         overloads for primitive types.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 * @tparam PhysicalElementComparator A relational comparator functor that compares individual values
 * rather than logical elements, defaults to `NaN` aware relational comparator that evaluates `NaN`
 * as greater than all other values.
 */
template <bool has_nested_columns,
          typename Nullate,
          typename PhysicalElementComparator = sorting_physical_element_comparator>
class device_row_comparator {
 public:
  friend class self_comparator;
  friend class two_table_comparator;

  /**
   * @brief Construct a function object for performing a lexicographic
   * comparison between the rows of two tables.
   *
   * @param check_nulls Indicates if any input column contains nulls.
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param l_dremel_device_views lhs table dremel device view for list type
   * @param r_dremel_device_views rhs table dremel device view for list type
   * @param depth Optional, device array the same length as a row that contains starting depths of
   * columns if they're nested, and 0 otherwise.
   * @param column_order Optional, device array the same length as a row that indicates the desired
   * ascending/descending order of each column in a row. If `nullopt`, it is assumed all columns are
   * sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   * values compare to all other for every column. If `nullopt`, then null precedence would be
   * `null_order::BEFORE` for all columns.
   * @param comparator Physical element relational comparison functor.
   */
  device_row_comparator(
    Nullate check_nulls,
    table_device_view lhs,
    table_device_view rhs,
    device_span<detail::dremel_device_view const> l_dremel_device_views,
    device_span<detail::dremel_device_view const> r_dremel_device_views,
    cuda::std::optional<device_span<int const>> depth                  = cuda::std::nullopt,
    cuda::std::optional<device_span<order const>> column_order         = cuda::std::nullopt,
    cuda::std::optional<device_span<null_order const>> null_precedence = cuda::std::nullopt,
    PhysicalElementComparator comparator                               = {}) noexcept
    : _lhs{lhs},
      _rhs{rhs},
      _l_dremel(l_dremel_device_views),
      _r_dremel(r_dremel_device_views),
      _check_nulls{check_nulls},
      _depth{depth},
      _column_order{column_order},
      _null_precedence{null_precedence},
      _comparator{comparator}
  {
  }

  /**
   * @brief Construct a function object for performing a lexicographic
   * comparison between the rows of two tables.
   * This is a special overload to allow device-side construction of the
   * comparator for cases where no preprocessing is needed, i.e. tables with
   * non-nested type columns.
   *
   * @param check_nulls Indicates if any input column contains nulls.
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param column_order Optional, device array the same length as a row that indicates the desired
   * ascending/descending order of each column in a row. If `nullopt`, it is assumed all columns are
   * sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   * values compare to all other for every column. If `nullopt`, then null precedence would be
   * `null_order::BEFORE` for all columns.
   * @param comparator Physical element relational comparison functor.
   */
  template <bool nested_disable = not has_nested_columns>
  __device__ device_row_comparator(
    Nullate check_nulls,
    table_device_view lhs,
    table_device_view rhs,
    cuda::std::optional<device_span<order const>> column_order         = cuda::std::nullopt,
    cuda::std::optional<device_span<null_order const>> null_precedence = cuda::std::nullopt,
    PhysicalElementComparator comparator                               = {}) noexcept
    requires(nested_disable)
    : _lhs{lhs},
      _rhs{rhs},
      _l_dremel{},
      _r_dremel{},
      _check_nulls{check_nulls},
      _depth{},
      _column_order{column_order},
      _null_precedence{null_precedence},
      _comparator{comparator}
  {
  }

  class dictionary_comparator {
   public:
    __device__ dictionary_comparator(column_device_view lhs,
                                     column_device_view rhs,
                                     PhysicalElementComparator comparator = {})
      : _lhs{lhs}, _rhs{rhs}, _comparator{comparator}
    {
    }

    template <typename KeyType>
    __device__ cuda::std::pair<cudf::detail::weak_ordering, int> operator()(
      size_type lhs_element_index, size_type rhs_element_index) const noexcept
      requires(cudf::is_relationally_comparable<KeyType, KeyType>())
    {
      auto const lidx = _lhs.element<cudf::dictionary32>(lhs_element_index).value();
      auto const ridx = _rhs.element<cudf::dictionary32>(rhs_element_index).value();
      auto const lhs =
        _lhs.child(cudf::dictionary_column_view::keys_column_index).element<KeyType>(lidx);
      auto const rhs =
        _rhs.child(cudf::dictionary_column_view::keys_column_index).element<KeyType>(ridx);
      return cuda::std::pair(_comparator(lhs, rhs), cuda::std::numeric_limits<int>::max());
    }

    template <typename KeyType>
    __device__ cuda::std::pair<cudf::detail::weak_ordering, int> operator()(
      size_type const, size_type const) const noexcept
      requires(not cudf::is_relationally_comparable<KeyType, KeyType>())
    {
      CUDF_UNREACHABLE("Key types are not comparable");
    }

   private:
    column_device_view _lhs;
    column_device_view _rhs;
    PhysicalElementComparator _comparator;
  };

  /**
   * @brief Performs a relational comparison between two elements in two columns.
   */
  class element_comparator {
   public:
    /**
     * @brief Construct type-dispatched function object for performing a
     * relational comparison between two elements.
     *
     * @note `lhs` and `rhs` may be the same.
     *
     * @param check_nulls Indicates if either input column contains nulls.
     * @param lhs The column containing the first element
     * @param rhs The column containing the second element (may be the same as lhs)
     * @param null_precedence Indicates how null values are ordered with other values
     * @param depth The depth of the column if part of a nested column @see
     * preprocessed_table::depths
     * @param comparator Physical element relational comparison functor.
     * @param l_dremel_device_view <>
     * @param r_dremel_device_view <>
     */
    __device__ element_comparator(Nullate check_nulls,
                                  column_device_view lhs,
                                  column_device_view rhs,
                                  null_order null_precedence                = null_order::BEFORE,
                                  int depth                                 = 0,
                                  PhysicalElementComparator comparator      = {},
                                  optional_dremel_view l_dremel_device_view = {},
                                  optional_dremel_view r_dremel_device_view = {})
      : _lhs{lhs},
        _rhs{rhs},
        _check_nulls{check_nulls},
        _null_precedence{null_precedence},
        _depth{depth},
        _l_dremel_device_view{l_dremel_device_view},
        _r_dremel_device_view{r_dremel_device_view},
        _comparator{comparator}
    {
    }

    /**
     * @brief Performs a relational comparison between the specified elements
     *
     * @param lhs_element_index The index of the first element
     * @param rhs_element_index The index of the second element
     * @return Indicates the relationship between the elements in the `lhs` and `rhs` columns, along
     * with the depth at which a null value was encountered.
     */
    template <typename Element>
    __device__ cuda::std::pair<cudf::detail::weak_ordering, int> operator()(
      size_type const lhs_element_index, size_type const rhs_element_index) const noexcept
      requires(cudf::is_relationally_comparable<Element, Element>() and
               not cudf::is_dictionary<Element>())
    {
      if (_check_nulls) {
        bool const lhs_is_null{_lhs.is_null(lhs_element_index)};
        bool const rhs_is_null{_rhs.is_null(rhs_element_index)};

        if (lhs_is_null or rhs_is_null) {
          return cuda::std::pair(
            cudf::detail::null_compare(lhs_is_null, rhs_is_null, _null_precedence), _depth);
        }
      }

      return cuda::std::pair(_comparator(_lhs.element<Element>(lhs_element_index),
                                         _rhs.element<Element>(rhs_element_index)),
                             cuda::std::numeric_limits<int>::max());
    }

    template <typename Element>
#ifndef NDEBUG
    __attribute__((noinline))
#endif
    __device__ cuda::std::pair<cudf::detail::weak_ordering, int>
    operator()(size_type const lhs_element_index, size_type const rhs_element_index) const noexcept
      requires(cudf::is_dictionary<Element>())
    {
      if (_check_nulls) {
        bool const lhs_is_null{_lhs.is_null(lhs_element_index)};
        bool const rhs_is_null{_rhs.is_null(rhs_element_index)};
        if (lhs_is_null or rhs_is_null) {
          return cuda::std::pair(
            cudf::detail::null_compare(lhs_is_null, rhs_is_null, _null_precedence), _depth);
        }
      }

      auto keys = _lhs.child(cudf::dictionary_column_view::keys_column_index);
      return cudf::type_dispatcher<dispatch_void_if_nested>(
        keys.type(),
        dictionary_comparator{_lhs, _rhs, _comparator},
        lhs_element_index,
        rhs_element_index);
    }

    /**
     * @brief Throws run-time error when columns types cannot be compared
     *        or if this class is instantiated with `has_nested_columns = false` but
     *        passed tables with nested columns
     *
     * @return Ordering
     */
    template <typename Element>
    __device__ cuda::std::pair<cudf::detail::weak_ordering, int> operator()(
      size_type const, size_type const) const noexcept
      requires(not cudf::is_relationally_comparable<Element, Element>() and
               (not has_nested_columns or not cudf::is_nested<Element>()))
    {
      CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
    }

    /**
     * @brief Compares two struct-type columns
     *
     * @param lhs_element_index The index of the first element
     * @param rhs_element_index The index of the second element
     * @return Indicates the relationship between the elements in the `lhs` and `rhs` columns, along
     * with the depth at which a null value was encountered.
     */
    template <typename Element>
#ifndef NDEBUG
    __attribute__((noinline))
#endif
    __device__ cuda::std::pair<cudf::detail::weak_ordering, int>
    operator()(size_type const lhs_element_index, size_type const rhs_element_index) const noexcept
      requires(has_nested_columns and cuda::std::is_same_v<Element, cudf::struct_view>)
    {
      column_device_view lcol = _lhs;
      column_device_view rcol = _rhs;
      int depth               = _depth;
      while (lcol.type().id() == type_id::STRUCT) {
        bool const lhs_is_null{lcol.is_null(lhs_element_index)};
        bool const rhs_is_null{rcol.is_null(rhs_element_index)};

        if (lhs_is_null or rhs_is_null) {
          cudf::detail::weak_ordering state =
            cudf::detail::null_compare(lhs_is_null, rhs_is_null, _null_precedence);
          return cuda::std::pair(state, depth);
        }

        if (lcol.num_child_columns() == 0) {
          return cuda::std::pair(cudf::detail::weak_ordering::EQUIVALENT,
                                 cuda::std::numeric_limits<int>::max());
        }

        lcol = structs_column_device_view(lcol).get_sliced_child(0);
        rcol = structs_column_device_view(rcol).get_sliced_child(0);
        ++depth;
      }

      return cudf::type_dispatcher<dispatch_void_if_nested>(
        lcol.type(),
        element_comparator{_check_nulls, lcol, rcol, _null_precedence, depth, _comparator},
        lhs_element_index,
        rhs_element_index);
    }

    /**
     * @brief Compares two list-type columns
     *
     * @param lhs_element_index The index of the first element
     * @param rhs_element_index The index of the second element
     * @return Indicates the relationship between the elements in the `lhs` and `rhs` columns, along
     * with the depth at which a null value was encountered.
     */
    template <typename Element>
#ifndef NDEBUG
    __attribute__((noinline))
#endif
    __device__ cuda::std::pair<cudf::detail::weak_ordering, int>
    operator()(size_type lhs_element_index, size_type rhs_element_index)
      requires(has_nested_columns and cuda::std::is_same_v<Element, cudf::list_view>)
    {
      auto const is_l_row_null = _lhs.is_null(lhs_element_index);
      auto const is_r_row_null = _rhs.is_null(rhs_element_index);
      if (is_l_row_null || is_r_row_null) {
        return cuda::std::pair(
          cudf::detail::null_compare(is_l_row_null, is_r_row_null, _null_precedence), _depth);
      }

      auto const l_max_def_level = _l_dremel_device_view->max_def_level;
      auto const r_max_def_level = _r_dremel_device_view->max_def_level;
      auto const l_def_levels    = _l_dremel_device_view->def_levels;
      auto const r_def_levels    = _r_dremel_device_view->def_levels;
      auto const l_rep_levels    = _l_dremel_device_view->rep_levels;
      auto const r_rep_levels    = _r_dremel_device_view->rep_levels;

      column_device_view lcol = _lhs.slice(lhs_element_index, 1);
      column_device_view rcol = _rhs.slice(rhs_element_index, 1);

      while (lcol.type().id() == type_id::LIST) {
        lcol = lists_column_device_view(lcol).get_sliced_child();
        rcol = lists_column_device_view(rcol).get_sliced_child();
      }

      auto const l_offsets = _l_dremel_device_view->offsets;
      auto const r_offsets = _r_dremel_device_view->offsets;
      auto l_start         = l_offsets[lhs_element_index];
      auto l_end           = l_offsets[lhs_element_index + 1];
      auto r_start         = r_offsets[rhs_element_index];
      auto r_end           = r_offsets[rhs_element_index + 1];

      auto comparator =
        element_comparator{_check_nulls, lcol, rcol, _null_precedence, _depth, _comparator};

      for (int l_dremel_index = l_start, r_dremel_index = r_start, element_index = 0;
           l_dremel_index < l_end and r_dremel_index < r_end;
           ++l_dremel_index, ++r_dremel_index) {
        auto const l_rep_level = l_rep_levels[l_dremel_index];
        auto const r_rep_level = r_rep_levels[r_dremel_index];

        if (l_rep_level != r_rep_level) {
          return l_rep_level < r_rep_level
                   ? cuda::std::pair(cudf::detail::weak_ordering::LESS, _depth)
                   : cuda::std::pair(cudf::detail::weak_ordering::GREATER, _depth);
        }

        auto const l_def_level = l_def_levels[l_dremel_index];
        auto const r_def_level = r_def_levels[r_dremel_index];

        if (l_def_level < l_max_def_level || r_def_level < r_max_def_level) {
          if ((lcol.nullable() and l_def_levels[l_dremel_index] == l_max_def_level - 1) or
              (rcol.nullable() and r_def_levels[r_dremel_index] == r_max_def_level - 1)) {
            ++element_index;
          }
          if (l_def_level == r_def_level) { continue; }
          return l_def_level < r_def_level
                   ? cuda::std::pair(cudf::detail::weak_ordering::LESS, _depth)
                   : cuda::std::pair(cudf::detail::weak_ordering::GREATER, _depth);
        }

        cudf::detail::weak_ordering state{cudf::detail::weak_ordering::EQUIVALENT};
        int last_null_depth                    = _depth;
        cuda::std::tie(state, last_null_depth) = cudf::type_dispatcher<dispatch_void_if_nested>(
          lcol.type(), comparator, element_index, element_index);
        if (state != cudf::detail::weak_ordering::EQUIVALENT) {
          return cuda::std::pair(state, _depth);
        }
        ++element_index;
      }

      return cuda::std::pair(detail::compare_elements(l_end - l_start, r_end - r_start), _depth);
    }

   private:
    column_device_view const _lhs;
    column_device_view const _rhs;
    Nullate const _check_nulls;
    null_order const _null_precedence;
    int const _depth;
    optional_dremel_view _l_dremel_device_view;
    optional_dremel_view _r_dremel_device_view;
    PhysicalElementComparator const _comparator;
  };

 public:
  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table compares
   * lexicographically less, greater, or equivalent to the row at `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of the row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return weak ordering comparison of the row in the `lhs` table relative to the row in the `rhs`
   * table
   */
  __device__ constexpr cudf::detail::weak_ordering operator()(
    size_type const lhs_index, size_type const rhs_index) const noexcept
  {
    int last_null_depth = cuda::std::numeric_limits<int>::max();
    size_type list_column_index{-1};
    for (size_type i = 0; i < _lhs.num_columns(); ++i) {
      if (_lhs.column(i).type().id() == type_id::LIST) { ++list_column_index; }

      int const depth = _depth.has_value() ? (*_depth)[i] : 0;
      if (depth > last_null_depth) { continue; }

      bool const ascending =
        _column_order.has_value() ? (*_column_order)[i] == order::ASCENDING : true;

      null_order const null_precedence =
        _null_precedence.has_value() ? (*_null_precedence)[i] : null_order::BEFORE;

      auto const [l_dremel_i, r_dremel_i] =
        _lhs.column(i).type().id() == type_id::LIST
          ? cuda::std::make_tuple(optional_dremel_view(_l_dremel[list_column_index]),
                                  optional_dremel_view(_r_dremel[list_column_index]))
          : cuda::std::make_tuple(optional_dremel_view{}, optional_dremel_view{});

      auto element_comp = element_comparator{_check_nulls,
                                             _lhs.column(i),
                                             _rhs.column(i),
                                             null_precedence,
                                             depth,
                                             _comparator,
                                             l_dremel_i,
                                             r_dremel_i};

      cudf::detail::weak_ordering state;
      cuda::std::tie(state, last_null_depth) =
        cudf::type_dispatcher(_lhs.column(i).type(), element_comp, lhs_index, rhs_index);

      if (state == cudf::detail::weak_ordering::EQUIVALENT) { continue; }

      return ascending ? state
                       : (state == cudf::detail::weak_ordering::GREATER
                            ? cudf::detail::weak_ordering::LESS
                            : cudf::detail::weak_ordering::GREATER);
    }
    return cudf::detail::weak_ordering::EQUIVALENT;
  }

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  device_span<detail::dremel_device_view const> _l_dremel;
  device_span<detail::dremel_device_view const> _r_dremel;
  Nullate _check_nulls;
  cuda::std::optional<device_span<int const>> _depth;
  cuda::std::optional<device_span<order const>> _column_order;
  cuda::std::optional<device_span<null_order const>> _null_precedence;
  PhysicalElementComparator _comparator;
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

/**
 * @brief Preprocessed table for use with lexicographical comparison
 *
 */
struct preprocessed_table {
  using table_device_view_owner = std::invoke_result_t<decltype(table_device_view::create),
                                                       table_view,
                                                       rmm::cuda_stream_view,
                                                       rmm::device_async_resource_ref>;

  /**
   * @brief Preprocess table for use with lexicographical comparison
   *
   * Sets up the table for use with lexicographical comparison. The resulting preprocessed table can
   * be passed to the constructor of `lexicographic::self_comparator` or
   * `lexicographic::two_table_comparator` to avoid preprocessing again.
   *
   * Note that the output of this factory function should not be used in `two_table_comparator` if
   * the input table contains lists-of-structs. In such cases, please use the overload
   * `preprocessed_table::create(table_view const&, table_view const&,...)` to preprocess both input
   * tables at the same time.
   *
   * @param table The table to preprocess
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all columns
   *        are sorted in ascending order.
   * @param null_precedence Optional, an array having the same length as the number of columns in
   *        the input tables that indicates how null values compare to all other. If it is empty,
   *        the order `null_order::BEFORE` will be used for all columns.
   * @param stream The stream to launch kernels and h->d copies on while preprocessing
   * @return A shared pointer to a preprocessed table
   */
  static std::shared_ptr<preprocessed_table> create(table_view const& table,
                                                    host_span<order const> column_order,
                                                    host_span<null_order const> null_precedence,
                                                    rmm::cuda_stream_view stream);

  /**
   * @brief Preprocess tables for use with lexicographical comparison
   *
   * Sets up the tables for use with lexicographical comparison. The resulting preprocessed tables
   * can be passed to the constructor of `lexicographic::self_comparator` or
   * `lexicographic::two_table_comparator` to avoid preprocessing again.
   *
   * This factory function performs some extra operations to guarantee that its output can be used
   * in `two_table_comparator` for all cases.
   *
   * @param lhs The lhs table to preprocess
   * @param rhs The rhs table to preprocess
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all columns
   *        are sorted in ascending order.
   * @param null_precedence Optional, an array having the same length as the number of columns in
   *        the input tables that indicates how null values compare to all other. If it is empty,
   *        the order `null_order::BEFORE` will be used for all columns.
   * @param stream The stream to launch kernels and h->d copies on while preprocessing
   * @return A pair of shared pointers to the preprocessed tables
   */
  static std::pair<std::shared_ptr<preprocessed_table>, std::shared_ptr<preprocessed_table>> create(
    table_view const& lhs,
    table_view const& rhs,
    host_span<order const> column_order,
    host_span<null_order const> null_precedence,
    rmm::cuda_stream_view stream);

 private:
  friend class self_comparator;
  friend class two_table_comparator;

  /**
   * @brief Create the output preprocessed table from intermediate preprocessing results
   *
   * @param preprocessed_input The table resulted from preprocessing
   * @param verticalized_col_depths The depths of each column resulting from decomposing struct
   *        columns in the original input table
   * @param transformed_columns Store the intermediate columns generated from transforming
   *        nested children columns into integers columns using `cudf::rank()`
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all columns
   *        are sorted in ascending order.
   * @param null_precedence Optional, an array having the same length as the number of columns in
   *        the input tables that indicates how null values compare to all other. If it is empty,
   *        the order `null_order::BEFORE` will be used for all columns.
   * @param has_ranked_children Flag indicating if the input table was preprocessed to transform
   *        any nested child column into an integer column using `cudf::rank`
   * @param stream The stream to launch kernels and h->d copies on while preprocessing
   * @return A shared pointer to a preprocessed table
   */
  static std::shared_ptr<preprocessed_table> create(
    table_view const& preprocessed_input,
    std::vector<int>&& verticalized_col_depths,
    std::vector<std::unique_ptr<column>>&& transformed_columns,
    host_span<order const> column_order,
    host_span<null_order const> null_precedence,
    bool has_ranked_children,
    rmm::cuda_stream_view stream);

  /**
   * @brief Construct a preprocessed table for use with lexicographical comparison
   *
   * Sets up the table for use with lexicographical comparison. The resulting preprocessed table can
   * be passed to the constructor of `lexicographic::self_comparator` to avoid preprocessing again.
   *
   * @param table The table to preprocess
   * @param column_order Optional, device array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all columns
   *        are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   *        values compare to all other for every column. If it is nullptr, then null precedence
   *        would be `null_order::BEFORE` for all columns.
   * @param depths The depths of each column resulting from decomposing struct columns.
   * @param dremel_data The dremel data for each list column. The length of this object is the
   *        number of list columns in the table.
   * @param dremel_device_views Device views into the dremel_data structs contained in the
   *        `dremel_data` parameter. For columns that are not list columns, this uvector will should
   *        contain an empty `dremel_device_view`. As such, this uvector has as many elements as
   *        there are columns in the table (unlike the `dremel_data` parameter, which is only as
   *        long as the number of list columns).
   * @param transformed_columns Store the intermediate columns generated from transforming
   *        nested children columns into integers columns using `cudf::rank()`
   * @param has_ranked_children Flag indicating if the input table was preprocessed to transform
   *        any lists-of-structs column having floating-point children using `cudf::rank`
   */
  preprocessed_table(table_device_view_owner&& table,
                     rmm::device_uvector<order>&& column_order,
                     rmm::device_uvector<null_order>&& null_precedence,
                     rmm::device_uvector<size_type>&& depths,
                     std::vector<detail::dremel_data>&& dremel_data,
                     rmm::device_uvector<detail::dremel_device_view>&& dremel_device_views,
                     std::vector<std::unique_ptr<column>>&& transformed_columns,
                     bool has_ranked_children);

  preprocessed_table(table_device_view_owner&& table,
                     rmm::device_uvector<order>&& column_order,
                     rmm::device_uvector<null_order>&& null_precedence,
                     rmm::device_uvector<size_type>&& depths,
                     std::vector<std::unique_ptr<column>>&& transformed_columns,
                     bool has_ranked_children);

  /**
   * @brief Implicit conversion operator to a `table_device_view` of the preprocessed table.
   *
   * @return table_device_view
   */
  operator table_device_view() { return *_t; }

  /**
   * @brief Get a device array containing the desired order of each column in the preprocessed table
   *
   * @return Device array containing respective column orders. If no explicit column orders were
   * specified during the creation of this object then this will be `nullopt`.
   */
  [[nodiscard]] cuda::std::optional<device_span<order const>> column_order() const
  {
    return _column_order.size() ? cuda::std::optional<device_span<order const>>(_column_order)
                                : cuda::std::nullopt;
  }

  /**
   * @brief Get a device array containing the desired null precedence of each column in the
   * preprocessed table
   *
   * @return Device array containing respective column null precedence. If no explicit column null
   * precedences were specified during the creation of this object then this will be `nullopt`.
   */
  [[nodiscard]] cuda::std::optional<device_span<null_order const>> null_precedence() const
  {
    return _null_precedence.size()
             ? cuda::std::optional<device_span<null_order const>>(_null_precedence)
             : cuda::std::nullopt;
  }

  /**
   * @brief Get a device array containing the depth of each column in the preprocessed table
   *
   * @see struct_linearize()
   *
   * @return std::optional<device_span<int const>> Device array containing respective column depths.
   * If there are no nested columns in the table then this will be `nullopt`.
   */
  [[nodiscard]] cuda::std::optional<device_span<int const>> depths() const
  {
    return _depths.size() ? cuda::std::optional<device_span<int const>>(_depths)
                          : cuda::std::nullopt;
  }

  [[nodiscard]] device_span<detail::dremel_device_view const> dremel_device_views() const
  {
    if (_dremel_device_views.has_value()) {
      return device_span<detail::dremel_device_view const>(*_dremel_device_views);
    } else {
      return {};
    }
  }

  template <typename PhysicalElementComparator>
  void check_physical_element_comparator()
  {
    if constexpr (!cuda::std::is_same_v<PhysicalElementComparator,
                                        sorting_physical_element_comparator>) {
      CUDF_EXPECTS(!_has_ranked_children,
                   "The input table has nested type children and they were transformed using a "
                   "different type of physical element comparator.");
    }
  }

 private:
  table_device_view_owner const _t;
  rmm::device_uvector<order> const _column_order;
  rmm::device_uvector<null_order> const _null_precedence;
  rmm::device_uvector<size_type> const _depths;

  cuda::std::optional<std::vector<detail::dremel_data>> _dremel_data;
  cuda::std::optional<rmm::device_uvector<detail::dremel_device_view>> _dremel_device_views;

  std::vector<std::unique_ptr<column>> _transformed_columns;

  bool const _has_ranked_children;
};

/**
 * @brief An owning object that can be used to lexicographically compare two rows of the same table
 *
 * This class can take a table_view and preprocess certain columns to allow for lexicographical
 * comparison. The preprocessed table and temporary data required for the comparison are created and
 * owned by this class.
 *
 * Alternatively, `self_comparator` can be constructed from an existing
 * `shared_ptr<preprocessed_table>` when sharing the same table among multiple comparators.
 *
 * This class can then provide a functor object that can used on the device.
 * The object of this class must outlive the usage of the device functor.
 */
class self_comparator {
 public:
  /**
   * @brief Construct an owning object for performing a lexicographic comparison between two rows of
   * the same table.
   *
   * @param t The table to compare
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all columns
   *        are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   *        values compare to all other for every column. If empty, then null precedence would be
   *        `null_order::BEFORE` for all columns.
   * @param stream The stream to construct this object on. Not the stream that will be used for
   *        comparisons using this object.
   */
  self_comparator(table_view const& t,
                  host_span<order const> column_order         = {},
                  host_span<null_order const> null_precedence = {},
                  rmm::cuda_stream_view stream                = cudf::get_default_stream())
    : d_t{preprocessed_table::create(t, column_order, null_precedence, stream)}
  {
  }

  /**
   * @brief Construct an owning object for performing a lexicographic comparison between two rows of
   * the same preprocessed table.
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it among
   * multiple comparators.
   *
   * @param t A table preprocessed for lexicographic comparison
   */
  self_comparator(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  /**
   * @brief Return the binary operator for comparing rows in the table.
   *
   * Returns a binary callable, `F`, with signature `bool F(size_type, size_type)`.
   *
   * `F(i,j)` returns true if and only if row `i` compares lexicographically less than row `j`.
   *
   * @note The operator overloads in sub-class `element_comparator` are templated via the
   *       `type_dispatcher` to help select an overload instance for each column in a table.
   *       So, `cudf::is_nested<Element>` will return `true` if the table has nested-type columns,
   *       but it will be a runtime error if template parameter `has_nested_columns != true`.
   *
   * @tparam has_nested_columns compile-time optimization for primitive types.
   *         This template parameter is to be used by the developer by querying
   *         `cudf::has_nested_columns(input)`. `true` compiles operator
   *         overloads for nested types, while `false` only compiles operator
   *         overloads for primitive types.
   * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
   * @tparam PhysicalElementComparator A relational comparator functor that compares individual
   *         values rather than logical elements, defaults to `NaN` aware relational comparator
   *         that evaluates `NaN` as greater than all other values.
   * @throw cudf::logic_error if the input table was preprocessed to transform any nested children
   *        columns into integer columns but `PhysicalElementComparator` is not
   *        `sorting_physical_element_comparator`.
   * @param nullate Indicates if any input column contains nulls.
   * @param comparator Physical element relational comparison functor.
   * @return A binary callable object.
   */
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalElementComparator = sorting_physical_element_comparator>
  auto less(Nullate nullate = {}, PhysicalElementComparator comparator = {}) const
  {
    d_t->check_physical_element_comparator<PhysicalElementComparator>();

    return less_comparator{
      device_row_comparator<has_nested_columns, Nullate, PhysicalElementComparator>{
        nullate,
        *d_t,
        *d_t,
        d_t->dremel_device_views(),
        d_t->dremel_device_views(),
        d_t->depths(),
        d_t->column_order(),
        d_t->null_precedence(),
        comparator}};
  }

  /// @copydoc less()
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalElementComparator = sorting_physical_element_comparator>
  auto less_equivalent(Nullate nullate = {}, PhysicalElementComparator comparator = {}) const
  {
    d_t->check_physical_element_comparator<PhysicalElementComparator>();

    return less_equivalent_comparator{
      device_row_comparator<has_nested_columns, Nullate, PhysicalElementComparator>{
        nullate,
        *d_t,
        *d_t,
        d_t->dremel_device_views(),
        d_t->dremel_device_views(),
        d_t->depths(),
        d_t->column_order(),
        d_t->null_precedence(),
        comparator}};
  }

 private:
  std::shared_ptr<preprocessed_table> d_t;
};

// @cond
template <typename Comparator>
struct strong_index_comparator_adapter {
  strong_index_comparator_adapter(Comparator const& comparator) : comparator{comparator} {}

  __device__ constexpr cudf::detail::weak_ordering operator()(
    lhs_index_type const lhs_index, rhs_index_type const rhs_index) const noexcept
  {
    return comparator(static_cast<cudf::size_type>(lhs_index),
                      static_cast<cudf::size_type>(rhs_index));
  }

  __device__ constexpr cudf::detail::weak_ordering operator()(
    rhs_index_type const rhs_index, lhs_index_type const lhs_index) const noexcept
  {
    auto const left_right_ordering =
      comparator(static_cast<cudf::size_type>(lhs_index), static_cast<cudf::size_type>(rhs_index));

    if (left_right_ordering == cudf::detail::weak_ordering::LESS) {
      return cudf::detail::weak_ordering::GREATER;
    } else if (left_right_ordering == cudf::detail::weak_ordering::GREATER) {
      return cudf::detail::weak_ordering::LESS;
    }
    return cudf::detail::weak_ordering::EQUIVALENT;
  }

  Comparator comparator;
};
// @endcond

/**
 * @brief An owning object that can be used to lexicographically compare rows of two different
 * tables
 *
 * This class takes two table_views and preprocesses certain columns to allow for lexicographical
 * comparison. The preprocessed table and temporary data required for the comparison are created and
 * owned by this class.
 *
 * Alternatively, `two_table_comparator` can be constructed from two existing
 * `shared_ptr<preprocessed_table>`s when sharing the same tables among multiple comparators.
 *
 * This class can then provide a functor object that can used on the device.
 * The object of this class must outlive the usage of the device functor.
 */
class two_table_comparator {
 public:
  /**
   * @brief Construct an owning object for performing a lexicographic comparison between rows of
   * two different tables.
   *
   * The left and right table are expected to have the same number of columns
   * and data types for each column.
   *
   * @param left The left table to compare
   * @param right The right table to compare
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all columns
   *        are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   *        values compare to all other for every column. If empty, then null precedence would be
   *        `null_order::BEFORE` for all columns.
   * @param stream The stream to construct this object on. Not the stream that will be used for
   *        comparisons using this object.
   */
  two_table_comparator(table_view const& left,
                       table_view const& right,
                       host_span<order const> column_order         = {},
                       host_span<null_order const> null_precedence = {},
                       rmm::cuda_stream_view stream                = cudf::get_default_stream());

  /**
   * @brief Construct an owning object for performing a lexicographic comparison between two rows of
   * the same preprocessed table.
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it among
   * multiple comparators.
   *
   * The preprocessed_table(s) should have been pre-generated together using the factory function
   * `preprocessed_table::create(table_view const&, table_view const&)`. Otherwise, the comparison
   * results between two tables may be incorrect.
   *
   * @param left A table preprocessed for lexicographic comparison
   * @param right A table preprocessed for lexicographic comparison
   */
  two_table_comparator(std::shared_ptr<preprocessed_table> left,
                       std::shared_ptr<preprocessed_table> right)
    : d_left_table{std::move(left)}, d_right_table{std::move(right)}
  {
  }

  /**
   * @brief Return the binary operator for comparing rows in the table.
   *
   * Returns a binary callable, `F`, with signatures
   * `bool F(lhs_index_type, rhs_index_type)` and
   * `bool F(rhs_index_type, lhs_index_type)`.
   *
   * `F(lhs_index_type i, rhs_index_type j)` returns true if and only if row
   * `i` of the left table compares lexicographically less than row `j` of the
   * right table.
   *
   * Similarly, `F(rhs_index_type i, lhs_index_type j)` returns true if and
   * only if row `i` of the right table compares lexicographically less than row
   * `j` of the left table.
   *
   * @note The operator overloads in sub-class `element_comparator` are templated via the
   *       `type_dispatcher` to help select an overload instance for each column in a table.
   *       So, `cudf::is_nested<Element>` will return `true` if the table has nested-type columns,
   *       but it will be a runtime error if template parameter `has_nested_columns != true`.
   *
   * @tparam has_nested_columns compile-time optimization for primitive types.
   *         This template parameter is to be used by the developer by querying
   *         `cudf::has_nested_columns(input)`. `true` compiles operator
   *         overloads for nested types, while `false` only compiles operator
   *         overloads for primitive types.
   * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
   * @tparam PhysicalElementComparator A relational comparator functor that compares individual
   *         values rather than logical elements, defaults to `NaN` aware relational comparator
   *         that evaluates `NaN` as greater than all other values.
   * @throw cudf::logic_error if the input tables were preprocessed to transform any nested children
   *        columns into integer columns but `PhysicalElementComparator` is not
   *        `sorting_physical_element_comparator`.
   * @param nullate Indicates if any input column contains nulls.
   * @param comparator Physical element relational comparison functor.
   * @return A binary callable object.
   */
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalElementComparator = sorting_physical_element_comparator>
  auto less(Nullate nullate = {}, PhysicalElementComparator comparator = {}) const
  {
    d_left_table->check_physical_element_comparator<PhysicalElementComparator>();
    d_right_table->check_physical_element_comparator<PhysicalElementComparator>();

    return less_comparator{strong_index_comparator_adapter{
      device_row_comparator<has_nested_columns, Nullate, PhysicalElementComparator>{
        nullate,
        *d_left_table,
        *d_right_table,
        d_left_table->dremel_device_views(),
        d_right_table->dremel_device_views(),
        d_left_table->depths(),
        d_left_table->column_order(),
        d_left_table->null_precedence(),
        comparator}}};
  }

  /// @copydoc less()
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalElementComparator = sorting_physical_element_comparator>
  auto less_equivalent(Nullate nullate = {}, PhysicalElementComparator comparator = {}) const
  {
    d_left_table->check_physical_element_comparator<PhysicalElementComparator>();
    d_right_table->check_physical_element_comparator<PhysicalElementComparator>();

    return less_equivalent_comparator{strong_index_comparator_adapter{
      device_row_comparator<has_nested_columns, Nullate, PhysicalElementComparator>{
        nullate,
        *d_left_table,
        *d_right_table,
        d_left_table->dremel_device_views(),
        d_right_table->dremel_device_views(),
        d_left_table->depths(),
        d_left_table->column_order(),
        d_left_table->null_precedence(),
        comparator}}};
  }

 private:
  std::shared_ptr<preprocessed_table> d_left_table;
  std::shared_ptr<preprocessed_table> d_right_table;
};

}  // namespace detail::row::lexicographic
}  // namespace CUDF_EXPORT cudf
