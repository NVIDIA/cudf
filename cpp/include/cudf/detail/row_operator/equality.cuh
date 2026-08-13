/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/structs/structs_column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail::row::equality {

/**
 * @brief Equality comparator functor that compares physical values rather than logical
 * elements like lists, strings, or structs. It evaluates `NaN` not equal to all other values for
 * IEEE-754 compliance.
 */
struct physical_equality_comparator {
  /**
   * @brief Operator for equality comparisons.
   *
   * Note that `NaN != NaN`, following IEEE-754.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return `true` if `lhs == rhs` else `false`
   */
  template <typename Element>
  __device__ constexpr bool operator()(Element const lhs, Element const rhs) const noexcept
  {
    return lhs == rhs;
  }
};

/**
 * @brief Equality comparator functor that compares physical values rather than logical
 * elements like lists, strings, or structs. It evaluates `NaN` as equal to other `NaN`s.
 */
struct nan_equal_physical_equality_comparator {
  /**
   * @brief Operator for equality comparison of non-floating point values.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return `true` if `lhs == rhs` else `false`
   */
  template <typename Element>
  __device__ constexpr bool operator()(Element const lhs, Element const rhs) const noexcept
    requires(not cuda::std::is_floating_point_v<Element>)
  {
    return lhs == rhs;
  }

  /**
   * @brief Operator for equality comparison of floating point values.
   *
   * Note that `NaN == NaN`.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return `true` if `lhs` == `rhs` else `false`
   */
  template <typename Element>
  __device__ constexpr bool operator()(Element const lhs, Element const rhs) const noexcept
    requires(cuda::std::is_floating_point_v<Element>)
  {
    return isnan(lhs) and isnan(rhs) ? true : lhs == rhs;
  }
};

/**
 * @brief Computes the equality comparison between 2 rows.
 *
 * Equality is determined by comparing rows element by element. The first mismatching element
 * returns false, representing unequal rows. If the rows are compared without mismatched elements,
 * the rows are equal.
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
 * @tparam PhysicalEqualityComparator A equality comparator functor that compares individual values
 * rather than logical elements, defaults to a comparator for which `NaN == NaN`.
 */
template <bool has_nested_columns,
          typename Nullate,
          typename PhysicalEqualityComparator = nan_equal_physical_equality_comparator>
class device_row_comparator {
  friend class self_comparator;
  friend class two_table_comparator;

 public:
  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table is equal to the row at
   * `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of the row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return `true` if row from the `lhs` table is equal to the row in the `rhs` table
   */
  __device__ constexpr bool operator()(size_type const lhs_index,
                                       size_type const rhs_index) const noexcept
  {
    auto equal_elements = [lhs_index, rhs_index, this](column_device_view l, column_device_view r) {
      return cudf::type_dispatcher(
        l.type(),
        element_comparator{check_nulls, l, r, nulls_are_equal, comparator},
        lhs_index,
        rhs_index);
    };

    return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(), equal_elements);
  }

 private:
  /**
   * @brief Construct a function object for performing equality comparison between the rows of two
   * tables.
   *
   * @param check_nulls Indicates if any input column contains nulls.
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   * @param comparator Physical element equality comparison functor.
   */
  device_row_comparator(Nullate check_nulls,
                        table_device_view lhs,
                        table_device_view rhs,
                        null_equality nulls_are_equal         = null_equality::EQUAL,
                        PhysicalEqualityComparator comparator = {}) noexcept
    : lhs{lhs},
      rhs{rhs},
      check_nulls{check_nulls},
      nulls_are_equal{nulls_are_equal},
      comparator{comparator}
  {
  }

  /**
   * @brief Performs an equality comparison between two elements in dictionary columns
   */
  class dictionary_comparator {
   public:
    __device__ dictionary_comparator(column_device_view lhs,
                                     column_device_view rhs,
                                     PhysicalEqualityComparator comparator = {}) noexcept
      : lhs{lhs}, rhs{rhs}, comparator{comparator}
    {
    }

    template <typename KeyType>
    __device__ bool operator()(size_type lhs_element_index,
                               size_type rhs_element_index) const noexcept
      requires(cudf::is_equality_comparable<KeyType, KeyType>())
    {
      auto const lidx = lhs.element<cudf::dictionary32>(lhs_element_index).value();
      auto const ridx = rhs.element<cudf::dictionary32>(rhs_element_index).value();
      auto const lh_elem =
        lhs.child(cudf::dictionary_column_view::keys_column_index).element<KeyType>(lidx);
      auto const rh_elem =
        rhs.child(cudf::dictionary_column_view::keys_column_index).element<KeyType>(ridx);
      return comparator(lh_elem, rh_elem);
    }

    template <typename KeyType, typename... Args>
    __device__ bool operator()(Args...) const noexcept
      requires(not cudf::is_equality_comparable<KeyType, KeyType>())
    {
      CUDF_UNREACHABLE("Key types are not comparable");
    }

   private:
    column_device_view lhs;
    column_device_view rhs;
    PhysicalEqualityComparator comparator;
  };

  /**
   * @brief Performs an equality comparison between two elements in two columns.
   */
  class element_comparator {
   public:
    /**
     * @brief Construct type-dispatched function object for comparing equality
     * between two elements.
     *
     * @note `lhs` and `rhs` may be the same.
     *
     * @param check_nulls Indicates if either input column contains nulls.
     * @param lhs The column containing the first element
     * @param rhs The column containing the second element (may be the same as lhs)
     * @param nulls_are_equal Indicates if two null elements are treated as equivalent
     * @param comparator Physical element equality comparison functor.
     */
    __device__ element_comparator(Nullate check_nulls,
                                  column_device_view lhs,
                                  column_device_view rhs,
                                  null_equality nulls_are_equal         = null_equality::EQUAL,
                                  PhysicalEqualityComparator comparator = {}) noexcept
      : lhs{lhs},
        rhs{rhs},
        check_nulls{check_nulls},
        nulls_are_equal{nulls_are_equal},
        comparator{comparator}
    {
    }

    /**
     * @brief Compares the specified elements for equality.
     *
     * @param lhs_element_index The index of the first element
     * @param rhs_element_index The index of the second element
     * @return True if lhs and rhs are equal or if both lhs and rhs are null and nulls are
     * considered equal (`nulls_are_equal` == `null_equality::EQUAL`)
     */
    template <typename Element>
    __device__ bool operator()(size_type const lhs_element_index,
                               size_type const rhs_element_index) const noexcept
      requires(cudf::is_equality_comparable<Element, Element>() and
               not cudf::is_dictionary<Element>())
    {
      if (check_nulls) {
        bool const lhs_is_null{lhs.is_null(lhs_element_index)};
        bool const rhs_is_null{rhs.is_null(rhs_element_index)};
        if (lhs_is_null and rhs_is_null) {
          return nulls_are_equal == null_equality::EQUAL;
        } else if (lhs_is_null != rhs_is_null) {
          return false;
        }
      }

      return comparator(lhs.element<Element>(lhs_element_index),
                        rhs.element<Element>(rhs_element_index));
    }

    template <typename Element>
#ifndef NDEBUG
    __attribute__((noinline))
#endif
    __device__ bool
    operator()(size_type const lhs_element_index, size_type const rhs_element_index) const noexcept
      requires(cudf::is_dictionary<Element>())
    {
      if (check_nulls) {
        bool const lhs_is_null{lhs.is_null(lhs_element_index)};
        bool const rhs_is_null{rhs.is_null(rhs_element_index)};
        if (lhs_is_null and rhs_is_null) {
          return nulls_are_equal == null_equality::EQUAL;
        } else if (lhs_is_null != rhs_is_null) {
          return false;
        }
      }

      return cudf::type_dispatcher<dispatch_void_if_nested>(
        lhs.child(cudf::dictionary_column_view::keys_column_index).type(),
        dictionary_comparator{lhs, rhs, comparator},
        lhs_element_index,
        rhs_element_index);
    }

    template <typename Element, typename... Args>
    __device__ bool operator()(Args...) const noexcept
      requires(not cudf::is_equality_comparable<Element, Element>() and
               (not has_nested_columns or not cudf::is_nested<Element>()))
    {
      CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
    }

    template <typename Element>
#ifndef NDEBUG
    __attribute__((noinline))
#endif
    __device__ bool
    operator()(size_type const lhs_element_index, size_type const rhs_element_index) const noexcept
      requires(has_nested_columns and cudf::is_nested<Element>())
    {
      column_device_view lcol = lhs.slice(lhs_element_index, 1);
      column_device_view rcol = rhs.slice(rhs_element_index, 1);
      while (lcol.type().id() == type_id::STRUCT || lcol.type().id() == type_id::LIST) {
        if (check_nulls) {
          auto lvalid = detail::make_validity_iterator<true>(lcol);
          auto rvalid = detail::make_validity_iterator<true>(rcol);
          if (nulls_are_equal == null_equality::UNEQUAL) {
            if (thrust::any_of(
                  thrust::seq, lvalid, lvalid + lcol.size(), cuda::std::logical_not<bool>()) or
                thrust::any_of(
                  thrust::seq, rvalid, rvalid + rcol.size(), cuda::std::logical_not<bool>())) {
              return false;
            }
          } else {
            if (not thrust::equal(thrust::seq, lvalid, lvalid + lcol.size(), rvalid)) {
              return false;
            }
          }
        }
        if (lcol.type().id() == type_id::STRUCT) {
          if (lcol.num_child_columns() == 0) { return true; }
          lcol = structs_column_device_view(lcol).get_sliced_child(0);
          rcol = structs_column_device_view(rcol).get_sliced_child(0);
        } else if (lcol.type().id() == type_id::LIST) {
          auto l_list_col = lists_column_device_view(lcol);
          auto r_list_col = lists_column_device_view(rcol);

          auto lsizes = make_list_size_iterator(l_list_col);
          auto rsizes = make_list_size_iterator(r_list_col);
          if (not thrust::equal(thrust::seq, lsizes, lsizes + lcol.size(), rsizes)) {
            return false;
          }

          lcol = l_list_col.get_sliced_child();
          rcol = r_list_col.get_sliced_child();
          if (lcol.size() != rcol.size()) { return false; }
        }
      }

      auto comp = column_comparator{
        element_comparator{check_nulls, lcol, rcol, nulls_are_equal, comparator}, lcol.size()};
      return type_dispatcher<dispatch_void_if_nested>(lcol.type(), comp);
    }

   private:
    /**
     * @brief Serially compare two columns for equality.
     *
     * When we want to get the equivalence of two columns by serially comparing all elements in
     * one column with the corresponding elements in the other column, this saves us from type
     * dispatching for each individual element in the range
     */
    struct column_comparator {
      element_comparator const comp;
      size_type const size;

      /**
       * @brief Serially compare two columns for equality.
       *
       * @return True if ALL elements compare equal, false otherwise
       */
      template <typename Element>
      __device__ bool operator()() const noexcept
        requires(cudf::is_equality_comparable<Element, Element>())
      {
        return thrust::all_of(thrust::seq,
                              cuda::counting_iterator<cudf::size_type>{0},
                              cuda::counting_iterator<cudf::size_type>{0} + size,
                              [this](auto i) { return comp.template operator()<Element>(i, i); });
      }

      template <typename Element, typename... Args>
      __device__ bool operator()(Args...) const noexcept
        requires(not cudf::is_equality_comparable<Element, Element>())
      {
        CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
      }
    };

    column_device_view const lhs;
    column_device_view const rhs;
    Nullate const check_nulls;
    null_equality const nulls_are_equal;
    PhysicalEqualityComparator const comparator;
  };

  table_device_view const lhs;
  table_device_view const rhs;
  Nullate const check_nulls;
  null_equality const nulls_are_equal;
  PhysicalEqualityComparator const comparator;
};

/**
 * @brief Comparator for performing equality comparisons between two rows of the same table.
 *
 */
class self_comparator {
 public:
  /**
   * @brief Construct an owning object for performing equality comparisons between two rows of the
   * same table.
   *
   * @param t The table to compare
   * @param stream The stream to construct this object on. Not the stream that will be used for
   * comparisons using this object.
   */
  self_comparator(table_view const& t, rmm::cuda_stream_view stream)
    : d_t(preprocessed_table::create(t, stream))
  {
  }

  /**
   * @brief Construct an owning object for performing equality comparisons between two rows of the
   * same table.
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it among
   * multiple comparators.
   *
   * @param t A table preprocessed for equality comparison
   */
  self_comparator(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  /**
   * @brief Get the comparison operator to use on the device
   *
   * Returns a binary callable, `F`, with signature `bool F(size_type, size_type)`.
   *
   * `F(i,j)` returns true if and only if row `i` compares equal to row `j`.
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
   * @tparam PhysicalEqualityComparator A equality comparator functor that compares individual
   * values rather than logical elements, defaults to a comparator for which `NaN == NaN`.
   * @param nullate Indicates if any input column contains nulls.
   * @param nulls_are_equal Indicates if nulls are equal.
   * @param comparator Physical element equality comparison functor.
   * @return A binary callable object
   */
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalEqualityComparator = nan_equal_physical_equality_comparator>
  auto equal_to(Nullate nullate                       = {},
                null_equality nulls_are_equal         = null_equality::EQUAL,
                PhysicalEqualityComparator comparator = {}) const noexcept
  {
    return device_row_comparator<has_nested_columns, Nullate, PhysicalEqualityComparator>{
      nullate, *d_t, *d_t, nulls_are_equal, comparator};
  }

 private:
  std::shared_ptr<preprocessed_table> d_t;
};

// @cond
template <typename Comparator>
struct strong_index_comparator_adapter {
  strong_index_comparator_adapter(Comparator const& comparator) : comparator{comparator} {}

  __device__ constexpr bool operator()(lhs_index_type const lhs_index,
                                       rhs_index_type const rhs_index) const noexcept
  {
    return comparator(static_cast<cudf::size_type>(lhs_index),
                      static_cast<cudf::size_type>(rhs_index));
  }

  __device__ constexpr bool operator()(rhs_index_type const rhs_index,
                                       lhs_index_type const lhs_index) const noexcept
  {
    return this->operator()(lhs_index, rhs_index);
  }

  Comparator const comparator;
};
// @endcond

/**
 * @brief An owning object that can be used to equality compare rows of two different tables.
 *
 * This class takes two table_views and preprocesses certain columns to allow for equality
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
   * @brief Construct an owning object for performing equality comparisons between two rows from two
   * tables.
   *
   * The left and right table are expected to have the same number of columns and data types for
   * each column.
   *
   * @throws std::invalid_argument if the tables have different number of columns or incompatible
   * column types
   *
   * @param left The left table to compare.
   * @param right The right table to compare.
   * @param stream The stream to construct this object on. Not the stream that will be used for
   * comparisons using this object.
   */
  two_table_comparator(table_view const& left,
                       table_view const& right,
                       rmm::cuda_stream_view stream);

  /**
   * @brief Construct an owning object for performing equality comparisons between two rows from two
   * tables.
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it among
   * multiple comparators.
   *
   * @throws std::invalid_argument if the tables have different number of columns
   *
   * @param left The left table preprocessed for equality comparison.
   * @param right The right table preprocessed for equality comparison.
   */
  two_table_comparator(std::shared_ptr<preprocessed_table> left,
                       std::shared_ptr<preprocessed_table> right);

  /**
   * @brief Return the binary operator for comparing rows in the table.
   *
   * Returns a binary callable, `F`, with signatures `bool F(lhs_index_type, rhs_index_type)` and
   * `bool F(rhs_index_type, lhs_index_type)`.
   *
   * `F(lhs_index_type i, rhs_index_type j)` returns true if and only if row `i` of the left table
   * compares equal to row `j` of the right table.
   *
   * Similarly, `F(rhs_index_type i, lhs_index_type j)` returns true if and only if row `i` of the
   * right table compares equal to row `j` of the left table.
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
   * @tparam PhysicalEqualityComparator A equality comparator functor that compares individual
   * values rather than logical elements, defaults to a `NaN == NaN` equality comparator.
   * @param nullate Indicates if any input column contains nulls.
   * @param nulls_are_equal Indicates if nulls are equal.
   * @param comparator Physical element equality comparison functor.
   * @return A binary callable object
   */
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalEqualityComparator = nan_equal_physical_equality_comparator>
  auto equal_to(Nullate nullate                       = {},
                null_equality nulls_are_equal         = null_equality::EQUAL,
                PhysicalEqualityComparator comparator = {}) const noexcept
  {
    return strong_index_comparator_adapter{
      device_row_comparator<has_nested_columns, Nullate, PhysicalEqualityComparator>(
        nullate, *d_left_table, *d_right_table, nulls_are_equal, comparator)};
  }

 private:
  std::shared_ptr<preprocessed_table> d_left_table;
  std::shared_ptr<preprocessed_table> d_right_table;
};

}  // namespace detail::row::equality
}  // namespace CUDF_EXPORT cudf
