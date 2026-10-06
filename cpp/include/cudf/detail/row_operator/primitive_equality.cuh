/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/row_operator/primitive_common.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail::row::primitive {

/**
 * @brief Performs an equality comparison between two elements in two columns.
 */
class element_equality_comparator {
 public:
  /**
   * @brief Compares the specified elements for equality.
   *
   * @param lhs The first column
   * @param rhs The second column
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   * @return True if lhs and rhs element are equal
   */
  template <typename Element, CUDF_ENABLE_IF(cudf::is_equality_comparable<Element, Element>())>
  __device__ bool operator()(column_device_view const& lhs,
                             column_device_view const& rhs,
                             size_type lhs_element_index,
                             size_type rhs_element_index) const
  {
    return cudf::detail::equality_compare(lhs.element<Element>(lhs_element_index),
                                          rhs.element<Element>(rhs_element_index));
  }

  // @cond
  template <typename Element, CUDF_ENABLE_IF(not cudf::is_equality_comparable<Element, Element>())>
  __device__ bool operator()(column_device_view const&,
                             column_device_view const&,
                             size_type,
                             size_type) const
  {
    CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
  }
  // @endcond
};

/**
 * @brief Performs a relational comparison between two elements in two tables.
 */
class row_equality_comparator {
 public:
  /**
   * @brief Construct a new row equality comparator object
   *
   * @param has_nulls Indicates if either input column contains nulls
   * @param lhs Preprocessed table containing the first element
   * @param rhs Preprocessed table containing the second element (may be the same as lhs)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   */
  row_equality_comparator(cudf::nullate::DYNAMIC const& has_nulls,
                          std::shared_ptr<cudf::detail::row::equality::preprocessed_table> lhs,
                          std::shared_ptr<cudf::detail::row::equality::preprocessed_table> rhs,
                          null_equality nulls_are_equal)
    : _has_nulls{has_nulls}, _lhs{*lhs}, _rhs{*rhs}, _nulls_are_equal{nulls_are_equal}
  {
    CUDF_EXPECTS(_lhs.num_columns() == _rhs.num_columns(), "Mismatched number of columns.");
  }

  /**
   * @brief Compares the specified rows for equality.
   *
   * @param lhs_row_index The index of the first row to compare (in the lhs table)
   * @param rhs_row_index The index of the second row to compare (in the rhs table)
   * @return true if both rows are equal, otherwise false
   */
  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) const
  {
    auto equal_elements = [this, lhs_row_index, rhs_row_index](column_device_view const& l,
                                                               column_device_view const& r) {
      // Handle null comparison for each element
      if (_has_nulls) {
        bool const lhs_is_null{l.is_null(lhs_row_index)};
        bool const rhs_is_null{r.is_null(rhs_row_index)};
        if (lhs_is_null and rhs_is_null) {
          return _nulls_are_equal == null_equality::EQUAL;
        } else if (lhs_is_null != rhs_is_null) {
          return false;
        }
      }

      // Both elements are non-null, compare their values
      element_equality_comparator comparator;
      return cudf::type_dispatcher<dispatch_primitive_type>(
        l.type(), comparator, l, r, lhs_row_index, rhs_row_index);
    };

    return thrust::equal(thrust::seq, _lhs.begin(), _lhs.end(), _rhs.begin(), equal_elements);
  }

  /**
   * @brief Compares the specified rows for equality.
   *
   * @param lhs_index The index of the first row to compare (in the lhs table)
   * @param rhs_index The index of the second row to compare (in the rhs table)
   * @return Boolean indicating if both rows are equal
   */
  __device__ bool operator()(cudf::detail::row::lhs_index_type lhs_index,
                             cudf::detail::row::rhs_index_type rhs_index) const
  {
    return (*this)(static_cast<size_type>(lhs_index), static_cast<size_type>(rhs_index));
  }

 private:
  cudf::nullate::DYNAMIC _has_nulls;
  table_device_view _lhs;
  table_device_view _rhs;
  null_equality _nulls_are_equal;
};

}  // namespace detail::row::primitive
}  // namespace CUDF_EXPORT cudf
