/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/lexicographic_common.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <type_traits>

namespace cudf::detail::row::primitive {

/**
 * @brief Performs a lexicographic comparison between rows of a numeric table.
 *
 * Uses the same reduced type map as primitive row equality so that comparisons of
 * numeric rows do not instantiate comparators for strings, dictionaries, or nested types.
 * NaNs compare equivalent to other NaNs and greater than all other non-null values.
 *
 * The table and ordering metadata must remain valid for the device use of this comparator.
 */
class row_lexicographic_comparator {
 public:
  /**
   * @brief Constructs a comparator for rows in the same numeric table.
   *
   * @param has_nulls Indicates if the input contains nulls
   * @param table Device view of a table whose columns must all be numeric
   * @param column_order Per-column sort order, or an empty span for all ascending
   * @param null_precedence Per-column null order, or an empty span for all nulls before
   */
  row_lexicographic_comparator(nullate::DYNAMIC has_nulls,
                               table_device_view table,
                               device_span<order const> column_order,
                               device_span<null_order const> null_precedence)
    : _has_nulls{has_nulls},
      _table{table},
      _column_order{column_order},
      _null_precedence{null_precedence}
  {
  }

  /**
   * @brief Compares two rows in lexicographic order.
   *
   * @param lhs_index Index of the first row
   * @param rhs_index Index of the second row
   * @return Weak ordering of the first row relative to the second row
   */
  __device__ weak_ordering operator()(size_type lhs_index, size_type rhs_index) const noexcept
  {
    for (size_type i = 0; i < _table.num_columns(); ++i) {
      auto const& col     = _table.column(i);
      auto state          = weak_ordering::EQUIVALENT;
      bool compare_values = true;
      if (_has_nulls) {
        bool const lhs_is_null = col.is_null(lhs_index);
        bool const rhs_is_null = col.is_null(rhs_index);
        if (lhs_is_null or rhs_is_null) {
          auto const null_precedence =
            _null_precedence.empty() ? null_order::BEFORE : _null_precedence[i];
          state          = null_compare(lhs_is_null, rhs_is_null, null_precedence);
          compare_values = false;
        }
      }

      if (compare_values) {
        state = cudf::type_dispatcher<dispatch_primitive_type>(
          col.type(), element_comparator{}, col, lhs_index, rhs_index);
      }

      if (state == weak_ordering::EQUIVALENT) { continue; }

      bool const ascending = _column_order.empty() || _column_order[i] == order::ASCENDING;
      return ascending
               ? state
               : (state == weak_ordering::LESS ? weak_ordering::GREATER : weak_ordering::LESS);
    }
    return weak_ordering::EQUIVALENT;
  }

 private:
  struct element_comparator {
    template <typename Element>
    __device__ weak_ordering operator()(column_device_view const& col,
                                        size_type lhs_index,
                                        size_type rhs_index) const noexcept
      requires(cudf::is_numeric<Element>())
    {
      return lexicographic::sorting_physical_element_comparator{}(col.element<Element>(lhs_index),
                                                                  col.element<Element>(rhs_index));
    }

    template <typename Element>
    __device__ weak_ordering operator()(column_device_view const&,
                                        size_type,
                                        size_type) const noexcept
      requires(not cudf::is_numeric<Element>())
    {
      CUDF_UNREACHABLE("Primitive lexicographic comparison requires numeric columns.");
    }
  };

  nullate::DYNAMIC _has_nulls;
  table_device_view _table;
  device_span<order const> _column_order;
  device_span<null_order const> _null_precedence;
};

/**
 * @brief Owns the device metadata for lexicographic comparisons within a numeric table.
 *
 * Host ordering policies may be released after construction. The input column data must remain
 * valid during comparisons. Use the returned device functor on the construction stream and keep
 * this object alive until all uses have been submitted to that stream.
 */
class lexicographic_comparator {
 public:
  /**
   * @brief Copies ordering policies and creates a device view of the numeric table.
   *
   * @param table Table whose columns must all be numeric
   * @param column_order Per-column sort order, or an empty span for all ascending
   * @param null_precedence Per-column null order, or an empty span for all nulls before
   * @param stream Stream used for initialization and comparisons
   */
  lexicographic_comparator(table_view const& table,
                           host_span<order const> column_order         = {},
                           host_span<null_order const> null_precedence = {},
                           cuda::stream_ref stream                     = cudf::get_default_stream())
    : _table{table_device_view::create(table, stream)},
      _column_order{cudf::detail::make_device_uvector_async(
        column_order, stream, cudf::get_current_device_resource_ref())},
      _null_precedence{cudf::detail::make_device_uvector_async(
        null_precedence, stream, cudf::get_current_device_resource_ref())}
  {
    if (not column_order.empty() or not null_precedence.empty()) {
      // Finish copying the host policies before the caller can release them.
      cudf::detail::sync_stream(stream);
    }
  }

  /**
   * @brief Returns a device functor that compares whether one row is less than another.
   *
   * @param has_nulls Indicates if the input contains nulls
   * @return A binary callable accepting two row indices
   */
  auto less(nullate::DYNAMIC has_nulls) const
  {
    return lexicographic::less_comparator{
      row_lexicographic_comparator{has_nulls, *_table, _column_order, _null_precedence}};
  }

 private:
  using table_device_view_owner = std::invoke_result_t<decltype(table_device_view::create),
                                                       table_view,
                                                       cuda::stream_ref,
                                                       rmm::device_async_resource_ref>;

  table_device_view_owner const _table;
  rmm::device_uvector<order> const _column_order;
  rmm::device_uvector<null_order> const _null_precedence;
};

}  // namespace cudf::detail::row::primitive
