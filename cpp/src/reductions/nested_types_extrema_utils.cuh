/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/reduction/detail/reduction_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace reduction {
namespace detail {

/**
 * @brief Binary operator ArgMin/ArgMax with index values into the input table.
 */
template <typename DeviceComparator>
struct row_arg_minmax_fn {
  size_type const num_rows;
  DeviceComparator const comp;
  bool const is_arg_min;

  row_arg_minmax_fn(size_type num_rows_, DeviceComparator comp_, bool const is_arg_min_)
    : num_rows{num_rows_}, comp{std::move(comp_)}, is_arg_min{is_arg_min_}
  {
  }

  // This function is explicitly prevented from inlining, because it calls to
  // `DeviceComparator::operator()` which is inlined and very heavy-weight. Inlining
  // this would result in huge code and significantly compile time when instantiated and
  // used with `thrust::reduce_by_key` or `thrust::scan_by_key`.
  __attribute__((noinline)) __device__ auto operator()(size_type lhs_idx, size_type rhs_idx) const
  {
    // The extra bounds checking is due to issue github.com/rapidsai/cudf/issues/9156 and
    // github.com/NVIDIA/thrust/issues/1525
    // where invalid random values may be passed here by thrust::reduce_by_key
    if (lhs_idx < 0 || lhs_idx >= num_rows) { return rhs_idx; }
    if (rhs_idx < 0 || rhs_idx >= num_rows) { return lhs_idx; }

    // Return `lhs_idx` iff:
    //   row(lhs_idx) <  row(rhs_idx) and finding ArgMin, or
    //   row(lhs_idx) >= row(rhs_idx) and finding ArgMax.
    return comp(lhs_idx, rhs_idx) == is_arg_min ? lhs_idx : rhs_idx;
  }
};

/**
 * @brief The null order when comparing a null with non-null elements. Currently support only the
 * default null order: nulls are compared as LESS than any other non-null elements.
 */
auto static constexpr DEFAULT_NULL_ORDER = cudf::null_order::BEFORE;

/**
 * @brief The utility class to provide a binary operator object for lexicographic comparison of
 * nested-type elements.
 *
 * The binary operator provided by this class has an explicit non-inline `operator()` method to
 * prevent excessive compile time when working with `thrust::reduce_by_key`.
 *
 * When it is a structs or a lists column, top-level NULLs are compared as larger than all other
 * non-null elements - if finding for ARGMIN, or smaller than all other non-null elements - if
 * finding for ARGMAX. This helps achieve the results of finding the min or max element when nulls
 * are excluded from the operations, returning null only when all the input elements are nulls.
 */
class arg_minmax_binop_generator {
 private:
  cudf::table_view const input_tview;
  bool const has_nulls;
  bool const is_min_op;
  cuda::stream_ref stream;

  // Contains data used in `row_comparator` below, thus needs to be kept alive as a member variable.
  std::unique_ptr<cudf::structs::detail::flattened_table> const flattened_input;

  // Contains data used in the returned binop, thus needs to be kept alive as a member variable.
  cudf::detail::row::lexicographic::self_comparator row_comparator;

  arg_minmax_binop_generator(column_view const& input_, bool is_min_op_, cuda::stream_ref stream_)
    : input_tview{cudf::table_view{{input_}}},
      has_nulls{cudf::has_nested_nulls(input_tview)},
      is_min_op{is_min_op_},
      stream{stream_},
      flattened_input{cudf::structs::detail::flatten_nested_columns(
        input_tview,
        {},
        std::vector<null_order>{DEFAULT_NULL_ORDER},
        cudf::structs::detail::column_nullability::MATCH_INCOMING,
        stream,
        cudf::get_current_device_resource_ref())},
      row_comparator{[&input_,
                      &input_tview     = input_tview,
                      &flattened_input = flattened_input,
                      is_min_op_,
                      stream_]() {
        if (is_min_op_ && input_.has_nulls()) {
          // If the input column is nested type (struct/list) and has nulls (at the top level), null
          // structs/lists are excluded from the operations. That is equivalent to considering
          // top-level nulls as larger than all other non-null elements (if finding for ARGMIN), or
          // smaller than all other non-null elements (if finding for ARGMAX).

          if (input_.type().id() == cudf::type_id::STRUCT) {
            // For struct type, it is simple: Just set a separate null order (`null_order::AFTER`)
            // for the top level column, which is stored at the first position in the null_orders
            // array resulted from struct flattening.
            auto null_orders    = flattened_input->null_orders();
            null_orders.front() = cudf::null_order::AFTER;
            return cudf::detail::row::lexicographic::self_comparator{
              flattened_input->flattened_columns(), {}, null_orders, stream_};
          } else {
            // For list type, we cannot set a separate null order for the top level column.
            // Thus, we have to workaround this by creating a dummy (empty) struct column view
            // having the same null mask as the input lists column.
            // This dummy column will have a different null order (`null_order::AFTER`).
            auto const null_orders =
              std::vector<null_order>{cudf::null_order::AFTER, DEFAULT_NULL_ORDER};
            auto const dummy_struct = column_view{data_type{type_id::STRUCT},
                                                  input_.size(),
                                                  nullptr,
                                                  input_.null_mask(),
                                                  input_.null_count(),
                                                  0,
                                                  {}};
            return cudf::detail::row::lexicographic::self_comparator{
              cudf::table_view{{dummy_struct, input_}}, {}, null_orders, stream_};
          }
        } else {
          return cudf::detail::row::lexicographic::self_comparator{
            input_tview, {}, std::vector<null_order>{DEFAULT_NULL_ORDER}, stream_};
        }
      }()}
  {
  }

 public:
  /**
   * @brief Generate the `less` comparator for the input table.
   * @return The `less` comparator
   */
  auto less() const { return row_comparator.less<true>(cudf::nullate::DYNAMIC{has_nulls}); }

  /**
   * @brief Generate the binary operator for ARGMIN/ARGMAX with index values into the input table.
   * @return The binary operator for ARGMIN/ARGMAX
   */
  auto binop() const { return row_arg_minmax_fn(input_tview.num_rows(), less(), is_min_op); }

  template <typename BinOp>
  static auto create(column_view const& input, cuda::stream_ref stream)
  {
    CUDF_EXPECTS(cudf::is_nested(input.type()),
                 "This utility class is designed exclusively for nested input types.");
    return arg_minmax_binop_generator(input,
                                      std::is_same_v<BinOp, cudf::reduction::detail::op::min> ||
                                        std::is_same_v<BinOp, cudf::DeviceMin>,
                                      stream);
  }

  template <cudf::aggregation::Kind K>
  static auto create(column_view const& input, cuda::stream_ref stream)
  {
    CUDF_EXPECTS(cudf::is_nested(input.type()),
                 "This utility class is designed exclusively for nested input types.");
    return arg_minmax_binop_generator(
      input, K == cudf::aggregation::MIN || K == cudf::aggregation::ARGMIN, stream);
  }
};

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
