/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "binary_ops.hpp"
#include "operation.cuh"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/lexicographic.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf::binops::compiled::detail {
template <class T, class... Ts>
inline constexpr bool is_any_v = std::disjunction<std::is_same<T, Ts>...>::value;

template <typename OptionalIterator, typename DeviceComparator>
struct device_comparison_functor {
  // Explicit constructor definition required to avoid a "no instance of constructor" compilation
  // error
  device_comparison_functor(OptionalIterator const optional_iter,
                            bool const is_lhs_scalar,
                            bool const is_rhs_scalar,
                            DeviceComparator const& comparator)
    : _optional_iter(optional_iter),
      _is_lhs_scalar(is_lhs_scalar),
      _is_rhs_scalar(is_rhs_scalar),
      _comparator(comparator)
  {
  }

  bool __device__ operator()(size_type i)
  {
    return _optional_iter[i].has_value() &&
           _comparator(cudf::detail::row::lhs_index_type{_is_lhs_scalar ? 0 : i},
                       cudf::detail::row::rhs_index_type{_is_rhs_scalar ? 0 : i});
  }

  OptionalIterator const _optional_iter;
  bool const _is_lhs_scalar;
  bool const _is_rhs_scalar;
  DeviceComparator const _comparator;
};

template <class BinaryOperator,
          typename PhysicalElementComparator =
            cudf::detail::row::lexicographic::sorting_physical_element_comparator>
void apply_struct_binary_op(mutable_column_view& out,
                            column_view const& lhs,
                            column_view const& rhs,
                            bool is_lhs_scalar,
                            bool is_rhs_scalar,
                            PhysicalElementComparator comparator,
                            cuda::stream_ref stream)
{
  auto const compare_orders = std::vector<order>(
    lhs.size(),
    is_any_v<BinaryOperator, ops::Greater, ops::GreaterEqual> ? order::DESCENDING
                                                              : order::ASCENDING);
  auto const tlhs = table_view{{lhs}};
  auto const trhs = table_view{{rhs}};
  auto const table_comparator =
    cudf::detail::row::lexicographic::two_table_comparator{tlhs, trhs, compare_orders, {}, stream};
  auto outd = column_device_view::create(out, stream);
  auto optional_iter =
    cudf::detail::make_optional_iterator<bool>(*outd, nullate::DYNAMIC{out.has_nulls()});
  auto const comparator_nulls = nullate::DYNAMIC{has_nested_nulls(tlhs) || has_nested_nulls(trhs)};

  auto tabulate_device_operator = [&](auto device_comparator) {
    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<size_type>(0),
      cuda::counting_iterator<size_type>(out.size()),
      out.begin<bool>(),
      device_comparison_functor{optional_iter, is_lhs_scalar, is_rhs_scalar, device_comparator});
  };
  if (cudf::detail::has_nested_columns(tlhs) || cudf::detail::has_nested_columns(trhs)) {
    is_any_v<BinaryOperator, ops::LessEqual, ops::GreaterEqual>
      ? tabulate_device_operator(
          table_comparator.less_equivalent<true>(comparator_nulls, comparator))
      : tabulate_device_operator(table_comparator.less<true>(comparator_nulls, comparator));
  } else {
    is_any_v<BinaryOperator, ops::LessEqual, ops::GreaterEqual>
      ? tabulate_device_operator(
          table_comparator.less_equivalent<false>(comparator_nulls, comparator))
      : tabulate_device_operator(table_comparator.less<false>(comparator_nulls, comparator));
  }
}

template <typename OptionalIteratorType, typename DeviceComparatorType>
struct struct_equality_functor {
  struct_equality_functor(OptionalIteratorType optional_iter,
                          DeviceComparatorType device_comparator,
                          bool is_lhs_scalar,
                          bool is_rhs_scalar,
                          bool preserve_output)
    : _optional_iter(optional_iter),
      _device_comparator(device_comparator),
      _is_lhs_scalar(is_lhs_scalar),
      _is_rhs_scalar(is_rhs_scalar),
      _preserve_output(preserve_output)
  {
  }

  auto __device__ operator()(size_type i) const noexcept
  {
    auto const lhs = cudf::detail::row::lhs_index_type{_is_lhs_scalar ? 0 : i};
    auto const rhs = cudf::detail::row::rhs_index_type{_is_rhs_scalar ? 0 : i};
    return _optional_iter[i].has_value() and (_device_comparator(lhs, rhs) == _preserve_output);
  }

 private:
  OptionalIteratorType _optional_iter;
  DeviceComparatorType _device_comparator;
  bool _is_lhs_scalar;
  bool _is_rhs_scalar;
  bool _preserve_output;
};

template <
  typename PhysicalEqualityComparator = cudf::detail::row::equality::physical_equality_comparator>
void apply_struct_equality_op(mutable_column_view& out,
                              column_view const& lhs,
                              column_view const& rhs,
                              bool is_lhs_scalar,
                              bool is_rhs_scalar,
                              binary_operator op,
                              PhysicalEqualityComparator comparator,
                              cuda::stream_ref stream)
{
  CUDF_EXPECTS(op == binary_operator::EQUAL || op == binary_operator::NOT_EQUAL ||
                 op == binary_operator::NULL_EQUALS || op == binary_operator::NULL_NOT_EQUALS,
               "Unsupported operator for these types",
               cudf::data_type_error);

  auto tlhs             = table_view{{lhs}};
  auto trhs             = table_view{{rhs}};
  auto table_comparator = cudf::detail::row::equality::two_table_comparator{tlhs, trhs, stream};

  auto outd = column_device_view::create(out, stream);
  auto optional_iter =
    cudf::detail::make_optional_iterator<bool>(*outd, nullate::DYNAMIC{out.has_nulls()});

  auto const comparator_helper = [&](auto const device_comparator) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(out.size()),
                      out.begin<bool>(),
                      struct_equality_functor<decltype(optional_iter), decltype(device_comparator)>(
                        optional_iter,
                        device_comparator,
                        is_lhs_scalar,
                        is_rhs_scalar,
                        op != binary_operator::NOT_EQUAL));
  };

  if (cudf::detail::has_nested_columns(tlhs) or cudf::detail::has_nested_columns(trhs)) {
    auto device_comparator = table_comparator.equal_to<true>(
      nullate::DYNAMIC{has_nested_nulls(tlhs) || has_nested_nulls(trhs)},
      null_equality::EQUAL,
      comparator);

    comparator_helper(device_comparator);
  } else {
    auto device_comparator = table_comparator.equal_to<false>(
      nullate::DYNAMIC{has_nested_nulls(tlhs) || has_nested_nulls(trhs)},
      null_equality::EQUAL,
      comparator);

    comparator_helper(device_comparator);
  }
}
}  // namespace cudf::binops::compiled::detail
