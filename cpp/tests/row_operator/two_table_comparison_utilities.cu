/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_operator_tests_utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/lexicographic.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_comparison(cudf::table_view lhs,
                                                   cudf::table_view rhs,
                                                   std::vector<cudf::order> const& column_order,
                                                   PhysicalElementComparator comparator,
                                                   rmm::cuda_stream_view stream,
                                                   cudf::memory_resources mr)
{
  // TODO: lexicographic::two_table_comparator still allocates from the current device resource.
  auto const table_comparator =
    cudf::detail::row::lexicographic::two_table_comparator{lhs, rhs, column_order, {}, stream};
  auto const lhs_it = cudf::detail::row::lhs_iterator(0);
  auto const rhs_it = cudf::detail::row::rhs_iterator(0);

  auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_id::BOOL8),
                                          lhs.num_rows(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr.get_output_mr());

  if (cudf::has_nested_columns(lhs) || cudf::has_nested_columns(rhs)) {
    thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                      lhs_it,
                      lhs_it + lhs.num_rows(),
                      rhs_it,
                      output->mutable_view().data<bool>(),
                      table_comparator.less<true>(cudf::nullate::NO{}, comparator));
  } else {
    thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                      lhs_it,
                      lhs_it + lhs.num_rows(),
                      rhs_it,
                      output->mutable_view().data<bool>(),
                      table_comparator.less<false>(cudf::nullate::NO{}, comparator));
  }
  return output;
}

template std::unique_ptr<cudf::column> two_table_comparison<physical_comparator_t>(
  cudf::table_view lhs,
  cudf::table_view rhs,
  std::vector<cudf::order> const& column_order,
  physical_comparator_t comparator,
  rmm::cuda_stream_view stream,
  cudf::memory_resources mr);
template std::unique_ptr<cudf::column> two_table_comparison<sorting_comparator_t>(
  cudf::table_view lhs,
  cudf::table_view rhs,
  std::vector<cudf::order> const& column_order,
  sorting_comparator_t comparator,
  rmm::cuda_stream_view stream,
  cudf::memory_resources mr);

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> sorted_order(
  std::shared_ptr<cudf::detail::row::lexicographic::preprocessed_table> preprocessed_input,
  cudf::size_type num_rows,
  bool has_nested,
  PhysicalElementComparator comparator,
  rmm::cuda_stream_view stream,
  cudf::memory_resources mr)
{
  auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<cudf::size_type>()),
                                          num_rows,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr.get_output_mr());
  auto const out_begin = output->mutable_view().begin<cudf::size_type>();
  thrust::sequence(
    rmm::exec_policy_nosync(stream, mr.get_temporary_mr()), out_begin, out_begin + num_rows, 0);

  auto const table_comparator =
    cudf::detail::row::lexicographic::self_comparator{preprocessed_input};
  if (has_nested) {
    auto const comp = table_comparator.less<true>(cudf::nullate::NO{}, comparator);
    thrust::stable_sort(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                        out_begin,
                        out_begin + num_rows,
                        comp);
  } else {
    auto const comp = table_comparator.less<false>(cudf::nullate::NO{}, comparator);
    thrust::stable_sort(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                        out_begin,
                        out_begin + num_rows,
                        comp);
  }

  return output;
}

template std::unique_ptr<cudf::column> sorted_order<physical_comparator_t>(
  std::shared_ptr<cudf::detail::row::lexicographic::preprocessed_table> preprocessed_input,
  cudf::size_type num_rows,
  bool has_nested,
  physical_comparator_t comparator,
  rmm::cuda_stream_view stream,
  cudf::memory_resources mr);
template std::unique_ptr<cudf::column> sorted_order<sorting_comparator_t>(
  std::shared_ptr<cudf::detail::row::lexicographic::preprocessed_table> preprocessed_input,
  cudf::size_type num_rows,
  bool has_nested,
  sorting_comparator_t comparator,
  rmm::cuda_stream_view stream,
  cudf::memory_resources mr);
