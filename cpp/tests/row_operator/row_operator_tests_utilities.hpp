/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <vector>

using physical_comparator_t = cudf::detail::row::lexicographic::physical_element_comparator;
using sorting_comparator_t  = cudf::detail::row::lexicographic::sorting_physical_element_comparator;
using physical_equality_t   = cudf::detail::row::equality::physical_equality_comparator;
using nan_equality_t        = cudf::detail::row::equality::nan_equal_physical_equality_comparator;

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> self_comparison(cudf::table_view input,
                                              std::vector<cudf::order> const& column_order,
                                              PhysicalElementComparator comparator,
                                              rmm::cuda_stream_view stream,
                                              cudf::memory_resources mr);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_comparison(cudf::table_view lhs,
                                                   cudf::table_view rhs,
                                                   std::vector<cudf::order> const& column_order,
                                                   PhysicalElementComparator comparator,
                                                   rmm::cuda_stream_view stream,
                                                   cudf::memory_resources mr);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> two_table_equality(cudf::table_view lhs,
                                                 cudf::table_view rhs,
                                                 std::vector<cudf::order> const& column_order,
                                                 PhysicalElementComparator comparator,
                                                 rmm::cuda_stream_view stream,
                                                 cudf::memory_resources mr);
template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> sorted_order(
  std::shared_ptr<cudf::detail::row::lexicographic::preprocessed_table> preprocessed_input,
  cudf::size_type num_rows,
  bool has_nested,
  PhysicalElementComparator comparator,
  rmm::cuda_stream_view stream,
  cudf::memory_resources mr);
