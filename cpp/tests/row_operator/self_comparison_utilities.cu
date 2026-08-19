/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_operator_tests_utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

// Including this declaration/defintion in two_table_comparison_utilities.cu causes
// the nvcc compiler to segfault when built with the debug (-g) flag.

template <typename PhysicalElementComparator>
std::unique_ptr<cudf::column> self_comparison(cudf::table_view input,
                                              std::vector<cudf::order> const& column_order,
                                              PhysicalElementComparator comparator,
                                              rmm::cuda_stream_view stream,
                                              cudf::memory_resources mr)
{
  auto const table_comparator =
    cudf::detail::row::lexicographic::self_comparator{input, column_order, {}, stream};

  auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_id::BOOL8),
                                          input.num_rows(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr.get_output_mr());

  if (cudf::has_nested_columns(input)) {
    thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{input.num_rows()},
                      cuda::counting_iterator<cudf::size_type>{0},
                      output->mutable_view().data<bool>(),
                      table_comparator.less<true>(cudf::nullate::NO{}, comparator));
  } else {
    thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{input.num_rows()},
                      cuda::counting_iterator<cudf::size_type>{0},
                      output->mutable_view().data<bool>(),
                      table_comparator.less<false>(cudf::nullate::NO{}, comparator));
  }
  return output;
}

template std::unique_ptr<cudf::column> self_comparison<physical_comparator_t>(
  cudf::table_view input,
  std::vector<cudf::order> const& column_order,
  physical_comparator_t comparator,
  rmm::cuda_stream_view stream,
  cudf::memory_resources mr);
template std::unique_ptr<cudf::column> self_comparison<sorting_comparator_t>(
  cudf::table_view input,
  std::vector<cudf::order> const& column_order,
  sorting_comparator_t comparator,
  rmm::cuda_stream_view stream,
  cudf::memory_resources mr);
