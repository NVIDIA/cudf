/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/scatter.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/transform.h>

#include <vector>

namespace cudf::detail {

/**
 * @brief Predicate to find indices at which LEAD/LAG evaluated to null.
 */
template <typename GatherMapIter>
class is_null_index_predicate_impl {
 public:
  is_null_index_predicate_impl(size_type input_size, GatherMapIter gather_)
    : _null_index{input_size}, _gather{gather_}
  {
  }

  bool __device__ operator()(size_type i) const { return _gather[i] == _null_index; }

 private:
  size_type const _null_index;  // Index value to use to output NULL for LEAD/LAG calculation.
  GatherMapIter _gather;        // Iterator for gather-map entries.
};

/**
 * @brief Helper to construct is_null_index_predicate_impl
 */
template <typename GatherMapIter>
is_null_index_predicate_impl<GatherMapIter> is_null_index_predicate(size_type input_size,
                                                                    GatherMapIter gather)
{
  return is_null_index_predicate_impl<GatherMapIter>{input_size, gather};
}

/**
 * @brief Helper function to calculate LEAD/LAG for nested-type input columns.
 *
 * @tparam PrecedingIterator Iterator-type that returns the preceding bounds
 * @tparam FollowingIterator Iterator-type that returns the following bounds
 * @param[in] op Aggregation kind.
 * @param[in] input Nested-type input column for LEAD/LAG calculation
 * @param[in] default_outputs Default values to use as outputs, if LEAD/LAG
 *                            offset crosses column/group boundaries
 * @param[in] preceding Iterator to retrieve preceding window bounds
 * @param[in] following Iterator to retrieve following window bounds
 * @param[in] row_offset Lead/Lag offset, indicating which row after/before
 *                       the current row is to be returned
 * @param[in] stream CUDA stream for device memory operations/allocations
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 */
template <typename PrecedingIter, typename FollowingIter>
std::unique_ptr<column> compute_lead_lag_for_nested(aggregation::Kind op,
                                                    column_view const& input,
                                                    column_view const& default_outputs,
                                                    PrecedingIter preceding,
                                                    FollowingIter following,
                                                    size_type row_offset,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(op == aggregation::LEAD || op == aggregation::LAG,
               "Unexpected aggregation type in compute_lead_lag_for_nested");
  CUDF_EXPECTS(cudf::have_same_types(input, default_outputs),
               "Defaults column type must match input column.",
               cudf::data_type_error);  // Because LEAD/LAG.

  CUDF_EXPECTS(default_outputs.is_empty() || (input.size() == default_outputs.size()),
               "Number of defaults must match input column.");

  // For LEAD(0)/LAG(0), no computation need be performed.
  // Return copy of input.
  if (row_offset == 0) { return std::make_unique<column>(input, stream, mr); }

  // Algorithm:
  //
  // 1. Construct gather_map with the LEAD/LAG offset applied to the indices.
  //    E.g. A gather_map of:
  //        {0, 1, 2, 3, ..., N-3, N-2, N-1}
  //    would select the input column, unchanged.
  //
  //    For LEAD(2), the following gather_map is used:
  //        {3, 4, 5, 6, ..., N-1, NULL_INDEX, NULL_INDEX}
  //    where `NULL_INDEX` selects `NULL` for the gather.
  //
  //    Similarly, LAG(2) is implemented using the following gather_map:
  //        {NULL_INDEX, NULL_INDEX, 0, 1, 2...}
  //
  // 2. Gather input column based on the gather_map.
  // 3. If default outputs are available, scatter contents of `default_outputs`
  //    to all positions where nulls where gathered in step 2.
  //
  // Note: Step 3 can be switched to use `copy_if_else()`, once it supports
  //       nested types.

  auto static constexpr size_data_type = data_type{type_to_id<size_type>()};

  auto gather_map_column = make_numeric_column(size_data_type,
                                               input.size(),
                                               mask_state::UNALLOCATED,
                                               stream,
                                               cudf::get_current_device_resource_ref());
  auto gather_map        = gather_map_column->mutable_view();

  auto const input_size = input.size();
  auto const null_index = input.size();
  if (op == aggregation::LEAD) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      gather_map.begin<size_type>(),
                      cuda::proclaim_return_type<size_type>(
                        [following, input_size, null_index, row_offset] __device__(size_type i) {
                          return (row_offset > following[i]) ? null_index : (i + row_offset);
                        }));
  } else {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      gather_map.begin<size_type>(),
                      cuda::proclaim_return_type<size_type>(
                        [preceding, input_size, null_index, row_offset] __device__(size_type i) {
                          return (row_offset > (preceding[i] - 1)) ? null_index : (i - row_offset);
                        }));
  }

  auto output_with_nulls = cudf::detail::gather(table_view{std::vector<column_view>{input}},
                                                gather_map_column->view(),
                                                out_of_bounds_policy::NULLIFY,
                                                cudf::negative_index_policy::NOT_ALLOWED,
                                                stream,
                                                mr);

  if (default_outputs.is_empty()) { return std::move(output_with_nulls->release()[0]); }

  // Must scatter defaults.
  auto scatter_map = rmm::device_uvector<size_type>(input.size(), stream);

  // Find all indices at which LEAD/LAG computed nulls previously.
  auto scatter_map_end =
    cudf::detail::copy_if(cuda::counting_iterator<size_type>{0},
                          cuda::counting_iterator<size_type>{input.size()},
                          scatter_map.begin(),
                          is_null_index_predicate(input.size(), gather_map.begin<size_type>()),
                          stream);

  scatter_map.resize(cuda::std::distance(scatter_map.begin(), scatter_map_end), stream);
  // Bail early, if all LEAD/LAG computations succeeded. No defaults need be substituted.
  if (scatter_map.is_empty()) { return std::move(output_with_nulls->release()[0]); }

  // Gather only those default values that are to be substituted.
  auto gathered_defaults =
    cudf::detail::gather(table_view{std::vector<column_view>{default_outputs}},
                         scatter_map,
                         out_of_bounds_policy::DONT_CHECK,
                         cudf::negative_index_policy::NOT_ALLOWED,
                         stream,
                         cudf::get_current_device_resource_ref());

  // Scatter defaults into locations where LEAD/LAG computed nulls.
  auto scattered_results = cudf::detail::scatter(
    table_view{std::vector<column_view>{gathered_defaults->release()[0]->view()}},
    scatter_map,
    table_view{std::vector<column_view>{output_with_nulls->release()[0]->view()}},
    stream,
    mr);
  return std::move(scattered_results->release()[0]);
}

}  // namespace cudf::detail
