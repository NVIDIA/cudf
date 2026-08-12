/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "join_common_utils.hpp"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left_input,
           table_view const& right_input,
           null_equality compare_nulls,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr)
{
  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input, right_input},
    stream,
    cudf::get_current_device_resource_ref());  // temporary objects returned

  // now rebuild the table views with the updated ones
  auto const left      = matched.second.front();
  auto const right     = matched.second.back();
  auto const has_nulls = cudf::has_nested_nulls(left) || cudf::has_nested_nulls(right)
                           ? cudf::nullable_join::YES
                           : cudf::nullable_join::NO;

  // For `inner_join`, we can freely choose either the `left` or `right` table to use for
  // building/probing the hash map. Because building is typically more expensive than probing, we
  // build the hash map from the smaller table.
  if (right.num_rows() > left.num_rows()) {
    cudf::hash_join hj_obj(left, has_nulls, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream);
    auto [right_result, left_result] = hj_obj.inner_join(right, std::nullopt, stream, mr);
    return std::pair(std::move(left_result), std::move(right_result));
  } else {
    cudf::hash_join hj_obj(right, has_nulls, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream);
    return hj_obj.inner_join(left, std::nullopt, stream, mr);
  }
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
left_join(table_view const& left_input,
          table_view const& right_input,
          null_equality compare_nulls,
          rmm::cuda_stream_view stream,
          rmm::device_async_resource_ref mr)
{
  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input, right_input},  // these should match
    stream,
    cudf::get_current_device_resource_ref());  // temporary objects returned
  // now rebuild the table views with the updated ones
  table_view const left  = matched.second.front();
  table_view const right = matched.second.back();
  auto const has_nulls   = cudf::has_nested_nulls(left) || cudf::has_nested_nulls(right)
                             ? cudf::nullable_join::YES
                             : cudf::nullable_join::NO;

  cudf::hash_join hj_obj(right, has_nulls, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream);
  return hj_obj.left_join(left, std::nullopt, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
full_join(table_view const& left_input,
          table_view const& right_input,
          null_equality compare_nulls,
          rmm::cuda_stream_view stream,
          rmm::device_async_resource_ref mr)
{
  // Make sure any dictionary columns have matched key sets.
  // This will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input, right_input},  // these should match
    stream,
    cudf::get_current_device_resource_ref());  // temporary objects returned
  // now rebuild the table views with the updated ones
  table_view const left  = matched.second.front();
  table_view const right = matched.second.back();
  auto const has_nulls   = cudf::has_nested_nulls(left) || cudf::has_nested_nulls(right)
                             ? cudf::nullable_join::YES
                             : cudf::nullable_join::NO;

  cudf::hash_join hj_obj(right, has_nulls, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream);
  return hj_obj.full_join(left, std::nullopt, stream, mr);
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left,
           table_view const& right,
           null_equality compare_nulls,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::inner_join(left, right, compare_nulls, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
left_join(table_view const& left,
          table_view const& right,
          null_equality compare_nulls,
          rmm::cuda_stream_view stream,
          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::left_join(left, right, compare_nulls, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
full_join(table_view const& left,
          table_view const& right,
          null_equality compare_nulls,
          rmm::cuda_stream_view stream,
          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::full_join(left, right, compare_nulls, stream, mr);
}

}  // namespace cudf
