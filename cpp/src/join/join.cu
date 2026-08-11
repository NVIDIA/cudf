/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "join_common_utils.hpp"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/join/hash_join.hpp>
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

#include <algorithm>
#include <memory>

namespace cudf {
namespace detail {

namespace {
bool has_dictionary_columns(table_view const& table)
{
  return std::any_of(table.begin(), table.end(), [](column_view const& column) {
    return column.type().id() == type_id::DICTIONARY32;
  });
}
}  // namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left_input,
           table_view const& right_input,
           null_equality compare_nulls,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr)
{
  auto join_tables = [&](table_view const& left, table_view const& right) {
    auto const has_nulls = cudf::has_nested_nulls(left) || cudf::has_nested_nulls(right)
                             ? cudf::nullable_join::YES
                             : cudf::nullable_join::NO;
    if (right.num_rows() > left.num_rows()) {
      cudf::hash_join::impl_type hj_obj(left,
                                        has_nulls == cudf::nullable_join::YES,
                                        compare_nulls,
                                        CUCO_DESIRED_LOAD_FACTOR,
                                        stream,
                                        cudf::get_current_device_resource_ref());
      auto [right_result, left_result] = hj_obj.inner_join(right, std::nullopt, stream, mr);
      return std::pair(std::move(left_result), std::move(right_result));
    }
    cudf::hash_join::impl_type hj_obj(right,
                                      has_nulls == cudf::nullable_join::YES,
                                      compare_nulls,
                                      CUCO_DESIRED_LOAD_FACTOR,
                                      stream,
                                      cudf::get_current_device_resource_ref());
    return hj_obj.inner_join(left, std::nullopt, stream, mr);
  };

  if (!has_dictionary_columns(left_input)) { return join_tables(left_input, right_input); }

  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input, right_input}, stream, cudf::get_current_device_resource_ref());
  return join_tables(matched.second.front(), matched.second.back());
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
left_join(table_view const& left_input,
          table_view const& right_input,
          null_equality compare_nulls,
          rmm::cuda_stream_view stream,
          rmm::device_async_resource_ref mr)
{
  auto join_tables = [&](table_view const& left, table_view const& right) {
    auto const has_nulls = cudf::has_nested_nulls(left) || cudf::has_nested_nulls(right)
                             ? cudf::nullable_join::YES
                             : cudf::nullable_join::NO;
    cudf::hash_join::impl_type hj_obj(right,
                                      has_nulls == cudf::nullable_join::YES,
                                      compare_nulls,
                                      CUCO_DESIRED_LOAD_FACTOR,
                                      stream,
                                      cudf::get_current_device_resource_ref());
    return hj_obj.left_join(left, std::nullopt, stream, mr);
  };

  if (!has_dictionary_columns(left_input)) { return join_tables(left_input, right_input); }

  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input, right_input}, stream, cudf::get_current_device_resource_ref());
  return join_tables(matched.second.front(), matched.second.back());
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
full_join(table_view const& left_input,
          table_view const& right_input,
          null_equality compare_nulls,
          rmm::cuda_stream_view stream,
          rmm::device_async_resource_ref mr)
{
  auto join_tables = [&](table_view const& left, table_view const& right) {
    auto const has_nulls = cudf::has_nested_nulls(left) || cudf::has_nested_nulls(right)
                             ? cudf::nullable_join::YES
                             : cudf::nullable_join::NO;
    cudf::hash_join::impl_type hj_obj(right,
                                      has_nulls == cudf::nullable_join::YES,
                                      compare_nulls,
                                      CUCO_DESIRED_LOAD_FACTOR,
                                      stream,
                                      cudf::get_current_device_resource_ref());
    return hj_obj.full_join(left, std::nullopt, stream, mr);
  };

  if (!has_dictionary_columns(left_input)) { return join_tables(left_input, right_input); }

  auto matched = cudf::dictionary::detail::match_dictionaries(
    {left_input, right_input}, stream, cudf::get_current_device_resource_ref());
  return join_tables(matched.second.front(), matched.second.back());
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
