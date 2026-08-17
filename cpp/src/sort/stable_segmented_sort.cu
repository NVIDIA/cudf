/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "segmented_sort_impl.cuh"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace detail {

std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return segmented_sorted_order_common<sort_method::STABLE>(
    keys, segment_offsets, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> stable_segmented_sort_by_key(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  return segmented_sort_by_key_common<sort_method::STABLE>(
    values, keys, segment_offsets, column_order, null_precedence, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_segmented_sorted_order(
    keys, segment_offsets, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> stable_segmented_sort_by_key(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_segmented_sort_by_key(
    values, keys, segment_offsets, column_order, null_precedence, stream, mr);
}

}  // namespace cudf
