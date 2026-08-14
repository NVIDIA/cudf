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

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/binary_search.h>

namespace cudf {
namespace detail {

rmm::device_uvector<size_type> get_segment_indices(size_type num_rows,
                                                   column_view const& offsets,
                                                   cuda::stream_ref stream)
{
  rmm::device_uvector<size_type> segment_ids(num_rows, stream);

  auto offset_begin  = offsets.begin<size_type>();
  auto offset_end    = offsets.end<size_type>();
  auto counting_iter = cuda::counting_iterator<size_type>{0};
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    counting_iter,
                    counting_iter + segment_ids.size(),
                    segment_ids.begin(),
                    [offset_begin, offset_end] __device__(auto idx) {
                      if (offset_begin == offset_end || idx < *offset_begin) { return idx; }
                      if (idx >= *(offset_end - 1)) { return idx + 1; }
                      return static_cast<size_type>(
                        *thrust::upper_bound(thrust::seq, offset_begin, offset_end, idx));
                    });
  return segment_ids;
}

std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  return segmented_sorted_order_common<sort_method::UNSTABLE>(
    keys, segment_offsets, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  return segmented_sort_by_key_common<sort_method::UNSTABLE>(
    values, keys, segment_offsets, column_order, null_precedence, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_sorted_order(
    keys, segment_offsets, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_sort_by_key(
    values, keys, segment_offsets, column_order, null_precedence, stream, mr);
}

}  // namespace cudf
