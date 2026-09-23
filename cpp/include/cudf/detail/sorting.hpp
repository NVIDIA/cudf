/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>
#include <vector>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> sorted_order(table_view const& input,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> stable_sorted_order(table_view const& input,
                                            std::vector<order> const& column_order,
                                            std::vector<null_order> const& null_precedence,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> sort_by_key(table_view const& values,
                                   table_view const& keys,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::rank
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> rank(column_view const& input,
                             rank_method method,
                             order column_order,
                             null_policy null_handling,
                             null_order null_precedence,
                             bool percentage,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> stable_sort_by_key(table_view const& values,
                                          table_view const& keys,
                                          std::vector<order> const& column_order,
                                          std::vector<null_order> const& null_precedence,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::segmented_sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_segmented_sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order,
  std::vector<null_order> const& null_precedence,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::segmented_sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_segmented_sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> stable_segmented_sort_by_key(table_view const& values,
                                                    table_view const& keys,
                                                    column_view const& segment_offsets,
                                                    std::vector<order> const& column_order,
                                                    std::vector<null_order> const& null_precedence,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::sort
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> sort(table_view const& values,
                            std::vector<order> const& column_order,
                            std::vector<null_order> const& null_precedence,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_sort
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> stable_sort(table_view const& values,
                                   std::vector<order> const& column_order,
                                   std::vector<null_order> const& null_precedence,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::segmented_top_k
 *
 */
std::unique_ptr<column> segmented_top_k(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order topk_order,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
