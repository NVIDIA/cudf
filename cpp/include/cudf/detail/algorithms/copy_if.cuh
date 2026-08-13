/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/device_scalar.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_select.cuh>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/stream_ref>
#include <thrust/copy.h>

namespace cudf::detail {

/**
 * @brief Helper to copy elements satisfying a predicate/stencil using CUB with pinned memory
 *
 * This function copies elements from the input range that satisfy the given predicate/stencil
 * to the output range, using CUB's DeviceSelect::FlaggedIf implementation with pinned memory
 * for efficient device-to-host transfer of the number of selected elements.
 *
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 * @tparam StencilIterator **[inferred]** Type of device-accessible stencil iterator
 * @tparam OutputIterator **[inferred]** Type of device-accessible output iterator
 * @tparam Predicate **[inferred]** Type of the unary predicate
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param stencil Device-accessible iterator to start of stencil values
 * @param result Device-accessible iterator to start of output values
 * @param predicate Unary predicate that returns true for elements to copy
 * @param stream CUDA stream to use
 * @return Iterator pointing to the end of the output range
 */
template <typename InputIterator,
          typename StencilIterator,
          typename OutputIterator,
          typename Predicate>
OutputIterator copy_if(InputIterator begin,
                       InputIterator end,
                       StencilIterator stencil,
                       OutputIterator result,
                       Predicate predicate,
                       rmm::cuda_stream_view stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  auto num_selected =
    cudf::detail::device_scalar<cuda::std::size_t>(stream, cudf::get_current_device_resource_ref());

  auto temp_storage_bytes = std::size_t{0};
  CUDF_CUDA_TRY(cub::DeviceSelect::FlaggedIf(nullptr,
                                             temp_storage_bytes,
                                             begin,
                                             stencil,
                                             result,
                                             num_selected.data(),
                                             num_items,
                                             predicate,
                                             stream.value()));

  auto d_temp_storage =
    rmm::device_buffer(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  CUDF_CUDA_TRY(cub::DeviceSelect::FlaggedIf(d_temp_storage.data(),
                                             temp_storage_bytes,
                                             begin,
                                             stencil,
                                             result,
                                             num_selected.data(),
                                             num_items,
                                             predicate,
                                             stream.value()));

  return result + num_selected.value(stream);
}

/**
 * @brief Helper to copy elements satisfying a predicate using CUB with pinned memory
 *
 * This function copies elements from the input range that satisfy the given predicate
 * to the output range, using CUB's DeviceSelect::If implementation with pinned memory
 * for efficient device-to-host transfer of the number of selected elements.
 *
 * @tparam Predicate **[inferred]** Type of the unary predicate
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 * @tparam OutputIterator **[inferred]** Type of device-accessible output iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param output Device-accessible iterator to start of output values
 * @param predicate Unary predicate that returns true for elements to copy
 * @param stream CUDA stream to use
 * @return Iterator pointing to the end of the output range
 */
template <typename Predicate, typename InputIterator, typename OutputIterator>
OutputIterator copy_if(InputIterator begin,
                       InputIterator end,
                       OutputIterator output,
                       Predicate predicate,
                       rmm::cuda_stream_view stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  // Device scalar to store the number of selected elements
  auto num_selected =
    cudf::detail::device_scalar<cuda::std::size_t>(stream, cudf::get_current_device_resource_ref());

  // First call to get temporary storage size
  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                      temp_storage_bytes,
                                      begin,
                                      output,
                                      num_selected.data(),
                                      num_items,
                                      predicate,
                                      stream.value()));

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  // Run copy_if
  CUDF_CUDA_TRY(cub::DeviceSelect::If(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      begin,
                                      output,
                                      num_selected.data(),
                                      num_items,
                                      predicate,
                                      stream.value()));

  // Copy number of selected elements back to host via pinned memory
  return output + num_selected.value(stream);
}

/**
 * @copydoc cudf::detail::copy_if
 *
 * This function performs the copy_if operation asynchronously.
 * It is useful when the calling function does not need the returned result
 * and therefore prevents a stream synchronization.
 */
template <typename Predicate, typename InputIterator, typename OutputIterator>
void copy_if_async(InputIterator begin,
                   InputIterator end,
                   OutputIterator output,
                   Predicate predicate,
                   rmm::cuda_stream_view stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  auto tmp_bytes = std::size_t{0};
  auto no_out    = cuda::make_discard_iterator<int>();
  CUDF_CUDA_TRY(cub::DeviceSelect::If(
    nullptr, tmp_bytes, begin, output, no_out, num_items, predicate, stream.value()));

  auto tmp_stg = rmm::device_buffer(tmp_bytes, stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(cub::DeviceSelect::If(
    tmp_stg.data(), tmp_bytes, begin, output, no_out, num_items, predicate, stream.value()));
}

/**
 * @copydoc cudf::detail::copy_if
 *
 * This function performs the copy_if operation asynchronously.
 * It is useful when the calling function does not need the returned result
 * and therefore prevents a stream synchronization.
 */
template <typename InputIterator,
          typename StencilIterator,
          typename OutputIterator,
          typename Predicate>
void copy_if_async(InputIterator begin,
                   InputIterator end,
                   StencilIterator stencil,
                   OutputIterator result,
                   Predicate predicate,
                   rmm::cuda_stream_view stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  auto tmp_bytes = std::size_t{0};
  auto no_out    = cuda::make_discard_iterator<int>();
  CUDF_CUDA_TRY(cub::DeviceSelect::FlaggedIf(
    nullptr, tmp_bytes, begin, stencil, result, no_out, num_items, predicate, stream.value()));

  auto tmp = rmm::device_buffer(tmp_bytes, stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(cub::DeviceSelect::FlaggedIf(
    tmp.data(), tmp_bytes, begin, stencil, result, no_out, num_items, predicate, stream.value()));
}

}  // namespace cudf::detail
