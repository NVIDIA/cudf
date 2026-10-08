/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/device_scalar.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/device/device_select.cuh>
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream>
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
                       cuda::stream_ref stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  auto num_selected =
    cudf::detail::device_scalar<cuda::std::size_t>(stream, cudf::get_current_device_resource_ref());

  auto env =
    cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream},
                              cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                         cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(cub::DeviceSelect::FlaggedIf(
    begin, stencil, result, num_selected.data(), num_items, predicate, env));

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
                       cuda::stream_ref stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  // Device scalar to store the number of selected elements
  auto num_selected =
    cudf::detail::device_scalar<cuda::std::size_t>(stream, cudf::get_current_device_resource_ref());

  auto env =
    cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream},
                              cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                         cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(
    cub::DeviceSelect::If(begin, output, num_selected.data(), num_items, predicate, env));

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
                   cuda::stream_ref stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  auto no_out = cuda::make_discard_iterator<int>();
  auto env =
    cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream},
                              cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                         cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(cub::DeviceSelect::If(begin, output, no_out, num_items, predicate, env));
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
                   cuda::stream_ref stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  auto no_out = cuda::make_discard_iterator<int>();
  auto env =
    cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream_t{}, stream},
                              cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                         cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(
    cub::DeviceSelect::FlaggedIf(begin, stencil, result, no_out, num_items, predicate, env));
}

}  // namespace cudf::detail
