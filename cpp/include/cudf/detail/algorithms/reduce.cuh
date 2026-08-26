/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/device_scalar.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/stream>

namespace cudf::detail {

/**
 * @brief Helper to reduce a device-accessible iterator using a binary operation
 *
 * This function performs a reduction operation on device data and returns the result
 * to the host using pinned memory for efficient transfer.
 *
 * @tparam Op **[inferred]** Type of the binary reduction operator
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 * @tparam OutputType **[inferred]** Type of the reduction result
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param init Initial value for the reduction
 * @param binary_op Binary reduction operator
 * @param stream CUDA stream to use
 * @return The reduction result
 */
template <typename Op,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
OutputType reduce(
  InputIterator begin, InputIterator end, OutputType init, Op binary_op, cuda::stream_ref stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  // Device scalar to store the result
  auto result =
    cudf::detail::device_scalar<OutputType>(stream, cudf::get_current_device_resource_ref());

  // Build environment with stream and memory resource for cub::DeviceReduce::Reduce
  auto env = cuda::std::execution::env{
    cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream.get()}},
    cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                               cudf::get_current_device_resource_ref()}};
  CUDF_CUDA_TRY(cub::DeviceReduce::Reduce(begin, result.data(), num_items, binary_op, init, env));

  // Copy result back to host via pinned memory
  return result.value(stream);
}

/**
 * @brief Helper to perform reduce-by-key operation using CUB with pinned memory for result transfer
 *
 * This function reduces values by consecutive equal keys, similar to thrust::reduce_by_key,
 * but uses CUB's implementation with pinned memory for efficient device-to-host transfer
 * of the number of unique keys.
 *
 * @tparam Op **[inferred]** Type of the binary reduction operator
 * @tparam KeysInputIterator **[inferred]** Type of device-accessible input iterator for keys
 * @tparam KeysOutputIterator **[inferred]** Type of device-accessible output iterator for unique
 * keys
 * @tparam ValuesInputIterator **[inferred]** Type of device-accessible input iterator for values
 * @tparam ValuesOutputIterator **[inferred]** Type of device-accessible output iterator for reduced
 * values
 *
 * @param keys_begin Device-accessible iterator to start of input keys
 * @param keys_end Device-accessible iterator to end of input keys
 * @param values_begin Device-accessible iterator to start of input values
 * @param keys_output Device-accessible iterator to start of output unique keys
 * @param values_output Device-accessible iterator to start of output reduced values
 * @param op Binary reduction operator
 * @param stream CUDA stream to use
 * @return A pair of iterators pointing to the end of the output key and value ranges
 */
template <typename Op,
          typename KeysInputIterator,
          typename KeysOutputIterator,
          typename ValuesInputIterator,
          typename ValuesOutputIterator>
cuda::std::pair<KeysOutputIterator, ValuesOutputIterator> reduce_by_key(
  KeysInputIterator keys_begin,
  KeysInputIterator keys_end,
  ValuesInputIterator values_begin,
  KeysOutputIterator keys_output,
  ValuesOutputIterator values_output,
  Op op,
  cuda::stream_ref stream)
{
  auto const num_items = cuda::std::distance(keys_begin, keys_end);

  // Device scalar to store the number of runs (unique keys)
  auto d_num_runs =
    cudf::detail::device_scalar<cuda::std::size_t>(stream, cudf::get_current_device_resource_ref());

  // First call to get temporary storage size
  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceReduce::ReduceByKey(nullptr,
                                               temp_storage_bytes,
                                               keys_begin,
                                               keys_output,
                                               values_begin,
                                               values_output,
                                               d_num_runs.data(),
                                               op,
                                               num_items,
                                               stream.get()));

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  // Run reduce-by-key
  CUDF_CUDA_TRY(cub::DeviceReduce::ReduceByKey(d_temp_storage.data(),
                                               temp_storage_bytes,
                                               keys_begin,
                                               keys_output,
                                               values_begin,
                                               values_output,
                                               d_num_runs.data(),
                                               op,
                                               num_items,
                                               stream.get()));

  // Copy number of runs back to host via pinned memory
  auto const num_runs = d_num_runs.value(stream);

  return {keys_output + num_runs, values_output + num_runs};
}

/**
 * @copydoc cudf::detail::reduce_by_key
 *
 * This function performs the reduce-by-key operation asynchronously.
 * It is useful when the calling function does not need the returned result
 * and therefore prevents a stream synchronization.
 *
 */
template <typename Op,
          typename KeysInputIterator,
          typename KeysOutputIterator,
          typename ValuesInputIterator,
          typename ValuesOutputIterator>
void reduce_by_key_async(KeysInputIterator keys_begin,
                         KeysInputIterator keys_end,
                         ValuesInputIterator values_begin,
                         KeysOutputIterator keys_output,
                         ValuesOutputIterator values_output,
                         Op op,
                         cuda::stream_ref stream)
{
  auto const num_items = cuda::std::distance(keys_begin, keys_end);

  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceReduce::ReduceByKey(nullptr,
                                               temp_storage_bytes,
                                               keys_begin,
                                               keys_output,
                                               values_begin,
                                               values_output,
                                               cuda::make_discard_iterator(),
                                               op,
                                               num_items,
                                               stream.get()));

  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  CUDF_CUDA_TRY(cub::DeviceReduce::ReduceByKey(d_temp_storage.data(),
                                               temp_storage_bytes,
                                               keys_begin,
                                               keys_output,
                                               values_begin,
                                               values_output,
                                               cuda::make_discard_iterator(),
                                               op,
                                               num_items,
                                               stream.get()));
}

/**
 * @brief Helper to transform device-accessible iterator using a unary operation and then reduce the
 * transformed values using a binary operation
 *
 * This function applies a unary transformation to each element in the input range, then
 * performs a reduction operation on the transformed values. The result is returned to the
 * host using pinned memory for efficient transfer.
 *
 * @tparam TransformationOp **[inferred]** Type of the unary transformation operator
 * @tparam ReductionOp **[inferred]** Type of the binary reduction operator
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 * @tparam OutputType **[inferred]** Type of the reduction result
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param transform_op Unary transformation operator to apply to each element
 * @param init Initial value for the reduction
 * @param reduce_op Binary reduction operator
 * @param stream CUDA stream to use
 * @return The reduction result
 */
template <typename TransformationOp,
          typename ReductionOp,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
OutputType transform_reduce(InputIterator begin,
                            InputIterator end,
                            TransformationOp transform_op,
                            OutputType init,
                            ReductionOp reduce_op,
                            cuda::stream_ref stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  // Device scalar to store the result
  auto result =
    cudf::detail::device_scalar<OutputType>(stream, cudf::get_current_device_resource_ref());

  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceReduce::TransformReduce(nullptr,
                                                   temp_storage_bytes,
                                                   begin,
                                                   result.data(),
                                                   num_items,
                                                   reduce_op,
                                                   transform_op,
                                                   init,
                                                   stream.get()));

  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(cub::DeviceReduce::TransformReduce(d_temp_storage.data(),
                                                   temp_storage_bytes,
                                                   begin,
                                                   result.data(),
                                                   num_items,
                                                   reduce_op,
                                                   transform_op,
                                                   init,
                                                   stream.get()));

  // Copy result back to host via pinned memory
  return result.value(stream);
}

/**
 * @brief Check if a predicate is true for all elements in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns true for all elements.
 *
 * @tparam TransformOp **[inferred]** Type of the unary transformation operator
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is true for all elements, false otherwise
 */
template <typename TransformOp, typename InputIterator>
bool all_of(InputIterator begin, InputIterator end, TransformOp op, cuda::stream_ref stream)
{
  return transform_reduce(begin, end, op, true, cuda::std::logical_and<bool>{}, stream);
}

/**
 * @brief Check if a predicate is true for any element in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns true for at least one element.
 *
 * @tparam TransformOp **[inferred]** Type of the unary transformation operator
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is true for any element, false otherwise
 */
template <typename TransformOp, typename InputIterator>
bool any_of(InputIterator begin, InputIterator end, TransformOp op, cuda::stream_ref stream)
{
  return transform_reduce(begin, end, op, false, cuda::std::logical_or<bool>{}, stream);
}

/**
 * @brief Check if a predicate is false for all elements in a device-accessible range
 *
 * This function applies a predicate to all elements in the range and returns true
 * if the predicate returns false for all elements (i.e., no element satisfies the predicate).
 *
 * @tparam TransformOp **[inferred]** Type of the predicate operator
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param op Predicate operator to apply to each element
 * @param stream CUDA stream to use
 * @return true if the predicate is false for all elements, false otherwise
 */
template <typename TransformOp, typename InputIterator>
bool none_of(InputIterator begin, InputIterator end, TransformOp op, cuda::stream_ref stream)
{
  return not any_of(begin, end, op, stream);
}

/**
 * @brief Helper to count elements satisfying a predicate using CUB with pinned memory
 *
 * This function counts elements in the input range that satisfy the given predicate,
 * using CUB's transform_reduce implementation with pinned memory for efficient
 * device-to-host transfer of the count.
 *
 * @tparam Predicate **[inferred]** Type of the unary predicate
 * @tparam InputIterator **[inferred]** Type of device-accessible input iterator
 *
 * @param begin Device-accessible iterator to start of input values
 * @param end Device-accessible iterator to end of input values
 * @param predicate Unary predicate that returns true for elements to count
 * @param stream CUDA stream to use
 * @return The count of elements satisfying the predicate
 */
template <typename Predicate, typename InputIterator>
cuda::std::size_t count_if(InputIterator begin,
                           InputIterator end,
                           Predicate predicate,
                           cuda::stream_ref stream)
{
  // Transform each element to 0 or 1 based on predicate, then sum
  auto transform_op = [predicate] __device__(auto const& val) -> cuda::std::size_t {
    return predicate(val) ? cuda::std::size_t{1} : cuda::std::size_t{0};
  };

  return transform_reduce(
    begin, end, transform_op, cuda::std::size_t{0}, cuda::std::plus<>{}, stream);
}

}  // namespace cudf::detail
