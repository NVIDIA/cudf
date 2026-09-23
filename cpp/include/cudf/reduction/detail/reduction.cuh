/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reduction_operators.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/cast_functor.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <optional>

namespace cudf {
namespace reduction {
namespace detail {
/**
 * @brief Compute the specified simple reduction over the input range of elements.
 *
 * @tparam Op               the reduction operator with device binary operator
 * @tparam InputIterator    the input column iterator
 * @tparam OutputType       the output type of reduction
 *
 * @param d_in      the begin iterator
 * @param num_items the number of items
 * @param op        the reduction operator
 * @param init      Optional initial value of the reduction
 * @param stream    CUDA stream used for device memory operations and kernel launches
 * @param mr        Device memory resource used to allocate the returned scalar's device memory
 * @returns Output scalar in device memory
 */
template <typename Op,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::simple_op<Op> op,
                               std::optional<OutputType> init,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
  requires(is_fixed_width<OutputType>() && not cudf::is_fixed_point<OutputType>())
{
  auto const binary_op     = cudf::detail::cast_functor<OutputType>(op.get_binary_op());
  auto const initial_value = init.value_or(op.template get_identity<OutputType>());
  using ScalarType         = cudf::scalar_type_t<OutputType>;
  auto result              = std::make_unique<ScalarType>(initial_value, true, stream, mr);

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            result->data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.get());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            result->data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.get());
  return result;
}

template <typename Op,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::simple_op<Op> op,
                               std::optional<OutputType> init,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
  requires(is_fixed_point<OutputType>())
{
  CUDF_FAIL(
    "This function should never be called. fixed_point reduce should always go through the reduce "
    "for the corresponding device_storage_type_t");
}

// @brief string_view specialization of simple reduction
template <typename Op,
          typename InputIterator,
          typename OutputType = cuda::std::iter_value_t<InputIterator>>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::simple_op<Op> op,
                               std::optional<OutputType> init,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
  requires(std::is_same_v<OutputType, string_view>)
{
  auto const binary_op     = cudf::detail::cast_functor<OutputType>(op.get_binary_op());
  auto const initial_value = init.value_or(op.template get_identity<OutputType>());
  auto dev_result          = cudf::detail::device_scalar<OutputType>{
    initial_value, stream, cudf::get_current_device_resource_ref()};

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            dev_result.data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.get());
  d_temp_storage =
    rmm::device_buffer{temp_storage_bytes, stream, cudf::get_current_device_resource_ref()};

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            dev_result.data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.get());

  return std::make_unique<cudf::string_scalar>(dev_result.value(stream), true, stream, mr);
}

/**
 * @brief compute reduction by the compound operator (reduce and transform)
 *
 * The reduction operator must have `intermediate::compute_result()` method.
 * This method performs reduction using binary operator `Op::Op` and transforms the
 * result to `OutputType` using `compute_result()` transform method.
 *
 * @tparam Op               the reduction operator with device binary operator
 * @tparam InputIterator    the input column iterator
 * @tparam OutputType       the output type of reduction
 *
 * @param d_in        the begin iterator
 * @param num_items   the number of items
 * @param op          the reduction operator
 * @param valid_count Number of valid items
 * @param ddof        Delta degrees of freedom used for standard deviation and variance
 * @param stream      CUDA stream used for device memory operations and kernel launches
 * @param mr          Device memory resource used to allocate the returned scalar's device memory
 * @returns Output scalar in device memory
 */
template <typename Op,
          typename InputIterator,
          typename OutputType,
          typename IntermediateType = cuda::std::iter_value_t<InputIterator>>
std::unique_ptr<scalar> reduce(InputIterator d_in,
                               cudf::size_type num_items,
                               op::compound_op<Op> op,
                               cudf::size_type valid_count,
                               cudf::size_type ddof,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  auto const binary_op     = cudf::detail::cast_functor<IntermediateType>(op.get_binary_op());
  auto const initial_value = op.template get_identity<IntermediateType>();

  cudf::detail::device_scalar<IntermediateType> intermediate_result{
    initial_value, stream, cudf::get_current_device_resource_ref()};

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            intermediate_result.data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.get());
  d_temp_storage =
    rmm::device_buffer{temp_storage_bytes, stream, cudf::get_current_device_resource_ref()};

  // Run reduction
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_in,
                            intermediate_result.data(),
                            num_items,
                            binary_op,
                            initial_value,
                            stream.get());

  // compute the result value from intermediate value in device
  using ScalarType = cudf::scalar_type_t<OutputType>;
  auto result      = std::make_unique<ScalarType>(OutputType{0}, true, stream, mr);
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     intermediate_result.data(),
                     1,
                     [dres = result->data(), op, valid_count, ddof] __device__(auto i) {
                       *dres = op.template compute_result<OutputType>(i, valid_count, ddof);
                     });
  return result;
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
