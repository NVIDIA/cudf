/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reduction_operators.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/cast_functor.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/execution>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <cstdint>
#include <optional>
#include <type_traits>

namespace cudf {
namespace reduction {
namespace detail {
inline auto make_reduction_env(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  using stream_property = cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>;
  using resource_property =
    cuda::std::execution::prop<cuda::mr::get_memory_resource_t, rmm::device_async_resource_ref>;
  return cuda::std::execution::env<stream_property, resource_property>{
    stream_property{cuda::get_stream_t{}, stream},
    resource_property{cuda::mr::get_memory_resource_t{}, mr}};
}

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

  auto env = make_reduction_env(stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(
    cub::DeviceReduce::Reduce(d_in, result->data(), num_items, binary_op, initial_value, env));
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

  auto env = make_reduction_env(stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(
    cub::DeviceReduce::Reduce(d_in, dev_result.data(), num_items, binary_op, initial_value, env));

  return std::make_unique<cudf::string_scalar>(dev_result.value(stream), true, stream, mr);
}

// Preserve CUB's architecture-specific tuning, changing only the collective used
// to combine expensive intermediate states for variance. cudf column sizes use 32-bit CUB offsets.
template <typename State, typename BinaryOp>
struct variance_reduce_policy {
  __host__ __device__ constexpr cub::ReducePolicy operator()(cuda::compute_capability cc) const
  {
    auto policy = cub::detail::reduce::
      policy_selector_from_types<State, std::make_unsigned_t<size_type>, BinaryOp>{}(cc);
    // Combining the states for variance is quite expensive. By using raking block reduce,
    // we use significantly less collective FP64 arithmetic than CUB's default policy.
    policy.multi_tile.reduce_algorithm  = cub::BLOCK_REDUCE_RAKING;
    policy.single_tile.reduce_algorithm = cub::BLOCK_REDUCE_RAKING;
    return policy;
  }
};

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

  auto const env = [&] {
    if constexpr (std::is_same_v<Op, op::variance> || std::is_same_v<Op, op::standard_deviation>) {
      return cuda::std::execution::env{
        make_reduction_env(stream, cudf::get_current_device_resource_ref()),
        cuda::execution::tune(
          variance_reduce_policy<IntermediateType, std::remove_cv_t<decltype(binary_op)>>{})};
    } else {
      return make_reduction_env(stream, cudf::get_current_device_resource_ref());
    }
  }();
  CUDF_CUDA_TRY(cub::DeviceReduce::Reduce(
    d_in, intermediate_result.data(), num_items, binary_op, initial_value, env));

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
