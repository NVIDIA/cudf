/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "jit/cache.hpp"
#include "jit/helpers.hpp"
#include "jit/parser.hpp"
#include "jit/util.hpp"
#include "rolling.hpp"
#include "rolling_jit.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace detail {

template <typename T>
std::string reflect_window_wrapper()
{
  if constexpr (std::is_same_v<T, fixed_window_wrapper>) {
    return "cudf::detail::fixed_window_wrapper";
  } else if constexpr (std::is_same_v<T, variable_window_wrapper>) {
    return "cudf::detail::variable_window_wrapper";
  } else if constexpr (std::is_same_v<T, preceding_window_wrapper>) {
    return "cudf::detail::preceding_window_wrapper";
  } else {
    static_assert(std::is_same_v<T, following_window_wrapper>, "Unsupported window wrapper type");
    return "cudf::detail::following_window_wrapper";
  }
}

inline std::unique_ptr<column> rolling_window_udf_impl(
  column_view const& input,
  std::string const& preceding_window_str,
  cudf::detail::window_wrapper_base const& preceding_window,
  std::string const& following_window_str,
  cudf::detail::window_wrapper_base const& following_window,
  size_type min_periods,
  rolling_aggregation const& agg,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "bitmask_type size does not match CUDA warp size");

  if (input.has_nulls()) {
    CUDF_FAIL("Currently the UDF version of rolling window does NOT support inputs with nulls.");
  }

  min_periods = std::max(min_periods, 0);

  auto& udf_agg = dynamic_cast<udf_aggregation const&>(agg);

  std::string hash = "prog_rolling." + std::to_string(std::hash<std::string>{}(udf_agg._source));

  std::string cuda_source;
  switch (udf_agg.kind) {
    case aggregation::Kind::PTX:
      cuda_source +=
        cudf::jit::parse_single_function_ptx(udf_agg._source,
                                             udf_agg._function_name,
                                             {{0, cudf::type_to_name(udf_agg._output_type) + " *"},
                                              {5, "void const *"}});  // args 0 and 5 are pointers
      break;
    case aggregation::Kind::CUDA:
      cuda_source += cudf::jit::parse_single_function_cuda(udf_agg._source, udf_agg._function_name);
      break;
    default: CUDF_FAIL("Unsupported UDF type.");
  }

  std::unique_ptr<column> output = make_numeric_column(
    udf_agg._output_type, input.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  auto output_view = output->mutable_view();
  cudf::detail::device_scalar<size_type> device_valid_count{
    0, stream, cudf::get_current_device_resource_ref()};

  std::string kernel_reflection =
    rtcx::reflect_template("cudf::rolling::jit::rolling_window_kernel",
                           cudf::type_to_name(input.type()),  // list of template arguments
                           cudf::type_to_name(output->type()),
                           udf_agg._operator_name,
                           preceding_window_str,
                           following_window_str);

  auto kernel =
    cudf::jit::get_udf_kernel("cudf/cpp/src/rolling/jit/kernel.cu", kernel_reflection, cuda_source);
  auto cfg = kernel.max_occupancy_config(0, 0);
  kernel.launch_with({cfg.min_grid_size},
                     {cfg.block_size},
                     0,
                     stream,
                     input.size(),
                     cudf::jit::get_data_ptr(input),
                     input.null_mask(),
                     cudf::jit::get_data_ptr(output->mutable_view()),
                     output_view.null_mask(),
                     device_valid_count.data(),
                     preceding_window,
                     following_window,
                     min_periods);

  output->set_null_count(output->size() - device_valid_count.value(stream));

  // check the stream for debugging
  CUDF_CHECK_CUDA(stream.value());

  return output;
}

// Applies a user-defined rolling window function to the values in a column.
template <typename PrecedingWindowIterator, typename FollowingWindowIterator>
std::unique_ptr<column> rolling_window_udf(column_view const& input,
                                           PrecedingWindowIterator preceding_window,
                                           FollowingWindowIterator following_window,
                                           size_type min_periods,
                                           rolling_aggregation const& agg,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  return rolling_window_udf_impl(input,
                                 reflect_window_wrapper<PrecedingWindowIterator>(),
                                 preceding_window,
                                 reflect_window_wrapper<FollowingWindowIterator>(),
                                 following_window,
                                 min_periods,
                                 agg,
                                 stream,
                                 mr);
}

}  // namespace detail
}  // namespace cudf
