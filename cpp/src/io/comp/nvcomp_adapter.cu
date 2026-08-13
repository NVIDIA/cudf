/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "nvcomp_adapter.cuh"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/transform.h>

namespace cudf::io::detail::nvcomp {

batched_args create_batched_nvcomp_args(device_span<device_span<uint8_t const> const> inputs,
                                        device_span<device_span<uint8_t> const> outputs,
                                        rmm::cuda_stream_view stream)
{
  auto const num_comp_chunks = inputs.size();
  rmm::device_uvector<void const*> input_data_ptrs(num_comp_chunks, stream);
  rmm::device_uvector<size_t> input_data_sizes(num_comp_chunks, stream);
  rmm::device_uvector<void*> output_data_ptrs(num_comp_chunks, stream);
  rmm::device_uvector<size_t> output_data_sizes(num_comp_chunks, stream);

  // Prepare the input vectors
  auto ins_it = cuda::make_zip_iterator(input_data_ptrs.begin(), input_data_sizes.begin());
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    inputs.begin(),
    inputs.end(),
    ins_it,
    [] __device__(auto const& in) { return cuda::std::make_tuple(in.data(), in.size()); });

  // Prepare the output vectors
  auto outs_it = cuda::make_zip_iterator(output_data_ptrs.begin(), output_data_sizes.begin());
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    outputs.begin(),
    outputs.end(),
    outs_it,
    [] __device__(auto const& out) { return cuda::std::make_tuple(out.data(), out.size()); });

  return {std::move(input_data_ptrs),
          std::move(input_data_sizes),
          std::move(output_data_ptrs),
          std::move(output_data_sizes)};
}

std::pair<rmm::device_uvector<void const*>, rmm::device_uvector<size_t>> create_get_temp_size_args(
  device_span<device_span<uint8_t const> const> inputs, rmm::cuda_stream_view stream)
{
  rmm::device_uvector<void const*> input_data_ptrs(inputs.size(), stream);
  rmm::device_uvector<size_t> input_data_sizes(inputs.size(), stream);

  auto ins_it = cuda::make_zip_iterator(input_data_ptrs.begin(), input_data_sizes.begin());
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    inputs.begin(),
    inputs.end(),
    ins_it,
    [] __device__(auto const& in) { return cuda::std::make_tuple(in.data(), in.size()); });

  return {std::move(input_data_ptrs), std::move(input_data_sizes)};
}

void update_compression_results(device_span<nvcompStatus_t const> nvcomp_stats,
                                device_span<size_t const> actual_output_sizes,
                                device_span<codec_exec_result> results,
                                rmm::cuda_stream_view stream)
{
  thrust::transform_if(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    nvcomp_stats.begin(),
    nvcomp_stats.end(),
    actual_output_sizes.begin(),
    results.begin(),
    results.begin(),
    [] __device__(auto const& nvcomp_status, auto const& size) {
      return codec_exec_result{size,
                               nvcomp_status == nvcompStatus_t::nvcompSuccess
                                 ? codec_status::SUCCESS
                                 : codec_status::FAILURE};
    },
    [] __device__(auto const& cudf_status) { return cudf_status.status != codec_status::SKIPPED; });
}

void update_compression_results(device_span<size_t const> actual_output_sizes,
                                device_span<codec_exec_result> results,
                                rmm::cuda_stream_view stream)
{
  thrust::transform_if(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    actual_output_sizes.begin(),
    actual_output_sizes.end(),
    results.begin(),
    results.begin(),
    [] __device__(auto const& size) { return codec_exec_result{size}; },
    [] __device__(auto const& results) { return results.status != codec_status::SKIPPED; });
}

void skip_unsupported_inputs(device_span<size_t> input_sizes,
                             device_span<codec_exec_result> results,
                             std::optional<size_t> max_valid_input_size,
                             rmm::cuda_stream_view stream)
{
  if (max_valid_input_size.has_value()) {
    auto status_size_it = cuda::make_zip_iterator(input_sizes.begin(), results.begin());
    thrust::transform_if(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      results.begin(),
      results.end(),
      input_sizes.begin(),
      status_size_it,
      [] __device__(auto const& status) {
        return cuda::std::pair{0, codec_exec_result{0, codec_status::SKIPPED}};
      },
      [max_size = max_valid_input_size.value()] __device__(size_t input_size) {
        return input_size > max_size;
      });
  }
}
std::pair<size_t, size_t> max_chunk_and_total_input_size(device_span<size_t const> input_sizes,
                                                         rmm::cuda_stream_view stream)
{
  auto const max =
    thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   input_sizes.begin(),
                   input_sizes.end(),
                   0ul,
                   cuda::maximum<size_t>());
  auto const sum =
    thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   input_sizes.begin(),
                   input_sizes.end());
  return {max, sum};
}

}  // namespace cudf::io::detail::nvcomp
