/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "strings/regex/glushkov.cuh"
#include "strings/regex/regex.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

#include <stdexcept>

namespace cudf {
namespace strings {
namespace detail {

constexpr auto regex_launch_kernel_block_size = 256;

template <typename ForEachFunction, typename ProgDevice>
CUDF_KERNEL void for_each_kernel(ForEachFunction fn, ProgDevice const d_prog, size_type size)
{
  extern __shared__ u_char shmem[];
  if (threadIdx.x == 0) { d_prog.store(shmem); }
  __syncthreads();
  auto s_prog = ProgDevice::load(d_prog, shmem);

  auto const thread_idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride     = s_prog.thread_count();
  if (thread_idx < stride) {
    for (auto idx = thread_idx; idx < size; idx += stride) {
      fn(idx, s_prog, thread_idx);
    }
  }
}

template <typename ForEachFunction, typename ProgDevice>
void launch_for_each_kernel(ForEachFunction fn,
                            ProgDevice& d_prog,
                            size_type size,
                            rmm::cuda_stream_view stream)
{
  auto [buffer_size, thread_count] = d_prog.compute_strided_working_memory(size);

  auto d_buffer = rmm::device_buffer(buffer_size, stream);
  d_prog.set_working_memory(d_buffer.data(), thread_count);

  auto const shmem_size = d_prog.compute_shared_memory_size();
  cudf::detail::grid_1d grid{thread_count, regex_launch_kernel_block_size};
  for_each_kernel<<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
    fn, d_prog, size);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename TransformFunction, typename ProgDevice, typename OutputType>
CUDF_KERNEL void transform_kernel(TransformFunction fn,
                                  ProgDevice const d_prog,
                                  OutputType* d_output,
                                  size_type size)
{
  extern __shared__ u_char shmem[];
  if (threadIdx.x == 0) { d_prog.store(shmem); }
  __syncthreads();
  auto s_prog = ProgDevice::load(d_prog, shmem);

  auto const thread_idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride     = s_prog.thread_count();
  if (thread_idx < stride) {
    for (auto idx = thread_idx; idx < size; idx += stride) {
      d_output[idx] = fn(idx, s_prog, thread_idx);
    }
  }
}

template <typename TransformFunction, typename ProgDevice, typename OutputType>
void launch_transform_kernel(TransformFunction fn,
                             ProgDevice& d_prog,
                             OutputType* d_output,
                             size_type size,
                             rmm::cuda_stream_view stream)
{
  auto [buffer_size, thread_count] = d_prog.compute_strided_working_memory(size);

  auto d_buffer = rmm::device_buffer(buffer_size, stream);
  d_prog.set_working_memory(d_buffer.data(), thread_count);

  auto const shmem_size = d_prog.compute_shared_memory_size();
  cudf::detail::grid_1d grid{thread_count, regex_launch_kernel_block_size};
  transform_kernel<<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
    fn, d_prog, d_output, size);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename SizeAndExecuteFunction, typename ProgDevice>
auto make_strings_children(SizeAndExecuteFunction size_and_exec_fn,
                           ProgDevice& d_prog,
                           size_type strings_count,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto output_sizes        = rmm::device_uvector<size_type>(strings_count, stream);
  size_and_exec_fn.d_sizes = output_sizes.data();

  auto [buffer_size, thread_count] = d_prog.compute_strided_working_memory(strings_count);

  auto d_buffer = rmm::device_buffer(buffer_size, stream);
  d_prog.set_working_memory(d_buffer.data(), thread_count);
  auto const shmem_size = d_prog.compute_shared_memory_size();
  cudf::detail::grid_1d grid{thread_count, 256};

  // Compute the output size for each row
  if (strings_count > 0) {
    for_each_kernel<<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
      size_and_exec_fn, d_prog, strings_count);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  // Convert the sizes to offsets
  auto [offsets, char_bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);
  size_and_exec_fn.d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // Now build the chars column
  rmm::device_uvector<char> chars(char_bytes, stream, mr);
  if (char_bytes > 0) {
    size_and_exec_fn.d_chars = chars.data();
    for_each_kernel<<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
      size_and_exec_fn, d_prog, strings_count);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  return std::make_pair(std::move(offsets), std::move(chars));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
