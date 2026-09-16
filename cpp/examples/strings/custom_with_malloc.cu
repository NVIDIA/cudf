/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

/**
 * @brief Reserve CUDA malloc heap size
 *
 * Call this function to change the CUDA malloc heap size limit.
 * This value depends on the total size of all the malloc()
 * calls needed for redact_kernel.
 *
 * @param heap_size Number of bytes to reserve
 *                  Default is 1GB
 */
void set_malloc_heap_size(size_t heap_size = 1073741824)  // 1GB
{
  size_t max_malloc_heap_size = 0;
  cudaDeviceGetLimit(&max_malloc_heap_size, cudaLimitMallocHeapSize);
  if (max_malloc_heap_size < heap_size) {
    max_malloc_heap_size = heap_size;
    if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_malloc_heap_size) != cudaSuccess) {
      fprintf(stderr, "could not set malloc heap size to %ldMB\n", (heap_size / (1024 * 1024)));
      throw std::runtime_error("");
    }
  }
}

/**
 * @brief Builds the output for each row
 *
 * This thread is called once per row in d_names.
 *
 * Note: This uses malloc() in a device kernel which works great
 * but is not very efficient. This can be useful for prototyping
 * on functions where performance is not yet important.
 * All calls to malloc() must have a corresponding free() call.
 * The separate free_kernel is launched for this purpose.
 *
 * @param d_names Column of names
 * @param d_visibilities Column of visibilities
 * @param redaction Redacted string replacement
 * @param d_output Output array of string_view objects
 */
__global__ static void redact_kernel(cudf::column_device_view const d_names,
                                     cudf::column_device_view const d_visibilities,
                                     cudf::string_view redaction,
                                     cudf::string_view* d_output)
{
  // The row index is resolved from the CUDA thread/block objects
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  // There may be more threads than actual rows
  if (index >= d_names.size()) return;

  auto const visible = cudf::string_view("public", 6);

  auto const name = d_names.element<cudf::string_view>(index);
  auto const vis  = d_visibilities.element<cudf::string_view>(index);
  if (vis == visible) {
    auto const space_idx    = name.find(' ');
    auto const first        = name.substr(0, space_idx);
    auto const last_initial = name.substr(space_idx + 1, 1);
    auto const output_size  = first.size_bytes() + last_initial.size_bytes() + 1;

    char* output_ptr = static_cast<char*>(malloc(output_size));
    d_output[index]  = cudf::string_view{output_ptr, output_size};

    // build output string
    memcpy(output_ptr, last_initial.data(), last_initial.size_bytes());
    output_ptr += last_initial.size_bytes();
    *output_ptr++ = ' ';
    memcpy(output_ptr, first.data(), first.size_bytes());
  } else {
    d_output[index] = cudf::string_view{redaction.data(), redaction.size_bytes()};
  }
}

/**
 * @brief Frees the temporary individual string objects created in the
 * redact_kernel
 *
 * Like malloc(), free() is not very efficient but must be called for
 * each malloc() to return the memory to the CUDA malloc heap.
 *
 * @param redaction Redacted string replacement (not to be freed)
 * @param d_output Output array of string_view objects to free
 */
__global__ static void free_kernel(cudf::string_view redaction,
                                   cudf::string_view* d_output,
                                   int count)
{
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= count) return;

  auto ptr = const_cast<char*>(d_output[index].data());
  if (ptr != redaction.data()) { free(ptr); }
}

std::unique_ptr<cudf::column> redact_strings(cudf::column_view const& names,
                                             cudf::column_view const& visibilities)
{
  // all device memory operations and kernel functions will run on this stream
  cuda::stream_ref stream = cudf::get_default_stream();

  set_malloc_heap_size();  // to illustrate adjusting the malloc heap

  auto const d_names        = cudf::column_device_view::create(names, stream);
  auto const d_visibilities = cudf::column_device_view::create(visibilities, stream);
  auto const d_redaction    = cudf::string_scalar(std::string("X X"), true, stream);

  constexpr int block_size = 128;  // this arbitrary size should be a power of 2
  auto const blocks        = (names.size() + block_size - 1) / block_size;

  nvtxRangePushA("redact_strings");

  // create a vector for the output strings' pointers
  auto str_ptrs = new rmm::device_uvector<cudf::string_view>(names.size(), stream);

  auto result = [&] {
    // build the output strings
    redact_kernel<<<blocks, block_size, 0, stream.get()>>>(
      *d_names, *d_visibilities, d_redaction.value(), str_ptrs->data());
    // create strings column from the string_view vector
    // this copies all the individual strings into a single output column
    return cudf::make_strings_column(*str_ptrs, cudf::string_view{nullptr, 0}, stream);
  }();

  // free the individual temporary memory pointers
  free_kernel<<<blocks, block_size, 0, stream.get()>>>(
    d_redaction.value(), str_ptrs->data(), names.size());
  delete str_ptrs;

  // wait for all of the above to finish
  stream.sync();

  nvtxRangePop();
  return result;
}
