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
 * @brief Builds the output for each row
 *
 * This thread is called once per row in d_names.
 *
 * @param d_names Column of names
 * @param d_visibilities Column of visibilities
 * @param redaction Redacted string replacement
 * @param working_memory Output memory for all rows
 * @param d_offsets Byte offset in `d_chars` for each row
 * @param d_output Output array of string_view objects
 */
__global__ static void redact_kernel(cudf::column_device_view const d_names,
                                     cudf::column_device_view const d_visibilities,
                                     cudf::string_view redaction,
                                     char* working_memory,
                                     cudf::size_type const* d_offsets,
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

    char* output_ptr = working_memory + d_offsets[index];
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
 * @brief Redacts each name per the corresponding visibility entry
 *
 * This implementation builds the individual strings into a fixed memory buffer
 * and then calls a factory function to gather them into a strings column.
 *
 * @param names Column of names
 * @param visibilities Column of visibilities
 * @return Redacted column of names
 */
std::unique_ptr<cudf::column> redact_strings(cudf::column_view const& names,
                                             cudf::column_view const& visibilities)
{
  // all device memory operations and kernel functions will run on this stream
  cuda::stream_ref stream = cudf::get_default_stream();

  auto const d_names        = cudf::column_device_view::create(names, stream);
  auto const d_visibilities = cudf::column_device_view::create(visibilities, stream);
  auto const d_redaction    = cudf::string_scalar(std::string("X X"), true, stream);

  constexpr int block_size = 128;  // this arbitrary size should be a power of 2
  auto const blocks        = (names.size() + block_size - 1) / block_size;

  nvtxRangePushA("redact_strings");

  auto const scv     = cudf::strings_column_view(names);
  auto const offsets = scv.offsets().begin<cudf::size_type>();

  // create working memory to hold the output of each string
  auto working_memory = rmm::device_uvector<char>(scv.chars_size(stream), stream);
  // create a vector for the output strings' pointers
  auto str_ptrs = rmm::device_uvector<cudf::string_view>(names.size(), stream);

  // build the output strings
  redact_kernel<<<blocks, block_size, 0, stream.get()>>>(*d_names,
                                                         *d_visibilities,
                                                         d_redaction.value(),
                                                         working_memory.data(),
                                                         offsets,
                                                         str_ptrs.data());

  // create strings column from the string_pairs;
  // this copies all the individual strings into a single output column
  auto result = cudf::make_strings_column(str_ptrs, cudf::string_view{nullptr, 0}, stream);
  // temporary memory cleanup cost here for str_ptrs and working_memory

  // wait for all of the above to finish
  stream.sync();

  nvtxRangePop();
  return result;
}
