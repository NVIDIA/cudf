/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compression.hpp"

#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <nvcomp.h>

#include <optional>

namespace cudf::io::detail::nvcomp {

struct batched_args {
  rmm::device_uvector<void const*> input_data_ptrs;
  rmm::device_uvector<size_t> input_data_sizes;
  rmm::device_uvector<void*> output_data_ptrs;
  rmm::device_uvector<size_t> output_data_sizes;
};

/**
 * @brief Split lists of src/dst device spans into lists of pointers/sizes.
 *
 * @param[in] inputs List of input buffers
 * @param[in] outputs List of output buffers
 * @param[in] stream CUDA stream to use
 */
batched_args create_batched_nvcomp_args(device_span<device_span<uint8_t const> const> inputs,
                                        device_span<device_span<uint8_t> const> outputs,
                                        cuda::stream_ref stream);

/**
 * @brief Prepares device arrays of input pointers and sizes for use with nvCOMP temp size APIs.
 */
std::pair<rmm::device_uvector<void const*>, rmm::device_uvector<size_t>> create_get_temp_size_args(
  device_span<device_span<uint8_t const> const> inputs, cuda::stream_ref stream);

/**
 * @brief Convert nvcomp statuses and output sizes into cuIO compression results.
 */
void update_compression_results(device_span<nvcompStatus_t const> nvcomp_stats,
                                device_span<size_t const> actual_output_sizes,
                                device_span<codec_exec_result> results,
                                cuda::stream_ref stream);

/**
 * @brief Fill the result array based on the actual output sizes.
 */
void update_compression_results(device_span<size_t const> actual_output_sizes,
                                device_span<codec_exec_result> results,
                                cuda::stream_ref stream);

/**
 * @brief Mark unsupported input chunks for skipping.
 */
void skip_unsupported_inputs(device_span<size_t> input_sizes,
                             device_span<codec_exec_result> results,
                             std::optional<size_t> max_valid_input_size,
                             cuda::stream_ref stream);

/**
 * @brief Returns the size of the largest input chunk and the total input size.
 */
std::pair<size_t, size_t> max_chunk_and_total_input_size(device_span<size_t const> input_sizes,
                                                         cuda::stream_ref stream);

}  // namespace cudf::io::detail::nvcomp
