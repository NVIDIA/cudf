/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/comp/compression.hpp"

#include <cudf/io/detail/nvcomp_adapter.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <optional>

namespace cudf::io::detail::nvcomp {
/**
 * @brief Device batch decompression of given type.
 *
 * @param[in] compression Compression type
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] results List of output status structures
 * @param[in] max_uncomp_chunk_size Maximum size of any single uncompressed chunk
 * @param[in] max_total_uncomp_size Maximum total size of uncompressed data
 * @param[in] stream CUDA stream to use
 */
void batched_decompress(compression_type compression,
                        device_span<device_span<uint8_t const> const> inputs,
                        device_span<device_span<uint8_t> const> outputs,
                        device_span<codec_exec_result> results,
                        size_t max_uncomp_chunk_size,
                        size_t max_total_uncomp_size,
                        cuda::stream_ref stream);

/**
 * @brief Return the amount of temporary space required in bytes for a given decompression
 * operation.
 *
 * The size returned reflects the size of the scratch buffer to be passed to
 * `batched_decompress_async`
 *
 * @param[in] compression Compression type
 * @param[in] num_chunks The number of decompression chunks to be processed
 * @param[in] max_uncomp_chunk_size Maximum size of any single uncompressed chunk
 * @param[in] max_total_uncomp_size Maximum total size of uncompressed data
 * @returns The total required size in bytes
 */
size_t batched_decompress_temp_size(compression_type compression,
                                    size_t num_chunks,
                                    size_t max_uncomp_chunk_size,
                                    size_t max_total_uncomp_size);

/**
 * @brief Return the amount of temporary space required in bytes for a given decompression
 * operation using synchronous nvcomp APIs.
 *
 * The size returned reflects the size of the scratch buffer to be passed to
 * `batched_decompress_async`. This version uses the sync APIs which are more precise, but
 * potentially require a kernel launch.
 *
 * @param[in] compression Compression type
 * @param[in] inputs Device span of compressed data chunks
 * @param[in] max_uncomp_chunk_size Maximum size of any single uncompressed chunk
 * @param[in] max_total_uncomp_size Maximum total size of uncompressed data
 * @param[in] stream CUDA stream to use
 * @returns The total required size in bytes
 */
[[nodiscard]] size_t batched_decompress_temp_size_ex(
  compression_type compression,
  device_span<device_span<uint8_t const> const> inputs,
  size_t max_uncomp_chunk_size,
  size_t max_total_uncomp_size,
  cuda::stream_ref stream);

[[nodiscard]] bool is_batched_decompress_temp_size_ex_supported(compression_type compression);

/**
 * @brief Gets the maximum size any chunk could compress to in the batch.
 *
 * @param compression Compression type
 * @param max_uncomp_chunk_size Size of the largest uncompressed chunk in the batch
 */
[[nodiscard]] size_t compress_max_output_chunk_size(compression_type compression,
                                                    size_t max_uncomp_chunk_size);

/**
 * @brief Gets input and output alignment requirements for compression.
 *
 * @param compression Compression type
 * @returns required alignment
 */
[[nodiscard]] size_t compress_required_alignment(compression_type compression);

/**
 * @brief Gets input and output alignment requirements for decompression.
 *
 * @param compression Compression type
 * @returns required alignment
 */
[[nodiscard]] size_t decompress_required_alignment(compression_type compression);

/**
 * @brief Maximum size of uncompressed chunks that can be compressed with nvCOMP.
 *
 * @param compression Compression type
 * @returns maximum chunk size
 */
[[nodiscard]] std::optional<size_t> compress_max_allowed_chunk_size(compression_type compression);

/**
 * @brief Loads the nvCOMP library.
 *
 * Can be used to load the nvCOMP library before its first use. Eager loading can help avoid issues
 * due to the device memory allocations performed during the dynamic loading of the library (e.g.
 * when loading after the memory pools have been created).
 */
void load_nvcomp_library();

/**
 * @brief Device batch compression of given type.
 *
 * @param[in] compression Compression type
 * @param[in] inputs List of input buffers
 * @param[out] outputs List of output buffers
 * @param[out] results List of output status structures
 * @param[in] stream CUDA stream to use
 */
void batched_compress(compression_type compression,
                      device_span<device_span<uint8_t const> const> inputs,
                      device_span<device_span<uint8_t> const> outputs,
                      device_span<codec_exec_result> results,
                      cuda::stream_ref stream);

}  // namespace cudf::io::detail::nvcomp
