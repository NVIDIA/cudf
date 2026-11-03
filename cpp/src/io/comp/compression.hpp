/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/detail/codec.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>

namespace CUDF_EXPORT cudf {
namespace io::detail {

/**
 * @brief Maximum size of uncompressed chunks that can be compressed.
 *
 * @param compression Compression type
 * @returns maximum chunk size
 */
[[nodiscard]] std::optional<size_t> compress_max_allowed_chunk_size(compression_type compression);

/**
 * @brief Gets input and output alignment requirements for the given compression type.
 *
 * @param compression Compression type
 * @returns required alignment
 */
[[nodiscard]] size_t compress_required_chunk_alignment(compression_type compression);

/**
 * @brief Gets the maximum size any chunk could compress to in the batch.
 *
 * @param compression Compression type
 * @param uncompressed_size Size of the largest uncompressed chunk in the batch
 */
[[nodiscard]] size_t max_compressed_size(compression_type compression, size_t uncompressed_size);

/**
 * @brief Aggregate results of compression into a single statistics object.
 *
 * @param inputs List of uncompressed input buffers
 * @param results List of compression results
 * @param stream CUDA stream to use
 * @return writer_compression_statistics
 */
[[nodiscard]] writer_compression_statistics collect_compression_statistics(
  device_span<device_span<uint8_t const> const> inputs,
  device_span<codec_exec_result const> results,
  cuda::stream_ref stream);

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
