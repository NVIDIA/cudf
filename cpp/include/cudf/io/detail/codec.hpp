/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file codec.hpp
 * @brief cuDF-IO compression and decompression API definitions
 */

#pragma once

#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <cstdint>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::detail {
/**
 * @addtogroup io_codec
 * @{
 * @file
 */

/**
 * @brief Status of a compression/decompression operation.
 */
enum class codec_status : uint8_t {
  SUCCESS,          ///< Successful, output is valid
  FAILURE,          ///< Failed, output is invalid (e.g. input is unsupported in some way)
  SKIPPED,          ///< Operation skipped (if conversion, uncompressed data can be used)
  OUTPUT_OVERFLOW,  ///< Output buffer is too small; operation can succeed with larger output
};

/**
 * @brief Summary of a compression/decompression operation.
 */
struct codec_exec_result {
  uint64_t bytes_written;  ///< Number of bytes written to the output buffer
  codec_status status;     ///< Status of the operation
};

/**
 * @brief Check if compression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported
 */
[[nodiscard]] bool is_compression_supported(compression_type compression);

/**
 * @brief Check if decompression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported
 */
[[nodiscard]] bool is_decompression_supported(compression_type compression);

/**
 * @brief Check if host compression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported by host engine
 */
[[nodiscard]] bool is_host_compression_supported(compression_type compression);

/**
 * @brief Check if device compression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported by device engine
 */
[[nodiscard]] bool is_device_compression_supported(compression_type compression);

/**
 * @brief Check if host decompression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported by host engine
 */
[[nodiscard]] bool is_host_decompression_supported(compression_type compression);

/**
 * @brief Check if device decompression is supported for the given compression type.
 *
 * @param compression Compression type
 * @return Boolean indicating if the compression type is supported by device engine
 */
[[nodiscard]] bool is_device_decompression_supported(compression_type compression);

/**
 * @brief Compress a host memory buffer.
 *
 * @param compression Compression type
 * @param src The input host buffer to compress
 * @return Vector containing the compressed output
 */
std::vector<uint8_t> compress(compression_type compression, host_span<uint8_t const> src);

/**
 * @brief Compress device memory buffers.
 *
 * @param compression Compression type
 * @param inputs Device memory buffers to compress
 * @param outputs Device memory buffers to store the compressed output
 * @param results Compression results
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void compress(compression_type compression,
              device_span<device_span<uint8_t const> const> inputs,
              device_span<device_span<uint8_t> const> outputs,
              device_span<codec_exec_result> results,
              cuda::stream_ref stream);

/**
 * @brief Decompresses a host memory buffer.
 *
 * @param compression Compression type
 * @param src The input host buffer to decompress
 * @return Vector containing the decompressed output
 */
[[nodiscard]] std::vector<uint8_t> decompress(compression_type compression,
                                              host_span<uint8_t const> src);

/**
 * @brief Decompresses a host memory buffer.
 *
 * @param compression Compression type
 * @param src The input host buffer to decompress
 * @param dst The host buffer to store decompressed output
 * @return Size of decompressed output
 */
size_t decompress(compression_type compression,
                  host_span<uint8_t const> src,
                  host_span<uint8_t> dst);

/**
 * @brief Decompresses device memory buffers.
 *
 * @param compression Compression type
 * @param inputs Device memory buffers to decompress
 * @param outputs Device memory buffers to store the decompressed output
 * @param results Compression results for each input buffer
 * @param max_uncomp_chunk_size Maximum size of any single uncompressed chunk
 * @param max_total_uncomp_size Maximum size of the total uncompressed data
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void decompress(compression_type compression,
                device_span<device_span<uint8_t const> const> inputs,
                device_span<device_span<uint8_t> const> outputs,
                device_span<codec_exec_result> results,
                size_t max_uncomp_chunk_size,
                size_t max_total_uncomp_size,
                cuda::stream_ref stream);

/** @} */  // end of group
}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
