/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvcomp_adapter.hpp"

#include "nvcomp_adapter.cuh"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>

#include <io/utilities/hostdevice_vector.hpp>
#include <nvcomp/deflate.h>
#include <nvcomp/gzip.h>
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#include <nvcomp/zstd.h>

#include <mutex>

namespace cudf::io::detail::nvcomp {
namespace {

[[nodiscard]] std::string nvcomp_status_to_string(nvcompStatus_t status)
{
#if NVCOMP_VER_MAJOR > 5 || (NVCOMP_VER_MAJOR == 5 && NVCOMP_VER_MINOR >= 2)
  if (auto const* str = nvcompGetStatusString(status); str != nullptr) { return str; }
  return "nvcompStatus_t(" + std::to_string(static_cast<int>(status)) + ")";
#else
  switch (status) {
    case nvcompStatus_t::nvcompSuccess: return "nvcompSuccess";
    case nvcompStatus_t::nvcompErrorInvalidValue: return "nvcompErrorInvalidValue";
    case nvcompStatus_t::nvcompErrorNotSupported: return "nvcompErrorNotSupported";
    case nvcompStatus_t::nvcompErrorCannotDecompress: return "nvcompErrorCannotDecompress";
    case nvcompStatus_t::nvcompErrorBadChecksum: return "nvcompErrorBadChecksum";
    case nvcompStatus_t::nvcompErrorCannotVerifyChecksums:
      return "nvcompErrorCannotVerifyChecksums";
    case nvcompStatus_t::nvcompErrorOutputBufferTooSmall: return "nvcompErrorOutputBufferTooSmall";
    case nvcompStatus_t::nvcompErrorWrongHeaderLength: return "nvcompErrorWrongHeaderLength";
    case nvcompStatus_t::nvcompErrorAlignment: return "nvcompErrorAlignment";
    case nvcompStatus_t::nvcompErrorChunkSizeTooLarge: return "nvcompErrorChunkSizeTooLarge";
    case nvcompStatus_t::nvcompErrorCudaError: return "nvcompErrorCudaError";
    case nvcompStatus_t::nvcompErrorInternal: return "nvcompErrorInternal";
    case nvcompStatus_t::nvcompErrorCannotCompress: return "nvcompErrorCannotCompress";
    case nvcompStatus_t::nvcompErrorWrongInputLength: return "nvcompErrorWrongInputLength";
#if NVCOMP_VER_MAJOR >= 5 and NVCOMP_VER_MINOR >= 1
    case nvcompStatus_t::nvcompErrorBatchSizeTooLarge: return "nvcompErrorBatchSizeTooLarge";
#endif
  }
  return "nvcompStatus_t(" + std::to_string(static_cast<int>(status)) + ")";
#endif
}

[[nodiscard]] std::string compression_type_name(compression_type compression)
{
  switch (compression) {
    case compression_type::SNAPPY: return "Snappy";
    case compression_type::ZSTD: return "Zstandard";
    case compression_type::DEFLATE: return "Deflate";
    case compression_type::LZ4: return "LZ4";
    case compression_type::GZIP: return "GZIP";
  }
  return "compression_type(" + std::to_string(static_cast<int>(compression)) + ")";
}

[[nodiscard]] std::optional<bool> use_hw_decompression()
{
  auto const env = getenv("LIBCUDF_HW_DECOMPRESSION");
  if (env == nullptr) { return std::nullopt; }
  std::string val{env};
  std::transform(
    val.begin(), val.end(), val.begin(), [](unsigned char c) { return std::toupper(c); });
  return val == "ON";
}

#define CHECK_NVCOMP_STATUS(status)                                   \
  do {                                                                \
    CUDF_EXPECTS(status == nvcompStatus_t::nvcompSuccess,             \
                 "nvCOMP error: " + nvcomp_status_to_string(status)); \
  } while (0)

#define UNSUPPORTED_COMPRESSION(compression)                                          \
  do {                                                                                \
    CUDF_FAIL("Unsupported compression type: " + compression_type_name(compression)); \
  } while (0)

// Dispatcher for nvcompBatched<format>DecompressGetTempSizeAsync
auto batched_decompress_get_temp_size_async(compression_type compression,
                                            size_t num_chunks,
                                            size_t max_uncompressed_chunk_bytes,
                                            size_t* temp_bytes,
                                            size_t max_total_uncompressed_bytes)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressGetTempSizeAsync(num_chunks,
                                                           max_uncompressed_chunk_bytes,
                                                           nvcompBatchedSnappyDecompressDefaultOpts,
                                                           temp_bytes,
                                                           max_total_uncompressed_bytes);
    case compression_type::ZSTD:
      return nvcompBatchedZstdDecompressGetTempSizeAsync(num_chunks,
                                                         max_uncompressed_chunk_bytes,
                                                         nvcompBatchedZstdDecompressDefaultOpts,
                                                         temp_bytes,
                                                         max_total_uncompressed_bytes);
    case compression_type::LZ4:
      return nvcompBatchedLZ4DecompressGetTempSizeAsync(num_chunks,
                                                        max_uncompressed_chunk_bytes,
                                                        nvcompBatchedLZ4DecompressDefaultOpts,
                                                        temp_bytes,
                                                        max_total_uncompressed_bytes);
    case compression_type::DEFLATE:
      return nvcompBatchedDeflateDecompressGetTempSizeAsync(
        num_chunks,
        max_uncompressed_chunk_bytes,
        nvcompBatchedDeflateDecompressDefaultOpts,
        temp_bytes,
        max_total_uncompressed_bytes);
    case compression_type::GZIP:
      return nvcompBatchedGzipDecompressGetTempSizeAsync(num_chunks,
                                                         max_uncompressed_chunk_bytes,
                                                         nvcompBatchedGzipDecompressDefaultOpts,
                                                         temp_bytes,
                                                         max_total_uncompressed_bytes);
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}

// Dispatcher for nvcompBatched<format>DecompressAsync
auto batched_decompress_async(compression_type compression,
                              std::optional<bool> use_hw_decompression,
                              void const* const* device_compressed_chunk_ptrs,
                              size_t const* device_compressed_chunk_bytes,
                              size_t const* device_uncompressed_buffer_bytes,
                              size_t* device_uncompressed_chunk_bytes,
                              size_t num_chunks,
                              void* device_temp_ptr,
                              size_t temp_bytes,
                              void* const* device_uncompressed_chunk_ptrs,
                              nvcompStatus_t* device_statuses,
                              rmm::cuda_stream_view stream)
{
  switch (compression) {
    case compression_type::SNAPPY: {
      auto opts = nvcompBatchedSnappyDecompressDefaultOpts;
      if (use_hw_decompression.has_value()) {
        opts.backend = *use_hw_decompression ? NVCOMP_DECOMPRESS_BACKEND_HARDWARE
                                             : NVCOMP_DECOMPRESS_BACKEND_CUDA;
      }
      return nvcompBatchedSnappyDecompressAsync(device_compressed_chunk_ptrs,
                                                device_compressed_chunk_bytes,
                                                device_uncompressed_buffer_bytes,
                                                device_uncompressed_chunk_bytes,
                                                num_chunks,
                                                device_temp_ptr,
                                                temp_bytes,
                                                device_uncompressed_chunk_ptrs,
                                                opts,
                                                device_statuses,
                                                stream.value());
    }
    case compression_type::ZSTD:
      return nvcompBatchedZstdDecompressAsync(device_compressed_chunk_ptrs,
                                              device_compressed_chunk_bytes,
                                              device_uncompressed_buffer_bytes,
                                              device_uncompressed_chunk_bytes,
                                              num_chunks,
                                              device_temp_ptr,
                                              temp_bytes,
                                              device_uncompressed_chunk_ptrs,
                                              nvcompBatchedZstdDecompressDefaultOpts,
                                              device_statuses,
                                              stream.value());
    case compression_type::DEFLATE:
      return nvcompBatchedDeflateDecompressAsync(device_compressed_chunk_ptrs,
                                                 device_compressed_chunk_bytes,
                                                 device_uncompressed_buffer_bytes,
                                                 device_uncompressed_chunk_bytes,
                                                 num_chunks,
                                                 device_temp_ptr,
                                                 temp_bytes,
                                                 device_uncompressed_chunk_ptrs,
                                                 nvcompBatchedDeflateDecompressDefaultOpts,
                                                 device_statuses,
                                                 stream.value());
    case compression_type::LZ4:
      return nvcompBatchedLZ4DecompressAsync(device_compressed_chunk_ptrs,
                                             device_compressed_chunk_bytes,
                                             device_uncompressed_buffer_bytes,
                                             device_uncompressed_chunk_bytes,
                                             num_chunks,
                                             device_temp_ptr,
                                             temp_bytes,
                                             device_uncompressed_chunk_ptrs,
                                             nvcompBatchedLZ4DecompressDefaultOpts,
                                             device_statuses,
                                             stream.value());
    case compression_type::GZIP:
      return nvcompBatchedGzipDecompressAsync(device_compressed_chunk_ptrs,
                                              device_compressed_chunk_bytes,
                                              device_uncompressed_buffer_bytes,
                                              device_uncompressed_chunk_bytes,
                                              num_chunks,
                                              device_temp_ptr,
                                              temp_bytes,
                                              device_uncompressed_chunk_ptrs,
                                              nvcompBatchedGzipDecompressDefaultOpts,
                                              device_statuses,
                                              stream.value());
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}

// Wrapper for nvcompBatched<format>CompressGetTempSizeAsync
nvcompStatus_t batched_compress_get_temp_size_async(compression_type compression,
                                                    size_t batch_size,
                                                    size_t max_uncompressed_chunk_bytes,
                                                    size_t* temp_size,
                                                    size_t max_total_uncompressed_bytes)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyCompressGetTempSizeAsync(batch_size,
                                                         max_uncompressed_chunk_bytes,
                                                         nvcompBatchedSnappyCompressDefaultOpts,
                                                         temp_size,
                                                         max_total_uncompressed_bytes);
      break;
    case compression_type::DEFLATE:
      return nvcompBatchedDeflateCompressGetTempSizeAsync(batch_size,
                                                          max_uncompressed_chunk_bytes,
                                                          nvcompBatchedDeflateCompressDefaultOpts,
                                                          temp_size,
                                                          max_total_uncompressed_bytes);
      break;
    case compression_type::ZSTD:
      return nvcompBatchedZstdCompressGetTempSizeAsync(batch_size,
                                                       max_uncompressed_chunk_bytes,
                                                       nvcompBatchedZstdCompressDefaultOpts,
                                                       temp_size,
                                                       max_total_uncompressed_bytes);
      break;
    case compression_type::LZ4:
      return nvcompBatchedLZ4CompressGetTempSizeAsync(batch_size,
                                                      max_uncompressed_chunk_bytes,
                                                      nvcompBatchedLZ4CompressDefaultOpts,
                                                      temp_size,
                                                      max_total_uncompressed_bytes);
      break;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}

size_t batched_compress_temp_size(compression_type compression,
                                  size_t batch_size,
                                  size_t max_uncompressed_chunk_bytes,
                                  size_t max_total_uncompressed_bytes)
{
  size_t temp_size             = 0;
  nvcompStatus_t nvcomp_status = nvcompStatus_t::nvcompSuccess;

  nvcomp_status = batched_compress_get_temp_size_async(compression,
                                                       batch_size,
                                                       max_uncompressed_chunk_bytes,
                                                       &temp_size,
                                                       max_total_uncompressed_bytes);

  CHECK_NVCOMP_STATUS(nvcomp_status);
  return temp_size;
}

// Dispatcher for nvcompBatched<format>CompressAsync
void batched_compress_async(compression_type compression,
                            void const* const* device_uncompressed_ptrs,
                            size_t const* device_uncompressed_bytes,
                            size_t max_uncompressed_chunk_bytes,
                            size_t batch_size,
                            void* device_temp_ptr,
                            size_t temp_bytes,
                            void* const* device_compressed_ptrs,
                            size_t* device_compressed_bytes,
                            nvcompStatus_t* device_nvcomp_statuses,
                            rmm::cuda_stream_view stream)
{
  nvcompStatus_t nvcomp_status = nvcompStatus_t::nvcompSuccess;
  switch (compression) {
    case compression_type::SNAPPY:
      nvcomp_status = nvcompBatchedSnappyCompressAsync(device_uncompressed_ptrs,
                                                       device_uncompressed_bytes,
                                                       max_uncompressed_chunk_bytes,
                                                       batch_size,
                                                       device_temp_ptr,
                                                       temp_bytes,
                                                       device_compressed_ptrs,
                                                       device_compressed_bytes,
                                                       nvcompBatchedSnappyCompressDefaultOpts,
                                                       device_nvcomp_statuses,
                                                       stream.value());
      break;
    case compression_type::DEFLATE:
      nvcomp_status = nvcompBatchedDeflateCompressAsync(device_uncompressed_ptrs,
                                                        device_uncompressed_bytes,
                                                        max_uncompressed_chunk_bytes,
                                                        batch_size,
                                                        device_temp_ptr,
                                                        temp_bytes,
                                                        device_compressed_ptrs,
                                                        device_compressed_bytes,
                                                        nvcompBatchedDeflateCompressDefaultOpts,
                                                        device_nvcomp_statuses,
                                                        stream.value());
      break;
    case compression_type::ZSTD:
      nvcomp_status = nvcompBatchedZstdCompressAsync(device_uncompressed_ptrs,
                                                     device_uncompressed_bytes,
                                                     max_uncompressed_chunk_bytes,
                                                     batch_size,
                                                     device_temp_ptr,
                                                     temp_bytes,
                                                     device_compressed_ptrs,
                                                     device_compressed_bytes,
                                                     nvcompBatchedZstdCompressDefaultOpts,
                                                     device_nvcomp_statuses,
                                                     stream.value());
      break;
    case compression_type::LZ4:
      nvcomp_status = nvcompBatchedLZ4CompressAsync(device_uncompressed_ptrs,
                                                    device_uncompressed_bytes,
                                                    max_uncompressed_chunk_bytes,
                                                    batch_size,
                                                    device_temp_ptr,
                                                    temp_bytes,
                                                    device_compressed_ptrs,
                                                    device_compressed_bytes,
                                                    nvcompBatchedLZ4CompressDefaultOpts,
                                                    device_nvcomp_statuses,
                                                    stream.value());
      break;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
  CHECK_NVCOMP_STATUS(nvcomp_status);
}

bool is_aligned(void const* ptr, std::uintptr_t alignment) noexcept
{
  return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

std::optional<std::string> is_compression_disabled_impl(compression_type compression,
                                                        feature_status_parameters params)
{
  switch (compression) {
    case compression_type::DEFLATE:
    case compression_type::LZ4:
    case compression_type::SNAPPY:
    case compression_type::ZSTD:
      if (not params.are_stable_integrations_enabled) {
        return "nvCOMP use is disabled through the `LIBCUDF_NVCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    default: return "Unsupported compression type";
  }
}

std::optional<std::string> is_decompression_disabled_impl(compression_type compression,
                                                          feature_status_parameters params)
{
  switch (compression) {
    case compression_type::GZIP: {
      if (not params.are_all_integrations_enabled) {
        return "GZIP decompression is experimental, you can enable it through "
               "`LIBCUDF_NVCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
    case compression_type::DEFLATE:
    case compression_type::LZ4:
    case compression_type::SNAPPY:
    case compression_type::ZSTD: {
      if (not params.are_stable_integrations_enabled) {
        return "nvCOMP use is disabled through the `LIBCUDF_NVCOMP_POLICY` environment variable.";
      }
      return std::nullopt;
    }
  }
  return "Unsupported compression type";
}

// Dispatcher for nvcompBatched<format>DecompressGetTempSizeSync
auto batched_decompress_get_temp_size_sync(compression_type compression,
                                           void const* const* device_compressed_chunk_ptrs,
                                           size_t const* device_compressed_chunk_bytes,
                                           size_t num_chunks,
                                           size_t max_uncompressed_chunk_bytes,
                                           size_t* temp_bytes,
                                           size_t max_total_uncompressed_bytes,
                                           nvcompStatus_t* device_statuses,
                                           cudaStream_t stream)
{
  switch (compression) {
    case compression_type::SNAPPY:
      return nvcompBatchedSnappyDecompressGetTempSizeSync(device_compressed_chunk_ptrs,
                                                          device_compressed_chunk_bytes,
                                                          num_chunks,
                                                          max_uncompressed_chunk_bytes,
                                                          temp_bytes,
                                                          max_total_uncompressed_bytes,
                                                          nvcompBatchedSnappyDecompressDefaultOpts,
                                                          device_statuses,
                                                          stream);
    case compression_type::ZSTD:
      return nvcompBatchedZstdDecompressGetTempSizeSync(device_compressed_chunk_ptrs,
                                                        device_compressed_chunk_bytes,
                                                        num_chunks,
                                                        max_uncompressed_chunk_bytes,
                                                        temp_bytes,
                                                        max_total_uncompressed_bytes,
                                                        nvcompBatchedZstdDecompressDefaultOpts,
                                                        device_statuses,
                                                        stream);
    case compression_type::LZ4:
      return nvcompBatchedLZ4DecompressGetTempSizeSync(device_compressed_chunk_ptrs,
                                                       device_compressed_chunk_bytes,
                                                       num_chunks,
                                                       max_uncompressed_chunk_bytes,
                                                       temp_bytes,
                                                       max_total_uncompressed_bytes,
                                                       nvcompBatchedLZ4DecompressDefaultOpts,
                                                       device_statuses,
                                                       stream);
    case compression_type::DEFLATE:
      return nvcompBatchedDeflateDecompressGetTempSizeSync(
        device_compressed_chunk_ptrs,
        device_compressed_chunk_bytes,
        num_chunks,
        max_uncompressed_chunk_bytes,
        temp_bytes,
        max_total_uncompressed_bytes,
        nvcompBatchedDeflateDecompressDefaultOpts,
        device_statuses,
        stream);
    case compression_type::GZIP:
      return nvcompBatchedGzipDecompressGetTempSizeSync(device_compressed_chunk_ptrs,
                                                        device_compressed_chunk_bytes,
                                                        num_chunks,
                                                        max_uncompressed_chunk_bytes,
                                                        temp_bytes,
                                                        max_total_uncompressed_bytes,
                                                        nvcompBatchedGzipDecompressDefaultOpts,
                                                        device_statuses,
                                                        stream);
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}

// Overload for internal use that takes device pointers and sizes directly
size_t batched_decompress_temp_size_ex(compression_type compression,
                                       device_span<void const* const> input_data_ptrs,
                                       device_span<size_t const> input_data_sizes,
                                       size_t max_uncomp_chunk_size,
                                       size_t max_total_uncomp_size,
                                       rmm::cuda_stream_view stream)
{
  if (is_batched_decompress_temp_size_ex_supported(compression)) {
    size_t temp_size = 0;
    auto d_statuses  = rmm::device_uvector<nvcompStatus_t>(input_data_ptrs.size(), stream);
    nvcompStatus_t const nvcomp_status =
      batched_decompress_get_temp_size_sync(compression,
                                            input_data_ptrs.data(),
                                            input_data_sizes.data(),
                                            input_data_ptrs.size(),
                                            max_uncomp_chunk_size,
                                            &temp_size,
                                            max_total_uncomp_size,
                                            d_statuses.data(),
                                            stream.value());
    if (nvcomp_status == nvcompStatus_t::nvcompSuccess) {
      auto const h_statuses = cudf::detail::make_host_vector(d_statuses, stream);
      auto const are_all_success =
        std::all_of(h_statuses.begin(), h_statuses.end(), [](nvcompStatus_t status) {
          return status == nvcompStatus_t::nvcompSuccess;
        });
      if (are_all_success) { return temp_size; }
    }
    CUDF_LOG_WARN(
      "batched_decompress_get_temp_size_sync failed, falling back to batched_decompress_temp_size");
  }
  // Fallback to the original batched decompress temp size calculation
  return batched_decompress_temp_size(
    compression, input_data_ptrs.size(), max_uncomp_chunk_size, max_total_uncomp_size);
}

}  // namespace

size_t batched_decompress_temp_size(compression_type compression,
                                    size_t num_chunks,
                                    size_t max_uncomp_chunk_size,
                                    size_t max_total_uncomp_size)
{
  size_t temp_size                   = 0;
  nvcompStatus_t const nvcomp_status = batched_decompress_get_temp_size_async(
    compression, num_chunks, max_uncomp_chunk_size, &temp_size, max_total_uncomp_size);
  CHECK_NVCOMP_STATUS(nvcomp_status);
  return temp_size;
}

bool is_batched_decompress_temp_size_ex_supported(compression_type compression)
{
  return compression == compression_type::ZSTD;
}

size_t batched_decompress_temp_size_ex(compression_type compression,
                                       device_span<device_span<uint8_t const> const> inputs,
                                       size_t max_uncomp_chunk_size,
                                       size_t max_total_uncomp_size,
                                       rmm::cuda_stream_view stream)
{
  auto const [d_input_ptrs, d_input_sizes] = create_get_temp_size_args(inputs, stream);

  return batched_decompress_temp_size_ex(
    compression, d_input_ptrs, d_input_sizes, max_uncomp_chunk_size, max_total_uncomp_size, stream);
}

void batched_decompress(compression_type compression,
                        device_span<device_span<uint8_t const> const> inputs,
                        device_span<device_span<uint8_t> const> outputs,
                        device_span<codec_exec_result> results,
                        size_t max_uncomp_chunk_size,
                        size_t max_total_uncomp_size,
                        rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(inputs.size() > 0, "inputs must be non-empty");
  CUDF_EXPECTS(inputs.size() == outputs.size(), "inputs and outputs must have the same size");
  CUDF_EXPECTS(inputs.size() == results.size(), "inputs and results must have the same size");
  CUDF_EXPECTS(max_total_uncomp_size > 0, "max_total_uncomp_size must be greater than 0");
  CUDF_EXPECTS(max_uncomp_chunk_size > 0, "max_uncomp_chunk_size must be greater than 0");

  auto const num_chunks = inputs.size();

  // cuDF inflate inputs converted to nvcomp inputs
  auto const nvcomp_args = create_batched_nvcomp_args(inputs, outputs, stream);
  rmm::device_uvector<size_t> actual_uncompressed_data_sizes(num_chunks, stream);
  rmm::device_uvector<nvcompStatus_t> nvcomp_statuses(num_chunks, stream);

  // Temporary space required for decompression
  auto const temp_size = batched_decompress_temp_size_ex(compression,
                                                         nvcomp_args.input_data_ptrs,
                                                         nvcomp_args.input_data_sizes,
                                                         max_uncomp_chunk_size,
                                                         max_total_uncomp_size,
                                                         stream);
  rmm::device_buffer scratch(temp_size, stream);

  auto const nvcomp_status = batched_decompress_async(compression,
                                                      use_hw_decompression(),
                                                      nvcomp_args.input_data_ptrs.data(),
                                                      nvcomp_args.input_data_sizes.data(),
                                                      nvcomp_args.output_data_sizes.data(),
                                                      actual_uncompressed_data_sizes.data(),
                                                      num_chunks,
                                                      scratch.data(),
                                                      scratch.size(),
                                                      nvcomp_args.output_data_ptrs.data(),
                                                      nvcomp_statuses.data(),
                                                      stream.value());
  CHECK_NVCOMP_STATUS(nvcomp_status);

  update_compression_results(nvcomp_statuses, actual_uncompressed_data_sizes, results, stream);
}

// Wrapper for nvcompBatched<format>CompressGetMaxOutputChunkSize
size_t compress_max_output_chunk_size(compression_type compression,
                                      size_t max_uncompressed_chunk_bytes)
{
  auto const capped_uncomp_bytes =
    std::min(compress_max_allowed_chunk_size(compression).value_or(max_uncompressed_chunk_bytes),
             max_uncompressed_chunk_bytes);

  size_t max_comp_chunk_size = 0;
  nvcompStatus_t status      = nvcompStatus_t::nvcompSuccess;
  switch (compression) {
    case compression_type::SNAPPY:
      status = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedSnappyCompressDefaultOpts, &max_comp_chunk_size);
      break;
    case compression_type::DEFLATE:
    case compression_type::GZIP: {
      // nvcompBatchedGzipCompressGetMaxOutputChunkSize is not yet available
      status = nvcompBatchedDeflateCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedDeflateCompressDefaultOpts, &max_comp_chunk_size);
      if (compression == compression_type::GZIP) {
        // GZIP adds 18 bytes for header and footer
        max_comp_chunk_size += 18;
      }
      break;
    }
    case compression_type::ZSTD:
      status = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedZstdCompressDefaultOpts, &max_comp_chunk_size);
      break;
    case compression_type::LZ4:
      status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
        capped_uncomp_bytes, nvcompBatchedLZ4CompressDefaultOpts, &max_comp_chunk_size);
      break;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
  CHECK_NVCOMP_STATUS(status);
  return max_comp_chunk_size;
}

void batched_compress(compression_type compression,
                      device_span<device_span<uint8_t const> const> inputs,
                      device_span<device_span<uint8_t> const> outputs,
                      device_span<codec_exec_result> results,
                      rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(inputs.size() > 0, "inputs must be non-empty");
  CUDF_EXPECTS(inputs.size() == outputs.size(), "inputs and outputs must have the same size");
  CUDF_EXPECTS(inputs.size() == results.size(), "inputs and results must have the same size");

  auto const num_chunks = inputs.size();

  auto nvcomp_args = create_batched_nvcomp_args(inputs, outputs, stream);

  skip_unsupported_inputs(
    nvcomp_args.input_data_sizes, results, compress_max_allowed_chunk_size(compression), stream);

  auto const [max_uncomp_chunk_size, total_uncomp_size] =
    max_chunk_and_total_input_size(nvcomp_args.input_data_sizes, stream);

  auto const temp_size =
    batched_compress_temp_size(compression, num_chunks, max_uncomp_chunk_size, total_uncomp_size);

  rmm::device_buffer scratch(temp_size, stream);
  CUDF_EXPECTS(is_aligned(scratch.data(), 8), "Compression failed, misaligned scratch buffer");

  rmm::device_uvector<size_t> actual_compressed_data_sizes(num_chunks, stream);
  rmm::device_uvector<nvcompStatus_t> nvcomp_statuses(num_chunks, stream);

  batched_compress_async(compression,
                         nvcomp_args.input_data_ptrs.data(),
                         nvcomp_args.input_data_sizes.data(),
                         max_uncomp_chunk_size,
                         num_chunks,
                         scratch.data(),
                         scratch.size(),
                         nvcomp_args.output_data_ptrs.data(),
                         actual_compressed_data_sizes.data(),
                         nvcomp_statuses.data(),
                         stream.value());

  update_compression_results(nvcomp_statuses, actual_compressed_data_sizes, results, stream);
}

feature_status_parameters::feature_status_parameters()
  : feature_status_parameters(nvcomp_integration::is_all_enabled(),
                              nvcomp_integration::is_stable_enabled())
{
}

feature_status_parameters::feature_status_parameters(bool all_enabled, bool stable_enabled)
  : lib_major_version{NVCOMP_VER_MAJOR},
    lib_minor_version{NVCOMP_VER_MINOR},
    lib_patch_version{NVCOMP_VER_PATCH},
    are_all_integrations_enabled{all_enabled},
    are_stable_integrations_enabled{stable_enabled}
{
}

// Represents all parameters required to determine status of a compression/decompression feature
using feature_status_inputs = std::pair<compression_type, feature_status_parameters>;
struct hash_feature_status_inputs {
  size_t operator()(feature_status_inputs const& fsi) const
  {
    // Outside of unit tests, the same `feature_status_parameters` value will always be passed
    // within a run; for simplicity, only use `compression_type` for the hash
    return std::hash<compression_type>{}(fsi.first);
  }
};

// Hash map type that stores feature status for different combinations of input parameters
using feature_status_memo_map =
  std::unordered_map<feature_status_inputs, std::optional<std::string>, hash_feature_status_inputs>;

std::optional<std::string> is_compression_disabled(compression_type compression,
                                                   feature_status_parameters params)
{
  static feature_status_memo_map comp_status_reason;
  static std::mutex memo_map_mutex;

  std::unique_lock memo_map_lock{memo_map_mutex};
  if (auto mem_res_it = comp_status_reason.find(feature_status_inputs{compression, params});
      mem_res_it != comp_status_reason.end()) {
    return mem_res_it->second;
  }

  // The rest of the function will execute only once per run, the memoized result will be returned
  // in all subsequent calls with the same compression type
  auto const reason                         = is_compression_disabled_impl(compression, params);
  comp_status_reason[{compression, params}] = reason;
  memo_map_lock.unlock();

  if (reason.has_value()) {
    CUDF_LOG_INFO("nvCOMP is disabled for %s compression; reason: %s",
                  compression_type_name(compression),
                  reason.value());
  } else {
    CUDF_LOG_INFO("nvCOMP is enabled for %s compression", compression_type_name(compression));
  }

  return reason;
}

std::optional<std::string> is_decompression_disabled(compression_type compression,
                                                     feature_status_parameters params)
{
  static feature_status_memo_map decomp_status_reason;
  static std::mutex memo_map_mutex;

  std::unique_lock memo_map_lock{memo_map_mutex};
  if (auto mem_res_it = decomp_status_reason.find(feature_status_inputs{compression, params});
      mem_res_it != decomp_status_reason.end()) {
    return mem_res_it->second;
  }

  // The rest of the function will execute only once per run, the memoized result will be returned
  // in all subsequent calls with the same compression type
  auto const reason                           = is_decompression_disabled_impl(compression, params);
  decomp_status_reason[{compression, params}] = reason;
  memo_map_lock.unlock();

  if (reason.has_value()) {
    CUDF_LOG_INFO("nvCOMP is disabled for %s decompression; reason: %s",
                  compression_type_name(compression),
                  reason.value());
  } else {
    CUDF_LOG_INFO("nvCOMP is enabled for %s decompression", compression_type_name(compression));
  }

  return reason;
}
size_t compress_required_alignment(compression_type compression)
{
  nvcompAlignmentRequirements_t alignments{};
  nvcompStatus_t status;
  switch (compression) {
    case compression_type::GZIP:
    case compression_type::DEFLATE:
      status = nvcompBatchedDeflateCompressGetRequiredAlignments(
        nvcompBatchedDeflateCompressDefaultOpts, &alignments);
      break;
    case compression_type::SNAPPY:
      status = nvcompBatchedSnappyCompressGetRequiredAlignments(
        nvcompBatchedSnappyCompressDefaultOpts, &alignments);
      break;
    case compression_type::ZSTD:
      status = nvcompBatchedZstdCompressGetRequiredAlignments(nvcompBatchedZstdCompressDefaultOpts,
                                                              &alignments);
      break;
    case compression_type::LZ4:
      status = nvcompBatchedLZ4CompressGetRequiredAlignments(nvcompBatchedLZ4CompressDefaultOpts,
                                                             &alignments);
      break;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
  CHECK_NVCOMP_STATUS(status);
  return std::max({alignments.input, alignments.output, alignments.temp});
}

size_t decompress_required_alignment(compression_type compression)
{
  nvcompAlignmentRequirements_t alignments{};
  nvcompStatus_t status;
  switch (compression) {
    case compression_type::GZIP:
    case compression_type::DEFLATE:
      status = nvcompBatchedDeflateDecompressGetRequiredAlignments(
        nvcompBatchedDeflateDecompressDefaultOpts, &alignments);
      break;
    case compression_type::SNAPPY:
      status = nvcompBatchedSnappyDecompressGetRequiredAlignments(
        nvcompBatchedSnappyDecompressDefaultOpts, &alignments);
      break;
    case compression_type::ZSTD:
      status = nvcompBatchedZstdDecompressGetRequiredAlignments(
        nvcompBatchedZstdDecompressDefaultOpts, &alignments);
      break;
    case compression_type::LZ4:
      status = nvcompBatchedLZ4DecompressGetRequiredAlignments(
        nvcompBatchedLZ4DecompressDefaultOpts, &alignments);
      break;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
  CHECK_NVCOMP_STATUS(status);
  return std::max({alignments.input, alignments.output, alignments.temp});
}

std::optional<size_t> compress_max_allowed_chunk_size(compression_type compression)
{
  switch (compression) {
    case compression_type::DEFLATE:
    case compression_type::GZIP: return nvcompDeflateCompressionMaxAllowedChunkSize;
    case compression_type::SNAPPY: return nvcompSnappyCompressionMaxAllowedChunkSize;
    case compression_type::ZSTD: return nvcompZstdCompressionMaxAllowedChunkSize;
    case compression_type::LZ4: return nvcompLZ4CompressionMaxAllowedChunkSize;
    default: UNSUPPORTED_COMPRESSION(compression);
  }
}

void load_nvcomp_library()
{
  static std::once_flag nvcomp_initialized_flag;

  std::call_once(nvcomp_initialized_flag, []() {
    auto const stream = cudf::get_default_stream();
    auto const mr     = cudf::get_current_device_resource_ref();

    // Allocate dummy input buffer and output buffer
    auto const d_input             = rmm::device_uvector<uint8_t>(1, stream, mr);
    auto const max_compressed_size = compress_max_output_chunk_size(compression_type::SNAPPY, 1);
    rmm::device_uvector<uint8_t> d_compressed(max_compressed_size, stream);

    // Prepare parameters for compression
    cudf::detail::hostdevice_vector<device_span<uint8_t const>> hd_inputs(1, stream);
    hd_inputs[0] = d_input;
    hd_inputs.host_to_device_async(stream);
    cudf::detail::hostdevice_vector<device_span<uint8_t>> hd_outputs(1, stream);
    hd_outputs[0] = d_compressed;
    hd_outputs.host_to_device_async(stream);
    cudf::detail::hostdevice_vector<codec_exec_result> hd_results(1, stream);
    hd_results[0] = codec_exec_result{0, codec_status::FAILURE};
    hd_results.host_to_device_async(stream);

    // Perform compression - this will execute an nvCOMP kernel
    batched_compress(compression_type::SNAPPY, hd_inputs, hd_outputs, hd_results, stream);
  });
}

}  // namespace cudf::io::detail::nvcomp
