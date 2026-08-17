/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "comp/compression.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/detail/contiguous_split.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/logger.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <limits>

namespace cudf::io::experimental {
namespace detail {

namespace {

constexpr uint32_t magic_number      = 0x4C425443;  ///< "CTBL" in little-endian
constexpr uint32_t format_version_v1 = 1;

/**
 * @brief Binary file format header for CudfTable (56 bytes)
 *
 * Layout: [magic(4)] [version=1(4)] [compression(4)] [block_size(4)]
 *         [metadata_version(4)] [reserved(4)]
 *         [metadata_length(8)] [uncompressed_data_length(8)]
 *         [num_blocks(8)] [compressed_data_length(8)]
 * Followed by: [metadata (variable, uncompressed)]
 *              [block_index (num_blocks * 16 bytes)]
 *              [block payloads (variable)]
 *
 * `metadata_version` records the pack() metadata layout version (see
 * cpp/src/copying/pack.cpp) so that readers reject files whose embedded
 * metadata they cannot parse instead of misinterpreting it.
 *
 * When `compression == NONE`, exactly one block holds all data and its payload
 * is written uncompressed (no codec invocation on write or read). An empty
 * table is encoded with `num_blocks == 0`, no block index, and no payload.
 */
struct cudftable_header {
  uint32_t magic{};
  uint32_t format_version{};
  uint32_t compression{};
  uint32_t block_size{};
  int32_t metadata_version{};  ///< Version of the embedded pack() metadata layout
  int32_t reserved{};          ///< Padding; kept zero for deterministic output
  uint64_t metadata_length{};
  uint64_t uncompressed_data_length{};
  uint64_t num_blocks{};
  uint64_t compressed_data_length{};

  cudftable_header() = default;
  cudftable_header(compression_type comp,
                   uint32_t blk_size,
                   uint64_t metadata_size,
                   uint64_t uncomp_data_size,
                   uint64_t n_blocks,
                   uint64_t comp_data_size,
                   int32_t meta_version)
    : magic{magic_number},
      format_version{format_version_v1},
      compression{static_cast<uint32_t>(comp)},
      block_size{blk_size},
      metadata_version{meta_version},
      metadata_length{metadata_size},
      uncompressed_data_length{uncomp_data_size},
      num_blocks{n_blocks},
      compressed_data_length{comp_data_size}
  {
  }
};

// Header is written/read via memcpy, so guard against accidental padding that
// would leave uninitialized bytes in the file.
static_assert(sizeof(cudftable_header) == 56);

/**
 * @brief Per-block size entry in the block index (16 bytes each).
 */
struct block_index_entry {
  uint64_t compressed_size;    ///< Size of the block payload on disk, in bytes
  uint64_t uncompressed_size;  ///< Size of the block after decompression, in bytes
};

/**
 * @brief Upload `size` bytes from `source` at byte `offset` into device `dst`.
 *
 * Prefers `device_read` when the datasource supports it; otherwise reads the
 * bytes into a pageable host staging buffer and copies them to the device.
 * Because `cudf::detail::memcpy_async` defers reads of pageable sources until
 * the stream executes the copy, the stream is synchronized before the staging
 * buffer goes out of scope.
 */
void upload_to_device(
  datasource* source, size_t offset, size_t size, void* dst, rmm::cuda_stream_view stream)
{
  if (size == 0) { return; }
  if (source->is_device_read_preferred(size)) {
    source->device_read(offset, size, static_cast<uint8_t*>(dst), stream);
  } else {
    auto host_buffer = source->host_read(offset, size);
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(dst, host_buffer->data(), size, stream));
    stream.synchronize();
  }
}

/**
 * @brief Write `size` bytes from device `src` to the sink.
 *
 * Uses `device_write` when the sink supports it; otherwise stages the bytes
 * through a host bounce buffer and writes via `host_write`.
 */
void write_from_device(data_sink* sink, void const* src, size_t size, rmm::cuda_stream_view stream)
{
  if (size == 0) { return; }
  if (sink->is_device_write_preferred(size)) {
    sink->device_write(src, size, stream);
  } else {
    auto host_buffer = cudf::detail::make_host_vector(
      cudf::device_span<uint8_t const>{static_cast<uint8_t const*>(src), size}, stream);
    sink->host_write(host_buffer.data(), size);
  }
}

/**
 * @brief Build input/output span tables for batched (de)compression.
 *
 * Returns pinned host vectors so the H2D transfer performed by
 * `make_device_uvector_async` is fast and safe without relying on pageable
 * copy semantics.
 */
auto make_io_span_tables(uint64_t num_blocks, rmm::cuda_stream_view stream)
{
  struct span_tables {
    cudf::detail::host_vector<cudf::device_span<uint8_t const>> inputs;
    cudf::detail::host_vector<cudf::device_span<uint8_t>> outputs;
  };
  return span_tables{
    cudf::detail::make_pinned_vector_async<cudf::device_span<uint8_t const>>(num_blocks, stream),
    cudf::detail::make_pinned_vector_async<cudf::device_span<uint8_t>>(num_blocks, stream)};
}

}  // anonymous namespace

void write_cudftable(data_sink* sink,
                     table_view const& input,
                     compression_type compression,
                     uint32_t block_size,
                     rmm::cuda_stream_view stream)
{
  auto const packed    = cudf::pack(input, stream, cudf::get_current_device_resource_ref());
  auto const data_size = packed.gpu_data->size();
  auto const mr        = cudf::get_current_device_resource_ref();

  // `NONE` path: one pass-through block (or zero blocks for empty data). No
  // codec invocation, and `block_size` is normalized to 0 on disk so that
  // readers and file inspectors see a clear "not applicable" marker.
  if (compression == compression_type::NONE) {
    if (block_size != cudftable_writer_options::default_block_size) {
      CUDF_LOG_WARN(
        "cudftable writer: block_size is ignored when compression is NONE; the "
        "data is always written as a single uncompressed block");
    }

    auto const num_blocks = data_size == 0 ? uint64_t{0} : uint64_t{1};
    std::vector<block_index_entry> block_index;
    if (num_blocks > 0) { block_index.push_back({data_size, data_size}); }

    auto const header = cudftable_header{compression,
                                         /*block_size=*/0,
                                         packed.metadata->size(),
                                         data_size,
                                         num_blocks,
                                         /*compressed=*/data_size,
                                         cudf::detail::packed_metadata_version};
    sink->host_write(&header, sizeof(cudftable_header));
    sink->host_write(packed.metadata->data(), header.metadata_length);
    if (num_blocks > 0) {
      sink->host_write(block_index.data(), num_blocks * sizeof(block_index_entry));
      write_from_device(sink, packed.gpu_data->data(), data_size, stream);
    }

    sink->flush();
    return;
  }

  // Compressed path: batched block compression. The compress() API transparently
  // routes per-block work between host and device engines, so check generic
  // support rather than device-only support.
  CUDF_EXPECTS(cudf::io::detail::is_compression_supported(compression),
               "Unsupported compression type for cudftable");
  CUDF_EXPECTS(block_size > 0, "block_size must be greater than zero for compressed cudftable");

  // Empty-table short-circuit: no payload, no block index.
  if (data_size == 0) {
    auto const header = cudftable_header{compression,
                                         block_size,
                                         packed.metadata->size(),
                                         /*uncomp=*/0,
                                         /*num_blocks=*/0,
                                         /*comp=*/0,
                                         cudf::detail::packed_metadata_version};
    sink->host_write(&header, sizeof(cudftable_header));
    sink->host_write(packed.metadata->data(), header.metadata_length);
    sink->flush();
    return;
  }

  auto const num_blocks          = (data_size + block_size - 1) / block_size;
  auto const max_comp_block_size = cudf::io::detail::max_compressed_size(compression, block_size);

  // Build input spans (views into packed.gpu_data) and output spans (into a
  // padded output buffer). Pinned host vectors keep the H2D span-table copy
  // stream-safe.
  rmm::device_buffer d_compressed(max_comp_block_size * num_blocks, stream, mr);

  auto [h_inputs, h_outputs] = make_io_span_tables(num_blocks, stream);
  for (uint64_t i = 0; i < num_blocks; ++i) {
    auto const offset      = i * block_size;
    auto const uncomp_size = std::min<uint64_t>(block_size, data_size - offset);
    h_inputs[i]            = cudf::device_span<uint8_t const>(
      static_cast<uint8_t const*>(packed.gpu_data->data()) + offset, uncomp_size);
    h_outputs[i] = cudf::device_span<uint8_t>(
      static_cast<uint8_t*>(d_compressed.data()) + i * max_comp_block_size, max_comp_block_size);
  }

  auto d_inputs  = cudf::detail::make_device_uvector_async(h_inputs, stream, mr);
  auto d_outputs = cudf::detail::make_device_uvector_async(h_outputs, stream, mr);
  auto d_results = rmm::device_uvector<cudf::io::detail::codec_exec_result>(num_blocks, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    d_results.data(), 0, num_blocks * sizeof(cudf::io::detail::codec_exec_result), stream));

  cudf::io::detail::compress(compression, d_inputs, d_outputs, d_results, stream);

  auto h_results = cudf::detail::make_host_vector(d_results, stream);

  std::vector<block_index_entry> block_index(num_blocks);
  uint64_t total_compressed = 0;
  for (uint64_t i = 0; i < num_blocks; ++i) {
    CUDF_EXPECTS(h_results[i].status == cudf::io::detail::codec_status::SUCCESS,
                 "cudftable block compression failed");
    auto const offset      = i * block_size;
    auto const uncomp_size = std::min<uint64_t>(block_size, data_size - offset);
    block_index[i]         = {h_results[i].bytes_written, uncomp_size};
    total_compressed += h_results[i].bytes_written;
  }

  // Compact the padded per-block outputs into a contiguous device buffer.
  rmm::device_buffer d_compacted(total_compressed, stream, mr);
  {
    auto h_dsts       = cudf::detail::make_pinned_vector_async<void*>(num_blocks, stream);
    auto h_srcs       = cudf::detail::make_pinned_vector_async<void const*>(num_blocks, stream);
    auto h_sizes      = cudf::detail::make_pinned_vector_async<std::size_t>(num_blocks, stream);
    uint64_t d_offset = 0;
    for (uint64_t i = 0; i < num_blocks; ++i) {
      h_srcs[i]  = static_cast<uint8_t const*>(d_compressed.data()) + i * max_comp_block_size;
      h_dsts[i]  = static_cast<uint8_t*>(d_compacted.data()) + d_offset;
      h_sizes[i] = block_index[i].compressed_size;
      d_offset += block_index[i].compressed_size;
    }

    CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
      h_dsts.data(), h_srcs.data(), h_sizes.data(), num_blocks, stream));
  }

  auto const header = cudftable_header{compression,
                                       block_size,
                                       packed.metadata->size(),
                                       data_size,
                                       num_blocks,
                                       total_compressed,
                                       cudf::detail::packed_metadata_version};
  sink->host_write(&header, sizeof(cudftable_header));
  sink->host_write(packed.metadata->data(), header.metadata_length);
  sink->host_write(block_index.data(), num_blocks * sizeof(block_index_entry));
  write_from_device(sink, d_compacted.data(), total_compressed, stream);

  sink->flush();
}

packed_table read_cudftable(datasource* source,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  auto const header_size = sizeof(cudftable_header);
  CUDF_EXPECTS(source->size() >= header_size, "File too small to contain a valid cudftable header");

  auto header = cudftable_header{};
  source->host_read(0, header_size, reinterpret_cast<uint8_t*>(&header));
  CUDF_EXPECTS(header.magic == magic_number, "Invalid magic number in cudftable header");
  CUDF_EXPECTS(header.format_version == format_version_v1,
               "Unsupported cudftable format version: " + std::to_string(header.format_version));
  CUDF_EXPECTS(header.metadata_version == cudf::detail::packed_metadata_version,
               "Unsupported cudftable packed metadata version");

  auto const comp = static_cast<compression_type>(header.compression);

  // Validate the header size fields with overflow-safe arithmetic before using
  // them to compute file offsets or size allocations. A forged header must not
  // be able to wrap a 64-bit sum/product and still satisfy the file-size check.
  auto const file_size   = static_cast<uint64_t>(source->size());
  auto const checked_add = [](uint64_t a, uint64_t b) {
    CUDF_EXPECTS(a <= std::numeric_limits<uint64_t>::max() - b,
                 "cudftable header size arithmetic overflow");
    return a + b;
  };
  CUDF_EXPECTS(
    header.num_blocks <= std::numeric_limits<uint64_t>::max() / sizeof(block_index_entry),
    "cudftable block index size overflow");

  auto const metadata_offset    = uint64_t{header_size};
  auto const block_index_offset = checked_add(metadata_offset, header.metadata_length);
  auto const block_index_size   = header.num_blocks * sizeof(block_index_entry);
  auto const blocks_offset      = checked_add(block_index_offset, block_index_size);
  CUDF_EXPECTS(checked_add(blocks_offset, header.compressed_data_length) == file_size,
               "File size mismatch for cudftable");

  auto packed = packed_columns{};
  packed.metadata->resize(header.metadata_length);
  source->host_read(metadata_offset, header.metadata_length, packed.metadata->data());

  std::vector<block_index_entry> block_index(header.num_blocks);
  if (header.num_blocks > 0) {
    source->host_read(
      block_index_offset, block_index_size, reinterpret_cast<uint8_t*>(block_index.data()));

    // Validate the block index against the header: the per-block sizes must
    // sum to the declared data lengths so that downstream upload/decompress
    // cannot write past the allocated device buffers.
    uint64_t total_compressed   = 0;
    uint64_t total_uncompressed = 0;
    for (auto const& e : block_index) {
      total_compressed += e.compressed_size;
      total_uncompressed += e.uncompressed_size;
    }
    CUDF_EXPECTS(total_compressed == header.compressed_data_length,
                 "cudftable block index compressed-size sum mismatch");
    CUDF_EXPECTS(total_uncompressed == header.uncompressed_data_length,
                 "cudftable block index uncompressed-size sum mismatch");
  } else {
    CUDF_EXPECTS(header.compressed_data_length == 0 && header.uncompressed_data_length == 0,
                 "cudftable with zero blocks must have zero data length");
  }

  if (comp == compression_type::NONE) {
    CUDF_EXPECTS(header.num_blocks <= 1, "Uncompressed cudftable must have at most one block");
    CUDF_EXPECTS(header.compressed_data_length == header.uncompressed_data_length,
                 "Uncompressed cudftable must have matching compressed and uncompressed sizes");

    packed.gpu_data =
      std::make_unique<rmm::device_buffer>(header.uncompressed_data_length, stream, mr);
    upload_to_device(
      source, blocks_offset, header.uncompressed_data_length, packed.gpu_data->data(), stream);

    auto unpacked_view = cudf::unpack(packed);
    return packed_table{unpacked_view, std::move(packed)};
  }

  // Compressed path: block-batched decompression. The decompress() API
  // transparently routes per-block work between host and device engines, so
  // check generic support rather than device-only support.
  CUDF_EXPECTS(cudf::io::detail::is_decompression_supported(comp),
               "Unsupported compression type in cudftable header");
  CUDF_EXPECTS(header.block_size > 0,
               "cudftable block_size must be greater than zero for compressed data");

  packed.gpu_data =
    std::make_unique<rmm::device_buffer>(header.uncompressed_data_length, stream, mr);

  // Short-circuit empty-table compressed encoding: no block index, no payload.
  if (header.num_blocks == 0) {
    auto unpacked_view = cudf::unpack(packed);
    return packed_table{unpacked_view, std::move(packed)};
  }

  rmm::device_buffer d_compressed(header.compressed_data_length, stream, mr);
  upload_to_device(
    source, blocks_offset, header.compressed_data_length, d_compressed.data(), stream);

  auto [h_inputs, h_outputs] = make_io_span_tables(header.num_blocks, stream);
  uint64_t comp_offset       = 0;
  uint64_t uncomp_offset     = 0;
  for (uint64_t i = 0; i < header.num_blocks; ++i) {
    h_inputs[i] = cudf::device_span<uint8_t const>(
      static_cast<uint8_t const*>(d_compressed.data()) + comp_offset,
      block_index[i].compressed_size);
    h_outputs[i] =
      cudf::device_span<uint8_t>(static_cast<uint8_t*>(packed.gpu_data->data()) + uncomp_offset,
                                 block_index[i].uncompressed_size);
    comp_offset += block_index[i].compressed_size;
    uncomp_offset += block_index[i].uncompressed_size;
  }

  auto d_inputs  = cudf::detail::make_device_uvector_async(h_inputs, stream, mr);
  auto d_outputs = cudf::detail::make_device_uvector_async(h_outputs, stream, mr);
  auto d_results =
    rmm::device_uvector<cudf::io::detail::codec_exec_result>(header.num_blocks, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    d_results.data(), 0, header.num_blocks * sizeof(cudf::io::detail::codec_exec_result), stream));

  cudf::io::detail::decompress(comp,
                               d_inputs,
                               d_outputs,
                               d_results,
                               header.block_size,
                               header.uncompressed_data_length,
                               stream);

  auto h_results = cudf::detail::make_host_vector(d_results, stream);
  for (uint64_t i = 0; i < header.num_blocks; ++i) {
    CUDF_EXPECTS(h_results[i].status == cudf::io::detail::codec_status::SUCCESS,
                 "cudftable block decompression failed");
  }

  auto unpacked_view = cudf::unpack(packed);
  return packed_table{unpacked_view, std::move(packed)};
}

}  // namespace detail

cudftable_writer_options_builder cudftable_writer_options::builder(sink_info const& sink,
                                                                   table_view const& table)
{
  return cudftable_writer_options_builder(sink, table);
}

cudftable_reader_options_builder cudftable_reader_options::builder(source_info src)
{
  return cudftable_reader_options_builder(std::move(src));
}

}  // namespace cudf::io::experimental
