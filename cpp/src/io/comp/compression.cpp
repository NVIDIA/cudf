/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compression.hpp"

#include "common_internal.hpp"
#include "cudf/utilities/memory_resource.hpp"
#include "gpuinflate.hpp"
#include "nvcomp_adapter.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/std/bit>

#include <BS_thread_pool.hpp>
#include <zlib.h>  // GZIP compression
#include <zstd.h>

#include <numeric>

namespace cudf::io::detail {

namespace {
/**
 * @brief GZIP host compressor (includes header)
 */
std::vector<std::uint8_t> compress_gzip(host_span<uint8_t const> src)
{
  z_stream zs;
  zs.zalloc   = Z_NULL;
  zs.zfree    = Z_NULL;
  zs.opaque   = Z_NULL;
  zs.avail_in = src.size();
  zs.next_in  = reinterpret_cast<unsigned char*>(const_cast<unsigned char*>(src.data()));

  std::vector<uint8_t> dst;
  zs.avail_out = 0;
  zs.next_out  = nullptr;

  constexpr int windowbits    = 15;
  constexpr int gzip_encoding = 16;
  int ret                     = deflateInit2(
    &zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, windowbits | gzip_encoding, 8, Z_DEFAULT_STRATEGY);
  CUDF_EXPECTS(ret == Z_OK, "GZIP DEFLATE compression initialization failed.");

  uint32_t const estcomplen = deflateBound(&zs, src.size());
  dst.resize(estcomplen);
  zs.avail_out = estcomplen;
  zs.next_out  = dst.data();

  ret = deflate(&zs, Z_FINISH);
  CUDF_EXPECTS(ret == Z_STREAM_END, "GZIP DEFLATE compression failed due to insufficient space!");
  dst.resize(std::distance(dst.data(), zs.next_out));

  ret = deflateEnd(&zs);
  CUDF_EXPECTS(ret == Z_OK, "GZIP DEFLATE compression failed at deallocation");

  return dst;
}

std::vector<std::uint8_t> compress_zstd(host_span<uint8_t const> src)
{
  auto check_error_code = [](size_t err_code, size_t line) {
    if (err_code != 0) {
      std::stringstream ss;
      ss << "CUDF failure at: " << __FILE__ << ":" << line << ": " << ZSTD_getErrorName(err_code)
         << std::endl;
      throw cudf::logic_error(ss.str());
    }
  };
  auto const compressed_size_estimate = ZSTD_compressBound(src.size());
  check_error_code(ZSTD_isError(compressed_size_estimate), __LINE__);
  std::vector<std::uint8_t> compressed_buffer(compressed_size_estimate);

  // This function compresses in a single frame
  auto const compressed_size_actual =
    ZSTD_compress(reinterpret_cast<void*>(compressed_buffer.data()),
                  compressed_size_estimate,
                  reinterpret_cast<void const*>(src.data()),
                  src.size(),
                  1);
  check_error_code(ZSTD_isError(compressed_size_actual), __LINE__);
  compressed_buffer.resize(compressed_size_actual);

  return compressed_buffer;
}

namespace snappy {

template <typename T>
[[nodiscard]] T load(uint8_t const* ptr)
{
  T value;
  std::memcpy(&value, ptr, sizeof(T));
  return value;
}

class hash_table {
  std::vector<uint16_t> tbl;
  static constexpr int hash_table_bits = 15;

 public:
  hash_table() : tbl(1 << hash_table_bits, 0) {}

  void clear() { std::fill(tbl.begin(), tbl.end(), 0); }

  [[nodiscard]] uint16_t* entry(uint32_t bytes)
  {
    constexpr uint32_t multiplier = 0x1e35a7bd;
    auto const hash               = (bytes * multiplier) >> (31 - hash_table_bits);
    return tbl.data() + hash / sizeof(uint16_t);
  }
};

uint8_t* emit_literal(uint8_t* out_begin, uint8_t const* literal_begin, uint8_t const* literal_end)
{
  auto const literal_size = literal_end - literal_begin;
  if (literal_size == 0) { return out_begin; }
  auto const n = literal_size - 1;

  auto out_it = out_begin;
  if (n < 60) {
    // Fits into a single tag byte
    *out_it++ = n << 2;
  } else {
    auto const log2_n = 31 - cuda::std::countl_zero(static_cast<uint32_t>(n));
    auto const count  = (log2_n >> 3) + 1;
    *out_it++         = (59 + count) << 2;
    std::memcpy(out_it, &n, count);
    out_it += count;
  }
  std::memcpy(out_it, literal_begin, literal_size);
  return out_it + literal_size;
}

uint8_t* emit_copy(uint8_t* out_begin, size_t offset, size_t len)
{
  while (len > 0) {
    auto const copy_len = std::min(len, 64ul);
    auto const out_val  = 2 + ((copy_len - 1) << 2) + (offset << 8);
    std::memcpy(out_begin, &out_val, 3);

    out_begin += 3;
    len -= copy_len;
  }
  return out_begin;
}

size_t compress_block(host_span<uint8_t const> input, hash_table& table, host_span<uint8_t> output)
{
  auto const [in_remain, out_remain] = [&]() -> std::pair<uint8_t const*, uint8_t*> {
    auto in_it  = input.begin();
    auto out_it = output.begin();

    // The algorithm reads 8 bytes at a time, so we need to ensure there are at least 8 bytes
    auto const input_max = input.end() - sizeof(uint64_t);
    while (in_it < input_max) {
      auto const next_emit     = in_it++;
      auto data                = load<uint64_t>(in_it);
      uint32_t stride          = 1;
      uint8_t const* candidate = nullptr;

      auto word_match_found = [&]() {
        if (input_max - in_it < 16) { return false; }
        for (size_t word_idx = 0; word_idx < 4; ++word_idx) {
          for (size_t byte_idx = 0; byte_idx < sizeof(uint32_t); ++byte_idx) {
            auto const offset = sizeof(uint32_t) * word_idx + byte_idx;
            auto* const entry = table.entry(static_cast<uint32_t>(data));
            candidate         = input.begin() + *entry;
            *entry            = in_it - input.data() + offset;

            if (load<uint32_t>(candidate) == static_cast<uint32_t>(data)) {
              *(out_it++) = offset * sizeof(uint32_t);
              std::memcpy(out_it, next_emit, offset + 1);
              in_it += offset;
              out_it += offset + 1;
              stride = 1;
              return true;
            }
            data >>= 8;
          }
          // Fetch the next eight bytes
          data = load<uint64_t>(in_it + sizeof(uint32_t) * (word_idx + 1));
        }
        in_it += 16;
        return false;
      }();

      if (not word_match_found) {
        // keep looking for a match with increasing stride
        while (true) {
          auto* const entry = table.entry(static_cast<uint32_t>(data));
          candidate         = input.begin() + *entry;
          *entry            = in_it - input.begin();
          if (static_cast<uint32_t>(data) == load<uint32_t>(candidate)) {
            stride = 1;
            break;
          }

          auto const next_input = in_it + stride;
          if (next_input > input_max) {
            // Reached the end of the input without finding a match
            return {next_emit, out_it};
          }

          data  = load<uint32_t>(next_input);
          in_it = next_input;
          stride += 1;
        }

        // Emit data prior to the match as literal
        out_it = emit_literal(out_it, next_emit, in_it);
      }

      // Emit match(es)
      do {
        auto const match_len = std::mismatch(in_it, input.end(), candidate).first - in_it;
        out_it               = emit_copy(out_it, in_it - candidate, match_len);

        in_it += match_len;
        if (in_it >= input_max) {
          // Reached the end of the input, no more matches to look for
          return {in_it, out_it};
        }
        data                                    = load<uint64_t>(in_it);
        *table.entry(load<uint32_t>(in_it - 1)) = in_it - input.begin() - 1;
        auto* const entry                       = table.entry(data);
        candidate                               = input.begin() + *entry;
        *entry                                  = in_it - input.begin();

      } while (static_cast<uint32_t>(data) == load<uint32_t>(candidate));
    }

    return {in_it, out_it};
  }();

  // Emit the remaining data as a literal
  return emit_literal(out_remain, in_remain, input.end()) - output.begin();
}

void append_varint(std::vector<uint8_t>& output, size_t v)
{
  while (v > 127) {
    output.push_back((v & 0x7F) | 0x80);
    v >>= 7;
  }
  output.push_back(v);
}

[[nodiscard]] std::vector<std::uint8_t> compress(host_span<uint8_t const> src)
{
  std::vector<uint8_t> dst;
  append_varint(dst, src.size());
  dst.reserve(dst.size() + max_compressed_size(compression_type::SNAPPY, src.size()));

  hash_table table;  // reuse hash table across blocks
  constexpr size_t block_size          = 1 << 16;
  auto const block_max_compressed_size = max_compressed_size(compression_type::SNAPPY, block_size);
  for (std::size_t src_offset = 0; src_offset < src.size(); src_offset += block_size) {
    // Compress data in blocks of limited size
    auto const block = src.subspan(src_offset, std::min(src.size() - src_offset, block_size));

    auto const previous_size = dst.size();
    auto const curr_block_max_comp_size =
      (block.size() == block_size) ? block_max_compressed_size
                                   : max_compressed_size(compression_type::SNAPPY, block.size());
    dst.resize(previous_size + curr_block_max_comp_size);
    auto const block_dst =
      host_span<uint8_t>{dst.data() + previous_size, dst.size() - previous_size};

    table.clear();
    auto const comp_block_size = compress_block(block, table, block_dst);
    dst.resize(previous_size + comp_block_size);
  }

  return dst;
}

}  // namespace snappy

void device_compress(compression_type compression,
                     device_span<device_span<uint8_t const> const> inputs,
                     device_span<device_span<uint8_t> const> outputs,
                     device_span<codec_exec_result> results,
                     rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  if (compression == compression_type::NONE or inputs.empty()) { return; }

  auto const nvcomp_type = to_nvcomp_compression(compression);
  auto nvcomp_disabled   = nvcomp_type.has_value() ? nvcomp::is_compression_disabled(*nvcomp_type)
                                                   : "invalid compression type";
  if (not nvcomp_disabled) {
    return nvcomp::batched_compress(*nvcomp_type, inputs, outputs, results, stream);
  }

  switch (compression) {
    case compression_type::SNAPPY: return gpu_snap(inputs, outputs, results, stream);
    default: CUDF_FAIL("Compression error: " + nvcomp_disabled.value());
  }
}

void host_compress(compression_type compression,
                   device_span<device_span<uint8_t const> const> inputs,
                   device_span<device_span<uint8_t> const> outputs,
                   device_span<codec_exec_result> results,
                   rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  if (compression == compression_type::NONE or inputs.empty()) { return; }

  auto const num_chunks = inputs.size();
  auto const h_inputs   = cudf::detail::make_host_vector_async(inputs, stream);
  auto const h_outputs  = cudf::detail::make_host_vector_async(outputs, stream);
  stream.synchronize();

  auto h_results = cudf::detail::make_pinned_vector<codec_exec_result>(results, stream);

  std::vector<std::future<std::pair<size_t, size_t>>> tasks;
  auto const num_streams =
    std::min<std::size_t>(num_chunks, cudf::detail::host_worker_pool().get_thread_count());
  auto const streams = cudf::detail::fork_streams(stream, num_streams);
  for (size_t i = 0; i < num_chunks; ++i) {
    auto const cur_stream = streams[i % streams.size()];
    if (h_results[i].status == codec_status::SKIPPED) { continue; }
    auto task = [d_in = h_inputs[i], d_out = h_outputs[i], cur_stream, compression, i]() {
      auto h_in = cudf::detail::make_pinned_vector<uint8_t>(d_in, cur_stream);

      auto const h_out = compress(compression, h_in);
      h_in.clear();  // Free pinned memory as soon as possible

      cudf::detail::cuda_memcpy<uint8_t>(d_out.subspan(0, h_out.size()), h_out, cur_stream);
      return std::pair<size_t, size_t>{i, h_out.size()};
    };
    tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(std::move(task)));
  }
  for (auto& task : tasks) {
    auto const [idx, bytes_written] = task.get();
    h_results[idx]                  = {bytes_written, codec_status::SUCCESS};
  }

  cudf::detail::cuda_memcpy<codec_exec_result>(results, h_results, stream);
}

[[nodiscard]] host_engine_state get_host_engine_state(compression_type compression)
{
  auto const has_host_support   = is_host_compression_supported(compression);
  auto const has_device_support = is_device_compression_supported(compression);
  CUDF_EXPECTS(has_host_support or has_device_support,
               "Unsupported compression type: " + compression_type_name(compression));
  if (not has_host_support) { return host_engine_state::OFF; }
  if (not has_device_support) { return host_engine_state::ON; }

  // If both host and device compression are supported, dispatch based on the environment variable
  auto const env_var = cudf::detail::getenv_or("LIBCUDF_HOST_COMPRESSION", std::string{"OFF"});

  if (env_var == "AUTO") {
    return host_engine_state::AUTO;
  } else if (env_var == "HYBRID") {
    return host_engine_state::HYBRID;
  } else if (env_var == "OFF") {
    return host_engine_state::OFF;
  } else if (env_var == "ON") {
    return host_engine_state::ON;
  }
  CUDF_FAIL("Invalid LIBCUDF_HOST_COMPRESSION value: " + env_var);
}

}  // namespace

std::optional<size_t> compress_max_allowed_chunk_size(compression_type compression)
{
  if (auto nvcomp_type = to_nvcomp_compression(compression);
      nvcomp_type.has_value() and not nvcomp::is_compression_disabled(*nvcomp_type)) {
    return nvcomp::compress_max_allowed_chunk_size(*nvcomp_type);
  }
  return std::nullopt;
}

[[nodiscard]] size_t compress_required_chunk_alignment(compression_type compression)
{
  auto nvcomp_type = to_nvcomp_compression(compression);
  if (compression == compression_type::NONE or not nvcomp_type.has_value() or
      nvcomp::is_compression_disabled(*nvcomp_type)) {
    return 1ul;
  }

  return nvcomp::compress_required_alignment(*nvcomp_type);
}

[[nodiscard]] size_t max_compressed_size(compression_type compression, size_t uncompressed_size)
{
  if (compression == compression_type::NONE) { return uncompressed_size; }

  if (auto nvcomp_type = to_nvcomp_compression(compression); nvcomp_type.has_value()) {
    return nvcomp::compress_max_output_chunk_size(*nvcomp_type, uncompressed_size);
  }
  CUDF_FAIL("Unsupported compression type: " + compression_type_name(compression));
}

std::vector<std::uint8_t> compress(compression_type compression, host_span<uint8_t const> src)
{
  CUDF_FUNC_RANGE();

  switch (compression) {
    case compression_type::GZIP: return detail::compress_gzip(src);
    case compression_type::SNAPPY: return detail::snappy::compress(src);
    case compression_type::ZSTD: return detail::compress_zstd(src);
    default:
      CUDF_FAIL("Unsupported compression type: " + detail::compression_type_name(compression));
  }
}

void compress(compression_type compression,
              device_span<device_span<uint8_t const> const> inputs,
              device_span<device_span<uint8_t> const> outputs,
              device_span<codec_exec_result> results,
              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  if (inputs.empty()) { return; }

  // sort inputs by size, largest first
  auto const [sorted_inputs, sorted_outputs, order] =
    sort_compression_tasks(inputs, outputs, stream, cudf::get_current_device_resource_ref());
  device_span<device_span<uint8_t const> const> inputs_view = sorted_inputs;
  device_span<device_span<uint8_t> const> outputs_view      = sorted_outputs;

  auto const split_idx =
    split_compression_tasks(inputs_view,
                            outputs_view,
                            get_host_engine_state(compression),
                            cudf::detail::getenv_or("LIBCUDF_HOST_COMPRESSION_THRESHOLD",
                                                    default_host_compression_auto_threshold),
                            cudf::detail::getenv_or("LIBCUDF_HOST_COMPRESSION_RATIO",
                                                    default_host_device_compression_cost_ratio),
                            stream);

  auto tmp_results = cudf::detail::make_device_uvector_async<detail::codec_exec_result>(
    results, stream, cudf::get_current_device_resource_ref());
  device_span<codec_exec_result> results_view = tmp_results;

  auto const streams = cudf::detail::fork_streams(stream, 2);
  detail::device_compress(compression,
                          inputs_view.subspan(split_idx, inputs_view.size() - split_idx),
                          outputs_view.subspan(split_idx, outputs_view.size() - split_idx),
                          results_view.subspan(split_idx, results_view.size() - split_idx),
                          streams[0]);
  detail::host_compress(compression,
                        inputs_view.subspan(0, split_idx),
                        outputs_view.subspan(0, split_idx),
                        results_view.subspan(0, split_idx),
                        streams[1]);
  cudf::detail::join_streams(streams, stream);

  copy_results_to_original_order(results_view, results, order, stream);
}

[[nodiscard]] bool is_host_compression_supported(compression_type compression)
{
  switch (compression) {
    case compression_type::GZIP:
    case compression_type::SNAPPY:
    case compression_type::ZSTD:
    case compression_type::NONE: return true;
    default: return false;
  }
}

[[nodiscard]] bool is_device_compression_supported(compression_type compression)
{
  auto const nvcomp_type = detail::to_nvcomp_compression(compression);
  switch (compression) {
    case compression_type::GZIP:
    case compression_type::LZ4:
    case compression_type::ZLIB:
    case compression_type::ZSTD:
      return not detail::nvcomp::is_compression_disabled(nvcomp_type.value());
    case compression_type::SNAPPY:
    case compression_type::NONE: return true;
    default: return false;
  }
}

[[nodiscard]] bool is_compression_supported(compression_type compression)
{
  return is_host_compression_supported(compression) or is_device_compression_supported(compression);
}

}  // namespace cudf::io::detail
