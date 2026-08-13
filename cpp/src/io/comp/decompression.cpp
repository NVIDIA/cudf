/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "decompression.hpp"

#include "common_internal.hpp"
#include "cudf/utilities/memory_resource.hpp"
#include "gpuinflate.hpp"
#include "nvcomp_adapter.hpp"
#include "unbz2.hpp"  // bz2 uncompress

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <zlib.h>  // uncompress
#include <zstd.h>

#include <cstdint>
#include <cstring>  // memset
#include <future>
#include <sstream>

namespace cudf::io::detail {

namespace {

#pragma pack(push, 1)
struct gz_file_header_s {
  uint8_t id1;        // 0x1f
  uint8_t id2;        // 0x8b
  uint8_t comp_mthd;  // compression method (0-7=reserved, 8=deflate)
  uint8_t flags;      // flags (GZIPHeaderFlag)
  uint8_t mtime[4];   // If non-zero: modification time (Unix format)  // NOLINT
  uint8_t xflags;     // Extra compressor-specific flags
  uint8_t os;         // OS id
};

struct zip_eocd_s  // end of central directory
{
  uint32_t sig;            // 0x0605'4b50
  uint16_t disk_id;        // number of this disk
  uint16_t start_disk;     // number of the disk with the start of the central directory
  uint16_t num_entries;    // number of entries in the central dir on this disk
  uint16_t total_entries;  // total number of entries in the central dir
  uint32_t cdir_size;      // size of the central directory
  uint32_t cdir_offset;    // offset of start of central directory with respect to the starting disk
  // number uint16_t comment_len;   // comment length (excluded from struct)
};

struct zip64_eocdl  // end of central dir locator
{
  uint32_t sig;         // 0x0706'4b50
  uint32_t disk_start;  // number of the disk with the start of the zip64 end of central directory
  uint64_t eocdr_ofs;   // relative offset of the zip64 end of central directory record
  uint32_t num_disks;   // total number of disks
};

struct zip_cdfh_s  // central directory file header
{
  uint32_t sig;          // 0x0201'4b50
  uint16_t ver;          // version made by
  uint16_t min_ver;      // version needed to extract
  uint16_t gp_flags;     // general purpose bit flag
  uint16_t comp_method;  // compression method
  uint16_t file_time;    // last mod file time
  uint16_t file_date;    // last mod file date
  uint32_t crc32;        // crc - 32
  uint32_t comp_size;    // compressed size
  uint32_t uncomp_size;  // uncompressed size
  uint16_t fname_len;    // filename length
  uint16_t extra_len;    // extra field length
  uint16_t comment_len;  // file comment length
  uint16_t start_disk;   // disk number start
  uint16_t int_fattr;    // internal file attributes
  uint32_t ext_fattr;    // external file attributes
  uint32_t hdr_ofs;      // relative offset of local header
};

struct zip_lfh_s {
  uint32_t sig;          // 0x0403'4b50
  uint16_t ver;          // version needed to extract
  uint16_t gp_flags;     // general purpose bit flag
  uint16_t comp_method;  // compression method
  uint16_t file_time;    // last mod file time
  uint16_t file_date;    // last mod file date
  uint32_t crc32;        // crc - 32
  uint32_t comp_size;    // compressed size
  uint32_t uncomp_size;  // uncompressed size
  uint16_t fname_len;    // filename length
  uint16_t extra_len;    // extra field length
};

#pragma pack(pop)

struct gz_archive_s {
  gz_file_header_s const* fhdr;
  uint16_t hcrc16;  // header crc16 if present
  uint16_t xlen;
  uint8_t const* fxtra;      // xlen bytes (optional)
  uint8_t const* fname;      // zero-terminated original filename if present
  uint8_t const* fcomment;   // zero-terminated comment if present
  uint8_t const* comp_data;  // compressed data
  size_t comp_len;           // Compressed data length
  uint32_t crc32;            // CRC32 of uncompressed data
  uint32_t isize;            // Input size modulo 2^32
};

struct zip_archive_s {
  zip_eocd_s const* eocd;    // end of central directory
  zip64_eocdl const* eocdl;  // end of central dir locator (optional)
  zip_cdfh_s const* cdfh;    // start of central directory file headers
};

bool ParseGZArchive(gz_archive_s* dst, uint8_t const* raw, size_t len)
{
  gz_file_header_s const* fhdr = nullptr;

  if (!dst) return false;
  memset(dst, 0, sizeof(gz_archive_s));
  if (len < sizeof(gz_file_header_s) + 8) return false;
  fhdr = reinterpret_cast<gz_file_header_s const*>(raw);
  if (fhdr->id1 != 0x1f || fhdr->id2 != 0x8b) return false;
  dst->fhdr = fhdr;
  raw += sizeof(gz_file_header_s);
  len -= sizeof(gz_file_header_s);
  if (fhdr->flags & GZIPHeaderFlag::fextra) {
    uint32_t xlen = 0;

    if (len < 2) return false;
    xlen = raw[0] | (raw[1] << 8);
    raw += 2;
    len -= 2;
    if (len < xlen) return false;
    dst->xlen  = (uint16_t)xlen;
    dst->fxtra = raw;
    raw += xlen;
    len -= xlen;
  }
  if (fhdr->flags & GZIPHeaderFlag::fname) {
    size_t l  = 0;
    uint8_t c = 0;
    do {
      if (l >= len) return false;
      c = raw[l];
      l++;
    } while (c != 0);
    dst->fname = raw;
    raw += l;
    len -= l;
  }
  if (fhdr->flags & GZIPHeaderFlag::fcomment) {
    size_t l  = 0;
    uint8_t c = 0;
    do {
      if (l >= len) return false;
      c = raw[l];
      l++;
    } while (c != 0);
    dst->fcomment = raw;
    raw += l;
    len -= l;
  }
  if (fhdr->flags & GZIPHeaderFlag::fhcrc) {
    if (len < 2) return false;
    dst->hcrc16 = raw[0] | (raw[1] << 8);
    raw += 2;
    len -= 2;
  }
  if (len < 8) return false;
  dst->crc32 = raw[len - 8] | (raw[len - 7] << 8) | (raw[len - 6] << 16) | (raw[len - 5] << 24);
  dst->isize = raw[len - 4] | (raw[len - 3] << 8) | (raw[len - 2] << 16) | (raw[len - 1] << 24);
  len -= 8;
  dst->comp_data = raw;
  dst->comp_len  = len;
  return (fhdr->comp_mthd == 8 && len > 0);
}

bool OpenZipArchive(zip_archive_s* dst, uint8_t const* raw, size_t len)
{
  memset(dst, 0, sizeof(zip_archive_s));
  // Find the end of central directory
  if (len >= sizeof(zip_eocd_s) + 2) {
    for (ptrdiff_t i = len - sizeof(zip_eocd_s) - 2;
         i + sizeof(zip_eocd_s) + 2 + 0xffff >= len && i >= 0;
         i--) {
      auto const* eocd = reinterpret_cast<zip_eocd_s const*>(raw + i);
      if (eocd->sig == 0x0605'4b50 &&
          eocd->disk_id == eocd->start_disk  // multi-file archives not supported
          && eocd->num_entries == eocd->total_entries &&
          eocd->cdir_size >= sizeof(zip_cdfh_s) * eocd->num_entries && eocd->cdir_offset < len &&
          i + *reinterpret_cast<uint16_t const*>(eocd + 1) <= static_cast<ptrdiff_t>(len)) {
        auto const* cdfh = reinterpret_cast<zip_cdfh_s const*>(raw + eocd->cdir_offset);
        dst->eocd        = eocd;
        if (std::cmp_greater_equal(i, sizeof(zip64_eocdl))) {
          auto const* eocdl = reinterpret_cast<zip64_eocdl const*>(raw + i - sizeof(zip64_eocdl));
          if (eocdl->sig == 0x0706'4b50) { dst->eocdl = eocdl; }
        }
        // Start of central directory
        if (cdfh->sig == 0x0201'4b50) { dst->cdfh = cdfh; }
      }
    }
  }
  return (dst->eocd && dst->cdfh);
}

/**
 * @brief Uncompresses a raw DEFLATE stream to a char vector.
 * The vector will be grown to match the uncompressed size
 * Optimized for the case where the initial size is the uncompressed
 * size truncated to 32-bit, and grows the buffer in 1GB increments.
 *
 * @param[out] dst Destination vector
 * @param[in] comp_data Raw compressed data
 * @param[in] comp_len Compressed data size
 */
void cpu_inflate_vector(std::vector<uint8_t>& dst, uint8_t const* comp_data, size_t comp_len)
{
  z_stream strm{};
  strm.next_in   = const_cast<Bytef*>(comp_data);
  strm.avail_in  = comp_len;
  strm.next_out  = dst.data();
  strm.avail_out = dst.size();
  auto zerr      = inflateInit2(&strm, -MAX_WBITS);  // for raw data without GZIP headers
  CUDF_EXPECTS(zerr == 0, "Error in DEFLATE stream: inflateInit2 failed");
  do {
    if (strm.avail_out == 0) {
      dst.resize(strm.total_out + (1 << 30));
      strm.avail_out = dst.size() - strm.total_out;
      strm.next_out  = reinterpret_cast<uint8_t*>(dst.data()) + strm.total_out;
    }
    zerr = inflate(&strm, Z_SYNC_FLUSH);
  } while ((zerr == Z_BUF_ERROR || zerr == Z_OK) && strm.avail_out == 0 &&
           strm.total_out == dst.size());
  dst.resize(strm.total_out);
  inflateEnd(&strm);
  CUDF_EXPECTS(zerr == Z_STREAM_END, "Error in DEFLATE stream: Z_STREAM_END not encountered");
}

/**
 * @brief ZLIB host decompressor (no header)
 */
size_t decompress_zlib(host_span<uint8_t const> src, host_span<uint8_t> dst)
{
  if (dst.empty()) { return 0; }

  z_stream strm{};
  strm.next_in   = const_cast<Bytef*>(src.data());
  strm.avail_in  = src.size();
  strm.next_out  = dst.data();
  strm.avail_out = dst.size();
  auto zerr      = inflateInit2(&strm, -MAX_WBITS);  // for raw data without GZIP headers
  CUDF_EXPECTS(zerr == 0, "Error in DEFLATE stream: inflateInit2 failed");

  zerr = inflate(&strm, Z_FINISH);
  CUDF_EXPECTS(zerr == Z_STREAM_END, "Error in DEFLATE stream: Z_STREAM_END not encountered");

  inflateEnd(&strm);
  return strm.total_out;
}

/**
 * @brief GZIP host decompressor (includes header)
 */
size_t decompress_gzip(host_span<uint8_t const> src, host_span<uint8_t> dst)
{
  gz_archive_s gz{};
  auto const parse_succeeded = ParseGZArchive(&gz, src.data(), src.size());
  CUDF_EXPECTS(parse_succeeded, "Failed to parse GZIP header");
  return decompress_zlib({gz.comp_data, gz.comp_len}, dst);
}

/**
 * @brief SNAPPY host decompressor
 */
size_t decompress_snappy(host_span<uint8_t const> src, host_span<uint8_t> dst)
{
  CUDF_EXPECTS(not src.empty(), "Empty Snappy decompress input", std::length_error);

  uint32_t uncompressed_size = 0, bytes_left = 0, dst_pos = 0;
  auto cur       = src.begin();
  auto const end = src.end();
  // Read uncompressed length (varint)
  {
    uint32_t l = 0, c = 0;
    uncompressed_size = 0;
    do {
      c              = *cur++;
      auto const lo7 = c & 0x7f;
      if (l >= 28 && c > 0xf) { return 0; }
      uncompressed_size |= lo7 << l;
      l += 7;
    } while (c > 0x7f && cur < end);
  }

  if (uncompressed_size == 0) {
    CUDF_EXPECTS(cur == end, "Non-empty compressed data for empty output in Snappy decompress");
    return 0;
  }
  // If the uncompressed size is not zero, the input must not be empty
  CUDF_EXPECTS(cur < end, "Missing data in Snappy decompress input");

  CUDF_EXPECTS(uncompressed_size <= dst.size(), "Output buffer too small for Snappy decompression");

  // Decode lz77
  dst_pos    = 0;
  bytes_left = uncompressed_size;
  do {
    uint32_t blen = *cur++;

    if (blen & 3) {
      // Copy
      uint32_t offset = 0;
      if (blen & 2) {
        // xxxxxx1x: copy with 6-bit length, 2-byte or 4-byte offset
        if (cur + 2 > end) break;
        offset = *reinterpret_cast<uint16_t const*>(cur);
        cur += 2;
        if (blen & 1)  // 4-byte offset
        {
          if (cur + 2 > end) break;
          offset |= (*reinterpret_cast<uint16_t const*>(cur)) << 16;
          cur += 2;
        }
        blen = (blen >> 2) + 1;
      } else {
        // xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
        if (cur >= end) break;
        offset = ((blen & 0xe0) << 3) | (*cur++);
        blen   = ((blen >> 2) & 7) + 4;
      }
      if (offset - 1u >= dst_pos || blen > bytes_left) break;
      bytes_left -= blen;
      do {
        dst[dst_pos] = dst[dst_pos - offset];
        dst_pos++;
      } while (--blen);
    } else {
      // xxxxxx00: literal
      blen >>= 2;
      if (blen >= 60) {
        uint32_t const num_bytes = blen - 59;
        if (cur + num_bytes >= end) break;
        blen = cur[0];
        if (num_bytes > 1) {
          blen |= cur[1] << 8;
          if (num_bytes > 2) {
            blen |= cur[2] << 16;
            if (num_bytes > 3) { blen |= cur[3] << 24; }
          }
        }
        cur += num_bytes;
      }
      blen++;
      if (cur + blen > end || blen > bytes_left) break;
      memcpy(dst.data() + dst_pos, cur, blen);
      cur += blen;
      dst_pos += blen;
      bytes_left -= blen;
    }
  } while (bytes_left && cur < end);
  CUDF_EXPECTS(bytes_left == 0, "Snappy Decompression failed");
  return uncompressed_size;
}

size_t decompress_zstd(host_span<uint8_t const> src, host_span<uint8_t> dst)
{
  auto check_error_code = [](size_t err_code, size_t line) {
    if (err_code != 0) {
      std::stringstream ss;
      ss << "CUDF failure at: " << __FILE__ << ":" << line << ": " << ZSTD_getErrorName(err_code)
         << std::endl;
      throw cudf::logic_error(ss.str());
    }
  };
  size_t const decompressed_bytes = ZSTD_decompress(reinterpret_cast<void*>(dst.data()),
                                                    dst.size(),
                                                    reinterpret_cast<void const*>(src.data()),
                                                    src.size());
  check_error_code(ZSTD_isError(decompressed_bytes), __LINE__);
  return decompressed_bytes;
}

struct source_properties {
  compression_type compression = compression_type::NONE;
  uint8_t const* comp_data     = nullptr;
  size_t comp_len              = 0;
  size_t uncomp_len            = 0;
};

source_properties get_source_properties(compression_type compression, host_span<uint8_t const> src)
{
  uint8_t const* comp_data = src.data();
  size_t comp_len          = src.size();
  size_t uncomp_len        = 0;

  switch (compression) {
    case compression_type::AUTO:
    case compression_type::GZIP: {
      gz_archive_s gz{};
      auto const parse_succeeded = ParseGZArchive(&gz, src.data(), src.size());
      if (compression != compression_type::AUTO) {
        CUDF_EXPECTS(parse_succeeded,
                     "Failed to parse GZIP header while fetching source properties");
      }
      if (parse_succeeded) {
        compression = compression_type::GZIP;
        comp_data   = gz.comp_data;
        comp_len    = gz.comp_len;
        uncomp_len  = gz.isize;
      }
      if (compression != compression_type::AUTO) { break; }
      [[fallthrough]];
    }
    case compression_type::ZIP: {
      zip_archive_s za{};
      auto const open_succeeded = OpenZipArchive(&za, src.data(), src.size());
      if (compression != compression_type::AUTO) {
        CUDF_EXPECTS(open_succeeded, "Failed to parse ZIP header while fetching source properties");
      }
      if (open_succeeded) {
        size_t cdfh_ofs = 0;
        for (uint16_t i = 0; i < za.eocd->num_entries; i++) {
          auto const* cdfh = reinterpret_cast<zip_cdfh_s const*>(
            reinterpret_cast<uint8_t const*>(za.cdfh) + cdfh_ofs);
          int const cdfh_len =
            sizeof(zip_cdfh_s) + cdfh->fname_len + cdfh->extra_len + cdfh->comment_len;
          if (cdfh_ofs + cdfh_len > za.eocd->cdir_size || cdfh->sig != 0x0201'4b50) {
            // Bad cdir
            break;
          }
          // For now, only accept with non-zero file sizes and DEFLATE
          if (cdfh->comp_method == 8 && cdfh->comp_size > 0 && cdfh->uncomp_size > 0) {
            size_t const lfh_ofs = cdfh->hdr_ofs;
            auto const* lfh      = reinterpret_cast<zip_lfh_s const*>(src.data() + lfh_ofs);
            if (lfh_ofs + sizeof(zip_lfh_s) <= src.size() && lfh->sig == 0x0403'4b50 &&
                lfh_ofs + sizeof(zip_lfh_s) + lfh->fname_len + lfh->extra_len <= src.size()) {
              if (lfh->comp_method == 8 && lfh->comp_size > 0 && lfh->uncomp_size > 0) {
                size_t const file_start =
                  lfh_ofs + sizeof(zip_lfh_s) + lfh->fname_len + lfh->extra_len;
                size_t const file_end = file_start + lfh->comp_size;
                if (file_end <= src.size()) {
                  // Pick the first valid file of non-zero size (only 1 file expected in archive)
                  compression = compression_type::ZIP;
                  comp_data   = src.data() + file_start;
                  comp_len    = lfh->comp_size;
                  uncomp_len  = lfh->uncomp_size;
                  break;
                }
              }
            }
          }
          cdfh_ofs += cdfh_len;
        }
      }
      if (compression != compression_type::AUTO) { break; }
      [[fallthrough]];
    }
    case compression_type::ZSTD: {
      auto const ret =
        ZSTD_findDecompressedSize(reinterpret_cast<void const*>(src.data()), src.size());
      uncomp_len = static_cast<size_t>(ret);
      if (compression != compression_type::AUTO) {
        CUDF_EXPECTS(ret != ZSTD_CONTENTSIZE_UNKNOWN,
                     "Decompressed ZSTD size cannot be determined");
        CUDF_EXPECTS(ret != ZSTD_CONTENTSIZE_ERROR, "Error determining decompressed ZSTD size");
      } else if (ret != ZSTD_CONTENTSIZE_UNKNOWN && ret != ZSTD_CONTENTSIZE_ERROR) {
        compression = compression_type::ZSTD;
      }
      if (compression != compression_type::AUTO) { break; }
      [[fallthrough]];
    }
      // Snappy is detected after ZSTD since the
      // magic characters checked at the beginning of the input buffer
      // are valid for ZSTD compression as well.
    case compression_type::SNAPPY: {
      uncomp_len     = 0;
      auto cur       = src.begin();
      auto const end = src.end();
      // Read uncompressed length (varint)
      {
        uint32_t l = 0, c = 0;
        do {
          c              = *cur++;
          auto const lo7 = c & 0x7f;
          if (l >= 28 && c > 0xf) {
            uncomp_len = 0;
            break;
          }
          uncomp_len |= lo7 << l;
          l += 7;
        } while (c > 0x7f && cur < end);
        if (compression != compression_type::AUTO) {
          CUDF_EXPECTS(uncomp_len != 0 && cur < end,
                       "Error in retrieving SNAPPY source properties");
        }
        if (uncomp_len != 0 && cur < end) { compression = compression_type::SNAPPY; }
      }
      if (compression != compression_type::AUTO) { break; }
      [[fallthrough]];
    }
    default: {
      uncomp_len = 0;
    }
  }

  return source_properties{compression, comp_data, comp_len, uncomp_len};
}

void device_decompress(compression_type compression,
                       device_span<device_span<uint8_t const> const> inputs,
                       device_span<device_span<uint8_t> const> outputs,
                       device_span<codec_exec_result> results,
                       size_t max_uncomp_chunk_size,
                       size_t max_total_uncomp_size,
                       rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  if (compression == compression_type::NONE or inputs.empty()) { return; }

  auto const nvcomp_type            = to_nvcomp_compression(compression);
  auto const nvcomp_disabled_reason = nvcomp_type.has_value()
                                        ? nvcomp::is_decompression_disabled(*nvcomp_type)
                                        : "invalid compression type";
  if (not nvcomp_disabled_reason) {
    return nvcomp::batched_decompress(
      *nvcomp_type, inputs, outputs, results, max_uncomp_chunk_size, max_total_uncomp_size, stream);
  }

  switch (compression) {
    case compression_type::BROTLI: return gpu_debrotli(inputs, outputs, results, stream);
    case compression_type::GZIP:
      return gpuinflate(inputs, outputs, results, gzip_header_included::YES, stream);
    case compression_type::SNAPPY: return gpu_unsnap(inputs, outputs, results, stream);
    case compression_type::ZLIB:
      return gpuinflate(inputs, outputs, results, gzip_header_included::NO, stream);
    default: CUDF_FAIL("Compression error: " + nvcomp_disabled_reason.value());
  }
}

void host_decompress(compression_type compression,
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

  std::vector<std::future<size_t>> tasks;
  auto const num_streams =
    std::min<std::size_t>(num_chunks, cudf::detail::host_worker_pool().get_thread_count());
  auto const streams = cudf::detail::fork_streams(stream, num_streams);
  for (size_t i = 0; i < num_chunks; ++i) {
    auto const cur_stream = streams[i % streams.size()];
    auto task = [d_in = h_inputs[i], d_out = h_outputs[i], cur_stream, compression]() -> size_t {
      auto h_in = cudf::detail::make_pinned_vector_async<uint8_t>(d_in, cur_stream);

      auto h_out             = cudf::detail::make_pinned_vector<uint8_t>(d_out.size(), cur_stream);
      auto const uncomp_size = decompress(compression, h_in, h_out);
      h_in.clear();  // Free pinned memory as soon as possible

      cudf::detail::cuda_memcpy<uint8_t>(d_out.subspan(0, uncomp_size),
                                         host_span<uint8_t>{h_out}.subspan(0, uncomp_size),
                                         cur_stream);
      return uncomp_size;
    };
    tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(std::move(task)));
  }
  auto h_results = cudf::detail::make_pinned_vector<codec_exec_result>(num_chunks, stream);
  for (auto i = 0ul; i < num_chunks; ++i) {
    h_results[i] = {tasks[i].get(), codec_status::SUCCESS};
  }

  cudf::detail::join_streams(streams, stream);
  cudf::detail::cuda_memcpy<codec_exec_result>(results, h_results, stream);
}

[[nodiscard]] host_engine_state get_host_engine_state(compression_type compression)
{
  auto const has_host_support   = is_host_decompression_supported(compression);
  auto const has_device_support = is_device_decompression_supported(compression);
  CUDF_EXPECTS(has_host_support or has_device_support,
               "Unsupported compression type: " + compression_type_name(compression));
  if (not has_host_support) { return host_engine_state::OFF; }
  if (not has_device_support) { return host_engine_state::ON; }

  // If both host and device compression are supported, dispatch based on the environment variable
  auto const env_var = cudf::detail::getenv_or("LIBCUDF_HOST_DECOMPRESSION", std::string{"OFF"});

  if (env_var == "AUTO") {
    return host_engine_state::AUTO;
  } else if (env_var == "HYBRID") {
    return host_engine_state::HYBRID;
  } else if (env_var == "OFF") {
    return host_engine_state::OFF;
  } else if (env_var == "ON") {
    return host_engine_state::ON;
  }
  CUDF_FAIL("Invalid LIBCUDF_HOST_DECOMPRESSION value: " + env_var);
}

}  // namespace

size_t get_uncompressed_size(compression_type compression, host_span<uint8_t const> src)
{
  return get_source_properties(compression, src).uncomp_len;
}

[[nodiscard]] size_t get_decompression_scratch_size(decompression_info const& di)
{
  if (di.type == compression_type::NONE or
      get_host_engine_state(di.type) == host_engine_state::ON) {
    return 0;
  }

  auto const nvcomp_type     = to_nvcomp_compression(di.type);
  auto const nvcomp_disabled = nvcomp_type.has_value()
                                 ? nvcomp::is_decompression_disabled(*nvcomp_type)
                                 : "invalid compression type";
  if (not nvcomp_disabled) {
    return nvcomp::batched_decompress_temp_size(
      nvcomp_type.value(), di.num_pages, di.max_page_decompressed_size, di.total_decompressed_size);
  }

  if (di.type == compression_type::BROTLI) return get_gpu_debrotli_scratch_size(di.num_pages);
  // only Brotli kernel requires scratch memory
  return 0;
}

[[nodiscard]] size_t get_decompression_scratch_size_ex(
  compression_type compression,
  device_span<device_span<uint8_t const> const> inputs,
  size_t max_uncomp_chunk_size,
  size_t max_total_uncomp_size,
  rmm::cuda_stream_view stream)
{
  if (compression == compression_type::NONE or
      get_host_engine_state(compression) == host_engine_state::ON) {
    return 0;
  }

  auto const nvcomp_type     = to_nvcomp_compression(compression);
  auto const nvcomp_disabled = nvcomp_type.has_value()
                                 ? nvcomp::is_decompression_disabled(*nvcomp_type)
                                 : "invalid compression type";
  CUDF_EXPECTS(
    !nvcomp_disabled,
    "Cannot compute decompression scratch size for " + compression_type_name(compression));
  return nvcomp::batched_decompress_temp_size_ex(
    nvcomp_type.value(), inputs, max_uncomp_chunk_size, max_total_uncomp_size, stream);
}

[[nodiscard]] bool is_decompression_scratch_size_ex_supported(compression_type compression)
{
  auto const nvcomp_type     = to_nvcomp_compression(compression);
  auto const nvcomp_disabled = nvcomp_type.has_value()
                                 ? nvcomp::is_decompression_disabled(*nvcomp_type)
                                 : "invalid compression type";
  if (nvcomp_disabled) { return false; }
  return nvcomp::is_batched_decompress_temp_size_ex_supported(nvcomp_type.value());
}

size_t decompress(compression_type compression,
                  host_span<uint8_t const> src,
                  host_span<uint8_t> dst)
{
  CUDF_FUNC_RANGE();

  switch (compression) {
    case compression_type::GZIP: return detail::decompress_gzip(src, dst);
    case compression_type::ZLIB: return detail::decompress_zlib(src, dst);
    case compression_type::SNAPPY: return detail::decompress_snappy(src, dst);
    case compression_type::ZSTD: return detail::decompress_zstd(src, dst);
    default:
      CUDF_FAIL("Unsupported compression type: " + detail::compression_type_name(compression));
  }
}

std::vector<uint8_t> decompress(compression_type compression, host_span<uint8_t const> src)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(src.data() != nullptr, "Decompression: Source cannot be nullptr");
  CUDF_EXPECTS(not src.empty(), "Decompression: Source size cannot be 0");

  auto srcprops = detail::get_source_properties(compression, src);
  CUDF_EXPECTS(srcprops.comp_data != nullptr and srcprops.comp_len > 0,
               "Unsupported compressed stream type");

  if (srcprops.uncomp_len <= 0) {
    srcprops.uncomp_len =
      srcprops.comp_len * 4 + 4096;  // In case uncompressed size isn't known in advance, assume
                                     // ~4:1 compression for initial size
  }

  if (srcprops.compression == compression_type::GZIP) {
    // INFLATE
    std::vector<uint8_t> dst(srcprops.uncomp_len);
    detail::decompress_gzip(src, dst);
    return dst;
  }
  if (srcprops.compression == compression_type::ZSTD) {
    std::vector<uint8_t> dst(srcprops.uncomp_len);
    auto const decompressed_bytes = detail::decompress_zstd(src, dst);
    CUDF_EXPECTS(decompressed_bytes == srcprops.uncomp_len,
                 "Error in ZSTD decompression: Mismatch in actual size of decompressed buffer and "
                 "estimated size ");
    dst.resize(decompressed_bytes);
    return dst;
  }
  if (srcprops.compression == compression_type::ZIP) {
    std::vector<uint8_t> dst(srcprops.uncomp_len);
    detail::cpu_inflate_vector(dst, srcprops.comp_data, srcprops.comp_len);
    return dst;
  }
  if (srcprops.compression == compression_type::BZIP2) {
    size_t src_ofs = 0;
    size_t dst_ofs = 0;
    int bz_err     = 0;
    std::vector<uint8_t> dst(srcprops.uncomp_len);
    do {
      size_t dst_len = srcprops.uncomp_len - dst_ofs;
      bz_err         = cpu_bz2_uncompress(
        srcprops.comp_data, srcprops.comp_len, dst.data() + dst_ofs, &dst_len, &src_ofs);
      if (bz_err == BZ_OUTBUFF_FULL) {
        // TBD: We could infer the compression ratio based on produced/consumed byte counts
        // in order to minimize realloc events and over-allocation
        dst_ofs = dst_len;
        dst_len = srcprops.uncomp_len + (srcprops.uncomp_len / 2);
        dst.resize(dst_len);
        srcprops.uncomp_len = dst_len;
      } else if (bz_err == 0) {
        srcprops.uncomp_len = dst_len;
        dst.resize(srcprops.uncomp_len);
      }
    } while (bz_err == BZ_OUTBUFF_FULL);
    CUDF_EXPECTS(bz_err == 0, "Decompression: error in stream");
    return dst;
  }
  if (srcprops.compression == compression_type::SNAPPY) {
    std::vector<uint8_t> dst(srcprops.uncomp_len);
    detail::decompress_snappy(src, dst);
    return dst;
  }

  CUDF_FAIL("Unsupported compressed stream type");
}

void decompress(compression_type compression,
                device_span<device_span<uint8_t const> const> inputs,
                device_span<device_span<uint8_t> const> outputs,
                device_span<detail::codec_exec_result> results,
                size_t max_uncomp_chunk_size,
                size_t max_total_uncomp_size,
                rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  if (inputs.empty()) { return; }

  // sort inputs by size, largest first
  auto const [sorted_inputs, sorted_outputs, order] =
    sort_decompression_tasks(inputs, outputs, stream, cudf::get_current_device_resource_ref());
  device_span<device_span<uint8_t const> const> inputs_view = sorted_inputs;
  device_span<device_span<uint8_t> const> outputs_view      = sorted_outputs;

  auto const split_idx =
    split_decompression_tasks(inputs_view,
                              outputs_view,
                              get_host_engine_state(compression),
                              cudf::detail::getenv_or("LIBCUDF_HOST_DECOMPRESSION_THRESHOLD",
                                                      default_host_decompression_auto_threshold),
                              cudf::detail::getenv_or("LIBCUDF_HOST_DECOMPRESSION_RATIO",
                                                      default_host_device_decompression_cost_ratio),
                              stream);

  auto tmp_results = cudf::detail::make_device_uvector_async<detail::codec_exec_result>(
    results, stream, cudf::get_current_device_resource_ref());
  device_span<codec_exec_result> results_view = tmp_results;

  // Chunks [0, split_idx) go to the host engine, [split_idx, end) to the device engine.
  auto const has_host_work   = split_idx > 0;
  auto const has_device_work = split_idx < inputs_view.size();

  // Only fork/join when both host and device have work.
  if (has_host_work and has_device_work) {
    auto const streams = cudf::detail::fork_streams(stream, 2);
    detail::device_decompress(compression,
                              inputs_view.subspan(split_idx, inputs_view.size() - split_idx),
                              outputs_view.subspan(split_idx, outputs_view.size() - split_idx),
                              results_view.subspan(split_idx, results_view.size() - split_idx),
                              max_uncomp_chunk_size,
                              max_total_uncomp_size,
                              streams[0]);
    detail::host_decompress(compression,
                            inputs_view.subspan(0, split_idx),
                            outputs_view.subspan(0, split_idx),
                            results_view.subspan(0, split_idx),
                            streams[1]);
    cudf::detail::join_streams(streams, stream);
  } else if (has_device_work) {
    detail::device_decompress(compression,
                              inputs_view,
                              outputs_view,
                              results_view,
                              max_uncomp_chunk_size,
                              max_total_uncomp_size,
                              stream);
  } else {
    detail::host_decompress(compression, inputs_view, outputs_view, results_view, stream);
  }

  copy_results_to_original_order(results_view, results, order, stream);
}

[[nodiscard]] bool is_host_decompression_supported(compression_type compression)
{
  switch (compression) {
    case compression_type::GZIP:
    case compression_type::SNAPPY:
    case compression_type::ZLIB:
    case compression_type::ZSTD:
    case compression_type::NONE: return true;
    default: return false;
  }
}

[[nodiscard]] bool is_device_decompression_supported(compression_type compression)
{
  auto const nvcomp_type = detail::to_nvcomp_compression(compression);
  switch (compression) {
    case compression_type::LZ4:
    case compression_type::ZSTD:
      return not detail::nvcomp::is_decompression_disabled(nvcomp_type.value());
    case compression_type::BROTLI:
    case compression_type::GZIP:
    case compression_type::SNAPPY:
    case compression_type::ZLIB:
    case compression_type::NONE: return true;
    default: return false;
  }
}

[[nodiscard]] bool is_decompression_supported(compression_type compression)
{
  return is_host_decompression_supported(compression) or
         is_device_decompression_supported(compression);
}

}  // namespace cudf::io::detail
