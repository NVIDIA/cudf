/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/error.hpp>

#include <src/io/parquet/compact_protocol_writer.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Builders for single-page Parquet files with DELTA-family encodings, used to test mini-block
// sizes no stock writer emits (the cudf and parquet-mr writers put 32 values in a mini-block,
// pyarrow and arrow-rs at most 64, while the format allows any multiple of 32). Callers pass
// the column values plus the DELTA block geometry (block_size, mini_block_count) and get back
// the complete file bytes.
//
// Both the page header and the file footer are serialized with cudf's production
// CompactProtocolWriter (the same CompactProtocolWriter::write() overloads the writer uses), so
// only the DELTA-encoded page body -- the part under test, which no stock writer emits -- is built
// by hand.

// Parquet metadata (page header + footer) serialization via the production compact protocol writer
namespace parquet_delta_test {

// serialize a V1 data page header for a flat REQUIRED column (no repetition/definition levels)
inline std::vector<uint8_t> serialize_data_page_header(int num_values,
                                                       cudf::io::parquet::Encoding encoding,
                                                       int64_t page_size)
{
  namespace pq = cudf::io::parquet;
  pq::PageHeader ph;
  ph.type                                       = pq::PageType::DATA_PAGE;
  ph.uncompressed_page_size                     = static_cast<int32_t>(page_size);
  ph.compressed_page_size                       = static_cast<int32_t>(page_size);
  ph.data_page_header.num_values                = num_values;
  ph.data_page_header.encoding                  = encoding;
  ph.data_page_header.definition_level_encoding = pq::Encoding::RLE;  // no levels
  ph.data_page_header.repetition_level_encoding = pq::Encoding::RLE;  // no levels
  std::vector<uint8_t> out;
  pq::detail::CompactProtocolWriter{&out}.write(ph);
  return out;
}

// serialize a V2 data page header (the LIST builders carry bit-packed repetition/definition levels)
inline std::vector<uint8_t> serialize_data_page_header_v2(int num_values,
                                                          int num_nulls,
                                                          int num_rows,
                                                          cudf::io::parquet::Encoding encoding,
                                                          int64_t definition_levels_byte_length,
                                                          int64_t repetition_levels_byte_length,
                                                          int64_t page_size)
{
  namespace pq = cudf::io::parquet;
  pq::PageHeader ph;
  ph.type                          = pq::PageType::DATA_PAGE_V2;
  ph.uncompressed_page_size        = static_cast<int32_t>(page_size);
  ph.compressed_page_size          = static_cast<int32_t>(page_size);
  auto& v2                         = ph.data_page_header_v2;
  v2.num_values                    = num_values;
  v2.num_nulls                     = num_nulls;
  v2.num_rows                      = num_rows;
  v2.encoding                      = encoding;
  v2.definition_levels_byte_length = static_cast<int32_t>(definition_levels_byte_length);
  v2.repetition_levels_byte_length = static_cast<int32_t>(repetition_levels_byte_length);
  v2.is_compressed                 = false;
  std::vector<uint8_t> out;
  pq::detail::CompactProtocolWriter{&out}.write(ph);
  return out;
}

// serialize a FileMetaData footer with the production CompactProtocolWriter
inline std::vector<uint8_t> serialize_footer(cudf::io::parquet::FileMetaData const& file_metadata)
{
  std::vector<uint8_t> out;
  cudf::io::parquet::detail::CompactProtocolWriter writer(&out);
  writer.write(file_metadata);
  return out;
}

}  // namespace parquet_delta_test

// DELTA_BINARY_PACKED stream encoder

// pack values (padded with 0 up to `count`) at `width` bits each, LSB-first, consecutively --
// the same layout the RLE/bit-packing hybrid and the delta mini-blocks use
inline void bitpack_into(std::vector<uint8_t>& out,
                         std::vector<uint64_t> const& vals,
                         int width,
                         int count)
{
  size_t const base = out.size();
  out.resize(base + static_cast<size_t>(count) * width / 8, 0);
  size_t pos = 0;
  for (auto const v : vals) {
    for (int b = 0; b < width; b++) {
      if ((v >> b) & 1) { out[base + pos / 8] |= 1 << (pos % 8); }
      pos++;
    }
  }
}

// append `v` to `out` as an unsigned LEB128 varint
inline void append_uleb128(std::vector<uint8_t>& out, uint64_t v)
{
  while (true) {
    uint8_t const b = v & 0x7f;
    v >>= 7;
    if (v) {
      out.push_back(b | 0x80);
    } else {
      out.push_back(b);
      return;
    }
  }
}

// append `v` to `out` as a zigzag-encoded LEB128 varint
inline void append_zigzag128(std::vector<uint8_t>& out, int64_t v)
{
  append_uleb128(out, (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63));
}

// complete DELTA_BINARY_PACKED stream: header (block_size, mini_block_count, value count, first
// value), then per block a zigzag min-delta, one bit-width byte per mini-block, and the
// bit-packed deltas
inline std::vector<uint8_t> encode_delta_binary_packed(std::vector<int64_t> const& values,
                                                       int block_size,
                                                       int mini_block_count)
{
  CUDF_EXPECTS(block_size % mini_block_count == 0 && (block_size / mini_block_count) % 32 == 0,
               "DELTA mini-block size (block_size / mini_block_count) must be a multiple of 32");
  std::vector<uint8_t> out;
  append_uleb128(out, block_size);
  append_uleb128(out, mini_block_count);
  append_uleb128(out, values.size());  // total value count, including the first value below
  append_zigzag128(out, values.empty() ? 0 : values.front());
  if (values.size() <= 1) { return out; }

  std::vector<int64_t> deltas(values.size() - 1);
  for (size_t i = 0; i + 1 < values.size(); i++) {
    deltas[i] = values[i + 1] - values[i];
  }

  int const vpm = block_size / mini_block_count;
  for (size_t bstart = 0; bstart < deltas.size(); bstart += block_size) {
    auto const bend      = std::min(bstart + block_size, deltas.size());
    auto const min_delta = *std::min_element(deltas.begin() + bstart, deltas.begin() + bend);
    append_zigzag128(out, min_delta);

    // per mini-block bit widths, then the packed deltas (empty trailing mini-blocks get width 0
    // and no data)
    std::vector<int> widths(mini_block_count, 0);
    std::vector<std::vector<uint64_t>> rel(mini_block_count);
    for (int m = 0; m < mini_block_count; m++) {
      auto const mstart = bstart + static_cast<size_t>(m) * vpm;
      auto const mend   = std::min(mstart + vpm, bend);
      for (size_t i = mstart; i < mend; i++) {
        auto const r = static_cast<uint64_t>(deltas[i] - min_delta);  // >= 0 by construction
        rel[m].push_back(r);
        int w = 0;
        while (r >> w) {
          w++;
        }
        widths[m] = std::max(widths[m], w);
      }
    }
    for (auto const w : widths) {
      out.push_back(static_cast<uint8_t>(w));
    }
    for (int m = 0; m < mini_block_count; m++) {
      if (!rel[m].empty()) { bitpack_into(out, rel[m], widths[m], vpm); }
    }
  }
  return out;
}

// raw DELTA_BINARY_PACKED header only (block/mini-block geometry, value count, first value), for
// negative tests that need a geometry encode_delta_binary_packed would reject. The reader validates
// the geometry in init_binary_block before decoding any deltas, so no mini-block data follows.
inline std::vector<uint8_t> encode_delta_binary_header(int block_size,
                                                       int mini_block_count,
                                                       int64_t value_count,
                                                       int64_t first_value)
{
  std::vector<uint8_t> out;
  append_uleb128(out, static_cast<uint64_t>(block_size));
  append_uleb128(out, static_cast<uint64_t>(mini_block_count));
  append_uleb128(out, static_cast<uint64_t>(value_count));
  append_zigzag128(out, first_value);
  return out;
}

// single-page file assembly

// V1 data page + footer around `body` for a single REQUIRED flat column "a"
inline std::vector<uint8_t> wrap_single_page_parquet(std::vector<uint8_t> const& body,
                                                     int num_values,
                                                     cudf::io::parquet::Type physical_type,
                                                     cudf::io::parquet::Encoding encoding,
                                                     bool utf8)
{
  namespace pq = cudf::io::parquet;

  auto const page_header =
    parquet_delta_test::serialize_data_page_header(num_values, encoding, body.size());

  int const data_page_offset = 4;  // immediately after the leading "PAR1" magic
  auto const chunk_size      = static_cast<int64_t>(page_header.size() + body.size());

  pq::FileMetaData file_metadata;
  file_metadata.version  = 1;
  file_metadata.num_rows = num_values;

  // schema: root group with a single REQUIRED leaf "a". The metadata structs are built in place
  // (emplace_back + reference) so that structs holding std::optional members are never copied
  // through an initializer_list, which trips GCC's -Wmaybe-uninitialized on the empty optionals.
  file_metadata.schema.reserve(2);
  auto& root           = file_metadata.schema.emplace_back();
  root.name            = "schema";
  root.num_children    = 1;
  root.repetition_type = pq::FieldRepetitionType::UNSPECIFIED;  // the root carries no repetition
  auto& col            = file_metadata.schema.emplace_back();
  col.type             = physical_type;
  col.repetition_type  = pq::FieldRepetitionType::REQUIRED;
  col.name             = "a";
  if (utf8) { col.converted_type = pq::ConvertedType::UTF8; }

  auto& row_group              = file_metadata.row_groups.emplace_back();
  row_group.total_byte_size    = chunk_size;
  row_group.num_rows           = num_values;
  auto& chunk                  = row_group.columns.emplace_back();
  chunk.file_offset            = data_page_offset;
  auto& meta                   = chunk.meta_data;
  meta.type                    = physical_type;
  meta.encodings               = {pq::Encoding::RLE, encoding};
  meta.path_in_schema          = {"a"};
  meta.codec                   = pq::Compression::UNCOMPRESSED;
  meta.num_values              = num_values;
  meta.total_uncompressed_size = chunk_size;
  meta.total_compressed_size   = chunk_size;
  meta.data_page_offset        = data_page_offset;

  auto const footer = parquet_delta_test::serialize_footer(file_metadata);

  std::vector<uint8_t> out;
  auto append = [&out](auto const& bytes) { out.insert(out.end(), bytes.begin(), bytes.end()); };
  out.insert(out.end(), {'P', 'A', 'R', '1'});
  append(page_header);
  append(body);
  append(footer);
  auto const flen = static_cast<uint32_t>(footer.size());
  for (int i = 0; i < 4; i++) {
    out.push_back((flen >> (8 * i)) & 0xff);
  }
  out.insert(out.end(), {'P', 'A', 'R', '1'});
  return out;
}

// complete file: one DELTA_BINARY_PACKED INT64 column "a"
inline std::vector<uint8_t> build_delta_binary_parquet(std::vector<int64_t> const& values,
                                                       int block_size,
                                                       int mini_block_count)
{
  auto const body = encode_delta_binary_packed(values, block_size, mini_block_count);
  return wrap_single_page_parquet(body,
                                  values.size(),
                                  cudf::io::parquet::Type::INT64,
                                  cudf::io::parquet::Encoding::DELTA_BINARY_PACKED,
                                  false);
}

// deterministic test data (self-contained splitmix64 so results never vary across platforms)
inline uint64_t delta_test_rand(uint64_t& state)
{
  state += 0x9e3779b97f4a7c15ull;
  uint64_t z = state;
  z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// values whose deltas vary within [-2, regime_max], with the magnitude regime switching every 64
// values so consecutive mini-blocks get different, non-zero bit widths
inline std::vector<int64_t> delta_test_int64_values(int n, uint64_t seed = 101)
{
  constexpr int64_t regime_max[] = {5, 220, 3000, 60000};
  std::vector<int64_t> out(n);
  int64_t v = 0;
  for (int i = 0; i < n; i++) {
    out[i]        = v;
    auto const hi = regime_max[(i / 64) % 4];
    v += static_cast<int64_t>(delta_test_rand(seed) % static_cast<uint64_t>(hi + 3)) - 2;
  }
  return out;
}

// string encodings

// strings mixing ASCII and valid non-ASCII UTF-8 sequences, with lengths varying in
// [1, max_length]; with shared_prefixes, each string keeps a random-length prefix of its
// predecessor so the DELTA_BYTE_ARRAY prefix-length stream also has varying non-zero deltas. The
// multi-byte code points exercise the byte-wise length/prefix/suffix reconstruction paths.
inline std::vector<std::string> delta_test_strings(int n,
                                                   bool shared_prefixes,
                                                   uint64_t seed     = 201,
                                                   size_t max_length = 20)
{
  constexpr char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  // 2- and 3-byte UTF-8 code points (e-acute, n-tilde, u-umlaut, euro, CJK, lambda) written as
  // explicit bytes so the encoding does not depend on the compiler's execution character set
  constexpr std::string_view utf8_tokens[] = {
    "\xc3\xa9", "\xc3\xb1", "\xc3\xbc", "\xe2\x82\xac", "\xe4\xb8\xad", "\xce\xbb"};
  std::vector<std::string> out;
  out.reserve(n);
  std::string prev;
  for (int i = 0; i < n; i++) {
    auto const length = 1 + static_cast<size_t>(delta_test_rand(seed) % max_length);
    std::string s;
    if (shared_prefixes && !prev.empty()) {
      auto const keep = delta_test_rand(seed) % (std::min(prev.size(), length) + 1);
      s               = prev.substr(0, keep);
    }
    while (s.size() < length) {
      auto const r         = delta_test_rand(seed);
      auto const remaining = length - s.size();
      // roughly one position in four appends a multi-byte token when it fits; the rest stay ASCII.
      // appending only whole tokens that fit keeps the byte length exactly `length`.
      if (remaining >= 2 && r % 4 == 0) {
        auto const& token = utf8_tokens[(r >> 2) % (sizeof(utf8_tokens) / sizeof(utf8_tokens[0]))];
        if (token.size() <= remaining) {
          s += token;
          continue;
        }
      }
      s += alphabet[r % (sizeof(alphabet) - 1)];
    }
    out.push_back(s);
    prev = std::move(s);
  }
  return out;
}

// complete file: one DELTA_LENGTH_BYTE_ARRAY string column "a" (delta-encoded lengths followed
// by the concatenated string bytes)
inline std::vector<uint8_t> build_delta_length_byte_array_parquet(
  std::vector<std::string> const& strings, int block_size, int mini_block_count)
{
  std::vector<int64_t> lengths(strings.size());
  std::transform(strings.begin(), strings.end(), lengths.begin(), [](auto const& s) {
    return static_cast<int64_t>(s.size());
  });
  auto body = encode_delta_binary_packed(lengths, block_size, mini_block_count);
  for (auto const& s : strings) {
    body.insert(body.end(), s.begin(), s.end());
  }
  return wrap_single_page_parquet(body,
                                  strings.size(),
                                  cudf::io::parquet::Type::BYTE_ARRAY,
                                  cudf::io::parquet::Encoding::DELTA_LENGTH_BYTE_ARRAY,
                                  true);
}

// complete file: one DELTA_BYTE_ARRAY string column "a" (front compression: delta-encoded
// shared-prefix lengths, then delta-encoded suffix lengths, then the concatenated suffixes)
inline std::vector<uint8_t> build_delta_byte_array_parquet(std::vector<std::string> const& strings,
                                                           int block_size,
                                                           int mini_block_count)
{
  std::vector<int64_t> prefix_lens, suffix_lens;
  std::string suffix_bytes;
  std::string prev;
  for (auto const& s : strings) {
    size_t lcp     = 0;
    auto const end = std::min(prev.size(), s.size());
    while (lcp < end && prev[lcp] == s[lcp]) {
      lcp++;
    }
    prefix_lens.push_back(lcp);
    suffix_lens.push_back(s.size() - lcp);
    suffix_bytes.append(s, lcp, std::string::npos);
    prev = s;
  }
  auto body                = encode_delta_binary_packed(prefix_lens, block_size, mini_block_count);
  auto const suffix_stream = encode_delta_binary_packed(suffix_lens, block_size, mini_block_count);
  body.insert(body.end(), suffix_stream.begin(), suffix_stream.end());
  body.insert(body.end(), suffix_bytes.begin(), suffix_bytes.end());
  return wrap_single_page_parquet(body,
                                  strings.size(),
                                  cudf::io::parquet::Type::BYTE_ARRAY,
                                  cudf::io::parquet::Encoding::DELTA_BYTE_ARRAY,
                                  true);
}

// LIST<INT64>: one optional list column "col" of optional int64 "element" (max_def_level 3,
// max_rep_level 1), no null lists or elements -- empty lists only. Emitted as a single
// uncompressed V2 data page whose rep/def levels are RLE/bit-packed hybrid runs.

// encode `levels` at `width` bits as one bit-packed hybrid run (padded to a multiple of 8)
inline std::vector<uint8_t> encode_levels_bit_packed(std::vector<int> const& levels, int width)
{
  auto const groups = (levels.size() + 7) / 8;
  std::vector<uint8_t> out;
  // RLE/bit-pack hybrid run header: LSB set marks a bit-packed run of `groups` 8-value groups
  append_uleb128(out, (groups << 1) | 1);
  std::vector<uint64_t> vals(levels.begin(), levels.end());
  bitpack_into(out, vals, width, groups * 8);
  return out;
}

// per-value repetition/definition levels for a list column with the shape given by `sizes`
// (max_def_level 3, max_rep_level 1, no null lists or elements -- empty lists only)
struct list_levels {
  std::vector<int> rep;
  std::vector<int> def;
  int num_nulls = 0;
};

inline list_levels compute_list_levels(std::vector<size_t> const& sizes)
{
  list_levels out;
  for (auto const size : sizes) {
    if (size == 0) {  // empty list: one level entry, def < max_def, no leaf value
      out.rep.push_back(0);
      out.def.push_back(1);
      out.num_nulls++;
      continue;
    }
    for (size_t j = 0; j < size; j++) {
      out.rep.push_back(j == 0 ? 0 : 1);
      out.def.push_back(3);
    }
  }
  return out;
}

// V2 data page + footer around the encoded `values` of a single list column "col" of leaf
// "element" with the given physical type and encoding
inline std::vector<uint8_t> wrap_single_page_list_parquet(
  list_levels const& levels,
  int num_rows,
  std::vector<uint8_t> const& values,
  cudf::io::parquet::Type leaf_physical_type,
  cudf::io::parquet::Encoding encoding,
  bool utf8)
{
  namespace pq = cudf::io::parquet;

  auto const num_values = static_cast<int>(levels.rep.size());
  auto const rep        = encode_levels_bit_packed(levels.rep, 1);  // max_rep_level 1
  auto const dfn        = encode_levels_bit_packed(levels.def, 2);  // max_def_level 3
  auto const page_size  = static_cast<int64_t>(rep.size() + dfn.size() + values.size());

  auto const page_header = parquet_delta_test::serialize_data_page_header_v2(
    num_values, levels.num_nulls, num_rows, encoding, dfn.size(), rep.size(), page_size);

  int const data_page_offset = 4;
  auto const chunk_size      = static_cast<int64_t>(page_header.size()) + page_size;

  pq::FileMetaData file_metadata;
  file_metadata.version  = 2;
  file_metadata.num_rows = num_rows;

  // schema: root -> optional LIST group "col" -> repeated group "list" -> optional leaf "element".
  // The metadata structs are built in place (emplace_back + reference) so that structs holding
  // std::optional members are never copied through an initializer_list, which trips GCC's
  // -Wmaybe-uninitialized on the empty optionals.
  file_metadata.schema.reserve(4);
  auto& root                 = file_metadata.schema.emplace_back();
  root.name                  = "schema";
  root.num_children          = 1;
  root.repetition_type       = pq::FieldRepetitionType::UNSPECIFIED;
  auto& list_col             = file_metadata.schema.emplace_back();
  list_col.repetition_type   = pq::FieldRepetitionType::OPTIONAL;
  list_col.name              = "col";
  list_col.num_children      = 1;
  list_col.converted_type    = pq::ConvertedType::LIST;
  list_col.logical_type      = pq::LogicalType{pq::LogicalType::LIST};
  auto& list_group           = file_metadata.schema.emplace_back();
  list_group.repetition_type = pq::FieldRepetitionType::REPEATED;
  list_group.name            = "list";
  list_group.num_children    = 1;
  auto& element              = file_metadata.schema.emplace_back();
  element.type               = leaf_physical_type;
  element.repetition_type    = pq::FieldRepetitionType::OPTIONAL;
  element.name               = "element";
  if (utf8) { element.converted_type = pq::ConvertedType::UTF8; }

  auto& row_group              = file_metadata.row_groups.emplace_back();
  row_group.total_byte_size    = chunk_size;
  row_group.num_rows           = num_rows;
  auto& chunk                  = row_group.columns.emplace_back();
  chunk.file_offset            = data_page_offset;
  auto& meta                   = chunk.meta_data;
  meta.type                    = leaf_physical_type;
  meta.encodings               = {pq::Encoding::RLE, encoding};
  meta.path_in_schema          = {"col", "list", "element"};
  meta.codec                   = pq::Compression::UNCOMPRESSED;
  meta.num_values              = num_values;  // counts level entries incl. empties
  meta.total_uncompressed_size = chunk_size;
  meta.total_compressed_size   = chunk_size;
  meta.data_page_offset        = data_page_offset;

  auto const footer = parquet_delta_test::serialize_footer(file_metadata);

  std::vector<uint8_t> out;
  auto append = [&out](auto const& bytes) { out.insert(out.end(), bytes.begin(), bytes.end()); };
  out.insert(out.end(), {'P', 'A', 'R', '1'});
  append(page_header);
  append(rep);
  append(dfn);
  append(values);
  append(footer);
  auto const flen = static_cast<uint32_t>(footer.size());
  for (int i = 0; i < 4; i++) {
    out.push_back((flen >> (8 * i)) & 0xff);
  }
  out.insert(out.end(), {'P', 'A', 'R', '1'});
  return out;
}

inline std::vector<uint8_t> build_delta_binary_list_parquet(
  std::vector<std::vector<int64_t>> const& lists, int block_size, int mini_block_count)
{
  std::vector<size_t> sizes(lists.size());
  std::vector<int64_t> leaf_values;
  for (size_t i = 0; i < lists.size(); i++) {
    sizes[i] = lists[i].size();
    leaf_values.insert(leaf_values.end(), lists[i].begin(), lists[i].end());
  }
  auto const values = encode_delta_binary_packed(leaf_values, block_size, mini_block_count);
  return wrap_single_page_list_parquet(compute_list_levels(sizes),
                                       lists.size(),
                                       values,
                                       cudf::io::parquet::Type::INT64,
                                       cudf::io::parquet::Encoding::DELTA_BINARY_PACKED,
                                       false);
}

// per-value repetition/definition levels for a list column whose leaves may be null (definition
// level 2). std::nullopt is a null leaf element; an empty inner vector is an empty list (definition
// level 1). Same LIST<INT64> shape as compute_list_levels (max_def_level 3, max_rep_level 1).
inline list_levels compute_list_levels_with_leaf_nulls(
  std::vector<std::vector<std::optional<int64_t>>> const& lists)
{
  list_levels out;
  for (auto const& list : lists) {
    if (list.empty()) {  // empty list: one level entry, def < max_def, no leaf value
      out.rep.push_back(0);
      out.def.push_back(1);
      out.num_nulls++;
      continue;
    }
    for (size_t j = 0; j < list.size(); j++) {
      out.rep.push_back(j == 0 ? 0 : 1);
      out.def.push_back(list[j].has_value() ? 3 : 2);  // 3: leaf present, 2: null leaf element
      if (not list[j].has_value()) { out.num_nulls++; }
    }
  }
  return out;
}

// complete file: one LIST<INT64> column whose leaves may be null. Only the non-null leaf values are
// DELTA_BINARY_PACKED encoded; the null leaves are carried by the definition levels.
inline std::vector<uint8_t> build_delta_binary_list_with_leaf_nulls_parquet(
  std::vector<std::vector<std::optional<int64_t>>> const& lists,
  int block_size,
  int mini_block_count)
{
  std::vector<int64_t> leaf_values;
  for (auto const& list : lists) {
    for (auto const& e : list) {
      if (e.has_value()) { leaf_values.push_back(*e); }
    }
  }
  auto const values = encode_delta_binary_packed(leaf_values, block_size, mini_block_count);
  return wrap_single_page_list_parquet(compute_list_levels_with_leaf_nulls(lists),
                                       lists.size(),
                                       values,
                                       cudf::io::parquet::Type::INT64,
                                       cudf::io::parquet::Encoding::DELTA_BINARY_PACKED,
                                       false);
}

// complete file: one LIST<STRING> column, leaf strings DELTA_LENGTH_BYTE_ARRAY encoded
inline std::vector<uint8_t> build_delta_length_byte_array_list_parquet(
  std::vector<std::vector<std::string>> const& lists, int block_size, int mini_block_count)
{
  std::vector<size_t> sizes(lists.size());
  std::vector<int64_t> lengths;
  std::string chars;
  for (size_t i = 0; i < lists.size(); i++) {
    sizes[i] = lists[i].size();
    for (auto const& s : lists[i]) {
      lengths.push_back(s.size());
      chars += s;
    }
  }
  auto body = encode_delta_binary_packed(lengths, block_size, mini_block_count);
  body.insert(body.end(), chars.begin(), chars.end());
  return wrap_single_page_list_parquet(compute_list_levels(sizes),
                                       lists.size(),
                                       body,
                                       cudf::io::parquet::Type::BYTE_ARRAY,
                                       cudf::io::parquet::Encoding::DELTA_LENGTH_BYTE_ARRAY,
                                       true);
}

// complete file: one LIST<STRING> column, leaf strings DELTA_BYTE_ARRAY (front compression)
// encoded over the flattened string sequence
inline std::vector<uint8_t> build_delta_byte_array_list_parquet(
  std::vector<std::vector<std::string>> const& lists, int block_size, int mini_block_count)
{
  std::vector<size_t> sizes(lists.size());
  std::vector<int64_t> prefix_lens, suffix_lens;
  std::string suffix_bytes;
  std::string prev;
  for (size_t i = 0; i < lists.size(); i++) {
    sizes[i] = lists[i].size();
    for (auto const& s : lists[i]) {
      size_t lcp     = 0;
      auto const end = std::min(prev.size(), s.size());
      while (lcp < end && prev[lcp] == s[lcp]) {
        lcp++;
      }
      prefix_lens.push_back(lcp);
      suffix_lens.push_back(s.size() - lcp);
      suffix_bytes.append(s, lcp, std::string::npos);
      prev = s;
    }
  }
  auto body                = encode_delta_binary_packed(prefix_lens, block_size, mini_block_count);
  auto const suffix_stream = encode_delta_binary_packed(suffix_lens, block_size, mini_block_count);
  body.insert(body.end(), suffix_stream.begin(), suffix_stream.end());
  body.insert(body.end(), suffix_bytes.begin(), suffix_bytes.end());
  return wrap_single_page_list_parquet(compute_list_levels(sizes),
                                       lists.size(),
                                       body,
                                       cudf::io::parquet::Type::BYTE_ARRAY,
                                       cudf::io::parquet::Encoding::DELTA_BYTE_ARRAY,
                                       true);
}

// lists of alphanumeric strings with the same shape as delta_test_lists (varying lengths 1..8,
// empties mixed in, a trailing empty list); the flattened string sequence comes from
// delta_test_strings so prefix and suffix lengths vary
inline std::vector<std::vector<std::string>> delta_test_string_lists(int n_lists,
                                                                     bool shared_prefixes,
                                                                     uint64_t seed     = 501,
                                                                     size_t max_length = 20)
{
  std::vector<size_t> lengths(n_lists);
  size_t n_leaf = 0;
  for (int i = 0; i < n_lists; i++) {
    bool const empty = (i + 1 == n_lists) || delta_test_rand(seed) % 6 == 0;
    lengths[i]       = empty ? 0 : 1 + delta_test_rand(seed) % 8;
    n_leaf += lengths[i];
  }
  auto const strings = delta_test_strings(n_leaf, shared_prefixes, seed, max_length);
  std::vector<std::vector<std::string>> out(n_lists);
  size_t pos = 0;
  for (int i = 0; i < n_lists; i++) {
    out[i].assign(strings.begin() + pos, strings.begin() + pos + lengths[i]);
    pos += lengths[i];
  }
  return out;
}

// lists of varying lengths 1..8 with empties mixed in (including a trailing empty list), leaf
// values from the varying-delta generator above
inline std::vector<std::vector<int64_t>> delta_test_lists(int n_lists, uint64_t seed = 401)
{
  std::vector<size_t> lengths(n_lists);
  size_t n_leaf = 0;
  for (int i = 0; i < n_lists; i++) {
    bool const empty = (i + 1 == n_lists) || delta_test_rand(seed) % 6 == 0;
    lengths[i]       = empty ? 0 : 1 + delta_test_rand(seed) % 8;
    n_leaf += lengths[i];
  }
  auto const values = delta_test_int64_values(n_leaf, seed);
  std::vector<std::vector<int64_t>> out(n_lists);
  size_t pos = 0;
  for (int i = 0; i < n_lists; i++) {
    out[i].assign(values.begin() + pos, values.begin() + pos + lengths[i]);
    pos += lengths[i];
  }
  return out;
}
