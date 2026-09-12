/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace {

using cudf::io::parquet::experimental::variant_basic_type;
using cudf::io::parquet::experimental::variant_primitive_type;

// The leaf value type exercised by the benchmark (nvbench "type" string axis).
enum class bench_variant_type : uint8_t {
  INT32,
  FLOAT,
  BOOL,
  STRING,
  ARRAY,
  DECIMAL32,
  DECIMAL64,
  DECIMAL128
};

bench_variant_type parse_bench_variant_type(std::string const& type_str)
{
  if (type_str == "int32_t") { return bench_variant_type::INT32; }
  if (type_str == "float") { return bench_variant_type::FLOAT; }
  if (type_str == "bool") { return bench_variant_type::BOOL; }
  if (type_str == "string") { return bench_variant_type::STRING; }
  if (type_str == "array") { return bench_variant_type::ARRAY; }
  if (type_str == "decimal32") { return bench_variant_type::DECIMAL32; }
  if (type_str == "decimal64") { return bench_variant_type::DECIMAL64; }
  if (type_str == "decimal128") { return bench_variant_type::DECIMAL128; }
  CUDF_FAIL("Unrecognized benchmark type: " + type_str);
}

// Shared by the encoded leaf values and the target column, so the cast measures decoding only.
constexpr uint8_t bench_decimal_scale = 2;

// Compose a value-metadata header byte from a basic type and its 6-bit value_header.
// See cpp/tests/io/experimental/variant_extract_test.cpp for the header byte layout.
constexpr uint8_t make_variant_header(variant_basic_type basic, uint8_t value_header)
{
  return static_cast<uint8_t>(static_cast<uint8_t>(basic) | (value_header << 2));
}

constexpr uint8_t make_variant_primitive_header(variant_primitive_type type)
{
  return make_variant_header(variant_basic_type::PRIMITIVE, static_cast<uint8_t>(type));
}

// Header byte for a short string of the given length (must fit in 6 bits: 0..63).
uint8_t make_variant_short_string_header(std::size_t length)
{
  CUDF_EXPECTS(length <= 63, "Short string length must fit in 6 bits (0..63)");
  return make_variant_header(variant_basic_type::SHORT_STRING, static_cast<uint8_t>(length));
}

// Header byte for an object value with 1-byte field ids and 1-byte offsets (value_header == 0).
constexpr uint8_t make_variant_object_header()
{
  return make_variant_header(variant_basic_type::OBJECT, 0);
}

// Header byte for an array value with 1-byte count and 1-byte offsets (value_header == 0).
constexpr uint8_t make_variant_array_header()
{
  return make_variant_header(variant_basic_type::ARRAY, 0);
}

// Append the low `width` bytes of `bits` to `out` in little-endian order.
void append_le(std::vector<uint8_t>& out, uint64_t bits, int width)
{
  for (int i = 0; i < width; ++i) {
    out.push_back(static_cast<uint8_t>((bits >> (8 * i)) & 0xff));
  }
}

// Build a V1 VARIANT metadata blob for a key dictionary. When `sorted` is true, callers must pass
// `keys` in ascending sorted order and the sorted-strings header bit is set to reflect that,
// letting field lookups take the binary-search path; when false, the sorted-strings bit is left
// unset so lookups fall back to a linear scan regardless of `keys`' actual order.
// Uses 2-byte offsets when the total string length exceeds 255 bytes; 1-byte otherwise.
// Header bits [7:6] = offset_size_minus_one; bit [4] = sorted_strings; bits [3:0] = version (1).
std::vector<uint8_t> build_metadata(std::vector<std::string> const& keys, bool sorted = true)
{
  constexpr uint8_t kVariantMetadataVersion  = 0x01;
  constexpr uint8_t kVariantMetadataSorted   = 0x10;
  constexpr int kMetadataOffsetSizeShift     = 6;
  constexpr uint32_t kMaxSingleByteOffsetSum = 255u;

  uint32_t total_key_bytes = 0;
  for (auto const& key : keys) {
    total_key_bytes += static_cast<uint32_t>(key.size());
  }

  int const offset_size = (total_key_bytes > kMaxSingleByteOffsetSum) ? 2 : 1;
  std::vector<uint8_t> out{static_cast<uint8_t>(kVariantMetadataVersion |
                                                (sorted ? kVariantMetadataSorted : 0) |
                                                ((offset_size - 1) << kMetadataOffsetSizeShift))};
  out.reserve(out.size() + static_cast<std::size_t>(offset_size) * (keys.size() + 2) +
              total_key_bytes);

  auto write_little_endian_offset = [&](uint32_t value) {
    for (int byte_index = 0; byte_index < offset_size; ++byte_index) {
      out.push_back(static_cast<uint8_t>(value >> (8 * byte_index)));
    }
  };
  write_little_endian_offset(static_cast<uint32_t>(keys.size()));

  uint32_t running_offset = 0;
  write_little_endian_offset(0u);
  for (auto const& key : keys) {
    running_offset += static_cast<uint32_t>(key.size());
    write_little_endian_offset(running_offset);
  }

  for (auto const& key : keys) {
    out.insert(out.end(), key.begin(), key.end());
  }
  return out;
}

// Wrap `inner` as the sole field (field id `fid`) of a 1-field VARIANT object.
// Uses 1-byte field_id_size and 1-byte field_offset_size (value_header=0).
std::vector<uint8_t> wrap_in_object(uint8_t fid, std::span<uint8_t const> inner)
{
  // Format: object_header(1) + num_fields(1) + fid(1) + offset[0]=0(1) + offset[1]=size(1) + data
  std::vector<uint8_t> out{
    make_variant_object_header(), 0x01, fid, 0x00, static_cast<uint8_t>(inner.size())};
  out.insert(out.end(), inner.begin(), inner.end());
  return out;
}

// Build the leaf VARIANT value blob for the requested type.
std::vector<uint8_t> build_leaf_value(bench_variant_type type)
{
  switch (type) {
    case bench_variant_type::INT32: {
      std::vector<uint8_t> out{make_variant_primitive_header(variant_primitive_type::INT32)};
      append_le(out, 42u, 4);
      return out;
    }
    case bench_variant_type::FLOAT: {
      std::vector<uint8_t> out{make_variant_primitive_header(variant_primitive_type::FLOAT32)};
      float const f = 1.0f;
      uint32_t u;
      std::memcpy(&u, &f, 4);
      append_le(out, u, 4);
      return out;
    }
    case bench_variant_type::BOOL:
      return {make_variant_primitive_header(variant_primitive_type::BOOLEAN_TRUE)};
    case bench_variant_type::STRING: {
      // Short string "hello" (5 bytes).
      auto const s = std::string{"hello"};
      std::vector<uint8_t> out{make_variant_short_string_header(s.size())};
      out.insert(out.end(), s.begin(), s.end());
      return out;
    }
    case bench_variant_type::DECIMAL32: {
      std::vector<uint8_t> out{make_variant_primitive_header(variant_primitive_type::DECIMAL4),
                               bench_decimal_scale};
      append_le(out, 1234u, 4);
      return out;
    }
    case bench_variant_type::DECIMAL64: {
      std::vector<uint8_t> out{make_variant_primitive_header(variant_primitive_type::DECIMAL8),
                               bench_decimal_scale};
      append_le(out, (static_cast<uint64_t>(1234u) << 32) | 5678u, 8);
      return out;
    }
    case bench_variant_type::DECIMAL128: {
      // The value needs more than 64 bits, so the decode is not measured on an all-zero high half.
      std::vector<uint8_t> out{make_variant_primitive_header(variant_primitive_type::DECIMAL16),
                               bench_decimal_scale};
      auto const unscaled = (static_cast<__uint128_t>(1234u) << 64) | 5678u;
      append_le(out, static_cast<uint64_t>(unscaled), 8);
      append_le(out, static_cast<uint64_t>(unscaled >> 64), 8);
      return out;
    }
    case bench_variant_type::ARRAY: {
      // VARIANT array of two INT32 values [42, 99]; element [1] is accessed in the benchmark.
      // 2 elements, offsets [0, 5, 10], then INT32(42) and INT32(99) (5 bytes each).
      std::vector<uint8_t> out{make_variant_array_header(), 0x02, 0x00, 0x05, 0x0a};
      out.push_back(make_variant_primitive_header(variant_primitive_type::INT32));
      append_le(out, 42u, 4);
      out.push_back(make_variant_primitive_header(variant_primitive_type::INT32));
      append_le(out, 99u, 4);
      return out;
    }
    default: CUDF_FAIL("Unsupported benchmark leaf type");
  }
}

// Build the full hit-row value blob by wrapping the leaf in `nesting` object levels.
// Keys a,b,c,d,e map to field IDs 0,1,2,3,4 in the shared dictionary.
// For path a.b.c.d.e the outermost object uses fid=0 ("a").
std::vector<uint8_t> build_hit_value(bench_variant_type type, int nesting)
{
  auto val = build_leaf_value(type);
  for (int i = nesting - 1; i >= 0; --i) {
    val = wrap_in_object(static_cast<uint8_t>(i), val);
  }
  return val;
}

// Build the miss-row value blob: a valid VARIANT that won't match the target path or type.
// For get_variant_field rows: a 1-level object keyed on "z" (field ID = nesting in the
// dictionary), so traversal fails at the first key lookup while the row remains non-null.
// For cast_variant rows (nesting=0, non-array): a different primitive type so the cast returns
// null.
std::vector<uint8_t> build_miss_value(int nesting, bool is_array, bench_variant_type type)
{
  if (nesting == 0 && !is_array) {
    // Wrong-type primitive for the cast path.
    switch (type) {
      case bench_variant_type::BOOL: {
        std::vector<uint8_t> out{make_variant_primitive_header(variant_primitive_type::INT32)};
        append_le(out, 0u, 4);
        return out;
      }
      default: return {make_variant_primitive_header(variant_primitive_type::BOOLEAN_TRUE)};
    }
  }
  // "z" is always the last key in the dictionary, at field ID = nesting.
  return wrap_in_object(static_cast<uint8_t>(nesting), build_leaf_value(type));
}

// Zero-pad the shorter of `hit_val`/`miss_val` so both end up the same length. VARIANT decoders
// only ever read the bytes their own header/offsets describe, so trailing padding is inert; this
// keeps a row's size from being a confound for hit vs. miss access-pattern benchmarking.
void pad_to_equal_size(std::vector<uint8_t>& hit_val, std::vector<uint8_t>& miss_val)
{
  auto const target_size = std::max(hit_val.size(), miss_val.size());
  hit_val.resize(target_size, uint8_t{0});
  miss_val.resize(target_size, uint8_t{0});
}

// Build a VARIANT struct column (STRUCT<list<uint8>, list<uint8>>) from per-row byte spans.
std::unique_ptr<cudf::column> build_variant_column(std::span<std::span<uint8_t const>> meta_rows,
                                                   std::span<std::span<uint8_t const>> val_rows,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto const n = static_cast<cudf::size_type>(meta_rows.size());

  auto build_list_col =
    [&](std::span<std::span<uint8_t const>> rows) -> std::unique_ptr<cudf::column> {
    std::vector<int32_t> offsets(n + 1, 0);
    auto const total_bytes = std::accumulate(
      rows.begin(), rows.end(), std::size_t{0}, [](std::size_t acc, auto const& row) {
        return acc + row.size();
      });
    std::vector<uint8_t> flat;
    flat.reserve(total_bytes);
    for (cudf::size_type i = 0; i < n; ++i) {
      flat.insert(flat.end(), rows[i].begin(), rows[i].end());
      offsets[i + 1] = static_cast<int32_t>(flat.size());
    }

    auto d_offsets =
      rmm::device_buffer{offsets.data(), offsets.size() * sizeof(int32_t), stream, mr};
    auto d_data = rmm::device_buffer{flat.data(), flat.size() * sizeof(uint8_t), stream, mr};

    auto off_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, n + 1, std::move(d_offsets), rmm::device_buffer{}, 0);
    auto data_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                                   static_cast<cudf::size_type>(flat.size()),
                                                   std::move(d_data),
                                                   rmm::device_buffer{},
                                                   0);

    return cudf::make_lists_column(n, std::move(off_col), std::move(data_col), 0, {});
  };

  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(build_list_col(meta_rows));
  children.emplace_back(build_list_col(val_rows));
  return cudf::make_structs_column(n, std::move(children), 0, {}, stream, mr);
}

// Keys for the shared metadata dictionary: a=0, b=1, ... plus "z" for miss rows.
// "z" is appended last; lexicographic order is preserved.
std::vector<std::string> get_dict_keys(int nesting)
{
  std::vector<std::string> keys;
  keys.reserve(nesting + 1);
  for (int i = 0; i < nesting; ++i) {
    keys.emplace_back(1, static_cast<char>('a' + i));
  }
  keys.emplace_back("z");
  return keys;
}

// Keys for the field-count benchmark: "f00", "f01", ..., "f{N-1}" plus "z" for miss rows.
// All sort before "z", maintaining the required lexicographic order.
std::vector<std::string> get_dict_keys_for_fields(int num_fields)
{
  std::vector<std::string> keys;
  keys.reserve(num_fields + 1);
  for (int i = 0; i < num_fields; ++i) {
    keys.emplace_back("f" + std::string(i < 10 ? "0" : "") + std::to_string(i));
  }
  keys.emplace_back("z");
  return keys;
}

// Dictionary for the "random" field_position benchmark. The queried key's text stays constant
// ("n_target") across every row, but its dictionary rank varies: `rank` filler keys prefixed 'a'
// (which sorts before 'n') precede it and `num_fields - 1 - rank` filler keys prefixed 'z' (which
// sorts after 'n') follow it, so "n_target" always lands at dictionary index `rank`. A trailing
// "~miss" key (prefixed '~', sorting after 'z') always lands at index num_fields, for miss rows.
std::vector<std::string> get_dict_keys_at_rank(int num_fields, int rank)
{
  auto const pad6 = [](int i) {
    auto s = std::to_string(i);
    return std::string(6 - s.size(), '0') + s;
  };
  std::vector<std::string> keys;
  keys.reserve(num_fields + 1);
  for (int i = 0; i < rank; ++i) {
    keys.emplace_back("a_" + pad6(i));
  }
  keys.emplace_back("n_target");
  for (int i = 0; i < num_fields - 1 - rank; ++i) {
    keys.emplace_back("z_" + pad6(i));
  }
  keys.emplace_back("~miss");
  return keys;
}

// Build a flat object with `num_fields` fields using 1-byte field IDs and 1-byte offsets.
// Field `target_fid` holds `inner`; all other fields hold a dummy BOOLEAN_TRUE.
std::vector<uint8_t> build_flat_object(int num_fields,
                                       int target_fid,
                                       std::span<uint8_t const> inner)
{
  // object_header(1) + num_fields(1) + field_ids(num_fields) + offsets(num_fields+1) + data
  std::vector<uint8_t> out{make_variant_object_header(), static_cast<uint8_t>(num_fields)};
  out.reserve(out.size() + static_cast<std::size_t>(3 * num_fields) + inner.size());
  for (int i = 0; i < num_fields; ++i) {
    out.push_back(static_cast<uint8_t>(i));
  }
  uint8_t running = 0;
  for (int i = 0; i < num_fields; ++i) {
    out.push_back(running);
    running += static_cast<uint8_t>(i == target_fid ? inner.size() : 1u);
  }
  out.push_back(running);  // sentinel offset after last field
  for (int i = 0; i < num_fields; ++i) {
    if (i == target_fid) {
      out.insert(out.end(), inner.begin(), inner.end());
    } else {
      out.push_back(make_variant_primitive_header(variant_primitive_type::BOOLEAN_TRUE));  // dummy
    }
  }
  return out;
}

// Dictionary key of the i-th leaf field: "f00", "f01", ...
std::string field_key(int i) { return "f" + std::string(i < 10 ? "0" : "") + std::to_string(i); }

// Build a VARIANT object from (field id, value) pairs. Ids must be ascending, which for a
// name-sorted dictionary is also name order, as the spec requires. Field offsets widen to 2 bytes
// once the values region outgrows a single byte.
std::vector<uint8_t> build_object(
  std::vector<std::pair<uint8_t, std::vector<uint8_t>>> const& fields)
{
  constexpr std::size_t max_single_byte_offset = 255;
  auto const values_bytes =
    std::accumulate(fields.begin(), fields.end(), std::size_t{0}, [](auto acc, auto const& field) {
      return acc + field.second.size();
    });
  int const offset_size = values_bytes > max_single_byte_offset ? 2 : 1;

  // object value_header: | is_large (1) | field_id_size-1 (2) | field_offset_size-1 (2) |
  std::vector<uint8_t> out{
    make_variant_header(variant_basic_type::OBJECT, static_cast<uint8_t>(offset_size - 1)),
    static_cast<uint8_t>(fields.size())};
  for (auto const& [id, value] : fields) {
    out.push_back(id);
  }
  std::size_t running = 0;
  for (auto const& [id, value] : fields) {
    append_le(out, running, offset_size);
    running += value.size();
  }
  append_le(out, running, offset_size);  // sentinel offset after the last field
  for (auto const& [id, value] : fields) {
    out.insert(out.end(), value.begin(), value.end());
  }
  return out;
}

// Dictionary key "item%03d" of the workload below. Zero padding makes the numeric order the
// lexicographic order, so key `n` sits at dictionary index `n - 1`.
std::string item_key(int n)
{
  auto const digits = std::to_string(n);
  return "item" + std::string(3 - digits.size(), '0') + digits;
}

// A bare VARIANT short string value.
std::vector<uint8_t> build_short_string(std::string_view s)
{
  std::vector<uint8_t> out{make_variant_short_string_header(s.size())};
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

// A VARIANT array value holding `elements` in order.
std::vector<uint8_t> build_array(std::vector<std::vector<uint8_t>> const& elements)
{
  constexpr std::size_t max_single_byte_offset = 255;
  auto const values_bytes =
    std::accumulate(elements.begin(), elements.end(), std::size_t{0}, [](auto acc, auto const& e) {
      return acc + e.size();
    });
  int const offset_size = values_bytes > max_single_byte_offset ? 2 : 1;

  // array value_header: | unused (3) | is_large (1) | offset_size-1 (2) |
  std::vector<uint8_t> out{
    make_variant_header(variant_basic_type::ARRAY, static_cast<uint8_t>(offset_size - 1)),
    static_cast<uint8_t>(elements.size())};
  std::size_t running = 0;
  for (auto const& element : elements) {
    append_le(out, running, offset_size);
    running += element.size();
  }
  append_le(out, running, offset_size);
  for (auto const& element : elements) {
    out.insert(out.end(), element.begin(), element.end());
  }
  return out;
}

// A VARIANT object whose fields are named rather than pre-assigned ids. Field ids are the keys'
// positions in the sorted dictionary, and the spec wants them in name order, which for a sorted
// dictionary is id order.
std::vector<uint8_t> build_named_object(
  std::vector<std::string> const& dict,
  std::vector<std::pair<std::string, std::vector<uint8_t>>> fields)
{
  std::ranges::sort(fields, {}, &std::pair<std::string, std::vector<uint8_t>>::first);

  std::vector<std::pair<uint8_t, std::vector<uint8_t>>> by_id;
  by_id.reserve(fields.size());
  for (auto& [key, value] : fields) {
    auto const entry = std::ranges::lower_bound(dict, key);
    CUDF_EXPECTS(entry != dict.end() && *entry == key, "Key missing from the VARIANT dictionary");
    by_id.emplace_back(static_cast<uint8_t>(std::distance(dict.begin(), entry)), std::move(value));
  }
  return build_object(by_id);
}

// Build the value blob of the multi-path workload: a root object of six fields whose `item016`
// child fans out into ~40 sibling sub-trees, most of them `{item085: [{item018: "..."}]}`. The
// paths in `workload_paths` below match this shape.
std::vector<uint8_t> build_workload_value(std::vector<std::string> const& dict)
{
  auto const fanout_leaf = [&](int n) {
    auto inner = build_named_object(dict, {{item_key(18), build_short_string("C_" + item_key(n))}});
    return build_named_object(dict, {{item_key(85), build_array({std::move(inner)})}});
  };

  std::vector<std::pair<std::string, std::vector<uint8_t>>> item016_fields;
  for (int n : {19, 20, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 54, 57, 58, 59, 61, 62}) {
    item016_fields.emplace_back(item_key(n), fanout_leaf(n));
  }
  // Sub-trees that several paths descend into past `item085` or `item027`.
  for (int n : {26, 55}) {
    auto element = build_named_object(
      dict,
      {{item_key(18), build_short_string("C_18")}, {item_key(30), build_short_string("C_30")}});
    auto item027 = build_named_object(
      dict,
      {{item_key(28), build_short_string("C_28")}, {item_key(29), build_short_string("C_29")}});
    item016_fields.emplace_back(
      item_key(n),
      build_named_object(
        dict,
        {{item_key(85), build_array({std::move(element)})}, {item_key(27), std::move(item027)}}));
  }
  for (int n : {50, 60}) {
    auto element = build_named_object(
      dict,
      {{item_key(51), build_short_string("C_51")}, {item_key(52), build_short_string("C_52")}});
    item016_fields.emplace_back(
      item_key(n), build_named_object(dict, {{item_key(85), build_array({std::move(element)})}}));
  }
  // `item056` is an object where one path expects an array, so that path misses.
  item016_fields.emplace_back(
    item_key(56),
    build_named_object(dict,
                       {{item_key(27),
                         build_named_object(dict,
                                            {{item_key(28), build_short_string("C_28")},
                                             {item_key(29), build_short_string("C_29")}})}}));

  auto item009 = build_named_object(
    dict,
    {{item_key(10),
      build_named_object(dict,
                         {{item_key(84),
                           build_array({build_named_object(
                             dict, {{item_key(11), build_short_string("C_011")}})})}})}});

  return build_named_object(
    dict,
    {{item_key(6), build_short_string("C_006")},
     {item_key(7), build_named_object(dict, {{item_key(8), build_short_string("C_008")}})},
     {item_key(9), std::move(item009)},
     {item_key(12),
      build_named_object(dict,
                         {{item_key(13), build_short_string("C_013")},
                          {item_key(14), build_short_string("C_014")},
                          {item_key(15), build_short_string("C_015")}})},
     {item_key(16), build_named_object(dict, std::move(item016_fields))},
     {item_key(63), build_short_string("C_063")}});
}

// The paths of the multi-path workload: a few shallow ones plus a wide fan-out that all shares the
// `$.item016` prefix.
std::vector<std::string> workload_paths()
{
  std::vector<std::string> paths{
    "$." + item_key(6),
    "$." + item_key(7) + "." + item_key(8),
    "$." + item_key(9) + "." + item_key(10) + "." + item_key(84) + "[0]." + item_key(11),
    "$." + item_key(12) + "." + item_key(13),
    "$." + item_key(12) + "." + item_key(14),
    "$." + item_key(12) + "." + item_key(15),
    "$." + item_key(63)};

  auto const under_016 = [](int n) { return "$." + item_key(16) + "." + item_key(n); };
  for (int n : {19, 20, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 54, 57, 58, 59, 61, 62}) {
    paths.push_back(under_016(n) + "." + item_key(85) + "[0]." + item_key(18));
  }
  for (int n : {26, 55}) {
    paths.push_back(under_016(n) + "." + item_key(85) + "[0]." + item_key(18));
    paths.push_back(under_016(n) + "." + item_key(85) + "[0]." + item_key(30));
    paths.push_back(under_016(n) + "." + item_key(27) + "." + item_key(28));
    paths.push_back(under_016(n) + "." + item_key(27) + "." + item_key(29));
  }
  for (int n : {50, 60}) {
    paths.push_back(under_016(n) + "." + item_key(85) + "[0]." + item_key(51));
    paths.push_back(under_016(n) + "." + item_key(85) + "[0]." + item_key(52));
  }
  paths.push_back(under_016(56) + "." + item_key(27) + "." + item_key(28));
  paths.push_back(under_016(56) + "." + item_key(27) + "." + item_key(29));
  paths.push_back(under_016(56) + "[0]." + item_key(18));
  return paths;
}

// Build the JSONPath-like extraction path.
// For nesting=2, type=array: "a.b[1]"
// For nesting=3, type=string: "a.b.c"
// For nesting=0, type=array:  "[1]"
std::string get_path(int nesting, bool is_array)
{
  std::string path;
  for (int i = 0; i < nesting; ++i) {
    if (i > 0) { path += '.'; }
    path += static_cast<char>('a' + i);
  }
  if (is_array) { path += "[1]"; }
  return path;
}

cudf::data_type get_target_type(bench_variant_type type)
{
  switch (type) {
    case bench_variant_type::FLOAT: return cudf::data_type{cudf::type_id::FLOAT32};
    case bench_variant_type::BOOL: return cudf::data_type{cudf::type_id::BOOL8};
    case bench_variant_type::STRING: return cudf::data_type{cudf::type_id::STRING};
    case bench_variant_type::DECIMAL32:
      return cudf::data_type{cudf::type_id::DECIMAL32, -bench_decimal_scale};
    case bench_variant_type::DECIMAL64:
      return cudf::data_type{cudf::type_id::DECIMAL64, -bench_decimal_scale};
    case bench_variant_type::DECIMAL128:
      return cudf::data_type{cudf::type_id::DECIMAL128, -bench_decimal_scale};
    // "array": element access yields INT32.
    case bench_variant_type::INT32:
    case bench_variant_type::ARRAY: return cudf::data_type{cudf::type_id::INT32};
    default: CUDF_FAIL("Unsupported benchmark target type");
  }
}

// Assign each row randomly as a hit or miss rather than using contiguous strided ranges, so the
// memory access pattern doesn't accidentally favour cache locality. Rows are spans aliasing
// `hit_val`/`miss_val` directly, avoiding a per-row byte copy.
std::vector<std::span<uint8_t const>> fill_val_rows(cudf::size_type num_rows,
                                                    std::span<uint8_t const> hit_val,
                                                    std::span<uint8_t const> miss_val,
                                                    int hit_rate)
{
  std::mt19937 rng{42};
  std::uniform_int_distribution<int> dist{0, 99};
  std::vector<std::span<uint8_t const>> val_rows;
  val_rows.reserve(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    val_rows.push_back((dist(rng) < hit_rate) ? hit_val : miss_val);
  }
  return val_rows;
}

}  // namespace

// Benchmarks cast_variant: each row's value IS the leaf primitive (no path traversal).
static void bench_variant_cast(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const type     = parse_bench_variant_type(state.get_string("type"));
  auto const hit_rate = static_cast<int>(state.get_int64("hit_rate"));

  auto const meta_blob = build_metadata(get_dict_keys(0));
  auto hit_val         = build_leaf_value(type);
  auto miss_val        = build_miss_value(0, /*is_array=*/false, type);
  pad_to_equal_size(hit_val, miss_val);

  std::vector<std::span<uint8_t const>> meta_spans(num_rows, std::span<uint8_t const>{meta_blob});
  auto val_spans = fill_val_rows(num_rows, hit_val, miss_val, hit_rate);
  auto col       = build_variant_column(meta_spans, val_spans, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  auto const target_type = get_target_type(type);
  auto const data_size   = static_cast<std::size_t>(num_rows) * (meta_blob.size() + hit_val.size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  mr                    = cudf::get_current_device_resource_ref();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::ignore = cudf::io::parquet::experimental::cast_variant(
      col->view().child(1), target_type, std::nullopt, stream, mr);
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_variant_cast)
  .set_name("bench_variant_cast")
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("type",
                   {"string", "float", "bool", "int32_t", "decimal32", "decimal64", "decimal128"})
  .add_int64_axis("hit_rate", {20, 80});

// Benchmarks get_variant_field with varying path depth (nesting >= 1). Casting is exercised
// separately by bench_variant_cast, so this isolates pure path-traversal cost.
static void bench_variant_extract_nesting(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const type     = parse_bench_variant_type(state.get_string("type"));
  auto const nesting  = static_cast<int>(state.get_int64("nesting"));
  auto const hit_rate = static_cast<int>(state.get_int64("hit_rate"));

  bool const is_array = (type == bench_variant_type::ARRAY);

  auto const meta_blob = build_metadata(get_dict_keys(nesting));
  auto hit_val         = build_hit_value(type, nesting);
  auto miss_val        = build_miss_value(nesting, is_array, type);
  pad_to_equal_size(hit_val, miss_val);

  std::vector<std::span<uint8_t const>> meta_spans(num_rows, std::span<uint8_t const>{meta_blob});
  auto val_spans = fill_val_rows(num_rows, hit_val, miss_val, hit_rate);
  auto col       = build_variant_column(meta_spans, val_spans, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));

  auto const path      = get_path(nesting, is_array);
  auto const data_size = static_cast<std::size_t>(num_rows) * (meta_blob.size() + hit_val.size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  mr                    = cudf::get_current_device_resource_ref();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::ignore = cudf::io::parquet::experimental::get_variant_field(
      col->view(), path, std::nullopt, stream, mr);
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_variant_extract_nesting)
  .set_name("bench_variant_extract_nesting")
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("type", {"string", "float", "bool", "int32_t", "array"})
  .add_int64_axis("nesting", {1, 5})
  .add_int64_axis("hit_rate", {20, 80});

// Benchmarks get_variant_field on a flat object, varying the total number of fields and whether
// the target field is first, last, or random (probes binary- vs linear-search cost). Type is
// fixed to int32_t to isolate field-lookup overhead; casting is exercised separately by
// bench_variant_cast. The "sorted" axis toggles the metadata's sorted-strings bit: when false,
// find_key_in_metadata always falls back to a linear scan regardless of field count.
// Seed for the "random" field_position axis, kept fixed so the per-row ranks it draws (and thus
// the benchmark's runtime characteristics) stay deterministic across runs.
static constexpr auto bench_variant_extract_fields_seed = 564;

static void bench_variant_extract_fields(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_fields    = static_cast<int>(state.get_int64("num_fields"));
  auto const field_pos_str = state.get_string("field_position");
  auto const hit_rate      = static_cast<int>(state.get_int64("hit_rate"));
  auto const sorted        = state.get_string("sorted") == "true";

  auto const leaf = build_leaf_value(bench_variant_type::INT32);

  std::unique_ptr<cudf::column> col;
  std::string path;
  std::size_t meta_size;
  std::size_t hit_size;

  if (field_pos_str == "random") {
    // Draw a rank per row rather than a single fixed target field for the whole benchmark, so
    // every row's dictionary places the queried key ("n_target") at a different index. This
    // measures the average-case lookup cost instead of repeatedly re-measuring one fixed
    // position, and it makes threads within a warp diverge the way real-world unsorted-position
    // data would (a single shared blob across all rows can't).
    std::vector<std::vector<uint8_t>> meta_blobs(num_fields);
    std::vector<std::vector<uint8_t>> hit_vals(num_fields);
    for (int rank = 0; rank < num_fields; ++rank) {
      meta_blobs[rank] = build_metadata(get_dict_keys_at_rank(num_fields, rank), sorted);
      hit_vals[rank]   = build_flat_object(num_fields, rank, leaf);
    }
    // Miss: object keyed on "~miss" (field ID = num_fields for every rank), so the lookup fails
    // no matter which row's dictionary/rank it is paired with.
    auto miss_val          = wrap_in_object(static_cast<uint8_t>(num_fields), leaf);
    auto const target_size = std::max(hit_vals.front().size(), miss_val.size());
    for (auto& hit_val : hit_vals) {
      hit_val.resize(target_size, uint8_t{0});
    }
    miss_val.resize(target_size, uint8_t{0});

    std::mt19937 rng{bench_variant_extract_fields_seed};
    std::uniform_int_distribution<int> rank_dist{0, num_fields - 1};
    std::uniform_int_distribution<int> hit_dist{0, 99};

    std::vector<std::span<uint8_t const>> meta_spans;
    std::vector<std::span<uint8_t const>> val_spans;
    meta_spans.reserve(num_rows);
    val_spans.reserve(num_rows);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      auto const rank = rank_dist(rng);
      meta_spans.emplace_back(meta_blobs[rank]);
      val_spans.emplace_back(hit_dist(rng) < hit_rate ? std::span<uint8_t const>{hit_vals[rank]}
                                                      : std::span<uint8_t const>{miss_val});
    }
    col = build_variant_column(meta_spans, val_spans, stream, mr);
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));

    path      = "n_target";
    meta_size = meta_blobs.front().size();
    hit_size  = hit_vals.front().size();
  } else {
    int const target_fid = (field_pos_str == "last") ? num_fields - 1 : 0;

    auto const meta_blob = build_metadata(get_dict_keys_for_fields(num_fields), sorted);
    auto hit_val         = build_flat_object(num_fields, target_fid, leaf);
    // Miss: object keyed on "z" (field ID = num_fields), so the lookup fails.
    auto miss_val = wrap_in_object(static_cast<uint8_t>(num_fields), leaf);
    pad_to_equal_size(hit_val, miss_val);

    std::vector<std::span<uint8_t const>> meta_spans(num_rows, std::span<uint8_t const>{meta_blob});
    auto val_spans = fill_val_rows(num_rows, hit_val, miss_val, hit_rate);
    col            = build_variant_column(meta_spans, val_spans, stream, mr);
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));

    path      = "f" + std::string(target_fid < 10 ? "0" : "") + std::to_string(target_fid);
    meta_size = meta_blob.size();
    hit_size  = hit_val.size();
  }

  auto const data_size = static_cast<std::size_t>(num_rows) * (meta_size + hit_size);

  auto mem_stats_logger = cudf::memory_stats_logger();
  mr                    = cudf::get_current_device_resource_ref();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::ignore = cudf::io::parquet::experimental::get_variant_field(
      col->view(), path, std::nullopt, stream, mr);
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_variant_extract_fields)
  .set_name("bench_variant_extract_fields")
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("num_fields", {1, 10, 100})
  .add_string_axis("field_position", {"first", "last", "random"})
  .add_int64_axis("hit_rate", {20, 80})
  .add_string_axis("sorted", {"true", "false"});

// One status column per path, for the benchmarks' `status` axis, or none when it is off. The
// extraction APIs read a status buffer as incoming status before overwriting it, so the buffers
// start at SUCCESS.
struct status_columns {
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<cudf::mutable_column_view> views;
};

status_columns make_status_columns(std::size_t num_paths,
                                   cudf::size_type num_rows,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  status_columns out;
  out.columns.reserve(num_paths);
  out.views.reserve(num_paths);
  for (std::size_t p = 0; p < num_paths; ++p) {
    out.columns.push_back(cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::UINT8}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr));
    auto view = out.columns.back()->mutable_view();
    CUDF_CUDA_TRY(
      cudaMemsetAsync(view.data<uint8_t>(), 0, static_cast<std::size_t>(num_rows), stream.value()));
    out.views.push_back(view);
  }
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  return out;
}

// Compares extracting many fields in one batched call against looping the single-field API, with
// and without a prefix shared by all the requested paths. Type is fixed to int32_t; both APIs
// decode, so the two sides of the comparison are end-to-end equivalent. The `status` axis asks both
// for per-row status, which the batched walk writes once per path and row.
static void bench_variant_extract_multi_field(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_fields    = static_cast<int>(state.get_int64("num_fields"));
  auto const hit_rate      = static_cast<int>(state.get_int64("hit_rate"));
  bool const shared_prefix = state.get_string("prefix") == "shared";
  bool const batched       = state.get_string("api") == "batched";
  bool const with_status   = state.get_string("status") == "on";

  // Dictionary: the shared parent key "a", the leaf keys f00..f{N-1}, and "z" for miss rows, in
  // sorted order. Field ids are dictionary indices, so "a" is 0 and leaf `i` is `i + 1`.
  std::vector<std::string> keys{"a"};
  for (int i = 0; i < num_fields; ++i) {
    keys.push_back(field_key(i));
  }
  keys.emplace_back("z");
  auto const meta_blob = build_metadata(keys);

  auto const leaf = build_leaf_value(bench_variant_type::INT32);
  std::vector<std::pair<uint8_t, std::vector<uint8_t>>> leaf_fields;
  for (int i = 0; i < num_fields; ++i) {
    leaf_fields.emplace_back(static_cast<uint8_t>(i + 1), leaf);
  }
  // Shared-prefix layout nests every leaf under "a"; the disjoint layout puts them at the top
  // level.
  auto const leaf_object = build_object(leaf_fields);
  auto hit_val           = shared_prefix ? build_object({{uint8_t{0}, leaf_object}}) : leaf_object;
  // Miss rows hold an object keyed on "z" alone, so every path fails at its first step.
  auto miss_val = build_object({{static_cast<uint8_t>(num_fields + 1), leaf}});
  pad_to_equal_size(hit_val, miss_val);

  std::vector<std::span<uint8_t const>> meta_spans(num_rows, std::span<uint8_t const>{meta_blob});
  auto val_spans = fill_val_rows(num_rows, hit_val, miss_val, hit_rate);
  auto col       = build_variant_column(meta_spans, val_spans, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  std::vector<std::string> path_strings;
  path_strings.reserve(num_fields);
  for (int i = 0; i < num_fields; ++i) {
    path_strings.push_back((shared_prefix ? "$.a." : "$.") + field_key(i));
  }
  std::vector<std::string_view> const paths(path_strings.begin(), path_strings.end());
  auto const target_type = cudf::data_type{cudf::type_id::INT32};
  std::vector<cudf::data_type> const target_types(num_fields, target_type);

  auto status = make_status_columns(with_status ? paths.size() : 0, num_rows, stream, mr);

  auto const data_size = static_cast<std::size_t>(num_rows) * (meta_blob.size() + hit_val.size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  mr                    = cudf::get_current_device_resource_ref();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (batched) {
      std::ignore = cudf::io::parquet::experimental::extract_variant_fields(
        col->view(), paths, target_types, status.views, stream, mr);
    } else {
      for (std::size_t p = 0; p < path_strings.size(); ++p) {
        std::ignore = cudf::io::parquet::experimental::extract_variant_field(
          col->view(),
          path_strings[p],
          target_type,
          with_status ? std::optional{status.views[p]} : std::nullopt,
          stream,
          mr);
      }
    }
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_variant_extract_multi_field)
  .set_name("bench_variant_extract_multi_field")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_int64_axis("num_fields", {1, 4, 16, 64})
  .add_string_axis("prefix", {"shared", "disjoint"})
  .add_string_axis("api", {"batched", "looped"})
  .add_string_axis("status", {"off", "on"})
  .add_int64_axis("hit_rate", {80});

// Compares batched against looped extraction on a workload shaped like a real one: an 85-key
// dictionary, a root object that nests most of its data under `item016`, and 50 paths that fan out
// below that shared prefix at a depth of four to five steps.
static void bench_variant_extract_workload(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  bool const batched     = state.get_string("api") == "batched";
  bool const with_status = state.get_string("status") == "on";

  std::vector<std::string> dict;
  dict.reserve(85);
  for (int n = 1; n <= 85; ++n) {
    dict.push_back(item_key(n));
  }

  auto const meta_blob = build_metadata(dict);
  auto const val_blob  = build_workload_value(dict);

  std::vector<std::span<uint8_t const>> meta_spans(num_rows, std::span<uint8_t const>{meta_blob});
  std::vector<std::span<uint8_t const>> val_spans(num_rows, std::span<uint8_t const>{val_blob});
  auto col = build_variant_column(meta_spans, val_spans, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  auto const path_strings = workload_paths();
  std::vector<std::string_view> const paths(path_strings.begin(), path_strings.end());
  auto const target_type = cudf::data_type{cudf::type_id::STRING};
  std::vector<cudf::data_type> const target_types(paths.size(), target_type);

  auto status = make_status_columns(with_status ? paths.size() : 0, num_rows, stream, mr);

  auto const data_size = static_cast<std::size_t>(num_rows) * (meta_blob.size() + val_blob.size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  mr                    = cudf::get_current_device_resource_ref();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (batched) {
      std::ignore = cudf::io::parquet::experimental::extract_variant_fields(
        col->view(), paths, target_types, status.views, stream, mr);
    } else {
      for (std::size_t p = 0; p < path_strings.size(); ++p) {
        std::ignore = cudf::io::parquet::experimental::extract_variant_field(
          col->view(),
          path_strings[p],
          target_type,
          with_status ? std::optional{status.views[p]} : std::nullopt,
          stream,
          mr);
      }
    }
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_variant_extract_workload)
  .set_name("bench_variant_extract_workload")
  .add_int64_axis("num_rows", {262144, 1048576})
  .add_string_axis("api", {"batched", "looped"})
  .add_string_axis("status", {"off", "on"});
