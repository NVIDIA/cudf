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
enum class bench_variant_type : uint8_t { INT32, FLOAT, BOOL, STRING, ARRAY };

bench_variant_type parse_bench_variant_type(std::string const& type_str)
{
  if (type_str == "int32_t") { return bench_variant_type::INT32; }
  if (type_str == "float") { return bench_variant_type::FLOAT; }
  if (type_str == "bool") { return bench_variant_type::BOOL; }
  if (type_str == "string") { return bench_variant_type::STRING; }
  if (type_str == "array") { return bench_variant_type::ARRAY; }
  CUDF_FAIL("Unrecognized benchmark type: " + type_str);
}

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

// Build a V1 VARIANT metadata blob for a sorted key dictionary. Callers must pass `keys` in
// ascending sorted order; the sorted-strings header bit is always set to reflect that.
// Uses 2-byte offsets when the total string length exceeds 255 bytes; 1-byte otherwise.
// Header bits [7:6] = offset_size_minus_one; bit [4] = sorted_strings; bits [3:0] = version (1).
std::vector<uint8_t> build_metadata(std::vector<std::string> const& keys)
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
  std::vector<uint8_t> out{static_cast<uint8_t>(kVariantMetadataVersion | kVariantMetadataSorted |
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
                                                   rmm::cuda_stream_view stream,
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
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  auto const target_type = get_target_type(type);
  auto const data_size   = static_cast<std::size_t>(num_rows) * (meta_blob.size() + hit_val.size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  mr                    = cudf::get_current_device_resource_ref();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::ignore =
      cudf::io::parquet::experimental::cast_variant(col->view().child(1), target_type, stream, mr);
  });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_variant_cast)
  .set_name("bench_variant_cast")
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("type", {"string", "float", "bool", "int32_t"})
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
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  auto const path      = get_path(nesting, is_array);
  auto const data_size = static_cast<std::size_t>(num_rows) * (meta_blob.size() + hit_val.size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  mr                    = cudf::get_current_device_resource_ref();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::ignore = cudf::io::parquet::experimental::get_variant_field(col->view(), path, stream, mr);
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
// the target field is first or last (probes binary search cost). Type is fixed to int32_t to
// isolate field-lookup overhead; casting is exercised separately by bench_variant_cast.
static void bench_variant_extract_fields(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_fields    = static_cast<int>(state.get_int64("num_fields"));
  auto const field_pos_str = state.get_string("field_position");
  auto const hit_rate      = static_cast<int>(state.get_int64("hit_rate"));

  int const target_fid = (field_pos_str == "last") ? (num_fields - 1) : 0;

  auto const meta_blob = build_metadata(get_dict_keys_for_fields(num_fields));
  auto const leaf      = build_leaf_value(bench_variant_type::INT32);
  auto hit_val         = build_flat_object(num_fields, target_fid, leaf);
  // Miss: object keyed on "z" (field ID = num_fields), so the lookup fails.
  auto miss_val = wrap_in_object(static_cast<uint8_t>(num_fields), leaf);
  pad_to_equal_size(hit_val, miss_val);

  std::vector<std::span<uint8_t const>> meta_spans(num_rows, std::span<uint8_t const>{meta_blob});
  auto val_spans = fill_val_rows(num_rows, hit_val, miss_val, hit_rate);
  auto col       = build_variant_column(meta_spans, val_spans, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  std::string const path =
    "f" + std::string(target_fid < 10 ? "0" : "") + std::to_string(target_fid);
  auto const data_size = static_cast<std::size_t>(num_rows) * (meta_blob.size() + hit_val.size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  mr                    = cudf::get_current_device_resource_ref();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::ignore = cudf::io::parquet::experimental::get_variant_field(col->view(), path, stream, mr);
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
  .add_string_axis("field_position", {"first", "last"})
  .add_int64_axis("hit_rate", {20, 80});
