/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

void append_le(std::vector<uint8_t>& out, uint64_t bits, int width)
{
  for (int i = 0; i < width; ++i) {
    out.push_back(static_cast<uint8_t>((bits >> (8 * i)) & 0xff));
  }
}

// Build a V1 VARIANT metadata blob for a sorted key dictionary.
// Uses 2-byte offsets when the total string length exceeds 255 bytes; 1-byte otherwise.
// Header bits [7:6] = offset_size_minus_one; bits [3:0] = version (1).
std::vector<uint8_t> build_metadata(std::vector<std::string> const& keys)
{
  uint32_t total = 0;
  for (auto const& k : keys)
    total += static_cast<uint32_t>(k.size());

  int const offset_size = (total > 255u) ? 2 : 1;
  std::vector<uint8_t> out{static_cast<uint8_t>(0x01 | ((offset_size - 1) << 6))};

  auto write_le = [&](uint32_t v) {
    for (int i = 0; i < offset_size; ++i)
      out.push_back(static_cast<uint8_t>(v >> (8 * i)));
  };
  write_le(static_cast<uint32_t>(keys.size()));

  uint32_t running = 0;
  write_le(0u);
  for (auto const& k : keys) {
    running += static_cast<uint32_t>(k.size());
    write_le(running);
  }

  for (auto const& k : keys) {
    out.insert(out.end(), k.begin(), k.end());
  }
  return out;
}

// Wrap `inner` as the sole field (field id `fid`) of a 1-field VARIANT object.
// Uses 1-byte field_id_size and 1-byte field_offset_size (value_header=0 → header=0x02).
std::vector<uint8_t> wrap_in_object(uint8_t fid, std::vector<uint8_t> const& inner)
{
  // Format: object_header(1) + num_fields(1) + fid(1) + offset[0]=0(1) + offset[1]=size(1) + data
  std::vector<uint8_t> out{0x02, 0x01, fid, 0x00, static_cast<uint8_t>(inner.size())};
  out.insert(out.end(), inner.begin(), inner.end());
  return out;
}

// Build the leaf VARIANT value blob for the requested type.
//
// Header byte composition: (physical_type_id << 2) | basic_type
//   PRIMITIVE basic_type = 0, so header = physical_type_id << 2
//   SHORT_STRING basic_type = 1, so header = (length << 2) | 1
//   ARRAY basic_type = 3, so header = (value_header << 2) | 3
//
// Physical type IDs used:
//   INT32    = 5  → header 0x14
//   FLOAT32  = 14 → header 0x38
//   BOOL_TRUE= 1  → header 0x04
std::vector<uint8_t> build_leaf_value(std::string const& type_str)
{
  if (type_str == "int32_t") {
    std::vector<uint8_t> out{0x14};
    append_le(out, 42u, 4);
    return out;
  }
  if (type_str == "float") {
    std::vector<uint8_t> out{0x38};
    float const f = 1.0f;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    append_le(out, u, 4);
    return out;
  }
  if (type_str == "bool") {
    return {0x04};  // BOOLEAN_TRUE
  }
  if (type_str == "string") {
    // Short string "hello" (5 bytes): (5 << 2) | 1 = 0x15
    return {0x15, 'h', 'e', 'l', 'l', 'o'};
  }
  // "array": VARIANT array of two INT32 values [42, 99]; element [1] is accessed in the benchmark.
  // Array header 0x03: basic_type=ARRAY(3), value_header=0 (1-byte count, 1-byte offsets).
  // 2 elements, offsets [0, 5, 10], then INT32(42) and INT32(99) (5 bytes each).
  std::vector<uint8_t> out{0x03, 0x02, 0x00, 0x05, 0x0a};
  out.push_back(0x14);
  append_le(out, 42u, 4);
  out.push_back(0x14);
  append_le(out, 99u, 4);
  return out;
}

// Build the full hit-row value blob by wrapping the leaf in `nesting` object levels.
// Keys a,b,c,d,e map to field IDs 0,1,2,3,4 in the shared dictionary.
// For path a.b.c.d.e the outermost object uses fid=0 ("a").
std::vector<uint8_t> build_hit_value(std::string const& type_str, int nesting)
{
  auto val = build_leaf_value(type_str);
  for (int i = nesting - 1; i >= 0; --i) {
    val = wrap_in_object(static_cast<uint8_t>(i), val);
  }
  return val;
}

// Build the miss-row value blob: a valid VARIANT that won't match the target path or type.
// For extract_variant_field rows: a 1-level object keyed on "z" (field ID = nesting in the
// dictionary), so traversal fails at the first key lookup while the row remains non-null.
// For cast_variant rows (nesting=0, non-array): a different primitive type so the cast returns
// null.
std::vector<uint8_t> build_miss_value(int nesting, bool is_array, std::string const& type_str)
{
  if (nesting == 0 && !is_array) {
    // Wrong-type primitive for the cast path.
    if (type_str == "bool") {
      std::vector<uint8_t> out{0x14};
      append_le(out, 0u, 4);
      return out;
    }
    return {0x04};  // BOOLEAN_TRUE
  }
  // "z" is always the last key in the dictionary, at field ID = nesting.
  return wrap_in_object(static_cast<uint8_t>(nesting), build_leaf_value(type_str));
}

// Build a VARIANT struct column (STRUCT<list<uint8>, list<uint8>>) from per-row byte vectors.
std::unique_ptr<cudf::column> build_variant_column(
  std::vector<std::vector<uint8_t>> const& meta_rows,
  std::vector<std::vector<uint8_t>> const& val_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const n = static_cast<cudf::size_type>(meta_rows.size());

  auto build_list_col =
    [&](std::vector<std::vector<uint8_t>> const& rows) -> std::unique_ptr<cudf::column> {
    std::vector<int32_t> offsets(n + 1, 0);
    std::vector<uint8_t> flat;
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
// Field `target_fid` holds `inner`; all other fields hold a dummy BOOLEAN_TRUE (0x04).
std::vector<uint8_t> build_flat_object(int num_fields,
                                       int target_fid,
                                       std::vector<uint8_t> const& inner)
{
  // object_header(1) + num_fields(1) + field_ids(num_fields) + offsets(num_fields+1) + data
  std::vector<uint8_t> out{0x02, static_cast<uint8_t>(num_fields)};
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
      out.push_back(0x04);  // BOOLEAN_TRUE dummy
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
    if (i > 0) path += '.';
    path += static_cast<char>('a' + i);
  }
  if (is_array) path += "[1]";
  return path;
}

cudf::data_type get_target_type(std::string const& type_str)
{
  if (type_str == "float") return cudf::data_type{cudf::type_id::FLOAT32};
  if (type_str == "bool") return cudf::data_type{cudf::type_id::BOOL8};
  if (type_str == "string") return cudf::data_type{cudf::type_id::STRING};
  // "int32_t" and "array" (element access yields INT32)
  return cudf::data_type{cudf::type_id::INT32};
}

}  // namespace

// Assign each row randomly as a hit or miss rather than using contiguous strided ranges,
// so the memory access pattern doesn't accidentally favour cache locality.
void fill_val_rows(std::vector<std::vector<uint8_t>>& val_rows,
                   std::vector<uint8_t> const& hit_val,
                   std::vector<uint8_t> const& miss_val,
                   int hit_rate)
{
  std::mt19937 rng{42};
  std::uniform_int_distribution<int> dist{0, 99};
  for (auto& row : val_rows) {
    row = (dist(rng) < hit_rate) ? hit_val : miss_val;
  }
}

// Benchmarks cast_variant: each row's value IS the leaf primitive (no path traversal).
static void bench_variant_cast(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const type_str = state.get_string("type");
  auto const hit_rate = static_cast<int>(state.get_int64("hit_rate"));

  auto const meta_blob = build_metadata(get_dict_keys(0));
  auto const hit_val   = build_leaf_value(type_str);
  auto const miss_val  = build_miss_value(0, /*is_array=*/false, type_str);

  std::vector<std::vector<uint8_t>> meta_rows(num_rows, meta_blob);
  std::vector<std::vector<uint8_t>> val_rows(num_rows);
  fill_val_rows(val_rows, hit_val, miss_val, hit_rate);

  auto col = build_variant_column(meta_rows, val_rows, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  auto const target_type = get_target_type(type_str);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::ignore =
      cudf::io::parquet::experimental::cast_variant(col->view().child(1), target_type, stream, mr);
  });
}

NVBENCH_BENCH(bench_variant_cast)
  .set_name("bench_variant_cast")
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("type", {"string", "float", "bool", "int32_t"})
  .add_int64_axis("hit_rate", {20, 80});

// Benchmarks extract_variant_field with varying path depth (nesting >= 1).
static void bench_variant_extract_nesting(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const type_str = state.get_string("type");
  auto const nesting  = static_cast<int>(state.get_int64("nesting"));
  auto const hit_rate = static_cast<int>(state.get_int64("hit_rate"));

  bool const is_array = (type_str == "array");

  auto const meta_blob = build_metadata(get_dict_keys(nesting));
  auto const hit_val   = build_hit_value(type_str, nesting);
  auto const miss_val  = build_miss_value(nesting, is_array, type_str);

  std::vector<std::vector<uint8_t>> meta_rows(num_rows, meta_blob);
  std::vector<std::vector<uint8_t>> val_rows(num_rows);
  fill_val_rows(val_rows, hit_val, miss_val, hit_rate);

  auto col = build_variant_column(meta_rows, val_rows, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  auto const target_type = get_target_type(type_str);
  auto const path        = get_path(nesting, is_array);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::ignore = cudf::io::parquet::experimental::extract_variant_field(
      col->view(), path, target_type, stream, mr);
  });
}

NVBENCH_BENCH(bench_variant_extract_nesting)
  .set_name("bench_variant_extract_nesting")
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("type", {"string", "float", "bool", "int32_t", "array"})
  .add_int64_axis("nesting", {1, 5})
  .add_int64_axis("hit_rate", {20, 80});

// Benchmarks extract_variant_field on a flat object, varying the total number of fields
// and whether the target field is first or last (probes binary search cost).
// Type is fixed to int32_t to isolate field-lookup overhead.
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
  auto const leaf      = build_leaf_value("int32_t");
  auto const hit_val   = build_flat_object(num_fields, target_fid, leaf);
  // Miss: object keyed on "z" (field ID = num_fields), so the lookup fails.
  auto const miss_val = wrap_in_object(static_cast<uint8_t>(num_fields), leaf);

  std::vector<std::vector<uint8_t>> meta_rows(num_rows, meta_blob);
  std::vector<std::vector<uint8_t>> val_rows(num_rows);
  fill_val_rows(val_rows, hit_val, miss_val, hit_rate);

  auto col = build_variant_column(meta_rows, val_rows, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  std::string const path =
    "f" + std::string(target_fid < 10 ? "0" : "") + std::to_string(target_fid);
  auto const target_type = cudf::data_type{cudf::type_id::INT32};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::ignore = cudf::io::parquet::experimental::extract_variant_field(
      col->view(), path, target_type, stream, mr);
  });
}

NVBENCH_BENCH(bench_variant_extract_fields)
  .set_name("bench_variant_extract_fields")
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("num_fields", {1, 10, 100})
  .add_string_axis("field_position", {"first", "last"})
  .add_int64_axis("hit_rate", {20, 80});
