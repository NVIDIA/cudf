/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/json/json.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/cstring>
#include <cuda/std/optional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace cudf {
namespace io::parquet::experimental {
namespace {

using basic_type     = variant_basic_type;
using primitive_type = variant_primitive_type;

constexpr int block_size_encode = 256;

// Sizes of the fixed-width fields in the encoded object value blob.
// We always use 1-byte field IDs (supports up to 255 keys), 1-byte num_elements,
// and 4-byte field offsets (supports values up to ~4 GB per row).
constexpr int FIELD_ID_SIZE     = 1;
constexpr int FIELD_OFFSET_SIZE = 4;
constexpr int NUM_ELEMENTS_SIZE = 1;  // is_large=0

// value_header for object: is_large(0) | (field_id_size-1)(0<<2) | (field_offset_size-1)(3)
// value_metadata = OBJECT(2) | (value_header << 2)
constexpr uint8_t OBJECT_VALUE_METADATA =
  static_cast<uint8_t>(basic_type::OBJECT) |
  (((0u << 4u) | (0u << 2u) | uint8_t{FIELD_OFFSET_SIZE - 1}) << 2u);

// ─── device helpers ───────────────────────────────────────────────────────────

__device__ void write_le(uint8_t*& out, uint64_t val, int bytes)
{
  for (int i = 0; i < bytes; ++i) {
    *out++ = static_cast<uint8_t>(val & 0xFFu);
    val >>= 8u;
  }
}

__device__ cuda::std::optional<int64_t> try_parse_int64(cudf::string_view s)
{
  auto const* data = s.data();
  auto const n     = s.size_bytes();
  if (n == 0) { return cuda::std::nullopt; }

  size_type i   = 0;
  bool negative = false;
  if (data[i] == '-') {
    negative = true;
    ++i;
  }
  if (i >= n || data[i] < '0' || data[i] > '9') { return cuda::std::nullopt; }

  // Accumulate as a negative value to correctly represent INT64_MIN.
  constexpr int64_t INT64_MIN_VAL = int64_t{-9223372036854775807LL - 1};
  int64_t result                  = 0;
  while (i < n) {
    char c = data[i];
    if (c < '0' || c > '9') { return cuda::std::nullopt; }
    int64_t d = c - '0';
    if (result < (INT64_MIN_VAL + d) / 10) { return cuda::std::nullopt; }
    result = result * 10 - d;
    ++i;
  }
  if (!negative) {
    if (result == INT64_MIN_VAL) { return cuda::std::nullopt; }
    return -result;
  }
  return result;
}

__device__ double parse_float64(cudf::string_view s)
{
  auto const* data = s.data();
  auto const n     = s.size_bytes();

  size_type i   = 0;
  bool negative = false;
  if (i < n && data[i] == '-') {
    negative = true;
    ++i;
  }

  double result = 0.0;
  while (i < n && data[i] >= '0' && data[i] <= '9') {
    result = result * 10.0 + (data[i] - '0');
    ++i;
  }

  if (i < n && data[i] == '.') {
    ++i;
    double factor = 0.1;
    while (i < n && data[i] >= '0' && data[i] <= '9') {
      result += (data[i] - '0') * factor;
      factor *= 0.1;
      ++i;
    }
  }

  if (i < n && (data[i] == 'e' || data[i] == 'E')) {
    ++i;
    bool exp_neg = false;
    if (i < n && data[i] == '-') {
      exp_neg = true;
      ++i;
    } else if (i < n && data[i] == '+') {
      ++i;
    }
    int exp = 0;
    while (i < n && data[i] >= '0' && data[i] <= '9') {
      exp = exp * 10 + (data[i] - '0');
      ++i;
    }
    double factor = 1.0;
    for (int j = 0; j < exp; ++j) {
      factor *= 10.0;
    }
    if (exp_neg) {
      result /= factor;
    } else {
      result *= factor;
    }
  }

  return negative ? -result : result;
}

// Returns true if s contains a float-indicating character ('.', 'e', 'E').
__device__ bool is_float_number(cudf::string_view s)
{
  for (size_type i = 0; i < s.size_bytes(); ++i) {
    char c = s.data()[i];
    if (c == '.' || c == 'e' || c == 'E') { return true; }
  }
  return false;
}

// Size in bytes of the VARIANT encoding for one JSON scalar value string.
// `raw` is the string returned by get_json_object with strip_quotes_from_single_strings=false.
__device__ size_type encoded_field_size(cudf::string_view raw)
{
  if (raw == cudf::string_view("null", 4)) { return 1; }
  if (raw == cudf::string_view("true", 4) || raw == cudf::string_view("false", 5)) { return 1; }

  if (raw.size_bytes() >= 2 && raw.data()[0] == '"') {
    // JSON string: strip surrounding quotes
    size_type str_len = static_cast<size_type>(raw.size_bytes()) - 2;
    if (str_len <= 63) { return 1 + str_len; }  // SHORT_STRING
    return 1 + 4 + str_len;                     // LONG_STRING (1 hdr + 4-byte len + bytes)
  }

  if (is_float_number(raw)) { return 9; }  // header + 8-byte double
  return 9;                                // header + 8-byte int64
}

// Write VARIANT bytes for one JSON scalar value string into `out`.
// Returns pointer past the last written byte.
__device__ uint8_t* write_field_value(uint8_t* out, cudf::string_view raw)
{
  auto make_prim_header = [](primitive_type pt) -> uint8_t {
    return static_cast<uint8_t>(basic_type::PRIMITIVE) | (static_cast<uint8_t>(pt) << 2u);
  };

  if (raw == cudf::string_view("null", 4)) {
    *out++ = make_prim_header(primitive_type::NULLVAL);
    return out;
  }
  if (raw == cudf::string_view("true", 4)) {
    *out++ = make_prim_header(primitive_type::BOOLEAN_TRUE);
    return out;
  }
  if (raw == cudf::string_view("false", 5)) {
    *out++ = make_prim_header(primitive_type::BOOLEAN_FALSE);
    return out;
  }

  if (raw.size_bytes() >= 2 && raw.data()[0] == '"') {
    auto const* str_start = raw.data() + 1;
    size_type str_len     = static_cast<size_type>(raw.size_bytes()) - 2;

    if (str_len <= 63) {
      *out++ =
        static_cast<uint8_t>(basic_type::SHORT_STRING) | (static_cast<uint8_t>(str_len) << 2u);
      cuda::std::memcpy(out, str_start, str_len);
      return out + str_len;
    }
    // LONG_STRING
    *out++         = make_prim_header(primitive_type::LONG_STRING);
    uint32_t len32 = static_cast<uint32_t>(str_len);
    cuda::std::memcpy(out, &len32, 4);
    out += 4;
    cuda::std::memcpy(out, str_start, str_len);
    return out + str_len;
  }

  if (is_float_number(raw)) {
    *out++     = make_prim_header(primitive_type::FLOAT64);
    double val = parse_float64(raw);
    cuda::std::memcpy(out, &val, sizeof(double));
    return out + sizeof(double);
  }

  auto parsed = try_parse_int64(raw);
  if (parsed.has_value()) {
    *out++       = make_prim_header(primitive_type::INT64);
    int64_t ival = *parsed;
    cuda::std::memcpy(out, &ival, sizeof(int64_t));
    return out + sizeof(int64_t);
  }
  // Failed to parse as INT64 (e.g. out-of-range): fall back to FLOAT64.
  *out++     = make_prim_header(primitive_type::FLOAT64);
  double val = parse_float64(raw);
  cuda::std::memcpy(out, &val, sizeof(double));
  return out + sizeof(double);
}

// ─── kernels ──────────────────────────────────────────────────────────────────

/**
 * @brief Compute the byte size of each row's VARIANT value blob.
 *
 * Null input rows produce size 0.  Non-null rows get the object blob size,
 * summing header + num_elements + field_ids + field_offsets + field values.
 */
CUDF_KERNEL __launch_bounds__(block_size_encode) void compute_value_sizes_kernel(
  device_span<column_device_view const> extracted,  // N columns in original order
  device_span<int32_t const> sorted_to_original,    // sorted field index → original col index
  size_type num_rows,
  size_type num_fields,
  bitmask_type const* input_null_mask,
  device_span<size_type> value_sizes)
{
  auto const tid    = cudf::detail::grid_1d::global_thread_id<block_size_encode>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size_encode>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (input_null_mask != nullptr && !cudf::bit_is_set(input_null_mask, row)) {
      value_sizes[row] = 0;
      continue;
    }

    size_type n_present    = 0;
    size_type values_bytes = 0;

    for (size_type si = 0; si < num_fields; ++si) {
      auto const orig = sorted_to_original[si];
      if (extracted[orig].is_null(row)) { continue; }
      ++n_present;
      values_bytes += encoded_field_size(extracted[orig].element<cudf::string_view>(row));
    }

    // 1 (value_metadata) + NUM_ELEMENTS_SIZE + n_present*FIELD_ID_SIZE
    // + (n_present+1)*FIELD_OFFSET_SIZE + values_bytes
    value_sizes[row] = 1 + NUM_ELEMENTS_SIZE + n_present * FIELD_ID_SIZE +
                       (n_present + 1) * FIELD_OFFSET_SIZE + values_bytes;
  }
}

/**
 * @brief Write the VARIANT value blob for each row into the pre-allocated output buffer.
 *
 * Uses prefix-summed @p value_offsets to locate each row's destination region.
 */
CUDF_KERNEL __launch_bounds__(block_size_encode) void write_values_kernel(
  device_span<column_device_view const> extracted,
  device_span<int32_t const> sorted_to_original,
  size_type num_rows,
  size_type num_fields,
  bitmask_type const* input_null_mask,
  device_span<size_type const> value_offsets,
  uint8_t* output)
{
  auto const tid    = cudf::detail::grid_1d::global_thread_id<block_size_encode>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size_encode>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (input_null_mask != nullptr && !cudf::bit_is_set(input_null_mask, row)) { continue; }

    uint8_t* out = output + value_offsets[row];

    // Header byte
    *out++ = OBJECT_VALUE_METADATA;

    // Count present fields and their individual sizes (first pass over fields)
    size_type n_present = 0;
    for (size_type si = 0; si < num_fields; ++si) {
      if (!extracted[sorted_to_original[si]].is_null(row)) { ++n_present; }
    }

    // num_elements
    write_le(out, static_cast<uint64_t>(n_present), NUM_ELEMENTS_SIZE);

    // field_ids (sorted dict index for each present field)
    for (size_type si = 0; si < num_fields; ++si) {
      auto const orig = sorted_to_original[si];
      if (extracted[orig].is_null(row)) { continue; }
      write_le(out, static_cast<uint64_t>(si), FIELD_ID_SIZE);
    }

    // field_offsets: cumulative offsets within the values region + sentinel
    size_type cur_offset = 0;
    for (size_type si = 0; si < num_fields; ++si) {
      auto const orig = sorted_to_original[si];
      if (extracted[orig].is_null(row)) { continue; }
      write_le(out, static_cast<uint64_t>(cur_offset), FIELD_OFFSET_SIZE);
      cur_offset += encoded_field_size(extracted[orig].element<cudf::string_view>(row));
    }
    write_le(out, static_cast<uint64_t>(cur_offset), FIELD_OFFSET_SIZE);  // sentinel

    // field values in sorted order
    for (size_type si = 0; si < num_fields; ++si) {
      auto const orig = sorted_to_original[si];
      if (extracted[orig].is_null(row)) { continue; }
      out = write_field_value(out, extracted[orig].element<cudf::string_view>(row));
    }
  }
}

// ─── host helpers ─────────────────────────────────────────────────────────────

// Build the fixed VARIANT metadata blob for a sorted list of key names.
// Layout: header(1) | dict_size(offset_size) | offsets[(N+1)*offset_size] | key_bytes
std::vector<uint8_t> build_metadata_blob(std::vector<std::string> const& sorted_names)
{
  size_t const N         = sorted_names.size();
  size_t total_key_bytes = 0;
  for (auto const& name : sorted_names) {
    total_key_bytes += name.size();
  }

  int offset_size = 1;
  if (total_key_bytes > 255 || N > 255) { offset_size = 2; }
  if (total_key_bytes > 65535 || N > 65535) { offset_size = 4; }

  auto write_le_host = [](std::vector<uint8_t>& buf, size_t val, int bytes) {
    for (int i = 0; i < bytes; ++i) {
      buf.push_back(static_cast<uint8_t>(val & 0xFFu));
      val >>= 8u;
    }
  };

  std::vector<uint8_t> blob;
  // header: version=1 | sorted=1 | unused=0 | offset_size-1
  blob.push_back(static_cast<uint8_t>(0x01u | (1u << 4u) | (uint8_t(offset_size - 1) << 6u)));

  write_le_host(blob, N, offset_size);  // dictionary_size

  // offsets[0..N] relative to start of string_data
  size_t cur = 0;
  for (auto const& name : sorted_names) {
    write_le_host(blob, cur, offset_size);
    cur += name.size();
  }
  write_le_host(blob, cur, offset_size);  // sentinel

  for (auto const& name : sorted_names) {
    for (char c : name) {
      blob.push_back(static_cast<uint8_t>(c));
    }
  }

  return blob;
}

// Build a list<uint8> column where every row contains the same `blob` bytes.
// Null rows (from input_null_mask) get 0-length list entries.
std::unique_ptr<column> make_constant_metadata_column(std::vector<uint8_t> const& blob,
                                                      size_type num_rows,
                                                      bitmask_type const* input_null_mask,
                                                      size_type null_count,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  size_type const m = static_cast<size_type>(blob.size());

  // Copy null mask to host so we can check validity per row while building offsets.
  std::vector<bitmask_type> h_null_mask;
  if (null_count > 0 && input_null_mask != nullptr) {
    size_t const mask_bytes = cudf::bitmask_allocation_size_bytes(num_rows);
    h_null_mask.resize(mask_bytes / sizeof(bitmask_type));
    cudf::detail::cuda_memcpy(host_span<bitmask_type>{h_null_mask},
                              device_span<bitmask_type const>{input_null_mask, h_null_mask.size()},
                              stream);
  }

  auto row_is_null = [&](size_type i) -> bool {
    if (h_null_mask.empty()) { return false; }
    return !cudf::bit_is_set(h_null_mask.data(), i);
  };

  // Build host offsets: each non-null row occupies m bytes
  std::vector<size_type> h_offsets(num_rows + 1);
  size_type running = 0;
  for (size_type i = 0; i < num_rows; ++i) {
    h_offsets[i] = running;
    if (!row_is_null(i)) { running += m; }
  }
  h_offsets[num_rows] = running;

  // Allocate child data: replicated blob for each non-null row
  size_type const total_bytes = running;
  rmm::device_buffer child_data(total_bytes, stream, mr);
  if (total_bytes > 0) {
    auto* dst = static_cast<uint8_t*>(child_data.data());
    // Copy host blob to device once, then tile it for each non-null row
    rmm::device_uvector<uint8_t> d_blob(
      blob.size(), stream, cudf::get_current_device_resource_ref());
    cudf::detail::cuda_memcpy_async(device_span<uint8_t>{d_blob.data(), d_blob.size()},
                                    host_span<uint8_t const>{blob.data(), blob.size()},
                                    stream);

    // Tile: for each non-null row, copy m bytes
    size_type write_pos = 0;
    for (size_type i = 0; i < num_rows; ++i) {
      if (!row_is_null(i)) {
        CUDF_CUDA_TRY(cudf::detail::memcpy_async(dst + write_pos, d_blob.data(), m, stream));
        write_pos += m;
      }
    }
  }

  auto d_offsets   = cudf::detail::make_device_uvector_async(h_offsets, stream, mr);
  auto offsets_col = std::make_unique<column>(data_type{type_id::INT32},
                                              static_cast<size_type>(h_offsets.size()),
                                              d_offsets.release(),
                                              rmm::device_buffer{},
                                              0);

  auto child_col = std::make_unique<column>(
    data_type{type_id::UINT8}, total_bytes, std::move(child_data), rmm::device_buffer{}, 0);

  // Null mask for the list column comes from the input
  rmm::device_buffer list_null_mask{};
  if (null_count > 0 && input_null_mask != nullptr) {
    list_null_mask = cudf::detail::copy_bitmask(input_null_mask, 0, num_rows, stream, mr);
  }

  return make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), null_count, std::move(list_null_mask));
}

}  // namespace

namespace detail {

std::unique_ptr<column> encode_strings_to_variant(cudf::strings_column_view const& input,
                                                  cudf::host_span<std::string const> column_names,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  size_type const num_rows   = input.size();
  size_type const num_fields = static_cast<size_type>(column_names.size());

  CUDF_EXPECTS(num_fields <= 255,
               "encode_strings_to_variant supports at most 255 fields",
               std::invalid_argument);

  // Empty output
  if (num_rows == 0) {
    auto empty_meta = cudf::make_lists_column(
      0, make_empty_column(type_id::INT32), make_empty_column(type_id::UINT8), 0, {});
    auto empty_val = cudf::make_lists_column(
      0, make_empty_column(type_id::INT32), make_empty_column(type_id::UINT8), 0, {});
    std::vector<std::unique_ptr<column>> empty_children;
    empty_children.push_back(std::move(empty_meta));
    empty_children.push_back(std::move(empty_val));
    return cudf::make_structs_column(0, std::move(empty_children), 0, {}, stream, mr);
  }

  // ── Sort field names ──────────────────────────────────────────────────────
  std::vector<size_t> sort_indices(num_fields);
  std::iota(sort_indices.begin(), sort_indices.end(), size_t{0});
  std::sort(sort_indices.begin(), sort_indices.end(), [&](size_t a, size_t b) {
    return column_names[a] < column_names[b];
  });
  // sorted_to_original[i] = original column index of the i-th sorted key
  std::vector<int32_t> h_sorted_to_original(num_fields);
  std::vector<std::string> sorted_names(num_fields);
  for (size_type i = 0; i < num_fields; ++i) {
    h_sorted_to_original[i] = static_cast<int32_t>(sort_indices[i]);
    sorted_names[i]         = std::string(column_names[sort_indices[i]]);
  }

  auto d_sorted_to_original = cudf::detail::make_device_uvector(
    h_sorted_to_original, stream, cudf::get_current_device_resource_ref());

  // ── Extract field values with get_json_object ─────────────────────────────
  cudf::get_json_object_options opts;
  opts.set_strip_quotes_from_single_strings(false);
  opts.set_missing_fields_as_nulls(true);

  std::vector<std::unique_ptr<column>> extracted_cols;
  extracted_cols.reserve(num_fields);
  for (size_type i = 0; i < num_fields; ++i) {
    std::string path = "$." + std::string(column_names[i]);
    cudf::string_scalar path_scalar(path, true, stream, cudf::get_current_device_resource_ref());
    extracted_cols.push_back(cudf::get_json_object(
      input, path_scalar, opts, stream, cudf::get_current_device_resource_ref()));
  }

  // ── Build device array of column_device_views ─────────────────────────────
  // column_device_view::create returns unique_ptr with a custom deleter; collect them to extend
  // lifetime, then copy the views themselves (which hold device pointers) to device.
  using cdv_ptr = std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>;
  std::vector<cdv_ptr> dv_holders;
  dv_holders.reserve(num_fields);
  std::vector<column_device_view> h_views;
  h_views.reserve(num_fields);
  for (auto const& col : extracted_cols) {
    dv_holders.push_back(column_device_view::create(col->view(), stream));
    h_views.push_back(*dv_holders.back());
  }
  auto d_views =
    cudf::detail::make_device_uvector(h_views, stream, cudf::get_current_device_resource_ref());

  // ── Input null mask ───────────────────────────────────────────────────────
  bitmask_type const* input_null_mask = input.null_mask();
  size_type const null_count          = input.null_count();

  // ── Compute per-row value blob sizes ─────────────────────────────────────
  rmm::device_uvector<size_type> value_sizes(
    num_rows, stream, cudf::get_current_device_resource_ref());
  {
    auto grid = cudf::detail::grid_1d{num_rows, block_size_encode};
    compute_value_sizes_kernel<<<grid.num_blocks, block_size_encode, 0, stream.value()>>>(
      d_views, d_sorted_to_original, num_rows, num_fields, input_null_mask, value_sizes);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // ── Prefix-sum to get per-row value offsets ───────────────────────────────
  rmm::device_uvector<size_type> value_offsets(num_rows + 1, stream, mr);
  {
    auto const zero = size_type{0};
    cudf::detail::cuda_memcpy_async(device_span<size_type>{value_offsets.data(), 1},
                                    host_span<size_type const>{&zero, 1},
                                    stream);
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           value_sizes.begin(),
                           value_sizes.end(),
                           value_offsets.begin() + 1);
  }

  // ── Allocate and write value blobs ────────────────────────────────────────
  size_type total_value_bytes{};
  cudf::detail::cuda_memcpy(host_span<size_type>{&total_value_bytes, 1},
                            device_span<size_type const>{value_offsets.data() + num_rows, 1},
                            stream);

  auto value_child_data = rmm::device_buffer(static_cast<size_t>(total_value_bytes), stream, mr);
  if (total_value_bytes > 0) {
    auto grid = cudf::detail::grid_1d{num_rows, block_size_encode};
    write_values_kernel<<<grid.num_blocks, block_size_encode, 0, stream.value()>>>(
      d_views,
      d_sorted_to_original,
      num_rows,
      num_fields,
      input_null_mask,
      device_span<size_type const>{value_offsets.data(), static_cast<size_t>(num_rows + 1)},
      static_cast<uint8_t*>(value_child_data.data()));
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // Build value list<uint8> column
  auto value_offsets_col = std::make_unique<column>(
    data_type{type_id::INT32}, num_rows + 1, value_offsets.release(), rmm::device_buffer{}, 0);
  auto value_child_col = std::make_unique<column>(data_type{type_id::UINT8},
                                                  total_value_bytes,
                                                  std::move(value_child_data),
                                                  rmm::device_buffer{},
                                                  0);

  rmm::device_buffer value_null_mask{};
  if (null_count > 0 && input_null_mask != nullptr) {
    value_null_mask = cudf::detail::copy_bitmask(input_null_mask, 0, num_rows, stream, mr);
  }
  auto value_col = make_lists_column(num_rows,
                                     std::move(value_offsets_col),
                                     std::move(value_child_col),
                                     null_count,
                                     std::move(value_null_mask));

  // ── Build metadata list<uint8> column ────────────────────────────────────
  auto metadata_blob = build_metadata_blob(sorted_names);
  auto metadata_col =
    make_constant_metadata_column(metadata_blob, num_rows, input_null_mask, null_count, stream, mr);

  // ── Assemble struct<metadata, value> ─────────────────────────────────────
  rmm::device_buffer struct_null_mask{};
  if (null_count > 0 && input_null_mask != nullptr) {
    struct_null_mask = cudf::detail::copy_bitmask(input_null_mask, 0, num_rows, stream, mr);
  }
  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(metadata_col));
  children.push_back(std::move(value_col));
  return make_structs_column(
    num_rows, std::move(children), null_count, std::move(struct_null_mask), stream, mr);
}

}  // namespace detail

std::unique_ptr<column> encode_strings_to_variant(cudf::strings_column_view const& input,
                                                  cudf::host_span<std::string const> column_names,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::encode_strings_to_variant(input, column_names, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
