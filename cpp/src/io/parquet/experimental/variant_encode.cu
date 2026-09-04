/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/json/json.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudf {
namespace io::parquet::experimental {
namespace {

using basic_type     = variant_basic_type;
using primitive_type = variant_primitive_type;

// Block size for encoding, one thread per row
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

CUDF_HOST_DEVICE uint8_t* write_le(uint8_t* out, uint64_t val, int bytes)
{
  for (int i = 0; i < bytes; ++i) {
    *out++ = static_cast<uint8_t>(val & 0xFFu);
    val >>= 8u;
  }
  return out;
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
      if (exp < 1000) { exp = exp * 10 + (data[i] - '0'); }
      ++i;
    }
    // Anything beyond the double range saturates.
    if (exp > 400) {
      return (exp_neg ? 0.0 : cuda::std::numeric_limits<double>::infinity()) *
             (negative ? -1.0 : 1.0);
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

// Validates `s` against the strict JSON number grammar (no leading zeros, digit required
// before/after '.', digit required after 'e'/'E', etc). Anything that fails this check is not
// a legal JSON number and must not be silently coerced into 0.0 by parse_float64.
__device__ bool is_valid_json_number(cudf::string_view s)
{
  auto const* data = s.data();
  auto const n     = s.size_bytes();
  if (n == 0) { return false; }

  size_type i = 0;
  if (data[i] == '-') {
    ++i;
    if (i >= n) { return false; }
  }
  if (data[i] == '0') {
    ++i;
  } else if (data[i] >= '1' && data[i] <= '9') {
    ++i;
    while (i < n && data[i] >= '0' && data[i] <= '9') {
      ++i;
    }
  } else {
    return false;
  }
  if (i < n && data[i] == '.') {
    ++i;
    if (i >= n || data[i] < '0' || data[i] > '9') { return false; }
    while (i < n && data[i] >= '0' && data[i] <= '9') {
      ++i;
    }
  }
  if (i < n && (data[i] == 'e' || data[i] == 'E')) {
    ++i;
    if (i < n && (data[i] == '+' || data[i] == '-')) { ++i; }
    if (i >= n || data[i] < '0' || data[i] > '9') { return false; }
    while (i < n && data[i] >= '0' && data[i] <= '9') {
      ++i;
    }
  }
  return i == n;
}

// Classification of a get_json_object result string, used to drive both size computation and
// encoding so the two stay in lock-step.
enum class scalar_kind : uint8_t { NUL, BOOL_TRUE, BOOL_FALSE, STRING, INT, FLOAT, INVALID };

// `raw` is the string returned by get_json_object with strip_quotes_from_single_strings=false.
// Anything that is not one of JSON null/true/false/string/number (in particular a nested
// object/array, or a malformed literal) is reported as INVALID rather than guessed at.
__device__ scalar_kind classify_scalar(cudf::string_view raw)
{
  if (raw.size_bytes() == 0) { return scalar_kind::INVALID; }
  if (raw == cudf::string_view("null", 4)) { return scalar_kind::NUL; }
  if (raw == cudf::string_view("true", 4)) { return scalar_kind::BOOL_TRUE; }
  if (raw == cudf::string_view("false", 5)) { return scalar_kind::BOOL_FALSE; }
  if (raw.data()[0] == '"') {
    if (raw.size_bytes() >= 2 && raw.data()[raw.size_bytes() - 1] == '"') {
      return scalar_kind::STRING;
    }
    return scalar_kind::INVALID;
  }
  if (!is_valid_json_number(raw)) { return scalar_kind::INVALID; }
  return is_float_number(raw) ? scalar_kind::FLOAT : scalar_kind::INT;
}

__device__ int hex_digit(char c)
{
  if (c >= '0' && c <= '9') { return c - '0'; }
  if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
  if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
  return -1;
}

// Encodes `cp` as UTF-8 into `out` (which must have room for up to 4 bytes) and returns the
// number of bytes written.
__device__ int encode_utf8(uint32_t cp, uint8_t* out)
{
  if (cp <= 0x7Fu) {
    out[0] = static_cast<uint8_t>(cp);
    return 1;
  }
  if (cp <= 0x7FFu) {
    out[0] = static_cast<uint8_t>(0xC0u | (cp >> 6u));
    out[1] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
    return 2;
  }
  if (cp <= 0xFFFFu) {
    out[0] = static_cast<uint8_t>(0xE0u | (cp >> 12u));
    out[1] = static_cast<uint8_t>(0x80u | ((cp >> 6u) & 0x3Fu));
    out[2] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
    return 3;
  }
  out[0] = static_cast<uint8_t>(0xF0u | (cp >> 18u));
  out[1] = static_cast<uint8_t>(0x80u | ((cp >> 12u) & 0x3Fu));
  out[2] = static_cast<uint8_t>(0x80u | ((cp >> 6u) & 0x3Fu));
  out[3] = static_cast<uint8_t>(0x80u | (cp & 0x3Fu));
  return 4;
}

// Computes the number of UTF-8 bytes that JSON-unescaping `[s, s+len)` (the content between the
// quotes of a JSON string) would produce, without writing anything. Returns nullopt if the
// content contains an invalid or truncated escape sequence.
__device__ cuda::std::optional<size_type> json_string_unescaped_size(char const* s, size_type len)
{
  size_type count = 0;
  size_type i     = 0;
  while (i < len) {
    char c = s[i];
    if (c != '\\') {
      ++count;
      ++i;
      continue;
    }
    ++i;
    if (i >= len) { return cuda::std::nullopt; }
    char e = s[i];
    switch (e) {
      case '"':
      case '\\':
      case '/':
      case 'b':
      case 'f':
      case 'n':
      case 'r':
      case 't':
        ++count;
        ++i;
        break;
      case 'u': {
        ++i;
        if (i + 4 > len) { return cuda::std::nullopt; }
        int cp = 0;
        for (int k = 0; k < 4; ++k) {
          int d = hex_digit(s[i + k]);
          if (d < 0) { return cuda::std::nullopt; }
          cp = (cp << 4) | d;
        }
        i += 4;
        auto codepoint = static_cast<uint32_t>(cp);
        if (codepoint >= 0xD800u && codepoint <= 0xDBFFu) {
          if (i + 6 > len || s[i] != '\\' || s[i + 1] != 'u') { return cuda::std::nullopt; }
          int lo = 0;
          for (int k = 0; k < 4; ++k) {
            int d = hex_digit(s[i + 2 + k]);
            if (d < 0) { return cuda::std::nullopt; }
            lo = (lo << 4) | d;
          }
          if (lo < 0xDC00 || lo > 0xDFFF) { return cuda::std::nullopt; }
          i += 6;
          codepoint =
            0x10000u + ((codepoint - 0xD800u) << 10u) + (static_cast<uint32_t>(lo) - 0xDC00u);
        } else if (codepoint >= 0xDC00u && codepoint <= 0xDFFFu) {
          return cuda::std::nullopt;  // unpaired low surrogate
        }
        uint8_t buf[4];
        count += encode_utf8(codepoint, buf);
        break;
      }
      default: return cuda::std::nullopt;
    }
  }
  return count;
}

// JSON-unescapes `[s, s+len)` into `out`. Assumes the content was already validated by
// json_string_unescaped_size (same grammar); returns the pointer past the last byte written.
__device__ uint8_t* write_json_string_unescaped(char const* s, size_type len, uint8_t* out)
{
  size_type i = 0;
  while (i < len) {
    char c = s[i];
    if (c != '\\') {
      *out++ = static_cast<uint8_t>(c);
      ++i;
      continue;
    }
    ++i;
    char e = s[i];
    switch (e) {
      case '"':
        *out++ = static_cast<uint8_t>('"');
        ++i;
        break;
      case '\\':
        *out++ = static_cast<uint8_t>('\\');
        ++i;
        break;
      case '/':
        *out++ = static_cast<uint8_t>('/');
        ++i;
        break;
      case 'b':
        *out++ = static_cast<uint8_t>('\b');
        ++i;
        break;
      case 'f':
        *out++ = static_cast<uint8_t>('\f');
        ++i;
        break;
      case 'n':
        *out++ = static_cast<uint8_t>('\n');
        ++i;
        break;
      case 'r':
        *out++ = static_cast<uint8_t>('\r');
        ++i;
        break;
      case 't':
        *out++ = static_cast<uint8_t>('\t');
        ++i;
        break;
      case 'u': {
        ++i;
        int cp = 0;
        for (int k = 0; k < 4; ++k) {
          cp = (cp << 4) | hex_digit(s[i + k]);
        }
        i += 4;
        auto codepoint = static_cast<uint32_t>(cp);
        if (codepoint >= 0xD800u && codepoint <= 0xDBFFu) {
          int lo = 0;
          for (int k = 0; k < 4; ++k) {
            lo = (lo << 4) | hex_digit(s[i + 2 + k]);
          }
          i += 6;
          codepoint =
            0x10000u + ((codepoint - 0xD800u) << 10u) + (static_cast<uint32_t>(lo) - 0xDC00u);
        }
        out += encode_utf8(codepoint, out);
        break;
      }
      default: break;
    }
  }
  return out;
}

// Size in bytes of the VARIANT encoding for one JSON scalar value string. If `error_flag` is
// non-null, sets *error_flag (via atomicExch) and returns a harmless placeholder size when `raw`
// is not a legal scalar (nested object/array or malformed literal) or contains an invalid string
// escape; callers passing a non-null flag must check it after the kernel completes and fail the
// whole encode rather than trust the size. Callers that have already validated every row via a
// prior checked pass (see write_values_kernel) may pass nullptr, since INVALID cannot recur for
// the same input.
__device__ size_type encoded_field_size(cudf::string_view raw, int32_t* error_flag = nullptr)
{
  switch (classify_scalar(raw)) {
    case scalar_kind::NUL:
    case scalar_kind::BOOL_TRUE:
    case scalar_kind::BOOL_FALSE: return 1;
    case scalar_kind::STRING: {
      size_type str_len        = static_cast<size_type>(raw.size_bytes()) - 2;
      auto const unescaped_len = json_string_unescaped_size(raw.data() + 1, str_len);
      if (!unescaped_len.has_value()) {
        if (error_flag != nullptr) { atomicExch(error_flag, 1); }
        return 1;
      }
      if (*unescaped_len <= 63) { return 1 + *unescaped_len; }  // SHORT_STRING
      return 1 + 4 + *unescaped_len;  // LONG_STRING (1 hdr + 4-byte len + bytes)
    }
    case scalar_kind::FLOAT:
    case scalar_kind::INT: return 9;  // header + 8-byte double/int64
    case scalar_kind::INVALID:
    default:
      if (error_flag != nullptr) { atomicExch(error_flag, 1); }
      return 1;
  }
}

// Write VARIANT bytes for one JSON scalar value string into `out`.
// Returns pointer past the last written byte. Assumes `raw` was already validated by a prior
// call to encoded_field_size that did not set the error flag.
__device__ uint8_t* write_field_value(uint8_t* out, cudf::string_view raw)
{
  auto make_prim_header = [](primitive_type pt) -> uint8_t {
    return static_cast<uint8_t>(basic_type::PRIMITIVE) | (static_cast<uint8_t>(pt) << 2u);
  };

  switch (classify_scalar(raw)) {
    case scalar_kind::NUL: {
      *out++ = make_prim_header(primitive_type::NULLVAL);
      return out;
    }
    case scalar_kind::BOOL_TRUE: {
      *out++ = make_prim_header(primitive_type::BOOLEAN_TRUE);
      return out;
    }
    case scalar_kind::BOOL_FALSE: {
      *out++ = make_prim_header(primitive_type::BOOLEAN_FALSE);
      return out;
    }
    case scalar_kind::STRING: {
      auto const* str_start    = raw.data() + 1;
      size_type str_len        = static_cast<size_type>(raw.size_bytes()) - 2;
      auto const unescaped_len = json_string_unescaped_size(str_start, str_len);
      if (!unescaped_len.has_value()) {
        // Should be unreachable: already validated by encoded_field_size.
        *out++ = make_prim_header(primitive_type::NULLVAL);
        return out;
      }
      auto const n = *unescaped_len;
      if (n <= 63) {
        *out++ = static_cast<uint8_t>(basic_type::SHORT_STRING) | (static_cast<uint8_t>(n) << 2u);
        return write_json_string_unescaped(str_start, str_len, out);
      }
      // LONG_STRING
      *out++         = make_prim_header(primitive_type::LONG_STRING);
      uint32_t len32 = static_cast<uint32_t>(n);
      cuda::std::memcpy(out, &len32, 4);
      out += 4;
      return write_json_string_unescaped(str_start, str_len, out);
    }
    case scalar_kind::FLOAT: {
      *out++     = make_prim_header(primitive_type::FLOAT64);
      double val = parse_float64(raw);
      cuda::std::memcpy(out, &val, sizeof(double));
      return out + sizeof(double);
    }
    case scalar_kind::INT: {
      auto parsed = try_parse_int64(raw);
      if (parsed.has_value()) {
        *out++       = make_prim_header(primitive_type::INT64);
        int64_t ival = *parsed;
        cuda::std::memcpy(out, &ival, sizeof(int64_t));
        return out + sizeof(int64_t);
      }
      // Failed to parse as INT64 (out-of-range): fall back to FLOAT64.
      *out++     = make_prim_header(primitive_type::FLOAT64);
      double val = parse_float64(raw);
      cuda::std::memcpy(out, &val, sizeof(double));
      return out + sizeof(double);
    }
    case scalar_kind::INVALID:
    default: {
      // Should be unreachable: already validated by encoded_field_size.
      *out++ = make_prim_header(primitive_type::NULLVAL);
      return out;
    }
  }
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
  device_span<int64_t> value_sizes,
  int32_t* error_flag)
{
  auto const tid    = cudf::detail::grid_1d::global_thread_id<block_size_encode>();
  auto const stride = cudf::detail::grid_1d::grid_stride<block_size_encode>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (input_null_mask != nullptr && !cudf::bit_is_set(input_null_mask, row)) {
      value_sizes[row] = 0;
      continue;
    }

    size_type n_present  = 0;
    int64_t values_bytes = 0;

    for (size_type si = 0; si < num_fields; ++si) {
      auto const orig = sorted_to_original[si];
      if (extracted[orig].is_null(row)) { continue; }
      ++n_present;
      values_bytes +=
        encoded_field_size(extracted[orig].element<cudf::string_view>(row), error_flag);
    }

    // 1 (value_metadata) + NUM_ELEMENTS_SIZE + n_present*FIELD_ID_SIZE
    // + (n_present+1)*FIELD_OFFSET_SIZE + values_bytes
    // Accumulated as int64_t: a row's total blob size (dominated by values_bytes, which sums
    // per-field encoded sizes that can each be up to ~4 GB) can exceed the int32 size_type range.
    value_sizes[row] = int64_t{1} + NUM_ELEMENTS_SIZE + n_present * FIELD_ID_SIZE +
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
    out = write_le(out, static_cast<uint64_t>(n_present), NUM_ELEMENTS_SIZE);

    // field_ids (sorted dict index for each present field)
    for (size_type si = 0; si < num_fields; ++si) {
      auto const orig = sorted_to_original[si];
      if (extracted[orig].is_null(row)) { continue; }
      out = write_le(out, static_cast<uint64_t>(si), FIELD_ID_SIZE);
    }

    // field_offsets: cumulative offsets within the values region + sentinel
    size_type cur_offset = 0;
    for (size_type si = 0; si < num_fields; ++si) {
      auto const orig = sorted_to_original[si];
      if (extracted[orig].is_null(row)) { continue; }
      out = write_le(out, static_cast<uint64_t>(cur_offset), FIELD_OFFSET_SIZE);
      cur_offset += encoded_field_size(extracted[orig].element<cudf::string_view>(row));
    }
    out = write_le(out, static_cast<uint64_t>(cur_offset), FIELD_OFFSET_SIZE);  // sentinel

    // field values in sorted order
    for (size_type si = 0; si < num_fields; ++si) {
      auto const orig = sorted_to_original[si];
      if (extracted[orig].is_null(row)) { continue; }
      out = write_field_value(out, extracted[orig].element<cudf::string_view>(row));
    }
  }
}

// ─── host helpers ─────────────────────────────────────────────────────────────

/**
 * @brief Build the fixed VARIANT metadata blob for a sorted list of key names.
 *
 * The returned pinned host buffer has layout:
 * header | dictionary size | offsets | key bytes.
 *
 * @param sorted_names Field names in lexicographic order
 * @param stream Stream used to allocate pinned host memory
 * @return Pinned host buffer containing the encoded metadata
 */
cudf::detail::host_vector<uint8_t> build_metadata_blob(std::span<std::string const> sorted_names,
                                                       cuda::stream_ref stream)
{
  auto const num_names       = sorted_names.size();
  auto const total_key_bytes = std::accumulate(
    sorted_names.begin(), sorted_names.end(), size_t{0}, [](size_t total, auto const& name) {
      return total + name.size();
    });

  int offset_size = 1;
  if (total_key_bytes > 255 || num_names > 255) { offset_size = 2; }
  if (total_key_bytes > 65535 || num_names > 65535) { offset_size = 4; }

  auto const blob_size = size_t{1} + offset_size + (num_names + 1) * offset_size + total_key_bytes;
  auto blob            = cudf::detail::make_pinned_vector<uint8_t>(blob_size, stream);
  auto out             = blob.data();

  // header: version=1 | sorted=1 | unused=0 | offset_size-1
  *out++ = static_cast<uint8_t>(0x01u | (1u << 4u) | (uint8_t(offset_size - 1) << 6u));
  out    = write_le(out, num_names, offset_size);

  size_t current_offset = 0;
  for (auto const& name : sorted_names) {
    out = write_le(out, current_offset, offset_size);
    current_offset += name.size();
  }
  out = write_le(out, current_offset, offset_size);

  for (auto const& name : sorted_names) {
    out = std::copy(name.begin(), name.end(), out);
  }
  return blob;
}

/**
 * @brief Build a list<uint8> column where every valid row contains the same metadata blob.
 *
 * Null rows have zero-length list entries.
 *
 * @param blob Metadata bytes in pinned host memory
 * @param num_rows Number of output rows
 * @param input_null_mask Offset-free input validity mask, or an empty span
 * @param null_count Number of null input rows
 * @param stream CUDA stream
 * @param mr Device memory resources
 * @return LIST column containing one metadata blob per valid row
 */
std::unique_ptr<column> make_constant_metadata_column(
  std::span<uint8_t const> blob,
  size_type num_rows,
  device_span<bitmask_type const> input_null_mask,
  size_type null_count,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  CUDF_EXPECTS(std::cmp_less_equal(blob.size(), std::numeric_limits<size_type>::max()),
               "encode_strings_to_variant metadata blob exceeds the LIST child size limit",
               std::overflow_error);
  CUDF_EXPECTS(std::cmp_less_equal(blob.size() * (num_rows - null_count),
                                   std::numeric_limits<size_type>::max()),
               "encode_strings_to_variant metadata column exceeds the LIST child size limit",
               std::overflow_error);
  auto const output_mr = mr.get_output_mr();
  auto const temp_mr   = mr.get_temporary_mr();
  auto const blob_size = static_cast<size_type>(blob.size());

  rmm::device_uvector<size_type> offsets(num_rows + 1, stream, output_mr);
  auto row_sizes = cudf::detail::make_counting_transform_iterator(
    size_type{0},
    cuda::proclaim_return_type<size_type>(
      [input_null_mask, blob_size, num_rows] __device__(size_type row) -> size_type {
    if (row == num_rows) { return 0; }
    return input_null_mask.empty() or cudf::bit_is_set(input_null_mask.data(), row) ? blob_size : 0;
      }));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         row_sizes,
                         row_sizes + num_rows + 1,
                         offsets.begin(),
                         size_type{0});

  auto total_bytes_host = cudf::detail::make_pinned_vector(
    device_span<size_type const>{offsets.data() + num_rows, 1}, stream);
  auto const total_bytes = total_bytes_host.front();

  rmm::device_buffer child_data(total_bytes, stream, output_mr);
  if (total_bytes > 0) {
    // This helper is private and is called with build_metadata_blob's pinned result.
    auto d_blob = cudf::detail::make_device_uvector_async(
      host_span<uint8_t const>{blob.data(), blob.size(), true}, stream, temp_mr);
    auto const dst_base = static_cast<uint8_t*>(child_data.data());
    auto src_iter = cuda::make_constant_iterator(static_cast<uint8_t const*>(d_blob.data()));
    auto dst_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t*>(
        [dst_base, offsets = offsets.data()] __device__(size_type row) -> uint8_t* {
          return dst_base + offsets[row];
        }));
    auto size_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<size_t>(
        [offsets = offsets.data()] __device__(size_type row) -> size_t {
          return static_cast<size_t>(offsets[row + 1] - offsets[row]);
        }));
    cudf::detail::batched_memcpy_async(
      src_iter, dst_iter, size_iter, static_cast<size_t>(num_rows), stream);
  }

  auto offsets_col = std::make_unique<column>(
    data_type{type_id::INT32}, num_rows + 1, offsets.release(), rmm::device_buffer{}, 0);
  auto child_col = std::make_unique<column>(
    data_type{type_id::UINT8}, total_bytes, std::move(child_data), rmm::device_buffer{}, 0);

  auto list_null_mask =
    input_null_mask.empty()
      ? rmm::device_buffer{}
      : cudf::detail::copy_bitmask(input_null_mask.data(), 0, num_rows, stream, output_mr);

  return make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), null_count, std::move(list_null_mask));
}

}  // namespace

namespace detail {

std::unique_ptr<column> encode_strings_to_variant(cudf::strings_column_view const& input,
                                                  std::span<std::string const> column_names,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr)
{
  auto const output_mr  = mr.get_output_mr();
  auto const temp_mr    = mr.get_temporary_mr();
  auto const num_rows   = input.size();
  auto const num_fields = static_cast<size_type>(column_names.size());

  CUDF_EXPECTS(num_fields <= 255,
               "encode_strings_to_variant supports at most 255 fields",
               std::invalid_argument);
  CUDF_EXPECTS(
    std::all_of(column_names.begin(),
                column_names.end(),
                [](auto const& name) { return name.find_first_of(".[") == std::string::npos; }),
    "encode_strings_to_variant does not support field names containing '.' or '['",
    std::invalid_argument);

  // Field names remain on the host because they are also used to construct JSON paths and metadata.
  std::vector<size_t> sort_indices(num_fields);
  std::iota(sort_indices.begin(), sort_indices.end(), size_t{0});
  std::sort(sort_indices.begin(), sort_indices.end(), [&](size_t lhs, size_t rhs) {
    return column_names[lhs] < column_names[rhs];
  });
  CUDF_EXPECTS(std::adjacent_find(sort_indices.begin(),
                                  sort_indices.end(),
                                  [&](size_t lhs, size_t rhs) {
                                    return column_names[lhs] == column_names[rhs];
                                  }) == sort_indices.end(),
               "encode_strings_to_variant does not accept duplicate field names",
               std::invalid_argument);

  if (num_rows == 0) {
    std::vector<std::unique_ptr<column>> empty_children;
    empty_children.push_back(cudf::make_empty_lists_column(data_type{type_id::UINT8}));
    empty_children.push_back(cudf::make_empty_lists_column(data_type{type_id::UINT8}));
    return cudf::make_structs_column(
      0, std::move(empty_children), 0, rmm::device_buffer{}, stream, output_mr);
  }

  auto h_sorted_to_original = cudf::detail::make_empty_pinned_vector<int32_t>(num_fields, stream);
  std::transform(sort_indices.begin(),
                 sort_indices.end(),
                 std::back_inserter(h_sorted_to_original),
                 [](size_t index) { return static_cast<int32_t>(index); });
  std::vector<std::string> sorted_names(num_fields);
  std::transform(sort_indices.begin(), sort_indices.end(), sorted_names.begin(), [&](size_t index) {
    return column_names[index];
  });

  auto d_sorted_to_original =
    cudf::detail::make_device_uvector_async(h_sorted_to_original, stream, temp_mr);

  cudf::get_json_object_options opts;
  opts.set_strip_quotes_from_single_strings(false);
  opts.set_missing_fields_as_nulls(true);

  std::vector<std::unique_ptr<column>> extracted_cols;
  extracted_cols.reserve(num_fields);
  for (auto const& column_name : column_names) {
    std::string path = "$." + column_name;
    cudf::string_scalar path_scalar(path, true, stream, temp_mr);
    extracted_cols.push_back(cudf::get_json_object(input, path_scalar, opts, stream, temp_mr));
  }

  // The holders keep each device view's nested allocations alive through both encoding kernels.
  using cdv_ptr = std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>;
  std::vector<cdv_ptr> dv_holders;
  dv_holders.reserve(num_fields);
  auto h_views = cudf::detail::make_empty_pinned_vector<column_device_view>(num_fields, stream);
  for (auto const& col : extracted_cols) {
    dv_holders.push_back(column_device_view::create(col->view(), stream));
    h_views.push_back(*dv_holders.back());
  }
  auto d_views = cudf::detail::make_device_uvector_async(h_views, stream, temp_mr);

  auto const null_count = input.null_count();
  auto owned_null_mask =
    null_count > 0
      ? cudf::detail::copy_bitmask(
          input.null_mask(), input.offset(), input.offset() + num_rows, stream, temp_mr)
      : rmm::device_buffer{};
  auto const input_null_mask =
    null_count > 0
      ? device_span<bitmask_type const>{static_cast<bitmask_type const*>(owned_null_mask.data()),
                                        static_cast<size_t>(cudf::num_bitmask_words(num_rows))}
      : device_span<bitmask_type const>{};

  // Store per-row sizes and then scan them in place, retaining 64-bit overflow detection without a
  // second 64-bit offsets allocation.
  rmm::device_uvector<int64_t> value_sizes(num_rows, stream, temp_mr);
  {
    cudf::detail::device_scalar<int32_t> error_flag(0, stream, temp_mr);
    auto grid = cudf::detail::grid_1d{num_rows, block_size_encode};
    compute_value_sizes_kernel<<<grid.num_blocks, block_size_encode, 0, stream.get()>>>(
      d_views,
      d_sorted_to_original,
      num_rows,
      num_fields,
      input_null_mask.data(),
      value_sizes,
      error_flag.data());
    CUDF_CUDA_TRY(cudaGetLastError());
    CUDF_EXPECTS(error_flag.value(stream) == 0,
                 "encode_strings_to_variant encountered a JSON value that is not a supported "
                 "scalar (nested object/array), or a malformed literal or string escape; "
                 "VARIANT encoding requires scalar, non-nested field values only",
                 std::invalid_argument);
  }

  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         value_sizes.begin(),
                         value_sizes.end(),
                         value_sizes.begin());

  auto total_value_bytes_host = cudf::detail::make_pinned_vector(
    device_span<int64_t const>{value_sizes.data() + num_rows - 1, 1}, stream);
  auto const total_value_bytes_wide = total_value_bytes_host.front();
  CUDF_EXPECTS(total_value_bytes_wide <= std::numeric_limits<size_type>::max(),
               "encode_strings_to_variant output exceeds the VARIANT value list<uint8> column "
               "size limit (a LIST column's offsets are always 32-bit)",
               std::overflow_error);
  auto const total_value_bytes = static_cast<size_type>(total_value_bytes_wide);

  rmm::device_uvector<size_type> value_offsets(num_rows + 1, stream, output_mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(value_offsets.data(), 0, sizeof(size_type), stream.get()));
  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    value_sizes.begin(),
                    value_sizes.end(),
                    value_offsets.begin() + 1,
                    cuda::proclaim_return_type<size_type>(
                      [] __device__(int64_t value) { return static_cast<size_type>(value); }));

  rmm::device_buffer value_child_data(total_value_bytes, stream, output_mr);
  if (total_value_bytes > 0) {
    auto grid = cudf::detail::grid_1d{num_rows, block_size_encode};
    write_values_kernel<<<grid.num_blocks, block_size_encode, 0, stream.get()>>>(
      d_views,
      d_sorted_to_original,
      num_rows,
      num_fields,
      input_null_mask.data(),
      value_offsets,
      static_cast<uint8_t*>(value_child_data.data()));
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto value_offsets_col = std::make_unique<column>(
    data_type{type_id::INT32}, num_rows + 1, value_offsets.release(), rmm::device_buffer{}, 0);
  auto value_child_col = std::make_unique<column>(data_type{type_id::UINT8},
                                                  total_value_bytes,
                                                  std::move(value_child_data),
                                                  rmm::device_buffer{},
                                                  0);
  auto value_null_mask =
    input_null_mask.empty()
      ? rmm::device_buffer{}
      : cudf::detail::copy_bitmask(input_null_mask.data(), 0, num_rows, stream, output_mr);
  auto value_col = make_lists_column(num_rows,
                                     std::move(value_offsets_col),
                                     std::move(value_child_col),
                                     null_count,
                                     std::move(value_null_mask));

  auto metadata_blob = build_metadata_blob(sorted_names, stream);
  auto metadata_col =
    make_constant_metadata_column(metadata_blob, num_rows, input_null_mask, null_count, stream, mr);

  auto struct_null_mask =
    input_null_mask.empty()
      ? rmm::device_buffer{}
      : cudf::detail::copy_bitmask(input_null_mask.data(), 0, num_rows, stream, output_mr);
  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(metadata_col));
  children.push_back(std::move(value_col));
  return make_structs_column(
    num_rows, std::move(children), null_count, std::move(struct_null_mask), stream, output_mr);
}

}  // namespace detail

std::unique_ptr<column> encode_strings_to_variant(cudf::strings_column_view const& input,
                                                  std::span<std::string const> column_names,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::encode_strings_to_variant(input, column_names, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
