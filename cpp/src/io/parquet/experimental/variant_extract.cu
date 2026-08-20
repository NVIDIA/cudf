/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/block_utils.cuh"
#include "variant_path.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/numeric>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace cudf {
namespace io::parquet::experimental {
namespace {

constexpr int variant_version_v1 = 1;

// Bytes consumed by the leading metadata byte common to every Variant value.
constexpr size_type variant_header_bytes = 1;

// Low 2 bits of a value's metadata byte: the basic type.
using basic_type = variant_basic_type;

// For a primitive value, the value_header is the physical type id of the payload.
using primitive_type = variant_primitive_type;

__device__ cuda::std::optional<uint64_t> read_uint64(device_span<uint8_t const> data,
                                                     size_type pos,
                                                     int width)
{
  if (cuda::std::cmp_greater(pos + width, data.size())) { return cuda::std::nullopt; }
  uint64_t v = 0;
  cuda::std::memcpy(&v, data.data() + pos, width);
  return v;
}

// Safely narrow a decoded value to size_type
__device__ cuda::std::optional<size_type> narrow_cast(cuda::std::optional<uint64_t> value)
{
  if (!value.has_value() ||
      cuda::std::cmp_greater(value.value(), cuda::std::numeric_limits<size_type>::max())) {
    return cuda::std::nullopt;
  }
  return static_cast<size_type>(value.value());
}

__device__ basic_type decode_basic_type(uint8_t value_metadata)
{
  return static_cast<basic_type>(value_metadata & 0x03);
}

__device__ uint8_t variant_value_header(uint8_t value_metadata)
{
  return (value_metadata >> 2) & 0x3F;
}

struct object_array_header {
  int field_offset_size;  // bytes per field_offset entry
  int field_id_size;      // bytes per field_id (0 for arrays)
  int num_elements_size;  // bytes holding num_elements
};

/**
 * @brief Decode the size fields packed into an object/array value header.
 *
 * For object and array values, the 6-bit value header (bits 2..7 of the value metadata byte)
 * encodes the widths used by the rest of the value. The layout differs between the two:
 *
 *   object value_header bits:  | is_large (1) | field_id_size-1 (2) | field_offset_size-1 (2) |
 *   array  value_header bits:  |          is_large (1)             | field_offset_size-1 (2) |
 *
 * where each `*_size-1` field stores (width in bytes - 1), so the decoded width is the field + 1
 * (1..4 bytes), and `is_large` selects the width of the `num_elements` field: 4 bytes if set,
 * else 1 byte. Arrays have no field ids, so `field_id_size` is 0.
 *
 * @param value_header The 6-bit value header (see variant_value_header)
 * @param is_object True for object values, false for array values
 * @return The decoded byte widths
 */
__device__ object_array_header decode_object_array_header(uint8_t value_header, bool is_object)
{
  auto const large_bit = is_object ? 4 : 2;
  bool const is_large  = (value_header >> large_bit) & 0x01;

  return {.field_offset_size = (value_header & 0x03) + 1,
          .field_id_size     = is_object ? ((value_header >> 2) & 0x03) + 1 : 0,
          .num_elements_size = is_large ? 4 : 1};
}

/**
 * @brief Compute the total encoded byte length of a single VARIANT value.
 *
 * Every value starts with a 1-byte value metadata header (`basic_type` in bits 0..1, `value_header`
 * in bits 2..7); the bytes that follow depend on the basic type:
 *
 *   - primitive (0): header + a fixed payload keyed by the primitive type id. Binary/long_string
 *     carry a 4-byte little-endian length prefix followed by that many payload bytes.
 *   - short_string (1): header + `value_header` payload bytes (the header is the string length).
 *   - object/array (2/3): header + num_elements + field-id list + field-offset list + values; the
 *     total values-region size is read from the trailing field_offset (the "sentinel" at index
 *     num_elements). See decode_object_array_header / locate_object_field for the sub-layout.
 *
 * @param enc The encoded value bytes (must begin at the value metadata byte)
 * @return The total length in bytes of the value, or nullopt if `enc` is empty/malformed or the
 *         type id is unrecognized
 */
__device__ cuda::std::optional<uint64_t> variant_value_length(device_span<uint8_t const> enc)
{
  if (enc.size() < 1) { return cuda::std::nullopt; }
  auto const value_metadata = enc[0];
  auto const btype          = decode_basic_type(value_metadata);
  auto const value_header   = variant_value_header(value_metadata);

  if (btype == basic_type::PRIMITIVE) {
    // The leading header byte plus a payload keyed by the physical type id
    uint64_t payload = 0;
    switch (static_cast<primitive_type>(value_header)) {
      case primitive_type::NULLVAL:
      case primitive_type::BOOLEAN_TRUE:
      case primitive_type::BOOLEAN_FALSE: break;  // no payload
      case primitive_type::INT8: payload = 1; break;
      case primitive_type::INT16: payload = 2; break;
      case primitive_type::INT32:
      case primitive_type::DATE:
      case primitive_type::FLOAT32: payload = 4; break;
      case primitive_type::INT64:
      case primitive_type::FLOAT64:
      case primitive_type::TIMESTAMP_MICROS:
      case primitive_type::TIMESTAMP_NTZ_MICROS:
      case primitive_type::TIME_NTZ_MICROS:
      case primitive_type::TIMESTAMP_NANOS:
      case primitive_type::TIMESTAMP_NTZ_NANOS: payload = 8; break;
      case primitive_type::DECIMAL4: payload = 1 + 4; break;    // scale + int32
      case primitive_type::DECIMAL8: payload = 1 + 8; break;    // scale + int64
      case primitive_type::DECIMAL16: payload = 1 + 16; break;  // scale + int128
      case primitive_type::UUID: payload = 16; break;
      case primitive_type::BINARY:
      case primitive_type::LONG_STRING: {
        constexpr int length_prefix_bytes = 4;
        auto const len = read_uint64(enc, variant_header_bytes, length_prefix_bytes);
        if (!len.has_value()) { return cuda::std::nullopt; }
        payload = length_prefix_bytes + len.value();
        break;
      }
      default: return cuda::std::nullopt;
    }
    return variant_header_bytes + payload;
  }

  if (btype == basic_type::SHORT_STRING) {
    // The value header is the payload length, following the header byte.
    return variant_header_bytes + static_cast<uint64_t>(value_header);
  }

  // Object / array: the encoded size is the header bytes (metadata byte, element count, optional
  // field-id list, and offset list)
  bool const is_object = btype == basic_type::OBJECT;
  auto const [offset_size, id_size, num_elements_size] =
    decode_object_array_header(value_header, is_object);

  auto const num_elements = read_uint64(enc, variant_header_bytes, num_elements_size);
  if (!num_elements.has_value()) { return cuda::std::nullopt; }
  auto const n = num_elements.value();

  auto const offsets_start = variant_header_bytes + num_elements_size + n * id_size;
  auto const values_base   = offsets_start + (n + 1) * offset_size;
  // Sentinel offset (entry n) holds the total size of the values region.
  auto const sentinel_pos = narrow_cast(offsets_start + n * offset_size);
  if (!sentinel_pos.has_value()) { return cuda::std::nullopt; }
  auto const sentinel = read_uint64(enc, sentinel_pos.value(), offset_size);
  if (!sentinel.has_value()) { return cuda::std::nullopt; }
  return values_base + sentinel.value();
}

/**
 * @brief Find the dictionary index of a key in a VARIANT metadata blob.
 *
 * The metadata blob is the per-row string dictionary, laid out as:
 *
 *   byte 0:          header        | version (4) | sorted (1) | unused (1) | offset_size-1 (2) |
 *   bytes 1..:       dictionary_size   (offset_size bytes, little-endian) = number of keys N
 *   next (N+1)*offset_size bytes:  offsets[0..N]   (offset_size bytes each, little-endian)
 *   remaining bytes: string_data   (concatenated UTF-8 key bytes)
 *
 * `offset_size` (1..4 bytes, from the header) is the width of every dictionary_size/offset entry.
 * Key `i` occupies `string_data[offsets[i] : offsets[i+1]]`; the trailing offset `offsets[N]` is
 * the total length of `string_data`. Offsets are relative to the start of `string_data`, i.e. to `1
 * + offset_size + (N+1)*offset_size`.
 *
 * @param meta The metadata blob bytes for a single row
 * @param key The key to search for
 * @return The dictionary index of `key`, or nullopt if absent or the blob is malformed
 */
__device__ cuda::std::optional<size_type> find_key_in_metadata(device_span<uint8_t const> meta,
                                                               cudf::string_view key)
{
  auto const meta_len = static_cast<size_type>(meta.size());
  if (meta_len < 1) { return cuda::std::nullopt; }

  auto const header = meta[0];
  int const version = header & 0x0F;
  if (version != variant_version_v1) { return cuda::std::nullopt; }
  int const offset_size = ((header >> 6) & 0x03) + 1;

  size_type pos          = 1;
  auto const num_entries = narrow_cast(read_uint64(meta, pos, offset_size));
  if (!num_entries.has_value()) { return cuda::std::nullopt; }
  pos += offset_size;

  auto const offsets_start = pos;
  auto const offsets_bytes = (static_cast<uint64_t>(num_entries.value()) + 1) * offset_size;
  if (cuda::std::cmp_greater(offsets_bytes, meta_len - offsets_start)) {
    return cuda::std::nullopt;
  }

  auto start_off = read_uint64(meta, offsets_start, offset_size);
  if (!start_off.has_value()) { return cuda::std::nullopt; }
  auto const strings_base = offsets_start + static_cast<size_type>(offsets_bytes);
  // Bytes available for dictionary string payloads
  auto const strings_extent = meta_len - strings_base;
  for (size_type i = 0; i < num_entries.value(); ++i) {
    auto const end_off = read_uint64(meta, offsets_start + (i + 1) * offset_size, offset_size);
    if (!end_off.has_value()) { return cuda::std::nullopt; }
    if (end_off.value() < start_off.value() || end_off.value() > strings_extent) {
      return cuda::std::nullopt;
    }
    cudf::string_view const entry{
      reinterpret_cast<char const*>(meta.data() + strings_base + start_off.value()),
      static_cast<size_type>(end_off.value() - start_off.value())};
    if (entry == key) { return i; }
    start_off = end_off;
  }
  return cuda::std::nullopt;
}

/**
 * @brief Locate the encoded bytes of a single field within an object value by field id.
 *
 * An object value is laid out as:
 *
 *   byte 0:        value metadata | basic_type=object (2) | value_header (6) |
 *   bytes 1..:     num_elements   (num_elements_size bytes) = number of fields N
 *   next N*field_id_size bytes:        field_ids[0..N-1]   (sorted by field name)
 *   next (N+1)*field_offset_size bytes: field_offsets[0..N] (relative to values_base)
 *   remaining bytes (values_base..):   the concatenated field values
 *
 * `num_elements_size`, `field_id_size`, and `field_offset_size` come from the value header (see
 * decode_object_array_header). The trailing offset `field_offsets[N]` is the total size of the
 * values region. This scans `field_ids` for `id`, then uses the matching `field_offsets[i]` to
 * slice out the field's value, whose length is derived from its own header via
 * variant_value_length.
 *
 * Per the spec, field ids/offsets are ordered by the corresponding field names (lexicographically),
 * but the values themselves may be in any order, so `field_offsets` are not necessarily monotonic —
 * hence the value length is taken from each field's own header rather than from offset deltas.
 *
 * @param val The object value bytes
 * @param id The dictionary index of the field to locate
 * @return The encoded bytes of the field value, or an empty span if `val` is not an object, the
 *         field is absent, or the blob is malformed
 */
__device__ device_span<uint8_t const> locate_object_field(device_span<uint8_t const> val, int id)
{
  auto const val_len = static_cast<size_type>(val.size());
  if (val_len < 1) { return {}; }
  auto const value_metadata = val[0];
  if (decode_basic_type(value_metadata) != basic_type::OBJECT) { return {}; }

  auto const [offset_size, id_size, num_elements_size] =
    decode_object_array_header(variant_value_header(value_metadata), true);

  size_type pos         = 1;
  auto const num_fields = narrow_cast(read_uint64(val, pos, num_elements_size));
  if (!num_fields.has_value()) { return {}; }
  pos += num_elements_size;

  auto const ids_start = pos;
  auto const ids_bytes = static_cast<uint64_t>(num_fields.value()) * id_size;
  if (ids_bytes > val_len - ids_start) { return {}; }

  auto const offsets_start = ids_start + static_cast<size_type>(ids_bytes);
  auto const offsets_bytes = (static_cast<uint64_t>(num_fields.value()) + 1) * offset_size;
  if (offsets_bytes > val_len - offsets_start) { return {}; }

  auto const values_base = offsets_start + static_cast<size_type>(offsets_bytes);
  // Maximum legitimate field-offset value: bytes available after values_base
  auto const values_extent = val_len - values_base;

  // Find the matching field ID and its start offset
  bool found           = false;
  uint64_t match_start = 0;
  for (size_type i = 0; i < num_fields.value(); ++i) {
    auto const current_id = read_uint64(val, ids_start + i * id_size, id_size);
    if (!current_id.has_value()) { return {}; }
    if (cuda::std::cmp_not_equal(current_id.value(), id)) { continue; }

    auto const match_offset = read_uint64(val, offsets_start + i * offset_size, offset_size);
    if (!match_offset.has_value()) { return {}; }
    if (match_offset.value() > values_extent) { return {}; }
    match_start = match_offset.value();
    found       = true;
    break;
  }
  if (!found) { return {}; }

  // Derive field's value length from its header
  auto const value     = val.subspan(values_base + match_start);
  auto const value_len = variant_value_length(value);
  if (!value_len.has_value()) { return {}; }
  auto const match_end = match_start + value_len.value();
  if (match_end > values_extent) { return {}; }
  return val.subspan(values_base + match_start, value_len.value());
}

// Parse an array value header and return the sub-span of the element at `index` (0-based) within
// `val`. Returns an empty span if `val` is not an array (`basic_type != array`), if `index` is out
// of bounds, or if the encoded data is truncated.
//
// Array layout per the Variant spec:
//   byte 0: header (basic_type=array in low 2 bits; value_header in high 6 bits)
//     value_header bits: (offset_size - 1) in bits 0-1, is_large in bit 2, bits 3-5 unused
//   num_elements: 1 byte if !is_large else 4 bytes (little-endian)
//   offsets:      (num_elements + 1) entries, each `offset_size` bytes, relative to the end of
//                 offsets
//   values:       concatenated element blobs
//
// Array element offsets are monotonically increasing, so the element length is taken directly from
// the offset delta (o1 - o0) rather than from the element's own header.
__device__ device_span<uint8_t const> locate_array_element(device_span<uint8_t const> value,
                                                           size_type index)
{
  if (index < 0) { return {}; }

  auto const value_size = static_cast<size_type>(value.size());
  if (value_size < 1) { return {}; }
  uint8_t const value_metadata = value[0];
  if (decode_basic_type(value_metadata) != basic_type::ARRAY) { return {}; }

  int const value_header = variant_value_header(value_metadata);
  [[maybe_unused]] auto const [offset_size, _, num_elements_size] =
    decode_object_array_header(value_header, false);

  size_type position            = 1;
  auto const num_elements_value = narrow_cast(read_uint64(value, position, num_elements_size));
  if (!num_elements_value.has_value()) { return {}; }
  auto const num_elements = num_elements_value.value();
  if (index >= num_elements) { return {}; }
  position += num_elements_size;

  size_type const offsets_start = position;
  // Computed in 64-bit because (num_elements + 1) * offset_size can exceed the signed `size_type`
  // range (which would be UB); the check below then rejects any array that overruns the value blob.
  auto const offsets_bytes = (static_cast<uint64_t>(num_elements) + 1) * offset_size;
  if (cuda::std::cmp_greater(offsets_bytes, value_size - offsets_start)) { return {}; }
  size_type const values_base = offsets_start + static_cast<size_type>(offsets_bytes);
  auto const values_extent    = value_size - values_base;

  auto const start_offset_pos = offsets_start + static_cast<uint64_t>(index) * offset_size;
  auto const end_offset_pos   = offsets_start + (static_cast<uint64_t>(index) + 1) * offset_size;
  if (cuda::std::cmp_greater(end_offset_pos + offset_size, value_size)) { return {}; }

  auto const start_offset = read_uint64(value, start_offset_pos, offset_size);
  auto const end_offset   = read_uint64(value, end_offset_pos, offset_size);
  if (!start_offset.has_value() || !end_offset.has_value()) { return {}; }
  auto const element_start = *start_offset;
  auto const element_end   = *end_offset;
  if (element_end < element_start || cuda::std::cmp_greater(element_end, values_extent)) {
    return {};
  }
  return value.subspan(values_base + element_start, element_end - element_start);
}

// The fixed-width signed integers a VARIANT value can be cast to: INT{8,16,32,64}.  Matches the
// exact width types (not e.g. __int128) since those are the only variant primitive int headers.
template <typename T>
constexpr bool is_variant_int =
  cudf::is_integral_not_bool<T>() && cudf::is_signed<T>() && !cuda::std::is_same_v<T, __int128_t>;

// The fixed-width primitive types (signed integers and floats) a VARIANT value can be decoded into.
template <typename T>
constexpr bool is_variant_numerical = is_variant_int<T> || cudf::is_floating_point<T>();

// The output types a VARIANT value can be cast to: the fixed-width signed integers, floats, bool,
// and strings.
template <typename T>
constexpr bool is_variant_castable = is_variant_numerical<T> || cuda::std::is_same_v<T, bool> ||
                                     cuda::std::is_same_v<T, cudf::string_view>;

// Maps a fixed-width output type to the VARIANT primitive type header id that encodes it.
template <typename T>
  requires(is_variant_numerical<T>)
__device__ constexpr primitive_type primitive_type_for()
{
  if constexpr (cuda::std::is_same_v<T, int8_t>) {
    return primitive_type::INT8;
  } else if constexpr (cuda::std::is_same_v<T, int16_t>) {
    return primitive_type::INT16;
  } else if constexpr (cuda::std::is_same_v<T, int32_t>) {
    return primitive_type::INT32;
  } else if constexpr (cuda::std::is_same_v<T, int64_t>) {
    return primitive_type::INT64;
  } else if constexpr (cuda::std::is_same_v<T, float>) {
    return primitive_type::FLOAT32;
  } else if constexpr (cuda::std::is_same_v<T, double>) {
    return primitive_type::FLOAT64;
  } else {
    CUDF_UNREACHABLE("primitive_type_for: T is not a supported variant primitive type");
    return primitive_type::NULLVAL;
  }
}

/**
 * @brief Decode a single VARIANT value blob into a fixed-width primitive of type `T`.
 *
 * Requires `basic_type == primitive` and a value header whose physical type id matches `T` exactly.
 */
template <typename T>
__device__ inline cuda::std::optional<T> decode_primitive(device_span<uint8_t const> enc)
{
  if (cuda::std::cmp_less(enc.size(), 1 + sizeof(T))) { return cuda::std::nullopt; }

  uint8_t const value_metadata = enc[0];
  if (decode_basic_type(value_metadata) != basic_type::PRIMITIVE ||
      variant_value_header(value_metadata) != static_cast<uint8_t>(primitive_type_for<T>())) {
    return cuda::std::nullopt;
  }
  return cudf::io::unaligned_load<T>(enc.data() + 1);
}

/**
 * @brief Decode a single VARIANT value blob into a bool.
 *
 * Boolean values carry no payload: the distinction between true and false is encoded entirely in
 * the primitive type header (`boolean_true` vs `boolean_false`).
 */
__device__ inline cuda::std::optional<bool> decode_bool(device_span<uint8_t const> enc)
{
  if (enc.empty()) { return cuda::std::nullopt; }
  uint8_t const value_metadata = enc[0];
  if (decode_basic_type(value_metadata) != basic_type::PRIMITIVE) { return cuda::std::nullopt; }
  auto const value_header = variant_value_header(value_metadata);
  if (value_header == static_cast<uint8_t>(primitive_type::BOOLEAN_TRUE)) { return true; }
  if (value_header == static_cast<uint8_t>(primitive_type::BOOLEAN_FALSE)) { return false; }
  return cuda::std::nullopt;
}

// Parse an array-index step token of the form "[<N>]" into its zero-based index. Returns nullopt
// for any malformed token or an index that does not fit in `size_type` (such an index is out of
// range for any array, so the caller treats it as a missing element).
__device__ cuda::std::optional<size_type> parse_index_step(cudf::string_view step)
{
  auto const step_size  = step.size_bytes();
  auto const* step_data = step.data();
  if (step_size < 3 || step_data[0] != '[' || step_data[step_size - 1] != ']') {
    return cuda::std::nullopt;
  }

  // Accumulate directly in `size_type`; the checked-arithmetic helpers reject the token if the
  // running value overflows, which means the index is out of range for any array and the caller
  // treats it as a missing element.
  size_type index = 0;
  for (size_type k = 1; k < step_size - 1; ++k) {
    char const c = step_data[k];
    if (c < '0' || c > '9') { return cuda::std::nullopt; }
    if (cuda::mul_overflow(index, index, size_type{10}) ||
        cuda::add_overflow(index, index, static_cast<size_type>(c - '0'))) {
      return cuda::std::nullopt;
    }
  }
  return index;
}

// Walk a path of object-key or array-index steps level by level starting at `val` and return
// the span of the final value (subspan of `val`). Returns an empty span on failure.
//
// Each path step is encoded in the `path` strings column as either:
//   - "<name>"  -> descend into an object by dictionary key, or
//   - "[<N>]"   -> descend into an array by zero-based integer index.
// The step kind is inferred from the first byte (`'['` means index).
__device__ device_span<uint8_t const> resolve_path(device_span<uint8_t const> meta,
                                                   device_span<uint8_t const> val,
                                                   column_device_view path)
{
  device_span<uint8_t const> sub_val = val;
  for (size_type i = 0; i < path.size(); ++i) {
    auto const step = path.element<cudf::string_view>(i);

    if (step.size_bytes() >= 1 && step.data()[0] == '[') {
      auto const index = parse_index_step(step);
      if (!index.has_value()) { return {}; }
      sub_val = locate_array_element(sub_val, index.value());
    } else {
      auto const field_id = find_key_in_metadata(meta, step);
      if (!field_id.has_value()) { return {}; }
      sub_val = locate_object_field(sub_val, field_id.value());
    }
    if (sub_val.empty()) { return {}; }
  }
  return sub_val;
}

__device__ cuda::std::optional<device_span<uint8_t const>> decode_string(
  device_span<uint8_t const> enc)
{
  auto const len = enc.size();
  if (len < 1) { return cuda::std::nullopt; }
  uint8_t const value_metadata = enc[0];
  auto const btype             = decode_basic_type(value_metadata);
  auto const value_header      = variant_value_header(value_metadata);

  if (btype == basic_type::SHORT_STRING) {
    // Short string: value_header = length
    std::size_t const str_len = value_header;
    if (1 + str_len > len) { return cuda::std::nullopt; }
    return enc.subspan(1, str_len);
  }
  if (btype == basic_type::PRIMITIVE &&
      value_header == static_cast<uint8_t>(primitive_type::LONG_STRING)) {
    // Long string: 1-byte header + 4-byte LE length + char bytes
    constexpr std::size_t long_string_prefix_bytes = 1 + sizeof(uint32_t);
    if (len < long_string_prefix_bytes) { return cuda::std::nullopt; }
    auto const str_len = cudf::io::unaligned_load<uint32_t>(enc.data() + 1);
    // Encoded length claims more char bytes than the buffer holds: truncated/malformed blob
    if (long_string_prefix_bytes + str_len > len) { return cuda::std::nullopt; }
    return enc.subspan(long_string_prefix_bytes, str_len);
  }
  return cuda::std::nullopt;
}

__device__ device_span<uint8_t const> list_row_span(cudf::lists_column_device_view const& col,
                                                    size_type row)
{
  auto const begin = col.offset_at(row);
  auto const end   = col.offset_at(row + 1);
  return {col.child().data<uint8_t>() + begin, static_cast<std::size_t>(end - begin)};
}

// Returns the metadata and value list bytes for a given row from device views
__device__ cuda::std::pair<device_span<uint8_t const>, device_span<uint8_t const>>
metadata_and_value_at(cudf::lists_column_device_view const& metadata,
                      cudf::lists_column_device_view const& values,
                      size_type row)
{
  return {list_row_span(metadata, row), list_row_span(values, row)};
}

constexpr int block_size = 256;

/**
 * @brief Resolves `path` in each VARIANT row and record the located field's size and source offset.
 *
 * For each non-null row, walks `path` to the target value and writes its byte length to
 * `d_sizes[row]` and its offset within the row's value blob to `d_src_offsets[row]`. Rows that are
 * null, or whose path does not resolve, are marked null in `d_null_mask` with a size of 0.
 */
CUDF_KERNEL __launch_bounds__(block_size) void locate_variant_fields_kernel(
  cudf::lists_column_device_view metadata,
  cudf::lists_column_device_view values,
  column_device_view path,
  device_span<size_type> d_sizes,
  device_span<size_type> d_src_offsets,
  bitmask_type* d_null_mask)
{
  auto const num_rows = static_cast<size_type>(d_sizes.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (!cudf::bit_is_set(d_null_mask, row)) {
      d_sizes[row]       = 0;
      d_src_offsets[row] = 0;
      continue;
    }

    auto const [meta, val] = metadata_and_value_at(metadata, values, row);

    auto const field = resolve_path(meta, val, path);
    if (field.empty()) {
      d_sizes[row]       = 0;
      d_src_offsets[row] = 0;
      cudf::clear_bit(d_null_mask, row);
      continue;
    }

    d_sizes[row]       = static_cast<size_type>(field.size());
    d_src_offsets[row] = static_cast<size_type>(field.data() - val.data());
  }
}

/**
 * @brief Per-row kernel: decode each VARIANT value blob into a fixed-width primitive of type `T`.
 *
 * Writes the decoded value to `d_output[row]` for non-null rows whose blob is a variant primitive
 * whose physical type id matches `T` exactly (e.g. an int16 value does not decode into an int32
 * output, and a float32 value does not decode into a float64 output; there is no widening). Rows
 * that are null, or whose value is not an exact-width match for `T`, are marked null in
 * `d_null_mask` with an output of 0.
 */
template <typename T>
CUDF_KERNEL __launch_bounds__(block_size) void cast_variant_primitive_kernel(
  cudf::lists_column_device_view values, device_span<T> d_output, bitmask_type* d_null_mask)
{
  auto const num_rows = static_cast<size_type>(d_output.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (!cudf::bit_is_set(d_null_mask, row)) {
      d_output[row] = 0;
      continue;
    }

    auto const val = list_row_span(values, row);

    auto const decoded = decode_primitive<T>(val);
    if (decoded.has_value()) {
      d_output[row] = *decoded;
    } else {
      d_output[row] = 0;
      cudf::clear_bit(d_null_mask, row);
    }
  }
}

/**
 * @brief Strings-children functor: decode each VARIANT value blob into a string.
 *
 * Used with `make_strings_children`, so it runs in two passes. On the sizing pass (`d_chars ==
 * nullptr`) it writes each decoded string's length to `d_sizes[row]`; on the write pass it copies
 * the decoded bytes to `d_chars` at `d_offsets[row]`. Rows that are null, or whose value does not
 * decode to a string, are marked null in `d_null_mask` with size 0.
 */
struct cast_variant_string_fn {
  cudf::lists_column_device_view d_values;
  bitmask_type* d_null_mask;
  size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type row)
  {
    if (!cudf::bit_is_set(d_null_mask, row)) {
      if (!d_chars) { d_sizes[row] = 0; }
      return;
    }

    auto const val = list_row_span(d_values, row);

    auto const str = decode_string(val);
    if (!str) {
      if (!d_chars) { d_sizes[row] = 0; }
      cudf::clear_bit(d_null_mask, row);
      return;
    }

    if (!d_chars) {
      d_sizes[row] = str->size();
    } else {
      cuda::std::memcpy(d_chars + d_offsets[row], str->data(), str->size());
    }
  }
};

void validate_variant_child(column_view const& child)
{
  CUDF_EXPECTS(child.type().id() == type_id::LIST,
               "VARIANT metadata/value column must be a list",
               std::invalid_argument);
  CUDF_EXPECTS(lists_column_view{child}.child().type().id() == type_id::UINT8,
               "VARIANT metadata/value column must be list<uint8>",
               std::invalid_argument);
}

struct cast_variant_fn {
  cudf::lists_column_device_view values;
  size_type num_rows;
  data_type desired_type;
  bitmask_type* d_null_mask;
  rmm::device_buffer null_mask;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(is_variant_numerical<T>)
  {
    rmm::device_buffer data{num_rows * sizeof(T), stream, mr};
    auto grid = cudf::detail::grid_1d{num_rows, block_size};
    cast_variant_primitive_kernel<T><<<grid.num_blocks, block_size, 0, stream.get()>>>(
      values, {static_cast<T*>(data.data()), static_cast<std::size_t>(num_rows)}, d_null_mask);
    CUDF_CUDA_TRY(cudaGetLastError());
    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                                    null_count);
  }

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(cuda::std::is_same_v<T, bool>)
  {
    rmm::device_buffer data{num_rows * sizeof(bool), stream, mr};

    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<size_type>(0),
      cuda::counting_iterator<size_type>(num_rows),
      static_cast<bool*>(data.data()),
      [values = this->values, d_null_mask = this->d_null_mask] __device__(size_type row) -> bool {
        if (!cudf::bit_is_set(d_null_mask, row)) { return false; }
        auto const val     = list_row_span(values, row);
        auto const decoded = decode_bool(val);
        if (decoded.has_value()) { return *decoded; }
        cudf::clear_bit(d_null_mask, row);
        return false;
      });

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                                    null_count);
  }

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(cuda::std::is_same_v<T, cudf::string_view>)
  {
    cast_variant_string_fn fn{values, d_null_mask, nullptr, nullptr, {}};
    auto [offsets_column, chars] =
      cudf::strings::detail::make_strings_children(fn, num_rows, stream, mr);
    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return make_strings_column(num_rows,
                               std::move(offsets_column),
                               chars.release(),
                               null_count,
                               null_count > 0 ? std::move(null_mask) : rmm::device_buffer{});
  }

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(not is_variant_castable<T>)
  {
    CUDF_FAIL("unsupported type for variant cast", std::invalid_argument);
  }
};

/**
 * @brief Classifies only the first (value_metadata) byte of enc; does not validate the remaining
 * payload. A recognized header returns its logical type even when the payload is truncated. Returns
 * nullopt for an empty blob or an unrecognized primitive type ID.
 */
__device__ cuda::std::optional<variant_logical_type> logical_type_of(device_span<uint8_t const> enc)
{
  if (enc.empty()) { return cuda::std::nullopt; }
  auto const value_metadata = enc[0];
  auto const btype          = decode_basic_type(value_metadata);

  if (btype == basic_type::SHORT_STRING) { return variant_logical_type::STRING; }
  if (btype == basic_type::OBJECT) { return variant_logical_type::OBJECT; }
  if (btype == basic_type::ARRAY) { return variant_logical_type::ARRAY; }

  switch (static_cast<primitive_type>(variant_value_header(value_metadata))) {
    case primitive_type::NULLVAL: return variant_logical_type::NULL_VALUE;
    case primitive_type::BOOLEAN_TRUE:
    case primitive_type::BOOLEAN_FALSE: return variant_logical_type::BOOLEAN;
    case primitive_type::INT8:
    case primitive_type::INT16:
    case primitive_type::INT32:
    case primitive_type::INT64: return variant_logical_type::LONG_VALUE;
    case primitive_type::FLOAT64: return variant_logical_type::DOUBLE_VALUE;
    case primitive_type::DECIMAL4:
    case primitive_type::DECIMAL8:
    case primitive_type::DECIMAL16: return variant_logical_type::DECIMAL;
    case primitive_type::DATE: return variant_logical_type::DATE;
    case primitive_type::TIMESTAMP_MICROS:
    case primitive_type::TIMESTAMP_NANOS: return variant_logical_type::TIMESTAMP;
    case primitive_type::TIMESTAMP_NTZ_MICROS:
    case primitive_type::TIMESTAMP_NTZ_NANOS: return variant_logical_type::TIMESTAMP_NTZ;
    case primitive_type::FLOAT32: return variant_logical_type::FLOAT_VALUE;
    case primitive_type::BINARY: return variant_logical_type::BINARY;
    case primitive_type::LONG_STRING: return variant_logical_type::STRING;
    case primitive_type::TIME_NTZ_MICROS: return variant_logical_type::TIME_NTZ;
    case primitive_type::UUID: return variant_logical_type::UUID;
    default: return cuda::std::nullopt;
  }
}

std::unique_ptr<column> build_path_column(cudf::host_span<std::string const> steps,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const depth = steps.size();

  std::string host_chars;
  std::vector<size_type> host_offsets(depth + 1);
  for (size_t i = 0; i < depth; ++i) {
    host_offsets[i] = static_cast<size_type>(host_chars.size());
    host_chars.append(steps[i]);
  }
  host_offsets[depth] = host_chars.size();

  auto d_offsets   = cudf::detail::make_device_uvector_async(host_offsets, stream, mr);
  auto offsets_col = std::make_unique<column>(data_type{type_id::INT32},
                                              static_cast<size_type>(host_offsets.size()),
                                              d_offsets.release(),
                                              rmm::device_buffer{},
                                              0);

  auto d_chars = cudf::detail::make_device_uvector(
    host_span<char const>{host_chars.data(), host_chars.size()}, stream, mr);
  return cudf::make_strings_column(
    depth, std::move(offsets_col), d_chars.release(), 0, rmm::device_buffer{});
}

}  // namespace

namespace detail {

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  // Validate the variant column
  CUDF_EXPECTS(variant_column.type().id() == type_id::STRUCT,
               "VARIANT column must be struct type",
               std::invalid_argument);
  CUDF_EXPECTS(variant_column.num_children() >= 2,
               "VARIANT struct must have at least two children",
               std::invalid_argument);
  validate_variant_child(variant_column.child(0));
  validate_variant_child(variant_column.child(1));

  // Validate the path even for empty input columns
  auto const steps = parse_variant_path(path);

  auto const num_rows = variant_column.size();
  if (num_rows == 0) {
    return cudf::make_lists_column(
      0, make_empty_column(type_id::INT32), make_empty_column(type_id::UINT8), 0, {});
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();

  auto path_column      = build_path_column(steps, stream, temp_mr);
  auto path_device_view = column_device_view::create(path_column->view(), stream);

  // Resolve children with respect to any slice/offset on the parent struct
  structs_column_view const variant_struct{variant_column};
  auto const meta_view = variant_struct.get_sliced_child(0, stream);
  auto const val_view  = variant_struct.get_sliced_child(1, stream);

  auto meta_device_view = column_device_view::create(meta_view, stream);
  auto val_device_view  = column_device_view::create(val_view, stream);
  cudf::lists_column_device_view meta_lists_device_view(*meta_device_view);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  rmm::device_uvector<size_type> d_sizes(num_rows, stream, temp_mr);
  // Caches the per-row intra-value byte offset
  rmm::device_uvector<size_type> d_src_offsets(num_rows, stream, temp_mr);
  auto null_mask =
    variant_column.nullable()
      ? cudf::detail::copy_bitmask(variant_column, stream, mr)
      : cudf::create_null_mask(variant_column.size(), mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  // Parse the path per row and compute the output sizes
  auto grid = cudf::detail::grid_1d{num_rows, block_size};
  locate_variant_fields_kernel<<<grid.num_blocks, block_size, 0, stream.get()>>>(
    meta_lists_device_view,
    val_lists_device_view,
    *path_device_view,
    d_sizes,
    d_src_offsets,
    d_null_mask);
  CUDF_CUDA_TRY(cudaGetLastError());

  // Convert sizes to offsets
  auto [offsets_column, total_bytes] =
    cudf::strings::detail::make_offsets_child_column(d_sizes, stream, mr);
  CUDF_EXPECTS(total_bytes <= std::numeric_limits<size_type>::max(),
               "VARIANT extracted bytes exceed cudf size_type limit",
               std::overflow_error);
  device_span<size_type const> d_offsets{offsets_column->view().data<size_type>(),
                                         static_cast<std::size_t>(num_rows + 1)};

  // Copy values into the output buffer
  auto val_child = make_numeric_column(
    data_type{type_id::UINT8}, total_bytes, mask_state::UNALLOCATED, stream, mr);
  if (total_bytes > 0) {
    auto const out_base = val_child->mutable_view().data<uint8_t>();
    auto src_iter       = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t const*>(
        [vlv   = val_lists_device_view,
         d_src = d_src_offsets.data()] __device__(size_type row) -> uint8_t const* {
          return vlv.child().template data<uint8_t>() + vlv.offset_at(row) + d_src[row];
        }));
    auto dst_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<uint8_t*>(
        [out_base, d_off = d_offsets.data()] __device__(size_type row) -> uint8_t* {
          return out_base + d_off[row];
        }));
    cudf::detail::batched_memcpy_async(src_iter, dst_iter, d_sizes.begin(), num_rows, stream);
  }

  auto const null_count = num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
  return make_lists_column(num_rows,
                           std::move(offsets_column),
                           std::move(val_child),
                           null_count,
                           null_count > 0 ? std::move(null_mask) : rmm::device_buffer{});
}

std::unique_ptr<column> cast_variant(column_view const& values,
                                     data_type desired_type,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  validate_variant_child(values);

  switch (desired_type.id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64:
    case type_id::FLOAT32:
    case type_id::FLOAT64:
    case type_id::BOOL8:
    case type_id::STRING: break;
    default: CUDF_FAIL("unsupported type for variant cast", std::invalid_argument);
  }

  size_type const num_rows = values.size();
  if (num_rows == 0) { return make_empty_column(desired_type); }

  auto val_device_view = column_device_view::create(values, stream);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  // Initialize the null mask from the values column (or all-valid)
  auto null_mask    = values.nullable()
                        ? cudf::detail::copy_bitmask(values, stream, mr)
                        : cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  return cudf::type_dispatcher(desired_type,
                               cast_variant_fn{val_lists_device_view,
                                               num_rows,
                                               desired_type,
                                               d_null_mask,
                                               std::move(null_mask),
                                               stream,
                                               mr});
}

std::unique_ptr<column> get_variant_type_id(column_view const& values,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  validate_variant_child(values);
  size_type const num_rows = values.size();
  if (num_rows == 0) { return make_empty_column(data_type{type_id::UINT8}); }

  auto val_device_view = column_device_view::create(values, stream);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  auto null_mask    = values.nullable()
                        ? cudf::detail::copy_bitmask(values, stream, mr)
                        : cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  rmm::device_buffer data{static_cast<std::size_t>(num_rows) * sizeof(uint8_t), stream, mr};

  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>(0),
    cuda::counting_iterator<size_type>(num_rows),
    static_cast<uint8_t*>(data.data()),
    [values = val_lists_device_view, d_null_mask] __device__(size_type row) -> uint8_t {
      if (!cudf::bit_is_set(d_null_mask, row)) { return 0; }
      auto const ltype = logical_type_of(list_row_span(values, row));
      if (ltype.has_value()) { return static_cast<uint8_t>(ltype.value()); }
      cudf::clear_bit(d_null_mask, row);
      return 0;
    });

  auto const null_count = num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
  return std::make_unique<column>(data_type{type_id::UINT8},
                                  num_rows,
                                  std::move(data),
                                  null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                                  null_count);
}

}  // namespace detail

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_variant_field(variant_column, path, stream, mr);
}

std::unique_ptr<column> cast_variant(column_view const& values,
                                     data_type desired_type,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::cast_variant(values, desired_type, stream, mr);
}

std::unique_ptr<column> get_variant_type_id(column_view const& values,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_variant_type_id(values, stream, mr);
}

std::unique_ptr<column> extract_variant_field(column_view const& variant_column,
                                              std::string_view path,
                                              data_type desired_type,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto value = detail::get_variant_field(
    variant_column, path, stream, cudf::get_current_device_resource_ref());
  return detail::cast_variant(value->view(), desired_type, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
