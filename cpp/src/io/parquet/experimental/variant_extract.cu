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
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/bit>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/numeric>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/concepts>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/scan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
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

// The status of a VARIANT operation.
using op_status = variant_operation_status;

__device__ cuda::std::optional<uint64_t> read_uint64(device_span<uint8_t const> data,
                                                     size_type pos,
                                                     int width)
{
  if (cuda::std::cmp_greater(pos + width, data.size())) { return cuda::std::nullopt; }
  uint64_t v = 0;
  cuda::std::memcpy(&v, data.data() + pos, width);
  return v;
}

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

// Parsed offset table of a metadata dictionary blob, produced once per row by
// `parse_metadata_dictionary` and reused for every `name_for_id` lookup along a row's path (a
// path may step through several objects, each needing id-to-name resolution against the same
// dictionary) instead of re-parsing and re-validating `meta` on every step.
struct metadata_dictionary {
  size_type offset_size;
  size_type offsets_start;
  size_type strings_base;
  size_type strings_declared;
  size_type num_entries;
};

/**
 * @brief Parse and validate a metadata dictionary blob's offset table.
 *
 * @param meta The metadata blob for this row
 * @return The parsed offset table, or a failure status if `meta` is malformed
 */
__device__ cuda::std::pair<metadata_dictionary, op_status> parse_metadata_dictionary(
  device_span<uint8_t const> meta)
{
  auto const meta_len = static_cast<size_type>(meta.size());
  if (meta_len < 1) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const meta_header = meta[0];
  if ((meta_header & 0x0F) != variant_version_v1) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const meta_offset_size = ((meta_header >> 6) & 0x03) + 1;

  size_type meta_pos          = 1;
  auto const num_meta_entries = narrow_cast(read_uint64(meta, meta_pos, meta_offset_size));
  if (!num_meta_entries.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
  meta_pos += meta_offset_size;

  auto const meta_offsets_start = meta_pos;
  auto const meta_offsets_bytes =
    (static_cast<uint64_t>(num_meta_entries.value()) + 1) * meta_offset_size;
  if (cuda::std::cmp_greater(meta_offsets_bytes, meta_len - meta_offsets_start)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const meta_strings_base   = meta_offsets_start + static_cast<size_type>(meta_offsets_bytes);
  auto const meta_strings_extent = meta_len - meta_strings_base;

  // Parquet VARIANT spec requires offsets[0] == 0; any other value is malformed.
  auto const first_off = read_uint64(meta, meta_offsets_start, meta_offset_size);
  if (!first_off.has_value() || first_off.value() != 0) {
    return {{}, op_status::MALFORMED_VARIANT};
  }

  // Read the terminal offset offsets[num_entries] up front: it is the authoritative size of the
  // string-data region, so every individual entry's end offset must be bounded by it below, not
  // just by the physical buffer remainder (`meta_strings_extent`) -- the same distinction the
  // object value's own sentinel/values_region check makes for field values.
  auto const terminal_off_pos =
    meta_offsets_start + static_cast<size_type>(num_meta_entries.value()) * meta_offset_size;
  auto const terminal_off = read_uint64(meta, terminal_off_pos, meta_offset_size);
  if (!terminal_off.has_value() ||
      cuda::std::cmp_greater(terminal_off.value(), meta_strings_extent)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }

  return {metadata_dictionary{.offset_size      = meta_offset_size,
                              .offsets_start    = meta_offsets_start,
                              .strings_base     = meta_strings_base,
                              .strings_declared = static_cast<size_type>(terminal_off.value()),
                              .num_entries      = num_meta_entries.value()},
          op_status::SUCCESS};
}

// O(1) name lookup by id: two offset reads into the metadata table. `field_id` may come directly
// from untrusted object data (an out-of-range dictionary index), so it is bounds checked against
// the metadata dictionary size before use, and the offset positions are computed in 64-bit
// arithmetic to avoid overflowing `size_type` for large ids.
__device__ cuda::std::optional<cudf::string_view> name_for_id(metadata_dictionary const& dict,
                                                              device_span<uint8_t const> meta,
                                                              size_type field_id)
{
  if (field_id < 0 || cuda::std::cmp_greater_equal(field_id, dict.num_entries)) {
    return cuda::std::nullopt;
  }
  auto const start_pos =
    static_cast<uint64_t>(dict.offsets_start) + static_cast<uint64_t>(field_id) * dict.offset_size;
  auto const end_pos = start_pos + static_cast<uint64_t>(dict.offset_size);
  if (cuda::std::cmp_greater(end_pos, meta.size())) { return cuda::std::nullopt; }
  auto const s = read_uint64(meta, static_cast<size_type>(start_pos), dict.offset_size);
  auto const e = read_uint64(meta, static_cast<size_type>(end_pos), dict.offset_size);
  if (!s.has_value() || !e.has_value()) { return cuda::std::nullopt; }
  if (e.value() < s.value() || cuda::std::cmp_greater(e.value(), dict.strings_declared)) {
    return cuda::std::nullopt;
  }
  return cudf::string_view{
    reinterpret_cast<char const*>(meta.data() + dict.strings_base + s.value()),
    static_cast<size_type>(e.value() - s.value())};
}

/**
 * @brief Locate the encoded bytes of a single field within an object value by field name.
 *
 * Object value layout, following the 1-byte value metadata header (basic_type=object in the low 2
 * bits; value_header in the high 6 bits, see decode_object_array_header):
 *
 *   bytes 1..:     num_elements   (num_elements_size bytes) = number of fields N
 *   next N*field_id_size bytes:        field_ids[0..N-1]   (sorted by field name)
 *   next (N+1)*field_offset_size bytes: field_offsets[0..N] (relative to values_base)
 *   remaining bytes (values_base..):   the concatenated field values
 *
 * `num_elements_size`, `field_id_size`, and `field_offset_size` come from the value header (see
 * decode_object_array_header). The trailing offset `field_offsets[N]` is the total size of the
 * values region.
 *
 * Per the spec, `field_ids[0..N-1]` are ordered by the corresponding field name
 * (lexicographically), not by the numeric id value, and the values themselves may be in any order,
 * so `field_offsets` are not necessarily monotonic -- hence the value length is taken from each
 * field's own header rather than from offset deltas.
 *
 * `field_ids[0..N-1]` is ordered by name (a per-object invariant, independent of whether the
 * metadata dictionary itself happens to be sorted), so it can be binary searched directly against
 * `key` without first resolving `key` to a dictionary id: each probe turns `field_ids[mid]` into
 * its dictionary name via an O(1) lookup in `meta` (the metadata offset table is indexed directly
 * by id) and compares that name against `key`, giving O(log N) with no separate name-to-id lookup
 * over the (potentially much larger) dictionary.
 *
 * Not using thrust::lower_bound since it does not propagate entry read failures.
 *
 * @param dict The already-parsed metadata dictionary offset table for `meta` (see
 *             parse_metadata_dictionary), reused across every path step so it is not re-parsed
 *             and re-validated once per object
 * @param meta The metadata blob for this row, used to resolve field ids to names
 * @param val The object value bytes
 * @param key The name of the field to locate
 * @return The encoded bytes of the field value, or an empty span if `val` is not an object, the
 *         field is absent, or either blob is malformed
 */
__device__ cuda::std::pair<device_span<uint8_t const>, op_status> locate_object_field(
  metadata_dictionary const& dict,
  device_span<uint8_t const> meta,
  device_span<uint8_t const> val,
  cudf::string_view key)
{
  auto const val_len = static_cast<size_type>(val.size());
  if (val_len < 1) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const value_metadata = val[0];
  if (decode_basic_type(value_metadata) != basic_type::OBJECT) {
    return {{}, op_status::MISSING_PATH};
  }

  auto const [offset_size, id_size, num_elements_size] =
    decode_object_array_header(variant_value_header(value_metadata), true);

  size_type pos         = 1;
  auto const num_fields = narrow_cast(read_uint64(val, pos, num_elements_size));
  if (!num_fields.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
  if (num_fields.value() == 0) { return {{}, op_status::MISSING_PATH}; }
  pos += num_elements_size;

  auto const ids_start = pos;
  auto const ids_bytes = static_cast<uint64_t>(num_fields.value()) * id_size;
  if (ids_bytes > val_len - ids_start) { return {{}, op_status::MALFORMED_VARIANT}; }

  auto const offsets_start = ids_start + static_cast<size_type>(ids_bytes);
  auto const offsets_bytes = (static_cast<uint64_t>(num_fields.value()) + 1) * offset_size;
  if (offsets_bytes > val_len - offsets_start) { return {{}, op_status::MALFORMED_VARIANT}; }

  auto const values_base   = offsets_start + static_cast<size_type>(offsets_bytes);
  auto const values_extent = val_len - values_base;

  // Read the sentinel (terminal offset at index num_fields) to get the authoritative end of the
  // values region.  Using the physical remainder (values_extent) would allow a malformed object
  // to reference bytes beyond the sentinel, passing validation despite corrupt data.
  auto const sentinel_raw =
    read_uint64(val, offsets_start + num_fields.value() * offset_size, offset_size);
  if (!sentinel_raw.has_value() || sentinel_raw.value() > static_cast<uint64_t>(values_extent)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const values_region = static_cast<size_type>(sentinel_raw.value());

  // Binary search field_ids[0..N-1] by resolving each probe to its name and comparing against
  // `key` directly
  bool found           = false;
  uint64_t match_start = 0;
  size_type lo         = 0;
  size_type hi         = num_fields.value();
  while (lo < hi) {
    size_type const mid = lo + (hi - lo) / 2;
    auto const probe_id = narrow_cast(read_uint64(val, ids_start + mid * id_size, id_size));
    if (!probe_id.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
    auto const probe_name = name_for_id(dict, meta, probe_id.value());
    if (!probe_name.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
    int const cmp = probe_name.value().compare(key);
    if (cmp == 0) {
      auto const match_offset = read_uint64(val, offsets_start + mid * offset_size, offset_size);
      if (!match_offset.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
      if (match_offset.value() > static_cast<uint64_t>(values_region)) {
        return {{}, op_status::MALFORMED_VARIANT};
      }
      match_start = match_offset.value();
      found       = true;
      break;
    }
    if (cmp < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (!found) { return {{}, op_status::MISSING_PATH}; }

  auto const value     = val.subspan(values_base + match_start);
  auto const value_len = variant_value_length(value);
  if (!value_len.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const match_end = match_start + value_len.value();
  if (match_end > static_cast<uint64_t>(values_region)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  return {val.subspan(values_base + match_start, value_len.value()), op_status::SUCCESS};
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
__device__ cuda::std::pair<device_span<uint8_t const>, op_status> locate_array_element(
  device_span<uint8_t const> value, size_type index)
{
  if (index < 0) { return {{}, op_status::MISSING_PATH}; }

  auto const value_size = static_cast<size_type>(value.size());
  if (value_size < 1) { return {{}, op_status::MALFORMED_VARIANT}; }
  uint8_t const value_metadata = value[0];
  if (decode_basic_type(value_metadata) != basic_type::ARRAY) {
    return {{}, op_status::MISSING_PATH};
  }

  int const value_header = variant_value_header(value_metadata);
  [[maybe_unused]] auto const [offset_size, _, num_elements_size] =
    decode_object_array_header(value_header, false);

  size_type position            = 1;
  auto const num_elements_value = narrow_cast(read_uint64(value, position, num_elements_size));
  if (!num_elements_value.has_value()) { return {{}, op_status::MALFORMED_VARIANT}; }
  auto const num_elements = num_elements_value.value();
  if (index >= num_elements) { return {{}, op_status::MISSING_PATH}; }
  position += num_elements_size;

  size_type const offsets_start = position;

  // Computed in 64-bit because (num_elements + 1) * offset_size can exceed the signed `size_type`
  // range (which would be UB); the check below then rejects any array that overruns the value blob.
  auto const offsets_bytes = (static_cast<uint64_t>(num_elements) + 1) * offset_size;
  if (cuda::std::cmp_greater(offsets_bytes, value_size - offsets_start)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  size_type const values_base = offsets_start + static_cast<size_type>(offsets_bytes);
  auto const values_extent    = value_size - values_base;
  // Read the terminal offset offsets[num_elements]; it is the spec-declared bound on the
  // values region and must be used instead of the physical extent so that an element whose
  // offset escapes the declared boundary is caught as malformed even when physical bytes
  // are present beyond it.
  auto const terminal_off_pos = offsets_start + static_cast<uint64_t>(num_elements) * offset_size;
  auto const terminal_off     = read_uint64(value, terminal_off_pos, offset_size);
  if (!terminal_off.has_value() || cuda::std::cmp_greater(*terminal_off, values_extent)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  // The spec requires offsets[0] == 0; a nonzero first offset silently skips leading
  // value bytes and can return a plausible result from a malformed array.
  auto const first_off = read_uint64(value, offsets_start, offset_size);
  if (!first_off.has_value() || *first_off != 0) { return {{}, op_status::MALFORMED_VARIANT}; }

  auto const start_offset_pos = offsets_start + static_cast<uint64_t>(index) * offset_size;
  auto const end_offset_pos   = offsets_start + (static_cast<uint64_t>(index) + 1) * offset_size;
  if (cuda::std::cmp_greater(end_offset_pos + offset_size, value_size)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const start_offset = read_uint64(value, start_offset_pos, offset_size);
  auto const end_offset   = read_uint64(value, end_offset_pos, offset_size);
  if (!start_offset.has_value() || !end_offset.has_value()) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  auto const element_start = *start_offset;
  auto const element_end   = *end_offset;
  if (element_end < element_start || cuda::std::cmp_greater(element_end, *terminal_off)) {
    return {{}, op_status::MALFORMED_VARIANT};
  }
  return {value.subspan(values_base + element_start, element_end - element_start),
          op_status::SUCCESS};
}

__device__ bool is_variant_null(device_span<uint8_t const> enc)
{
  if (enc.empty()) { return false; }
  auto const vm = enc[0];
  return decode_basic_type(vm) == basic_type::PRIMITIVE &&
         variant_value_header(vm) == static_cast<uint8_t>(primitive_type::NULLVAL);
}

// The fixed-width signed integers a VARIANT value can be cast to: INT{8,16,32,64}.  Matches the
// exact width types (not e.g. __int128) since those are the only variant primitive int headers.
template <typename T>
concept variant_int = cuda::std::signed_integral<T> && !cuda::std::same_as<T, __int128_t>;

// The microsecond/nanosecond timestamps a VARIANT value can be decoded into.
template <typename T>
concept variant_timestamp =
  cuda::std::same_as<T, cudf::timestamp_us> || cuda::std::same_as<T, cudf::timestamp_ns>;

// The fixed-width primitive types (signed integers, floats, timestamps) a VARIANT value can be
// decoded into.
template <typename T>
concept variant_fixed_width =
  variant_int<T> || cuda::std::floating_point<T> || variant_timestamp<T>;

// The output types a VARIANT value can be cast to: the fixed-width signed integers, floats,
// timestamps, decimals, bool, and strings.
template <typename T>
concept variant_castable = variant_fixed_width<T> || cudf::concepts::fixed_point<T> ||
                           cudf::concepts::boolean<T> || cudf::concepts::string_view<T>;

// Whether a VARIANT primitive type header encodes `T`.
template <variant_fixed_width T>
__device__ constexpr bool matches_primitive_type(primitive_type ptype)
{
  if constexpr (cuda::std::is_same_v<T, int8_t>) {
    return ptype == primitive_type::INT8;
  } else if constexpr (cuda::std::is_same_v<T, int16_t>) {
    return ptype == primitive_type::INT16;
  } else if constexpr (cuda::std::is_same_v<T, int32_t>) {
    return ptype == primitive_type::INT32;
  } else if constexpr (cuda::std::is_same_v<T, int64_t>) {
    return ptype == primitive_type::INT64;
  } else if constexpr (cuda::std::is_same_v<T, float>) {
    return ptype == primitive_type::FLOAT32;
  } else if constexpr (cuda::std::is_same_v<T, double>) {
    return ptype == primitive_type::FLOAT64;
  }
  // cuDF timestamps carry no timezone, so a timestamp target also accepts the matching NTZ
  // encoding; Spark's TIMESTAMP and TIMESTAMP_NTZ both map to TIMESTAMP_MICROSECONDS.
  else if constexpr (cuda::std::is_same_v<T, cudf::timestamp_us>) {
    return ptype == primitive_type::TIMESTAMP_MICROS ||
           ptype == primitive_type::TIMESTAMP_NTZ_MICROS;
  } else {
    static_assert(cuda::std::is_same_v<T, cudf::timestamp_ns>,
                  "T is not a supported variant primitive type");
    return ptype == primitive_type::TIMESTAMP_NANOS || ptype == primitive_type::TIMESTAMP_NTZ_NANOS;
  }
}

/**
 * @brief Decode a single VARIANT value blob into a fixed-width primitive of type `T`.
 *
 * Requires `basic_type == primitive` and a value header accepted by `matches_primitive_type<T>`.
 */
template <typename T>
__device__ inline cuda::std::optional<T> decode_primitive(device_span<uint8_t const> enc)
{
  if (cuda::std::cmp_less(enc.size(), 1 + sizeof(T))) { return cuda::std::nullopt; }

  uint8_t const value_metadata = enc[0];
  if (decode_basic_type(value_metadata) != basic_type::PRIMITIVE ||
      !matches_primitive_type<T>(
        static_cast<primitive_type>(variant_value_header(value_metadata)))) {
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
__device__ cuda::std::pair<device_span<uint8_t const>, op_status> resolve_path(
  device_span<uint8_t const> meta,
  device_span<uint8_t const> val,
  column_device_view path,
  cuda::std::optional<cuda::std::pair<metadata_dictionary, op_status>>& dict)
{
  device_span<uint8_t const> sub_val = val;
  for (size_type i = 0; i < path.size(); ++i) {
    auto const step = path.element<cudf::string_view>(i);
    if (step.size_bytes() >= 1 && step.data()[0] == '[') {
      auto const index = parse_index_step(step);
      if (!index.has_value()) { return {{}, op_status::MISSING_PATH}; }
      auto const [span, st] = locate_array_element(sub_val, index.value());
      if (st != op_status::SUCCESS) { return {{}, st}; }
      sub_val = span;
    } else {
      if (!dict.has_value()) { dict = parse_metadata_dictionary(meta); }
      if (dict->second != op_status::SUCCESS) { return {{}, dict->second}; }
      auto const [span, st] = locate_object_field(dict->first, meta, sub_val, step);
      if (st != op_status::SUCCESS) { return {{}, st}; }
      sub_val = span;
    }

    // VARIANT null before the end of the path is missing_path per spec.
    if (i + 1 < path.size() && is_variant_null(sub_val)) { return {{}, op_status::MISSING_PATH}; }
    // A zero-length resolved value is not decodable; the value-only path drops the row.
    if (sub_val.empty()) { return {{}, op_status::MALFORMED_VARIANT}; }
  }

  // Terminal VARIANT null: return the bytes with variant_null status.
  if (is_variant_null(sub_val)) { return {sub_val, op_status::VARIANT_NULL}; }
  return {sub_val, op_status::SUCCESS};
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

__device__ cuda::std::pair<device_span<uint8_t const>, device_span<uint8_t const>>
metadata_and_value_at(cudf::lists_column_device_view const& metadata,
                      cudf::lists_column_device_view const& values,
                      size_type row)
{
  return {list_row_span(metadata, row), list_row_span(values, row)};
}

constexpr int block_size = 256;

// Returns true for every primitive_type ID that variant_value_length maps to a known payload
// size in its `basic_type::PRIMITIVE` switch, i.e. every ID other than its `default` case.
__device__ bool is_recognized_primitive_type(primitive_type ptype)
{
  switch (ptype) {
    case primitive_type::NULLVAL:
    case primitive_type::BOOLEAN_TRUE:
    case primitive_type::BOOLEAN_FALSE:
    case primitive_type::INT8:
    case primitive_type::INT16:
    case primitive_type::INT32:
    case primitive_type::INT64:
    case primitive_type::FLOAT64:
    case primitive_type::DECIMAL4:
    case primitive_type::DECIMAL8:
    case primitive_type::DECIMAL16:
    case primitive_type::DATE:
    case primitive_type::TIMESTAMP_MICROS:
    case primitive_type::TIMESTAMP_NTZ_MICROS:
    case primitive_type::FLOAT32:
    case primitive_type::BINARY:
    case primitive_type::LONG_STRING:
    case primitive_type::TIME_NTZ_MICROS:
    case primitive_type::TIMESTAMP_NANOS:
    case primitive_type::TIMESTAMP_NTZ_NANOS:
    case primitive_type::UUID: return true;
    default: return false;
  }
}

/**
 * @brief Status helper for fixed-width primitive targets: classifies why `decode_primitive<T>`
 * failed to decode `val`, per `variant_operation_status` semantics.
 */
template <variant_fixed_width T>
__device__ op_status cast_status_for_primitive(device_span<uint8_t const> val)
{
  if (val.empty()) { return op_status::MALFORMED_VARIANT; }
  if (is_variant_null(val)) { return op_status::VARIANT_NULL; }
  if (decode_primitive<T>(val).has_value()) { return op_status::SUCCESS; }
  if (decode_basic_type(val[0]) != basic_type::PRIMITIVE) { return op_status::TYPE_MISMATCH; }
  auto const ptype = static_cast<primitive_type>(variant_value_header(val[0]));
  if (matches_primitive_type<T>(ptype)) { return op_status::MALFORMED_VARIANT; }
  return is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                             : op_status::MALFORMED_VARIANT;
}

// The spec allows a scale in [0, 38] for every decimal width.
constexpr int variant_decimal_max_scale = 38;

// The largest power of ten each representation holds; no value fits past it.
constexpr int max_int128_pow10 = cuda::std::numeric_limits<__int128_t>::digits10;
constexpr int max_int64_pow10  = cuda::std::numeric_limits<int64_t>::digits10;

// Multiply `value` by 10^exp, or return nullopt if the result does not fit in `__int128_t`.
__device__ cuda::std::optional<__int128_t> constexpr multiply_pow10(__int128_t value, int64_t exp)
{
  // Zero is representable at every scale, while any other value overflows past 10^38.
  if (value == 0) { return 0; }
  if (exp > max_int128_pow10) { return cuda::std::nullopt; }

  constexpr __int128_t max_over_10 = cuda::std::numeric_limits<__int128_t>::max() / 10;
  constexpr __int128_t min_over_10 = cuda::std::numeric_limits<__int128_t>::min() / 10;
  for (int64_t i = 0; i < exp; ++i) {
    if (value > max_over_10 || value < min_over_10) { return cuda::std::nullopt; }
    value *= 10;
  }
  return value;
}

// Divide `value` by 10^exp, truncating toward zero.
__device__ __int128_t constexpr divide_pow10(__int128_t value, int64_t exp)
{
  using numeric::detail::ipow;

  // Any `__int128_t` is smaller than 10^39, so a larger divisor truncates it away
  if (exp > max_int128_pow10) { return 0; }
  auto const exponent = static_cast<int32_t>(exp);

  // 128-bit division is a slow software sequence; use the 64-bit one when both operands fit.
  constexpr __int128_t i64_max = cuda::std::numeric_limits<int64_t>::max();
  constexpr __int128_t i64_min = cuda::std::numeric_limits<int64_t>::min();
  if (exponent <= max_int64_pow10 && value <= i64_max && value >= i64_min) {
    return static_cast<int64_t>(value) / ipow<int64_t, numeric::Radix::BASE_10>(exponent);
  }
  return value / ipow<__int128_t, numeric::Radix::BASE_10>(exponent);
}

__device__ int constexpr variant_decimal_unscaled_width(primitive_type ptype)
{
  switch (ptype) {
    case primitive_type::DECIMAL4: return 4;
    case primitive_type::DECIMAL8: return 8;
    case primitive_type::DECIMAL16: return 16;
    default: return 0;
  }
}

/**
 * @brief Decode a single VARIANT decimal value blob into the representation of a cuDF fixed-point
 * type, rescaled to `desired_scale`.
 *
 * Encoded as the value metadata byte, a one-byte scale (fractional digit count), then the
 * little-endian two's-complement unscaled integer. Any of the three widths decodes into any `Rep`
 * whose range fits the rescaled value; digits below `desired_scale` truncate toward zero.
 *
 * @return The rescaled representation, valid only when the returned status is `SUCCESS`
 */
template <typename Rep>
__device__ cuda::std::pair<Rep, op_status> decode_decimal(device_span<uint8_t const> enc,
                                                          int desired_scale)
{
  auto const fail = [](op_status status) { return cuda::std::pair<Rep, op_status>{Rep{}, status}; };

  if (enc.empty()) { return fail(op_status::MALFORMED_VARIANT); }
  if (is_variant_null(enc)) { return fail(op_status::VARIANT_NULL); }
  if (decode_basic_type(enc[0]) != basic_type::PRIMITIVE) { return fail(op_status::TYPE_MISMATCH); }

  auto const ptype = static_cast<primitive_type>(variant_value_header(enc[0]));
  auto const width = variant_decimal_unscaled_width(ptype);
  if (width == 0) {
    return fail(is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                                    : op_status::MALFORMED_VARIANT);
  }

  constexpr size_type scale_bytes = 1;
  if (cuda::std::cmp_less(enc.size(), variant_header_bytes + scale_bytes + width)) {
    return fail(op_status::MALFORMED_VARIANT);
  }
  int const encoded_scale = enc[variant_header_bytes];
  if (encoded_scale > variant_decimal_max_scale) { return fail(op_status::MALFORMED_VARIANT); }

  auto const* unscaled_data = enc.data() + variant_header_bytes + scale_bytes;
  auto const unscaled       = [&]() -> __int128_t {
    switch (width) {
      case 4: return cudf::io::unaligned_load<int32_t>(unscaled_data);
      case 8: return cudf::io::unaligned_load<int64_t>(unscaled_data);
      case 16: return cudf::io::unaligned_load<__int128_t>(unscaled_data);
      default: return 0;  // should be unreachable
    }
  }();

  // The encoded value is `unscaled * 10^-encoded_scale`; the output is `rep * 10^desired_scale`.
  auto const shift = -static_cast<int64_t>(encoded_scale) - desired_scale;
  __int128_t rescaled{};
  if (shift >= 0) {
    auto const scaled = multiply_pow10(unscaled, shift);
    if (!scaled.has_value()) { return fail(op_status::OVERFLOW); }
    rescaled = scaled.value();
  } else {
    rescaled = divide_pow10(unscaled, -shift);
  }

  if constexpr (!cuda::std::is_same_v<Rep, __int128_t>) {
    if (rescaled < static_cast<__int128_t>(cuda::std::numeric_limits<Rep>::min()) ||
        rescaled > static_cast<__int128_t>(cuda::std::numeric_limits<Rep>::max())) {
      return fail(op_status::OVERFLOW);
    }
  }
  return {static_cast<Rep>(rescaled), op_status::SUCCESS};
}

/**
 * @brief Decides whether a row should be decoded, shared by every cast path.
 *
 * A skipped row has its null bit cleared and its status recorded here. Zeroing its output element
 * is left to the caller, since the element type is not known here.
 *
 * @return True when the row's value blob should be decoded
 */
__device__ bool should_decode_row(size_type row, bitmask_type* d_null_mask, op_status* d_status)
{
  auto const is_valid = cudf::bit_is_set(d_null_mask, row);

  if (d_status == nullptr) { return is_valid; }

  if (d_status[row] != op_status::SUCCESS) {
    if (is_valid) { cudf::clear_bit(d_null_mask, row); }
    return false;
  }
  // Status column is always non-nullable, so ROW_NULL is what reports the cleared null bit.
  if (!is_valid) {
    d_status[row] = op_status::ROW_NULL;
    return false;
  }
  return true;
}

/**
 * @brief Where one trie slot's value was found within a row's value blob.
 *
 * An unresolved slot has size `invalid_slot_size`, and the slots below it report its `status`.
 */
constexpr size_type invalid_slot_size = -1;

struct slot_result {
  size_type src_offset;
  size_type size;
  op_status status;

  [[nodiscard]] __device__ bool is_valid() const { return size != invalid_slot_size; }
};

__device__ slot_result make_slot_result(device_span<uint8_t const> field,
                                        uint8_t const* val_base,
                                        op_status status)
{
  if (field.empty()) { return {0, invalid_slot_size, status}; }
  return {
    static_cast<size_type>(field.data() - val_base), static_cast<size_type>(field.size()), status};
}

// The span a slot located, or an empty span if it did not resolve.
__device__ device_span<uint8_t const> slot_span(device_span<uint8_t const> val,
                                                slot_result const& result)
{
  if (!result.is_valid()) { return {}; }
  return val.subspan(result.src_offset, result.size);
}

// Tries up to this deep are walked with a per-thread stack in local memory. The stack is indexed
// by depth at run time, so it cannot live in registers, but local memory is per-thread and the
// hardware coalesces it across a warp. A deeper trie uses a global scratch allocation instead.
constexpr size_type max_local_trie_depth = 16;

// Global scratch is per thread, so its grid is capped to keep the allocation independent of the row
// count, and its total is capped because the trie depth is bounded only by the number of paths.
constexpr int max_global_scratch_blocks        = 256;
constexpr std::size_t max_global_scratch_bytes = std::size_t{32} * 1024 * 1024;

// Dynamic shared memory up to this size needs no opt-in on any supported architecture
constexpr std::size_t max_shared_trie_bytes = std::size_t{48} * 1024;

/**
 * @brief Resolves a whole trie of VARIANT paths in each row, recording each path's result.
 *
 * Slots are visited in pre-order keeping one located value per depth, so a shared prefix resolves
 * once per row and a failed prefix fails its whole subtree.
 *
 * Path `p`'s size and source offset for each row go to `d_sizes[p * num_rows + row]` and
 * `d_src_offsets`; null and unresolved rows get size 0 and are cleared in `d_null_masks[p]`.
 * `d_statuses`, when not empty, receives each path's per-row status.
 *
 * @tparam UseLocalScratch Keep the per-depth values in local memory rather than `d_scratch`
 */
template <bool UseLocalScratch>
CUDF_KERNEL __launch_bounds__(block_size) void locate_variant_field_trie_kernel(
  cudf::lists_column_device_view metadata,
  cudf::lists_column_device_view values,
  column_device_view steps,
  device_span<size_type const> slot_steps,
  device_span<size_type const> slot_depth,
  device_span<size_type const> output_offsets,
  device_span<size_type const> output_paths,
  bitmask_type const* d_row_valid,
  bool stage_trie,
  device_span<size_type> d_sizes,
  device_span<size_type> d_src_offsets,
  device_span<bitmask_type* const> d_null_masks,
  device_span<op_status* const> d_statuses,  // empty when no status was requested
  device_span<slot_result> d_scratch)
{
  auto const num_rows  = values.size();
  auto const num_slots = static_cast<size_type>(slot_depth.size());
  auto const num_paths = static_cast<size_type>(d_null_masks.size());
  auto const tid       = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride    = cudf::detail::grid_1d::grid_stride<block_size>();

  // The trie arrays are read for every (row, slot) pair but are the same for the whole block, so
  // they are staged in shared memory when the launch reserved room for them
  if (stage_trie) {
    extern __shared__ size_type shared_trie[];
    auto* next       = shared_trie;
    auto const stage = [&](device_span<size_type const>& span) {
      for (auto i = static_cast<std::size_t>(threadIdx.x); i < span.size(); i += block_size) {
        next[i] = span[i];
      }
      span = device_span<size_type const>{next, span.size()};
      next += span.size();
    };
    stage(slot_steps);
    stage(slot_depth);
    stage(output_offsets);
    stage(output_paths);
    __syncthreads();
  }

  // Global scratch is depth-major so that a warp's accesses at each level are coalesced
  [[maybe_unused]] cuda::std::array<slot_result, UseLocalScratch ? max_local_trie_depth : 1>
    local_stack;
  auto located = [&](size_type depth) -> slot_result& {
    if constexpr (UseLocalScratch) {
      return local_stack[depth];
    } else {
      return d_scratch[static_cast<std::size_t>(depth) * stride + tid];
    }
  };

  for (auto row = tid; row < num_rows; row += stride) {
    bool const row_valid = d_row_valid == nullptr || cudf::bit_is_set(d_row_valid, row);

    if (!row_valid) {
      for (size_type path = 0; path < num_paths; ++path) {
        auto const out     = static_cast<int64_t>(path) * num_rows + row;
        d_sizes[out]       = 0;
        d_src_offsets[out] = 0;
        cudf::clear_bit(d_null_masks[path], row);
        if (!d_statuses.empty()) { d_statuses[path][row] = op_status::ROW_NULL; }
      }
      continue;
    }

    auto const [meta, val] = metadata_and_value_at(metadata, values, row);
    cuda::std::optional<cuda::std::pair<metadata_dictionary, op_status>> dict;

    for (size_type slot = 0; slot < num_slots; ++slot) {
      auto const depth = slot_depth[slot];
      // An empty value blob resolves nothing, and a slot under one that did not resolve reports
      // its parent's reason rather than what probing an empty value would say.
      auto const parent = depth == 0 ? val : slot_span(val, located(depth - 1));
      auto const parent_status =
        depth == 0 ? (val.empty() ? op_status::MALFORMED_VARIANT : op_status::SUCCESS)
                   : located(depth - 1).status;

      auto& result = located(depth);
      if (parent.empty()) {
        result = {0, invalid_slot_size, parent_status};
      } else {
        auto const [field, status] =
          resolve_path(meta,
                       parent,
                       steps.slice(slot_steps[slot], slot_steps[slot + 1] - slot_steps[slot]),
                       dict);
        result = make_slot_result(field, val.data(), status);
      }

      for (auto out_idx = output_offsets[slot]; out_idx < output_offsets[slot + 1]; ++out_idx) {
        auto const path = output_paths[out_idx];
        auto const out  = static_cast<int64_t>(path) * num_rows + row;
        if (!d_statuses.empty()) { d_statuses[path][row] = result.status; }
        if (result.is_valid()) {
          d_sizes[out]       = result.size;
          d_src_offsets[out] = result.src_offset;
        } else {
          d_sizes[out]       = 0;
          d_src_offsets[out] = 0;
          cudf::clear_bit(d_null_masks[path], row);
        }
      }
    }
  }
}

/**
 * @brief Adds the number of valid rows in each of `d_null_masks` to `d_valid_counts`.
 *
 * Each mask's words are spread over `blocks_per_path` consecutive blocks, so that a few long masks
 * still fill the GPU. The counts must start at zero, since every block adds its share to them.
 */
CUDF_KERNEL __launch_bounds__(block_size) void count_valid_rows_kernel(
  device_span<bitmask_type* const> d_null_masks,
  size_type num_rows,
  size_type words_per_path,
  size_type blocks_per_path,
  device_span<int64_t> d_valid_counts)
{
  auto constexpr bits_per_word = static_cast<size_type>(cudf::detail::size_in_bits<bitmask_type>());
  auto const block             = static_cast<size_type>(blockIdx.x);
  auto const path              = block / blocks_per_path;
  auto const* mask             = d_null_masks[path];
  size_type num_valid          = 0;
  for (auto word_idx = (block % blocks_per_path) * block_size + static_cast<size_type>(threadIdx.x);
       word_idx < words_per_path;
       word_idx += blocks_per_path * block_size) {
    auto word = mask[word_idx];
    // `create_null_mask(ALL_VALID)` sets the whole last word, including the bits past the last row,
    // so those have to come off before they are counted as valid rows
    auto const rows_in_word = num_rows - word_idx * bits_per_word;
    if (rows_in_word < bits_per_word) { word &= cuda::bitmask<bitmask_type>(0, rows_in_word); }
    num_valid += cuda::std::popcount(word);
  }

  using block_reduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  auto const block_valid = block_reduce(temp_storage).Sum(num_valid);
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int64_t, cuda::thread_scope_device>{d_valid_counts[path]}.fetch_add(
      block_valid, cuda::memory_order_relaxed);
  }
}

// Counts the valid rows of every mask in one launch
void count_valid_rows(device_span<bitmask_type* const> d_null_masks,
                      size_type num_rows,
                      device_span<int64_t> d_valid_counts,
                      cuda::stream_ref stream)
{
  constexpr size_type words_per_thread = 8;
  auto const words_per_path            = static_cast<size_type>(num_bitmask_words(num_rows));
  auto const blocks_per_path =
    cuda::ceil_div(words_per_path, static_cast<size_type>(block_size) * words_per_thread);
  auto const num_paths = static_cast<size_type>(d_null_masks.size());
  count_valid_rows_kernel<<<num_paths * blocks_per_path, block_size, 0, stream.get()>>>(
    d_null_masks, num_rows, words_per_path, blocks_per_path, d_valid_counts);
  CUDF_CUDA_TRY(cudaGetLastError());
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
// `d_status`, when present, is an in-out buffer: it is read as incoming status from a prior
// `get_variant_field` call (rows already marked non-success are propagated without decoding), then
// overwritten in place with the final per-row status. Callers with no real incoming status must
// pre-fill every row with `op_status::SUCCESS` before calling.
template <typename T>
CUDF_KERNEL __launch_bounds__(block_size) void cast_variant_primitive_kernel(
  cudf::lists_column_device_view values,
  device_span<T> d_output,
  bitmask_type* d_null_mask,
  op_status* d_status)  // nullptr when no status was requested
{
  auto const num_rows = static_cast<size_type>(d_output.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (!should_decode_row(row, d_null_mask, d_status)) {
      d_output[row] = T{};
      continue;
    }

    auto const val     = list_row_span(values, row);
    auto const decoded = decode_primitive<T>(val);
    if (decoded.has_value()) {
      d_output[row] = *decoded;
      if (d_status != nullptr) { d_status[row] = op_status::SUCCESS; }
    } else {
      d_output[row] = T{};
      cudf::clear_bit(d_null_mask, row);
      if (d_status != nullptr) { d_status[row] = cast_status_for_primitive<T>(val); }
    }
  }
}

/**
 * @brief Per-row kernel: decode each VARIANT decimal value blob into a fixed-point representation
 * of type `Rep`, rescaled to `desired_scale`. Same null and status protocol as
 * `cast_variant_primitive_kernel`.
 */
template <typename Rep>
CUDF_KERNEL __launch_bounds__(block_size) void cast_variant_decimal_kernel(
  cudf::lists_column_device_view values,
  device_span<Rep> d_output,
  int desired_scale,
  bitmask_type* d_null_mask,
  op_status* d_status)  // nullptr when no status was requested
{
  auto const num_rows = static_cast<size_type>(d_output.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    if (!should_decode_row(row, d_null_mask, d_status)) {
      d_output[row] = Rep{};
      continue;
    }

    auto const [value, status] = decode_decimal<Rep>(list_row_span(values, row), desired_scale);
    if (status == op_status::SUCCESS) {
      d_output[row] = value;
    } else {
      d_output[row] = Rep{};
      cudf::clear_bit(d_null_mask, row);
    }
    if (d_status != nullptr) { d_status[row] = status; }
  }
}

__device__ op_status cast_status_for_bool(device_span<uint8_t const> val)
{
  if (val.empty()) { return op_status::MALFORMED_VARIANT; }
  if (is_variant_null(val)) { return op_status::VARIANT_NULL; }
  if (decode_bool(val).has_value()) { return op_status::SUCCESS; }
  if (decode_basic_type(val[0]) != basic_type::PRIMITIVE) { return op_status::TYPE_MISMATCH; }
  // Boolean values carry no payload, so a BOOLEAN_TRUE/FALSE header can never be truncated;
  // decode_bool would have succeeded above.  Any remaining primitive ID is a type mismatch when
  // recognised, or malformed when not.
  auto const ptype = static_cast<primitive_type>(variant_value_header(val[0]));
  return is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                             : op_status::MALFORMED_VARIANT;
}

__device__ op_status cast_status_for_string(device_span<uint8_t const> val)
{
  if (val.empty()) { return op_status::MALFORMED_VARIANT; }
  if (is_variant_null(val)) { return op_status::VARIANT_NULL; }
  if (decode_string(val).has_value()) { return op_status::SUCCESS; }
  auto const btype = decode_basic_type(val[0]);
  if (btype == basic_type::SHORT_STRING) { return op_status::MALFORMED_VARIANT; }
  if (btype == basic_type::PRIMITIVE) {
    auto const ptype = static_cast<primitive_type>(variant_value_header(val[0]));
    // LONG_STRING is a recognized string type whose payload was truncated.
    if (ptype == primitive_type::LONG_STRING) { return op_status::MALFORMED_VARIANT; }
    return is_recognized_primitive_type(ptype) ? op_status::TYPE_MISMATCH
                                               : op_status::MALFORMED_VARIANT;
  }
  // OBJECT, ARRAY, or other non-primitive basic types: well-formed, just not a string.
  return op_status::TYPE_MISMATCH;
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
  // In-out status tracking (optional: d_status non-null to enable; status is always non-nullable).
  // Read as incoming status from a prior `get_variant_field` call, then overwritten in place with
  // the final status on the sizing pass.
  op_status* d_status{nullptr};

  __device__ void operator()(size_type row)
  {
    // Status is only written on the sizing pass (d_chars == nullptr). On the writing pass the
    // null mask may already be cleared from the sizing pass, so we must not re-inspect it to
    // write status (that would misidentify a decode-failed row as a SQL-null row).
    bool const is_sizing_pass = (d_chars == nullptr);

    if (d_status) {
      // Status column is always non-nullable; row_null replaces the null bit.
      auto const s = d_status[row];
      if (s != op_status::SUCCESS) {
        if (is_sizing_pass) { d_sizes[row] = 0; }
        if (cudf::bit_is_set(d_null_mask, row)) { cudf::clear_bit(d_null_mask, row); }
        return;
      }
      if (!cudf::bit_is_set(d_null_mask, row)) {
        if (is_sizing_pass) {
          d_sizes[row]  = 0;
          d_status[row] = op_status::ROW_NULL;
        }
        return;
      }
    } else {
      if (!cudf::bit_is_set(d_null_mask, row)) {
        if (is_sizing_pass) { d_sizes[row] = 0; }
        return;
      }
    }

    auto const val = list_row_span(d_values, row);

    auto const str = decode_string(val);
    if (!str) {
      if (is_sizing_pass) { d_sizes[row] = 0; }
      cudf::clear_bit(d_null_mask, row);
      if (is_sizing_pass && d_status) { d_status[row] = cast_status_for_string(val); }
      return;
    }

    if (is_sizing_pass) {
      d_sizes[row] = str->size();
    } else {
      cuda::std::memcpy(d_chars + d_offsets[row], str->data(), str->size());
    }
    if (is_sizing_pass && d_status) { d_status[row] = op_status::SUCCESS; }
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

// Statuses are written one per row of the input, in place, so the buffer has to line up with it.
// `input_name` names the column the row count must match, for the error message.
void validate_status_column(std::optional<mutable_column_view> const& status,
                            size_type num_rows,
                            std::string_view input_name)
{
  if (!status.has_value()) { return; }
  CUDF_EXPECTS(!status->nullable(),
               "status column must not be nullable; use row_null for SQL-null rows",
               std::invalid_argument);
  CUDF_EXPECTS(
    status->type().id() == type_id::UINT8, "status column must be UINT8", std::invalid_argument);
  CUDF_EXPECTS(status->size() == num_rows,
               "status column must have the same number of rows as " + std::string{input_name},
               std::invalid_argument);
}

// The output types a VARIANT value can be decoded into, i.e. those `cast_variant` dispatches on.
void validate_cast_type(data_type desired_type)
{
  switch (desired_type.id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64:
    case type_id::FLOAT32:
    case type_id::FLOAT64:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS:
    case type_id::BOOL8:
    case type_id::STRING:
    case type_id::DECIMAL32:
    case type_id::DECIMAL64:
    case type_id::DECIMAL128: break;
    default: CUDF_FAIL("unsupported type for variant cast", std::invalid_argument);
  }
}

// Copies the located value of every (path, row) pair, `Index` being wide enough to count them all
template <typename Index>
void copy_located_values(cudf::lists_column_device_view const& values,
                         device_span<size_type const> d_src_offsets,
                         device_span<size_type const> d_sizes,
                         device_span<size_type* const> d_offsets,
                         device_span<uint8_t* const> d_out,
                         size_type num_rows,
                         cuda::stream_ref stream)
{
  auto const rows     = cuda::fast_mod_div<Index>{num_rows};
  auto const src_iter = cuda::make_transform_iterator(
    cuda::counting_iterator<Index>{0},
    cuda::proclaim_return_type<uint8_t const*>(
      [values, d_src = d_src_offsets.data(), rows] __device__(Index i) -> uint8_t const* {
        return values.child().template data<uint8_t>() +
               values.offset_at(static_cast<size_type>(i % rows)) + d_src[i];
      }));
  auto const dst_iter = cuda::make_transform_iterator(
    cuda::counting_iterator<Index>{0},
    cuda::proclaim_return_type<uint8_t*>(
      [d_off = d_offsets.data(), d_dst = d_out.data(), rows] __device__(Index i) -> uint8_t* {
        auto const path = i / rows;
        return d_dst[path] + d_off[path][i - path * static_cast<Index>(rows)];
      }));
  cudf::detail::batched_memcpy_async(src_iter, dst_iter, d_sizes.begin(), d_sizes.size(), stream);
}

struct cast_variant_fn {
  cudf::lists_column_device_view values;
  size_type num_rows;
  data_type desired_type;
  bitmask_type* d_null_mask;
  cuda::device_buffer<cudf::bitmask_type> null_mask{
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED)};
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;
  // In-out status tracking; null when no status was requested.
  op_status* d_status{nullptr};

  template <variant_fixed_width T>
  std::unique_ptr<column> operator()()
  {
    rmm::device_buffer data{num_rows * sizeof(T), stream, mr};
    auto const grid = cudf::detail::grid_1d{num_rows, block_size};
    auto const d_out =
      device_span<T>{static_cast<T*>(data.data()), static_cast<std::size_t>(num_rows)};
    cast_variant_primitive_kernel<T>
      <<<grid.num_blocks, block_size, 0, stream.get()>>>(values, d_out, d_null_mask, d_status);
    CUDF_CUDA_TRY(cudaGetLastError());

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0
                                      ? std::move(null_mask)
                                      : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                    null_count);
  }

  template <cudf::concepts::fixed_point T>
  std::unique_ptr<column> operator()()
  {
    using Rep = typename T::rep;
    rmm::device_buffer data{num_rows * sizeof(Rep), stream, mr};
    auto const grid = cudf::detail::grid_1d{num_rows, block_size};
    auto const d_out =
      device_span<Rep>{static_cast<Rep*>(data.data()), static_cast<std::size_t>(num_rows)};
    cast_variant_decimal_kernel<Rep><<<grid.num_blocks, block_size, 0, stream.get()>>>(
      values, d_out, desired_type.scale(), d_null_mask, d_status);
    CUDF_CUDA_TRY(cudaGetLastError());

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0
                                      ? std::move(null_mask)
                                      : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                    null_count);
  }

  template <cudf::concepts::boolean T>
  std::unique_ptr<column> operator()()
  {
    rmm::device_buffer data{num_rows * sizeof(bool), stream, mr};

    auto* dp_s = d_status;

    thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>(0),
                     cuda::counting_iterator<size_type>(num_rows),
                     [vals  = this->values,
                      d_out = static_cast<bool*>(data.data()),
                      dnm   = this->d_null_mask,
                      dp_s] __device__(size_type row) {
                       if (!should_decode_row(row, dnm, dp_s)) {
                         d_out[row] = false;
                         return;
                       }
                       // The row's null bit is still set here, so a failure can clear it unguarded.
                       auto const fail = [&](op_status s) {
                         d_out[row] = false;
                         cudf::clear_bit(dnm, row);
                         if (dp_s) { dp_s[row] = s; }
                       };
                       auto const val     = list_row_span(vals, row);
                       auto const decoded = decode_bool(val);
                       if (!decoded) { return fail(cast_status_for_bool(val)); }
                       d_out[row] = *decoded;
                       if (dp_s) { dp_s[row] = op_status::SUCCESS; }
                     });

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0
                                      ? std::move(null_mask)
                                      : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                    null_count);
  }

  template <cudf::concepts::string_view T>
  std::unique_ptr<column> operator()()
  {
    cast_variant_string_fn fn{values, d_null_mask, nullptr, nullptr, {}, d_status};
    auto [offsets_column, chars] =
      cudf::strings::detail::make_strings_children(fn, num_rows, stream, mr);

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return make_strings_column(num_rows,
                               std::move(offsets_column),
                               chars.release(),
                               null_count,
                               null_count > 0
                                 ? std::move(null_mask)
                                 : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  }

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(not variant_castable<T>)
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

  auto host_offsets = cudf::detail::make_pinned_vector_async<size_type>(depth + 1, stream);
  for (size_t i = 0; i < depth; ++i) {
    host_offsets[i + 1] = host_offsets[i] + static_cast<size_type>(steps[i].size());
  }
  auto host_chars = cudf::detail::make_pinned_vector_async<char>(host_offsets[depth], stream);
  for (size_t i = 0; i < depth; ++i) {
    std::ranges::copy(steps[i], host_chars.begin() + host_offsets[i]);
  }

  auto d_offsets = cudf::detail::make_device_uvector_async(host_offsets, stream, mr);
  auto offsets_col =
    std::make_unique<column>(data_type{type_id::INT32},
                             static_cast<size_type>(host_offsets.size()),
                             d_offsets.release(),
                             cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                             0);

  auto d_chars = cudf::detail::make_device_uvector_async(host_chars, stream, mr);
  return cudf::make_strings_column(depth,
                                   std::move(offsets_col),
                                   d_chars.release(),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

}  // namespace

namespace detail {

std::unique_ptr<table> get_variant_fields(column_view const& variant_column,
                                          std::span<std::string_view const> paths,
                                          std::span<mutable_column_view const> statuses,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources resources)
{
  auto const mr      = resources.get_output_mr();
  auto const temp_mr = resources.get_temporary_mr();

  // Validate the variant column
  CUDF_EXPECTS(variant_column.type().id() == type_id::STRUCT,
               "VARIANT column must be struct type",
               std::invalid_argument);
  CUDF_EXPECTS(variant_column.num_children() >= 2,
               "VARIANT struct must have at least two children",
               std::invalid_argument);
  validate_variant_child(variant_column.child(0));
  validate_variant_child(variant_column.child(1));

  auto const num_paths = static_cast<size_type>(paths.size());
  auto const num_rows  = variant_column.size();

  // Validate the status columns even for empty inputs
  auto const want_status = !statuses.empty();
  CUDF_EXPECTS(!want_status || statuses.size() == paths.size(),
               "status columns must be empty or one per path",
               std::invalid_argument);
  for (auto const& status : statuses) {
    validate_status_column(status, num_rows, "variant_column");
  }

  if (num_paths == 0) { return std::make_unique<table>(std::vector<std::unique_ptr<column>>{}); }

  // Validate and merge the paths even for empty input columns
  auto const trie = build_variant_path_trie(paths);

  if (num_rows == 0) {
    std::vector<std::unique_ptr<column>> output(num_paths);
    std::ranges::generate(output, [] {
      return cudf::make_lists_column(0,
                                     make_empty_column(type_id::INT32),
                                     make_empty_column(type_id::UINT8),
                                     0,
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
    });
    return std::make_unique<table>(std::move(output));
  }

  auto steps_column      = build_path_column(trie.steps, stream, temp_mr);
  auto steps_device_view = column_device_view::create(steps_column->view(), stream);
  // The trie arrays are staged with a single copy, since each transfer costs more than its contents
  auto h_trie_arrays = cudf::detail::make_pinned_vector_async<size_type>(
    trie.slot_steps.size() + trie.slot_depth.size() + trie.output_offsets.size() +
      trie.output_paths.size(),
    stream);
  auto next_host = h_trie_arrays.begin();
  for (auto const* array :
       {&trie.slot_steps, &trie.slot_depth, &trie.output_offsets, &trie.output_paths}) {
    next_host = std::ranges::copy(*array, next_host).out;
  }
  auto const d_trie_arrays =
    cudf::detail::make_device_uvector_async(h_trie_arrays, stream, temp_mr);
  auto const* next_array = d_trie_arrays.data();
  auto const next_span   = [&](std::vector<size_type> const& array) {
    auto const span = device_span<size_type const>{next_array, array.size()};
    next_array += array.size();
    return span;
  };
  auto const d_slot_steps     = next_span(trie.slot_steps);
  auto const d_slot_depth     = next_span(trie.slot_depth);
  auto const d_output_offsets = next_span(trie.output_offsets);
  auto const d_output_paths   = next_span(trie.output_paths);

  // Resolve children with respect to any slice/offset on the parent struct
  structs_column_view const variant_struct{variant_column};
  auto const meta_view = variant_struct.get_sliced_child(0, stream);
  auto const val_view  = variant_struct.get_sliced_child(1, stream);

  auto meta_device_view = column_device_view::create(meta_view, stream);
  auto val_device_view  = column_device_view::create(val_view, stream);
  cudf::lists_column_device_view meta_lists_device_view(*meta_device_view);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  // Input row validity, copied so that it is indexable by row regardless of any slice offset
  auto const row_mask =
    variant_column.nullable()
      ? cudf::detail::copy_bitmask(variant_column, stream, temp_mr)
      : cudf::detail::create_null_mask(0, mask_state::UNALLOCATED, stream, temp_mr);
  auto const* d_row_valid = reinterpret_cast<bitmask_type const*>(row_mask.data());

  // Per-path outputs are contiguous, so each path's sizes can be scanned on their own. Only each
  // path has to fit in size_type, so the (path, row) index across all of them is 64-bit.
  auto const num_outputs = static_cast<int64_t>(num_paths) * num_rows;
  cuda::device_buffer<size_type> d_sizes(stream, temp_mr, num_outputs, cuda::no_init);
  cuda::device_buffer<size_type> d_src_offsets(stream, temp_mr, num_outputs, cuda::no_init);

  // One null mask per output column, narrowed from all-valid by the walk
  std::vector<cuda::device_buffer<bitmask_type>> null_masks;
  null_masks.reserve(num_paths);
  auto h_null_masks = cudf::detail::make_pinned_vector_async<bitmask_type*>(num_paths, stream);
  for (size_type path_idx = 0; path_idx < num_paths; ++path_idx) {
    null_masks.push_back(cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr));
    h_null_masks[path_idx] = null_masks.back().data();
  }
  auto const d_null_masks = cudf::detail::make_device_uvector_async(h_null_masks, stream, temp_mr);

  // One caller-owned status buffer per path, written by the walk
  auto h_statuses =
    cudf::detail::make_pinned_vector_async<op_status*>(want_status ? num_paths : 0, stream);
  std::ranges::transform(statuses, h_statuses.begin(), [](mutable_column_view const& status) {
    return reinterpret_cast<op_status*>(status.data<uint8_t>());
  });
  auto const d_statuses = cudf::detail::make_device_uvector_async(h_statuses, stream, temp_mr);

  // Resolve the whole trie per row and compute the output sizes. The walk keeps one located value
  // per trie level, so only a pathologically deep trie needs scratch outside the thread.
  auto const trie_depth = 1 + *std::max_element(trie.slot_depth.begin(), trie.slot_depth.end());
  bool const use_local_scratch = trie_depth <= max_local_trie_depth;
  auto const grid              = cudf::detail::grid_1d{num_rows, block_size};
  // Blocks that fit in the scratch budget, at least one however deep the trie is
  auto const scratch_per_block =
    static_cast<std::size_t>(block_size) * trie_depth * sizeof(slot_result);
  auto const scratch_blocks = std::max(std::size_t{1},
                                       std::min(static_cast<std::size_t>(max_global_scratch_blocks),
                                                max_global_scratch_bytes / scratch_per_block));
  auto const num_blocks =
    use_local_scratch ? grid.num_blocks
                      : std::min(grid.num_blocks, static_cast<thread_index_type>(scratch_blocks));
  auto const scratch_size =
    use_local_scratch ? 0 : static_cast<std::size_t>(num_blocks) * block_size * trie_depth;
  cuda::device_buffer<slot_result> d_scratch(stream, temp_mr, scratch_size, cuda::no_init);

  // Room for the trie arrays in shared memory, kept within the default per-block limit so that the
  // launch needs no opt-in; a trie too large for it is read from global memory instead
  auto const trie_bytes = sizeof(size_type) * (d_slot_steps.size() + d_slot_depth.size() +
                                               d_output_offsets.size() + d_output_paths.size());
  bool const stage_trie = trie_bytes <= max_shared_trie_bytes;
  auto const launch     = [&](auto use_local) {
    // Reserving any shared memory otherwise lets the driver shrink L1, which the metadata lookups
    // depend on; at many paths that costs more than staging the trie saves
    CUDF_CUDA_TRY(cudaFuncSetAttribute(locate_variant_field_trie_kernel<decltype(use_local)::value>,
                                       cudaFuncAttributePreferredSharedMemoryCarveout,
                                       cudaSharedmemCarveoutMaxL1));
    locate_variant_field_trie_kernel<decltype(use_local)::value>
      <<<num_blocks, block_size, stage_trie ? trie_bytes : 0, stream.get()>>>(
        meta_lists_device_view,
        val_lists_device_view,
        *steps_device_view,
        d_slot_steps,
        d_slot_depth,
        d_output_offsets,
        d_output_paths,
        d_row_valid,
        stage_trie,
        d_sizes,
        d_src_offsets,
        d_null_masks,
        d_statuses,
        d_scratch);
  };
  if (use_local_scratch) {
    launch(cuda::std::true_type{});
  } else {
    launch(cuda::std::false_type{});
  }
  CUDF_CUDA_TRY(cudaGetLastError());

  // Allocate each path's output offsets column; the copy below finds them through `d_offsets`
  std::vector<std::unique_ptr<column>> offsets_columns;
  offsets_columns.reserve(num_paths);
  auto h_offsets = cudf::detail::make_pinned_vector_async<size_type*>(num_paths, stream);
  for (size_type path_idx = 0; path_idx < num_paths; ++path_idx) {
    auto offsets_column = make_numeric_column(
      data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
    h_offsets[path_idx] = offsets_column->mutable_view().data<size_type>();
    offsets_columns.push_back(std::move(offsets_column));
  }
  auto const d_offsets = cudf::detail::make_device_uvector_async(h_offsets, stream, temp_mr);

  // Byte totals and valid counts share a buffer, since each extra readback costs a synchronization
  auto d_path_counts = cudf::detail::make_zeroed_device_uvector_async<int64_t>(
    2 * static_cast<std::size_t>(num_paths), stream, temp_mr);
  auto const d_byte_totals  = device_span<int64_t>{d_path_counts}.first(num_paths);
  auto const d_valid_counts = device_span<int64_t>{d_path_counts}.last(num_paths);

  // One scan per path; `exclusive_scan_by_key` was slower once rows outnumber paths
  for (size_type path_idx = 0; path_idx < num_paths; ++path_idx) {
    // The scan takes one input per output; the extra trailing input is never read for its own
    // output element.
    auto const path_sizes = cudf::detail::make_counting_transform_iterator(
      size_type{0},
      cuda::proclaim_return_type<size_type>(
        [sizes = d_sizes.data() + static_cast<int64_t>(path_idx) * num_rows, num_rows] __device__(
          size_type row) -> size_type { return row < num_rows ? sizes[row] : 0; }));
    // Offsets are narrowed to size_type on the way out while the scan accumulates in 64 bits, so
    // the total still reveals an overflow that the offsets themselves would have wrapped away.
    auto const offsets_out = cudf::detail::make_sizes_to_offsets_iterator(
      h_offsets[path_idx], h_offsets[path_idx] + num_rows + 1, d_byte_totals.data() + path_idx);
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                           path_sizes,
                           path_sizes + num_rows + 1,
                           offsets_out,
                           int64_t{0});
  }

  count_valid_rows(d_null_masks, num_rows, d_valid_counts, stream);

  auto const path_counts  = cudf::detail::make_pinned_vector(d_path_counts, stream);
  auto const byte_totals  = host_span<int64_t const>{path_counts}.first(num_paths);
  auto const valid_counts = host_span<int64_t const>{path_counts}.last(num_paths);

  // Allocate each path's output bytes now that its total is known
  std::vector<std::unique_ptr<column>> value_children;
  value_children.reserve(num_paths);
  auto h_out              = cudf::detail::make_pinned_vector_async<uint8_t*>(num_paths, stream);
  int64_t all_paths_bytes = 0;
  for (size_type path_idx = 0; path_idx < num_paths; ++path_idx) {
    auto const total_bytes = byte_totals[path_idx];
    CUDF_EXPECTS(total_bytes <= std::numeric_limits<size_type>::max(),
                 "VARIANT extracted bytes exceed cudf size_type limit",
                 std::overflow_error);
    auto value_child = make_numeric_column(data_type{type_id::UINT8},
                                           static_cast<size_type>(total_bytes),
                                           mask_state::UNALLOCATED,
                                           stream,
                                           mr);
    h_out[path_idx]  = value_child->mutable_view().data<uint8_t>();
    all_paths_bytes += total_bytes;
    value_children.push_back(std::move(value_child));
  }

  // Copy the located values of every (path, row) pair in one pass
  if (all_paths_bytes > 0) {
    auto const d_out = cudf::detail::make_device_uvector_async(h_out, stream, temp_mr);
    // 64-bit division costs the copy a few percent, so it is only paid when the index needs it
    if (num_outputs <= std::numeric_limits<size_type>::max()) {
      copy_located_values<size_type>(
        val_lists_device_view, d_src_offsets, d_sizes, d_offsets, d_out, num_rows, stream);
    } else {
      copy_located_values<int64_t>(
        val_lists_device_view, d_src_offsets, d_sizes, d_offsets, d_out, num_rows, stream);
    }
  }

  std::vector<std::unique_ptr<column>> output;
  output.reserve(num_paths);
  for (size_type path_idx = 0; path_idx < num_paths; ++path_idx) {
    auto const null_count = num_rows - static_cast<size_type>(valid_counts[path_idx]);
    output.push_back(make_lists_column(
      num_rows,
      std::move(offsets_columns[path_idx]),
      std::move(value_children[path_idx]),
      null_count,
      null_count > 0 ? std::move(null_masks[path_idx])
                     : cudf::detail::create_null_mask(0, mask_state::UNALLOCATED, stream, mr)));
  }

  return std::make_unique<table>(std::move(output));
}

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          std::optional<mutable_column_view> status,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  std::array const paths{path};
  std::vector<mutable_column_view> statuses;
  if (status.has_value()) { statuses.push_back(*status); }
  return std::move(
    get_variant_fields(variant_column, paths, statuses, stream, mr)->release().front());
}

std::unique_ptr<column> cast_variant(column_view const& values,
                                     data_type desired_type,
                                     std::optional<mutable_column_view> status,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  validate_variant_child(values);
  validate_cast_type(desired_type);

  size_type const num_rows = values.size();

  // Validate status before the empty-values fast path so callers always get
  // std::invalid_argument for a malformed status column, even when values is empty.
  validate_status_column(status, num_rows, "the values column");

  if (num_rows == 0) { return make_empty_column(desired_type); }

  auto val_device_view = column_device_view::create(values, stream);
  cudf::lists_column_device_view val_lists_device_view(*val_device_view);

  auto null_mask    = values.nullable()
                        ? cudf::detail::copy_bitmask(values, stream, mr)
                        : cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = null_mask.data();

  return cudf::type_dispatcher(
    desired_type,
    cast_variant_fn{
      val_lists_device_view,
      num_rows,
      desired_type,
      d_null_mask,
      std::move(null_mask),
      stream,
      mr,
      status.has_value() ? reinterpret_cast<op_status*>(status->data<uint8_t>()) : nullptr});
}

std::unique_ptr<table> extract_variant_fields(column_view const& variant_column,
                                              std::span<std::string_view const> paths,
                                              std::span<data_type const> desired_types,
                                              std::span<mutable_column_view const> statuses,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources resources)
{
  CUDF_EXPECTS(paths.size() == desired_types.size(),
               "VARIANT paths and desired types must have the same size",
               std::invalid_argument);

  for (auto const desired_type : desired_types) {
    validate_cast_type(desired_type);
  }

  auto const want_status = !statuses.empty();
  // Each path's status is filled by the walk, then read back by that path's cast as incoming status
  // and overwritten in place with the final status.
  auto const values =
    get_variant_fields(variant_column, paths, statuses, stream, resources.as_temporary());

  std::vector<std::unique_ptr<column>> output;
  output.reserve(paths.size());
  for (size_type path_idx = 0; path_idx < values->num_columns(); ++path_idx) {
    output.push_back(cast_variant(values->get_column(path_idx).view(),
                                  desired_types[path_idx],
                                  want_status ? std::optional{statuses[path_idx]} : std::nullopt,
                                  stream,
                                  resources.get_output_mr()));
  }
  return std::make_unique<table>(std::move(output));
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
  auto* d_null_mask = null_mask.data();

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
                                  null_count > 0
                                    ? std::move(null_mask)
                                    : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                  null_count);
}

}  // namespace detail

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          std::optional<mutable_column_view> status,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_variant_field(variant_column, path, status, stream, mr);
}

std::unique_ptr<column> cast_variant(column_view const& values,
                                     data_type desired_type,
                                     std::optional<mutable_column_view> status,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::cast_variant(values, desired_type, status, stream, mr);
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
                                              std::optional<mutable_column_view> status,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const temp_mr = cudf::get_current_device_resource_ref();

  if (status.has_value()) {
    // `status` is filled by `get_variant_field`, then read back by `cast_variant` as incoming
    // status and overwritten in place with the final per-row status.
    auto value = detail::get_variant_field(variant_column, path, status, stream, temp_mr);
    return detail::cast_variant(value->view(), desired_type, status, stream, mr);
  }

  auto value = detail::get_variant_field(variant_column, path, std::nullopt, stream, temp_mr);
  return detail::cast_variant(value->view(), desired_type, std::nullopt, stream, mr);
}

std::unique_ptr<table> get_variant_fields(column_view const& variant_column,
                                          std::span<std::string_view const> paths,
                                          std::span<mutable_column_view const> statuses,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_variant_fields(variant_column, paths, statuses, stream, mr);
}

std::unique_ptr<table> extract_variant_fields(column_view const& variant_column,
                                              std::span<std::string_view const> paths,
                                              std::span<data_type const> desired_types,
                                              std::span<mutable_column_view const> statuses,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_variant_fields(variant_column, paths, desired_types, statuses, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
