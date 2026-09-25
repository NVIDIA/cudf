/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/protobuf/types.cuh"

#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/atomic>
#include <cuda/std/limits>

namespace cudf::io::protobuf::detail {

// ============================================================================
// Device helper functions
// ============================================================================

__device__ inline bool read_varint(uint8_t const* cur,
                                   uint8_t const* end,
                                   uint64_t& out,
                                   int& bytes)
{
  out       = 0;
  bytes     = 0;
  int shift = 0;
  // Protobuf varint uses 7 bits per byte with MSB as continuation flag.
  // A 64-bit value requires at most ceil(64/7) = 10 bytes.
  while (cur < end && bytes < MAX_VARINT_BYTES) {
    uint8_t b = *cur++;
    // For the 10th byte (bytes == 9, shift == 63), only the lowest bit is valid
    if (bytes == 9 && (b & 0xFE) != 0) {
      return false;  // Invalid: 10th byte has more than 1 significant bit
    }
    out |= (static_cast<uint64_t>(b & 0x7Fu) << shift);
    bytes++;
    if ((b & 0x80u) == 0) { return true; }
    shift += 7;
  }
  return false;
}

__device__ inline void set_error_once(int* error_flag, proto_decode_error error)
{
  int expected = 0;
  cuda::atomic_ref<int, cuda::thread_scope_device> ref(*error_flag);
  ref.compare_exchange_strong(expected, error_code(error), cuda::memory_order_relaxed);
}

void set_error_once_async(int* error_flag, proto_decode_error error, rmm::cuda_stream_view stream);

__device__ inline int get_wire_type_size(int wt, uint8_t const* cur, uint8_t const* end)
{
  switch (wt) {
    case wire_type_value(proto_wire_type::VARINT): {
      // Need to scan to find the end of varint
      int count = 0;
      while (cur < end && count < MAX_VARINT_BYTES) {
        if ((*cur++ & 0x80u) == 0) { return count + 1; }
        count++;
      }
      return -1;  // Invalid varint
    }
    case wire_type_value(proto_wire_type::I64BIT):
      // Check if there's enough data for 8 bytes
      if (end - cur < 8) return -1;
      return 8;
    case wire_type_value(proto_wire_type::I32BIT):
      // Check if there's enough data for 4 bytes
      if (end - cur < 4) return -1;
      return 4;
    case wire_type_value(proto_wire_type::LEN): {
      uint64_t len;
      int n;
      if (!read_varint(cur, end, len, n)) return -1;
      if (len > static_cast<uint64_t>(end - cur - n) ||
          len > static_cast<uint64_t>(cuda::std::numeric_limits<int>::max() - n)) {
        return -1;
      }
      return n + static_cast<int>(len);
    }
    case wire_type_value(proto_wire_type::SGROUP): {
      auto const* start = cur;
      int depth         = 1;
      while (cur < end && depth > 0) {
        uint64_t key;
        int key_bytes;
        if (!read_varint(cur, end, key, key_bytes)) return -1;
        cur += key_bytes;

        int inner_wt = static_cast<int>(key & 0x7);
        if (inner_wt == wire_type_value(proto_wire_type::EGROUP)) {
          --depth;
          if (depth == 0) { return static_cast<int>(cur - start); }
        } else if (inner_wt == wire_type_value(proto_wire_type::SGROUP)) {
          if (++depth > 32) return -1;
        } else {
          int inner_size = -1;
          switch (inner_wt) {
            case wire_type_value(proto_wire_type::VARINT): {
              uint64_t dummy;
              int vbytes;
              if (!read_varint(cur, end, dummy, vbytes)) return -1;
              inner_size = vbytes;
              break;
            }
            case wire_type_value(proto_wire_type::I64BIT): inner_size = 8; break;
            case wire_type_value(proto_wire_type::LEN): {
              uint64_t len;
              int len_bytes;
              if (!read_varint(cur, end, len, len_bytes)) return -1;
              if (len > static_cast<uint64_t>(cuda::std::numeric_limits<int>::max() - len_bytes)) {
                return -1;
              }
              inner_size = len_bytes + static_cast<int>(len);
              break;
            }
            case wire_type_value(proto_wire_type::I32BIT): inner_size = 4; break;
            default: return -1;
          }
          if (inner_size < 0 || cur + inner_size > end) return -1;
          cur += inner_size;
        }
      }
      return -1;
    }
    case wire_type_value(proto_wire_type::EGROUP): return 0;
    default: return -1;
  }
}

__device__ inline bool skip_field(uint8_t const* cur,
                                  uint8_t const* end,
                                  int wt,
                                  uint8_t const*& out_cur)
{
  // A bare end-group is only valid while a start-group payload is being parsed recursively inside
  // get_wire_type_size(wire_type_value(proto_wire_type::SGROUP)).
  // The scan/count kernels should never accept it as a standalone field because Spark CPU treats
  // unmatched end-groups as malformed protobuf.
  if (wt == wire_type_value(proto_wire_type::EGROUP)) { return false; }

  int size = get_wire_type_size(wt, cur, end);
  if (size < 0) return false;
  // Ensure we don't skip past the end of the buffer
  if (cur + size > end) return false;
  out_cur = cur + size;
  return true;
}

/**
 * Get the data offset and length for a field at current position.
 * Returns true on success, false on error.
 */
__device__ inline bool get_field_data_location(
  uint8_t const* cur, uint8_t const* end, int wt, int32_t& data_offset, int32_t& data_length)
{
  if (wt == wire_type_value(proto_wire_type::LEN)) {
    // For length-delimited, read the length prefix
    uint64_t len;
    int len_bytes;
    if (!read_varint(cur, end, len, len_bytes)) return false;
    if (len > static_cast<uint64_t>(end - cur - len_bytes) ||
        len > static_cast<uint64_t>(cuda::std::numeric_limits<int>::max())) {
      return false;
    }
    data_offset = len_bytes;  // offset past the length prefix
    data_length = static_cast<int32_t>(len);
  } else {
    // For fixed-size and varint fields
    int field_size = get_wire_type_size(wt, cur, end);
    if (field_size < 0) return false;
    data_offset = 0;
    data_length = field_size;
  }
  return true;
}

CUDF_HOST_DEVICE inline size_t flat_index(size_t row, size_t width, size_t col)
{
  return row * width + col;
}

__device__ inline bool checked_add_int32(int32_t lhs, int32_t rhs, int32_t& out)
{
  auto const sum = static_cast<int64_t>(lhs) + rhs;
  if (sum < cuda::std::numeric_limits<int32_t>::min() ||
      sum > cuda::std::numeric_limits<int32_t>::max()) {
    return false;
  }
  out = static_cast<int32_t>(sum);
  return true;
}

__device__ inline bool check_message_bounds(int32_t start,
                                            int32_t end_pos,
                                            cudf::size_type total_size,
                                            int* error_flag)
{
  if (start < 0 || end_pos < start || end_pos > total_size) {
    set_error_once(error_flag, proto_decode_error::BOUNDS);
    return false;
  }
  return true;
}

struct proto_tag {
  int field_number;
  int wire_type;
};

__device__ inline bool decode_tag(uint8_t const*& cur,
                                  uint8_t const* end,
                                  proto_tag& tag,
                                  int* error_flag)
{
  uint64_t key;
  int key_bytes;
  if (!read_varint(cur, end, key, key_bytes)) {
    set_error_once(error_flag, proto_decode_error::VARINT);
    return false;
  }

  cur += key_bytes;
  uint64_t fn = key >> 3;
  if (fn == 0 || fn > static_cast<uint64_t>(max_field_number)) {
    set_error_once(error_flag, proto_decode_error::FIELD_NUMBER);
    return false;
  }
  tag.field_number = static_cast<int>(fn);
  tag.wire_type    = static_cast<int>(key & 0x7);
  return true;
}

/**
 * Load a little-endian value from unaligned memory.
 * Reads bytes individually to avoid unaligned-access issues on GPU.
 */
template <typename T>
__device__ inline T load_le(uint8_t const* p);

template <>
__device__ inline uint32_t load_le<uint32_t>(uint8_t const* p)
{
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

template <>
__device__ inline uint64_t load_le<uint64_t>(uint8_t const* p)
{
  uint64_t v = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v |= (static_cast<uint64_t>(p[i]) << (8 * i));
  }
  return v;
}

/**
 * O(1) lookup of field_number -> field_index using a direct-mapped table.
 * Falls back to linear search when the table is empty (field numbers too large).
 */
// Keep this definition in the header so all CUDA translation units can inline it.
__device__ __forceinline__ int lookup_field(int field_number,
                                            int const* lookup_table,
                                            int lookup_table_size,
                                            field_descriptor const* field_descs,
                                            int num_fields)
{
  if (lookup_table != nullptr && field_number > 0 && field_number < lookup_table_size) {
    return lookup_table[field_number];
  }
  for (int f = 0; f < num_fields; f++) {
    if (field_descs[f].field_number == field_number) return f;
  }
  return -1;
}

}  // namespace cudf::io::protobuf::detail
