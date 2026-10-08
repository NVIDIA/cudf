/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/utilities/column_type_histogram.hpp"

#include <cuda/std/limits>

#include <cstdint>

namespace cudf {
namespace io {
namespace csv {
// Offsets relative to the input replace pointers so inputs below the compact null sentinel can
// stage 8-byte entries; larger inputs use the same layout with 64-bit offsets.
template <typename OffsetT>
struct string_offset_pair {
  using offset_type                        = OffsetT;
  static constexpr offset_type null_offset = cuda::std::numeric_limits<offset_type>::max();

  offset_type offset{null_offset};
  uint32_t length{0};
};

using compact_string_offset_pair = string_offset_pair<uint32_t>;
using wide_string_offset_pair    = string_offset_pair<uint64_t>;

static_assert(sizeof(compact_string_offset_pair) == 8);

// Offsets are at most data_size (an empty trailing field), so the sentinel stays unreachable.
[[nodiscard]] constexpr bool use_compact_string_offsets(size_t data_size)
{
  return data_size < compact_string_offset_pair::null_offset;
}

namespace column_parse {
/**
 * @brief Per-column parsing flags used for dtype detection and data conversion
 */
enum : uint8_t {
  disabled       = 0,   ///< data is not read
  enabled        = 1,   ///< data is read and parsed as usual
  inferred       = 2,   ///< infer the dtype
  as_default     = 4,   ///< no special decoding
  as_hexadecimal = 8,   ///< decode with base-16
  as_datetime    = 16,  ///< decode as date and/or time
};
using flags = uint8_t;

}  // namespace column_parse

}  // namespace csv
}  // namespace io
}  // namespace cudf
