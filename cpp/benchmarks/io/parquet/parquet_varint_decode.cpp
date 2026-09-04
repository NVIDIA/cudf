/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Microbenchmark for CompactProtocolReader varint decoding (get_varint via get_u32/get_u64), the
// hot loop that parses Parquet Thrift-compact metadata. Each axis point fills a host buffer with
// fixed-width unsigned LEB128 varints and times decoding them, so throughput shows how decode cost
// scales with encoded length while exercising the overflow guard on each byte. Only valid varints
// are measured; the overlong-input throw path is covered by the unit tests.

#include "io/parquet/compact_protocol_reader.hpp"

#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

using cudf::io::parquet::detail::CompactProtocolReader;

// Varints decoded per timed sample; large enough to amortize per-sample timing overhead.
constexpr std::size_t num_varints = 1 << 20;

// Append `value` to `buf` as an unsigned LEB128 varint (7 payload bits per byte, LSB first), the
// encoding CompactProtocolReader::get_varint decodes.
void append_uleb128(std::vector<uint8_t>& buf, uint64_t value)
{
  while (value >= 0x80) {
    buf.push_back(static_cast<uint8_t>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  buf.push_back(static_cast<uint8_t>(value));
}

// Largest value of `T` that still encodes in exactly `num_bytes` LEB128 bytes, so every varint has
// the requested width and its final byte lands on the overflow guard's boundary.
template <typename T>
T max_value_for_width(int num_bytes)
{
  constexpr int digits = std::numeric_limits<T>::digits;
  return (7 * num_bytes >= digits) ? std::numeric_limits<T>::max()
                                   : static_cast<T>((T{1} << (7 * num_bytes)) - 1);
}

// Buffer of `num_varints` copies of `value`, each encoded in `num_bytes` LEB128 bytes.
std::vector<uint8_t> make_varint_buffer(uint64_t value, int num_bytes)
{
  std::vector<uint8_t> buf;
  buf.reserve(num_varints * static_cast<std::size_t>(num_bytes));
  for (std::size_t i = 0; i < num_varints; ++i) {
    append_uleb128(buf, value);
  }
  return buf;
}

}  // namespace

// Decode `num_varints` unsigned varints of a fixed byte width per sample, timing only the decode
// loop. `T` selects the width; the two `BM_*` entry points below instantiate it for u32 and u64.
template <typename T>
void bm_parquet_varint_decode(nvbench::state& state)
{
  auto const num_bytes = static_cast<int>(state.get_int64("varint_bytes"));
  auto const value     = max_value_for_width<T>(num_bytes);
  auto const buffer    = make_varint_buffer(value, num_bytes);
  auto const expected  = static_cast<uint64_t>(value) * num_varints;

  state.add_element_count(num_varints, "varints");
  state.exec(nvbench::exec_tag::no_gpu | nvbench::exec_tag::timer,
             [&](nvbench::launch&, auto& timer) {
               CompactProtocolReader cp(buffer.data(), buffer.size());
               timer.start();
               uint64_t checksum = 0;
               for (std::size_t i = 0; i < num_varints; ++i) {
                 if constexpr (std::is_same_v<T, uint32_t>) {
                   checksum += cp.get_u32();
                 } else {
                   checksum += cp.get_u64();
                 }
               }
               timer.stop();
               // Consume `checksum` to block dead-code elimination and validate the decode.
               CUDF_EXPECTS(checksum == expected, "Unexpected decoded varint sum");
             });
}

void BM_parquet_varint_decode_u32(nvbench::state& state)
{
  bm_parquet_varint_decode<uint32_t>(state);
}

void BM_parquet_varint_decode_u64(nvbench::state& state)
{
  bm_parquet_varint_decode<uint64_t>(state);
}

// u32 varints span 1..5 encoded bytes (32 bits at 7 payload bits per byte, rounded up).
NVBENCH_BENCH(BM_parquet_varint_decode_u32)
  .set_name("parquet_varint_decode_u32")
  .set_is_cpu_only(true)
  .add_int64_axis("varint_bytes", {1, 2, 3, 4, 5});

// u64 varints span 1..10 encoded bytes (64 bits at 7 payload bits per byte, rounded up).
NVBENCH_BENCH(BM_parquet_varint_decode_u64)
  .set_name("parquet_varint_decode_u64")
  .set_is_cpu_only(true)
  .add_int64_axis("varint_bytes", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
