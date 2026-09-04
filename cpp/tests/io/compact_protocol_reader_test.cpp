/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/testing_main.hpp>

#include <src/io/parquet/compact_protocol_reader.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// CompactProtocolReader::get_varint on crafted byte runs, decoded host-side: an overlong varint
// (value too large for the target type) throws std::overflow_error; a within-width unterminated run
// terminates cleanly at EOF; well-formed and empty inputs decode to defined values.
TEST(CompactProtocolReaderVarintTest, OverlongU32)
{
  // A seven-byte run whose value exceeds uint32_t: the value-bound check rejects the fifth byte
  // 0x85 at shift 28 (0x85 > max() >> 28 == 0x0F), so get_varint throws rather than truncating.
  std::vector<uint8_t> const bytes{0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x07};
  cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
  EXPECT_THROW(cp.get_u32(), std::overflow_error);
}

TEST(CompactProtocolReaderVarintTest, OverlongU64)
{
  // A value exceeding uint64_t: the value-bound check rejects the tenth byte 0x8B at shift 63
  // (0x8B > max() >> 63 == 0x01), so get_varint throws rather than truncating.
  std::vector<uint8_t> const bytes{
    0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8B, 0x8A, 0x00};
  cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
  EXPECT_THROW(cp.get_u64(), std::overflow_error);
}

TEST(CompactProtocolReaderVarintTest, OverflowAtWidthBoundaryU32)
{
  // Terminating fifth group at shift 28: the shift is within uint32_t's 32 bits, but 0x10 << 28
  // sets bit 32, so the value overflows the target type. The shift-count check alone would truncate
  // it to 0; the value check rejects it.
  std::vector<uint8_t> const bytes{0x80, 0x80, 0x80, 0x80, 0x10};
  cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
  EXPECT_THROW(cp.get_u32(), std::overflow_error);
}

TEST(CompactProtocolReaderVarintTest, OverflowAtWidthBoundaryU64)
{
  // Terminating tenth group at shift 63: the shift is within uint64_t's 64 bits, but 0x02 << 63
  // sets bit 64, so the value overflows the target type and get_varint rejects it.
  std::vector<uint8_t> const bytes{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02};
  cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
  EXPECT_THROW(cp.get_u64(), std::overflow_error);
}

TEST(CompactProtocolReaderVarintTest, OverlongSignedThrows)
{
  // get_i32/get_i64 forward through get_zigzag to get_varint<U>, so they now propagate
  // std::overflow_error on an overlong varint instead of being noexcept. Pin that contract with the
  // same boundary-overflow byte runs the unsigned tests use.
  {
    std::vector<uint8_t> const bytes{0x80, 0x80, 0x80, 0x80, 0x10};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_THROW(cp.get_i32(), std::overflow_error);
  }
  {
    std::vector<uint8_t> const bytes{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_THROW(cp.get_i64(), std::overflow_error);
  }
}

TEST(CompactProtocolReaderVarintTest, UnterminatedRunAtEof)
{
  // Continuation bytes running to end-of-buffer with no terminator: getb() yields 0 at EOF, which
  // ends the loop. A run that stays within the target width returns a defined value...
  {
    // u32: three 0x7F groups (shifts 0/7/14) then EOF -> 0x1FFFFF, all three bytes consumed.
    std::vector<uint8_t> const bytes(3, 0xFF);
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u32(), 0x1F'FFFFu);
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
  {
    // u64: five 0x7F groups (shifts 0..28) then EOF -> 0x7FFFFFFFF.
    std::vector<uint8_t> const bytes(5, 0xFF);
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u64(), 0x7'FFFF'FFFFUL);
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
  {
    // ...but a run spanning past the width is overlong and throws, terminator or not.
    std::vector<uint8_t> const bytes(6, 0xFF);
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_THROW(cp.get_u32(), std::overflow_error);
  }
}

TEST(CompactProtocolReaderVarintTest, WellFormedValues)
{
  {
    // Back-to-back single-byte varints and the classic two-byte encoding of 300:
    // 0x2C | 0x02<<7 = 300.
    std::vector<uint8_t> const bytes{0x00, 0x7F, 0xAC, 0x02};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u32(), 0u);
    EXPECT_EQ(cp.get_u32(), 127u);
    EXPECT_EQ(cp.get_u32(), 300u);
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
  {
    // Maximum five-byte u32 reaching shift 28: 0x7F | 0x7F<<7 | 0x7F<<14 | 0x7F<<21 | 0x0F<<28.
    std::vector<uint8_t> const bytes{0xFF, 0xFF, 0xFF, 0xFF, 0x0F};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u32(), std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
  {
    // Nine-byte u64: payload groups 0x6F,0x1B,0x2F,0x4D,0x78,0x2C,0x51,0x11,0x01 at shifts
    // 0,7,...,56 reassemble 0x0123456789ABCDEF.
    std::vector<uint8_t> const bytes{0xEF, 0x9B, 0xAF, 0xCD, 0xF8, 0xAC, 0xD1, 0x91, 0x01};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u64(), 0x0123'4567'89AB'CDEFUL);
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
  {
    // Maximum ten-byte u64 reaching shift 63: nine 0x7F groups then 0x01<<63.
    std::vector<uint8_t> const bytes{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u64(), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
}

TEST(CompactProtocolReaderVarintTest, EmptyBuffer)
{
  // Zero-length input: getb() returns 0 without dereferencing past m_end, so the loop terminates
  // on its first iteration with a defined, zero value.
  std::vector<uint8_t> const bytes{};
  cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
  EXPECT_EQ(cp.get_u32(), 0u);
  EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
}

TEST(CompactProtocolReaderVarintTest, NullBufferWithNonZeroLengthThrows)
{
  // A null buffer with a positive length has no backing storage; construction rejects it rather
  // than leaving the reader positioned to read past a null base.
  EXPECT_THROW(cudf::io::parquet::detail::CompactProtocolReader(nullptr, 4), std::invalid_argument);
}

TEST(CompactProtocolReaderVarintTest, NullBufferZeroLengthIsDefinedEmpty)
{
  // A null buffer with zero length is valid and yields the fully-defined empty state: get_varint
  // returns 0 without reading past the null base. Constructed with an explicit nullptr so the
  // null-base branch of init() is covered regardless of std::vector::data()'s empty-buffer value.
  cudf::io::parquet::detail::CompactProtocolReader cp(nullptr, 0);
  EXPECT_EQ(cp.get_u32(), 0u);
  EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(0));
}

CUDF_TEST_PROGRAM_MAIN()
