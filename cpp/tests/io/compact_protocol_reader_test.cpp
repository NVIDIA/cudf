/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <src/io/parquet/compact_protocol_reader.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

TEST(CompactProtocolReaderVarintTest, OverflowAtWidthBoundaryU32)
{
  // Fifth group at shift 28: 0x10 << 28 sets bit 32, overflowing uint32_t.
  std::vector<uint8_t> const bytes{0x80, 0x80, 0x80, 0x80, 0x10};
  cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
  EXPECT_THROW(cp.get_u32(), std::overflow_error);
}
TEST(CompactProtocolReaderVarintTest, OverlongSignedThrows)
{
  // get_i32 forwards through get_zigzag to get_varint<U> and propagates overflow_error.
  std::vector<uint8_t> const bytes{0x80, 0x80, 0x80, 0x80, 0x10};
  cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
  EXPECT_THROW(cp.get_i32(), std::overflow_error);
}

TEST(CompactProtocolReaderVarintTest, WellFormedListHeader)
{
  // Inline size (0x3C: type 0xC, size 3) and escaped size both return {type, size}.
  {
    std::vector<uint8_t> const bytes{0x3C};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_listh(), (std::pair<uint8_t, uint32_t>{0xC, 3u}));
  }
  {
    std::vector<uint8_t> const bytes{0xF0, 0xAC, 0x02};  // escaped size 300
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_listh(), (std::pair<uint8_t, uint32_t>{0x0, 300u}));
  }
}

TEST(CompactProtocolReaderVarintTest, UnterminatedRunAtEof)
{
  // getb() yields 0 at EOF, ending the loop. A within-width run returns a defined value.
  {
    std::vector<uint8_t> const bytes(3, 0xFF);  // shifts 0/7/14 -> 0x1FFFFF
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u32(), 0x1F'FFFFu);
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
  {
    std::vector<uint8_t> const bytes(5, 0xFF);  // shifts 0..28 -> 0x7FFFFFFFF
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u64(), 0x7'FFFF'FFFFUL);
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
}

TEST(CompactProtocolReaderVarintTest, WellFormedValues)
{
  {
    // 0x2C | 0x02<<7 = 300.
    std::vector<uint8_t> const bytes{0x00, 0x7F, 0xAC, 0x02};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u32(), 0u);
    EXPECT_EQ(cp.get_u32(), 127u);
    EXPECT_EQ(cp.get_u32(), 300u);
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
  {
    // Maximum five-byte u32: 0x7F | 0x7F<<7 | 0x7F<<14 | 0x7F<<21 | 0x0F<<28.
    std::vector<uint8_t> const bytes{0xFF, 0xFF, 0xFF, 0xFF, 0x0F};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u32(), std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
  {
    // Nine-byte u64 reassembling 0x0123456789ABCDEF.
    std::vector<uint8_t> const bytes{0xEF, 0x9B, 0xAF, 0xCD, 0xF8, 0xAC, 0xD1, 0x91, 0x01};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u64(), 0x0123'4567'89AB'CDEFUL);
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
  {
    // Maximum ten-byte u64: nine 0x7F groups then 0x01<<63.
    std::vector<uint8_t> const bytes{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01};
    cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
    EXPECT_EQ(cp.get_u64(), std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
  }
}

TEST(CompactProtocolReaderVarintTest, EmptyBuffer)
{
  std::vector<uint8_t> const bytes{};
  cudf::io::parquet::detail::CompactProtocolReader cp(bytes.data(), bytes.size());
  EXPECT_EQ(cp.get_u32(), 0u);
  EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(bytes.size()));
}

TEST(CompactProtocolReaderVarintTest, NullBufferWithNonZeroLengthThrows)
{
  // A null base with a positive length has no backing storage; construction rejects it.
  EXPECT_THROW(cudf::io::parquet::detail::CompactProtocolReader(nullptr, 4), std::invalid_argument);
}

TEST(CompactProtocolReaderVarintTest, NullBufferZeroLengthIsDefinedEmpty)
{
  cudf::io::parquet::detail::CompactProtocolReader cp(nullptr, 0);
  EXPECT_EQ(cp.get_u32(), 0u);
  EXPECT_EQ(cp.bytecount(), static_cast<ptrdiff_t>(0));
}

CUDF_TEST_PROGRAM_MAIN()
