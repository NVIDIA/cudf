/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/avro.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

struct AvroReaderTest : public cudf::test::BaseFixture {};

TEST_F(AvroReaderTest, MetadataVarintWidth)
{
  // The Avro magic followed by the metadata item count as a 10-byte varint.
  auto const read_with_last_group = [](uint8_t first_group, uint8_t last_group) {
    std::vector<uint8_t> file{'O', 'b', 'j', 0x01, first_group};
    file.insert(file.end(), 8, 0xFF);
    file.push_back(last_group);
    return cudf::io::read_avro(
      cudf::io::avro_reader_options::builder(
        cudf::io::source_info{cudf::host_span<uint8_t const>(file.data(), file.size())})
        .build());
  };

  // The last group holds only bit 63, so 0x02 and a continuation byte do not fit in 64 bits
  EXPECT_THROW(read_with_last_group(0xFF, 0x02), std::overflow_error);
  EXPECT_THROW(read_with_last_group(0xFF, 0x81), std::overflow_error);
  // The largest positive count decodes, and is then rejected as too large for the input
  EXPECT_THROW(read_with_last_group(0xFE, 0x01), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
