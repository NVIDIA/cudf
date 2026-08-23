/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/io/avro.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <cstring>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> length_prefixed_string(std::string const& text)
{
  std::vector<uint8_t> out;
  // Avro strings are preceded by a zigzag-encoded byte length.
  uint64_t encoded_length = static_cast<uint64_t>(text.size()) << 1;
  while (encoded_length >= 0x80) {
    out.push_back(static_cast<uint8_t>(0x80u | (encoded_length & 0x7fu)));
    encoded_length >>= 7;
  }
  out.push_back(static_cast<uint8_t>(encoded_length));
  out.insert(out.end(), text.begin(), text.end());
  return out;
}

// Builds a minimal valid one-row Avro file with a single int column "a" holding the value 0.
std::vector<uint8_t> make_valid_avro_file()
{
  std::string const schema = R"({"type":"record","name":"r","fields":[{"name":"a","type":"int"}]})";

  std::vector<uint8_t> data{'O', 'b', 'j', 0x01};
  auto metadata_item  = length_prefixed_string("avro.schema");
  auto metadata_value = length_prefixed_string(schema);
  data.push_back(0x02);  // one metadata entry, zigzag encoded
  data.insert(data.end(), metadata_item.begin(), metadata_item.end());
  data.insert(data.end(), metadata_value.begin(), metadata_value.end());
  data.push_back(0x00);               // end of the metadata map
  data.insert(data.end(), 16, 0x00);  // sync marker

  data.push_back(0x02);               // one row in the block, zigzag encoded
  data.push_back(0x02);               // one byte of row data
  data.push_back(0x00);               // row value: int 0
  data.insert(data.end(), 16, 0x00);  // block sync marker
  return data;
}

}  // namespace

struct AvroReaderTest : public cudf::test::BaseFixture {};

TEST_F(AvroReaderTest, ValidSingleRowFile)
{
  auto data          = make_valid_avro_file();
  auto const options = cudf::io::avro_reader_options::builder(
                         cudf::io::source_info(cudf::host_span<uint8_t>(data.data(), data.size())))
                         .build();

  auto const result = cudf::io::read_avro(options);
  EXPECT_EQ(result.tbl->num_rows(), 1);
  ASSERT_EQ(result.tbl->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->get_column(0),
                                 cudf::test::fixed_width_column_wrapper<int32_t>{{0}});
}

TEST_F(AvroReaderTest, OverlongMetadataVarintThrows)
{
  // Avro magic followed by a metadata item count encoded as an over-long varint: more
  // continuation bytes than fit in an int64.
  std::vector<uint8_t> data{'O', 'b', 'j', 0x01};
  for (int i = 0; i < 12; ++i) {
    data.push_back(0x80);
  }
  data.push_back(0x01);

  auto const options = cudf::io::avro_reader_options::builder(
                         cudf::io::source_info(cudf::host_span<uint8_t>(data.data(), data.size())))
                         .build();

  try {
    cudf::io::read_avro(options);
    FAIL() << "expected cudf::logic_error";
  } catch (cudf::logic_error const& e) {
    EXPECT_NE(std::strstr(e.what(), "Invalid varint"), nullptr)
      << "unexpected error message: " << e.what();
  }
}
