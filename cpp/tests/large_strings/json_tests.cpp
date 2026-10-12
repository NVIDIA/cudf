/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../io/json/json_utils.hpp"
#include "large_strings_fixture.hpp"

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/json.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

struct JsonLargeReaderTest : public cudf::test::StringsLargeTest,
                             public testing::WithParamInterface<cudf::io::compression_type> {
 public:
  void set_batch_size(size_t batch_size_upper_bound)
  {
    setenv("LIBCUDF_JSON_BATCH_SIZE", std::to_string(batch_size_upper_bound).c_str(), 1);
  }

  ~JsonLargeReaderTest() { unsetenv("LIBCUDF_JSON_BATCH_SIZE"); }
};

// Parametrize qualifying JSON tests for multiple compression types
INSTANTIATE_TEST_SUITE_P(JsonLargeReaderTest,
                         JsonLargeReaderTest,
                         ::testing::Values(cudf::io::compression_type::GZIP,
                                           cudf::io::compression_type::NONE));

TEST_P(JsonLargeReaderTest, MultiBatch)
{
  // this test runs over 1.5 hours when racecheck is used
  if (getenv("LIBCUDF_RACECHECK_ENABLED")) { GTEST_SKIP(); }

  cudf::io::compression_type const comptype = GetParam();

  std::string json_string = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";

  std::size_t const batch_size_upper_bound = std::numeric_limits<int32_t>::max() / 16;
  // set smaller batch_size to reduce file size and execution time
  this->set_batch_size(batch_size_upper_bound);

  constexpr std::size_t expected_file_size = 1.5 * static_cast<double>(batch_size_upper_bound);
  std::size_t const log_repetitions =
    static_cast<std::size_t>(std::ceil(std::log2(expected_file_size / json_string.size())));

  json_string.reserve(json_string.size() * (1UL << log_repetitions));
  for (std::size_t i = 0; i < log_repetitions; i++) {
    json_string += json_string;
  }

  std::vector<std::uint8_t> cdata;
  if (comptype != cudf::io::compression_type::NONE) {
    cdata = cudf::io::detail::compress(
      comptype,
      cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(json_string.data()),
                                     json_string.size()));
  } else
    cdata = std::vector<uint8_t>(
      reinterpret_cast<uint8_t const*>(json_string.data()),
      reinterpret_cast<uint8_t const*>(json_string.data()) + json_string.size());

  constexpr int num_sources = 2;
  std::vector<cudf::host_span<std::byte>> hostbufs(
    num_sources,
    cudf::host_span<std::byte>(reinterpret_cast<std::byte*>(json_string.data()),
                               json_string.size()));
  std::vector<cudf::host_span<std::byte>> chostbufs(
    num_sources,
    cudf::host_span<std::byte>(reinterpret_cast<std::byte*>(cdata.data()), cdata.size()));

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options json_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{
        cudf::host_span<cudf::host_span<std::byte>>(hostbufs.data(), hostbufs.size())})
      .lines(true)
      .compression(cudf::io::compression_type::NONE)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);
  cudf::io::json_reader_options cjson_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{
        cudf::host_span<cudf::host_span<std::byte>>(chostbufs.data(), chostbufs.size())})
      .lines(true)
      .compression(comptype)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  // Read full test data via existing, nested JSON lines reader
  cudf::io::table_with_metadata current_reader_table = cudf::io::read_json(cjson_lines_options);

  auto datasources  = cudf::io::datasource::create(json_lines_options.get_source().host_buffers());
  auto cdatasources = cudf::io::datasource::create(cjson_lines_options.get_source().host_buffers());

  // Test small byte ranges that require many batches and large ranges that exceed a batch.
  std::vector<std::size_t> chunk_sizes{batch_size_upper_bound / 4,
                                       static_cast<std::size_t>(batch_size_upper_bound * 2)};

  for (auto chunk_size : chunk_sizes) {
    auto const tables =
      split_byte_range_reading<std::int64_t>(datasources,
                                             cdatasources,
                                             json_lines_options,
                                             cjson_lines_options,
                                             chunk_size,
                                             cudf::get_default_stream(),
                                             cudf::get_current_device_resource_ref());

    auto table_views = std::vector<cudf::table_view>(tables.size());
    std::transform(tables.begin(), tables.end(), table_views.begin(), [](auto& table) {
      return table.tbl->view();
    });
    auto result = cudf::concatenate(table_views);

    // Verify that the data read via chunked reader matches the data read via nested JSON reader
    // cannot use EQUAL due to concatenate removing null mask
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(current_reader_table.tbl->view(), result->view());
  }
}

TEST_P(JsonLargeReaderTest, MultiBatchWithNulls)
{
  cudf::io::compression_type const comptype = GetParam();

  // The goal of this test is to ensure that column schema from the first
  // batch is enforced on all following batches in the JSON reader. The column
  // ordering from the first batch is applied to batches 2 and 3.
  std::string json_string_b1 = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";
  std::string json_string_b2 = R"(
    { "a": { "y" : 6}, "c": 11 }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";
  std::string json_string_b3 = R"(
    { "b" : [1, 2, 3], "a": { "y" : 6}}
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": 12 }
    { "a": { "y" : 6}, "b" : [6      ], "c": 13 }
    { "a": { "y" : 6}, "b" : [7      ], "c": 14 })";

  // Set the batch size to the size of the first json string, `json_string_b1`.
  std::size_t const batch_size_upper_bound = json_string_b1.size();
  // set smaller batch_size to reduce file size and execution time
  this->set_batch_size(batch_size_upper_bound);

  auto json_string = json_string_b1 + json_string_b2 + json_string_b3;
  std::vector<std::uint8_t> cdata;
  if (comptype != cudf::io::compression_type::NONE) {
    cdata = cudf::io::detail::compress(
      comptype,
      cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(json_string.data()),
                                     json_string.size()));
  } else
    cdata = std::vector<uint8_t>(
      reinterpret_cast<uint8_t const*>(json_string.data()),
      reinterpret_cast<uint8_t const*>(json_string.data()) + json_string.size());

  constexpr int num_sources = 2;
  std::vector<cudf::host_span<std::byte>> chostbufs(
    num_sources,
    cudf::host_span<std::byte>(reinterpret_cast<std::byte*>(cdata.data()), cdata.size()));

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options cjson_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{
        cudf::host_span<cudf::host_span<std::byte>>(chostbufs.data(), chostbufs.size())})
      .lines(true)
      .compression(comptype)
      .recovery_mode(cudf::io::json_recovery_mode_t::FAIL);

  // Read full test data via existing, nested JSON lines reader
  CUDF_EXPECT_NO_THROW(cudf::io::read_json(cjson_lines_options));
}

TEST_P(JsonLargeReaderTest, MultiBatchDoubleBufferInput)
{
  cudf::io::compression_type const comptype = GetParam();

  // This test constructs a JSON input of size two times the batch size but sets the batch boundary
  // after the start of the last record in the batch i.e. the input is constructed such that the
  // size of the last record is approximately the same as the size of all preceding records. Since
  // the reader now ends up reading twice the allowed batch size per batch, it has to split the read
  // buffer in two, each part of size <= the batch size.
  std::string json_string      = R"(
    { "a": { "y" : 6}, "b" : [1, 2, 3], "c": "11" }
    { "a": { "y" : 6}, "b" : [4, 5   ], "c": "12" }
    { "a": { "y" : 6}, "b" : [6      ], "c": "13" }
    { "a": { "y" : 6}, "b" : [7      ], "c": "14" }
    )";
  std::size_t const batch_size = json_string.size() + 1;
  // set smaller batch_size to reduce file size and execution time
  this->set_batch_size(batch_size);

  std::string really_long_string    = R"(libcudf)";
  std::size_t const log_repetitions = static_cast<std::size_t>(
    std::floor(std::log2(static_cast<double>(json_string.size()) / really_long_string.size())));
  really_long_string.reserve(really_long_string.size() * (1UL << log_repetitions));
  for (std::size_t i = 0; i < log_repetitions; i++) {
    really_long_string += really_long_string;
  }
  std::string last_line = R"({ "a": { "y" : 6}, "b" : [1, 2, 3], "c": ")";
  last_line += really_long_string + "\" }\n";
  json_string += last_line;

  std::vector<std::uint8_t> cdata;
  if (comptype != cudf::io::compression_type::NONE) {
    cdata = cudf::io::detail::compress(
      comptype,
      cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(json_string.data()),
                                     json_string.size()));
  } else {
    cdata = std::vector<uint8_t>(
      reinterpret_cast<uint8_t const*>(json_string.data()),
      reinterpret_cast<uint8_t const*>(json_string.data()) + json_string.size());
  }

  constexpr int num_sources = 3;
  std::vector<cudf::host_span<std::byte>> chostbufs(
    num_sources,
    cudf::host_span<std::byte>(reinterpret_cast<std::byte*>(cdata.data()), cdata.size()));

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options cjson_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{
        cudf::host_span<cudf::host_span<std::byte>>(chostbufs.data(), chostbufs.size())})
      .lines(true)
      .compression(comptype);

  // Read full test data via existing, nested JSON lines reader
  auto const result = cudf::io::read_json(cjson_lines_options);

  ASSERT_EQ(result.tbl->num_columns(), 3);
  ASSERT_EQ(result.tbl->num_rows(), 15);

  ASSERT_EQ(result.metadata.schema_info.size(), 3);
  EXPECT_EQ(result.metadata.schema_info[0].name, "a");
  EXPECT_EQ(result.metadata.schema_info[1].name, "b");
  EXPECT_EQ(result.metadata.schema_info[2].name, "c");

  EXPECT_EQ(result.tbl->get_column(2).type().id(), cudf::type_id::STRING);
  auto expected_c_col       = std::vector<std::string>{"11", "12", "13", "14", really_long_string};
  auto single_src_ccol_size = expected_c_col.size();
  expected_c_col.resize(single_src_ccol_size * num_sources);
  for (int i = 1; i <= num_sources - 1; i++)
    std::copy(expected_c_col.begin(),
              expected_c_col.begin() + single_src_ccol_size,
              expected_c_col.begin() + (i * single_src_ccol_size));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    result.tbl->get_column(2),
    cudf::test::strings_column_wrapper(expected_c_col.begin(), expected_c_col.end()));
}

TEST_P(JsonLargeReaderTest, OverBatchLimitLine)
{
  cudf::io::compression_type const comptype = GetParam();

  // This test constructs a JSONL input of size three times the batch limit. The input contains a
  // single JSONL which will be completely read in the first batch itself. Since we cannot divide a
  // single line, we expect the test to throw
  std::string json_string           = R"({ "a": { "y" : 6}, "b" : [1, 2, 3], "c": ")";
  std::string really_long_string    = R"(libcudf)";
  std::size_t const log_repetitions = 5;
  really_long_string.reserve(really_long_string.size() * (1UL << log_repetitions));
  for (std::size_t i = 0; i < log_repetitions; i++) {
    really_long_string += really_long_string;
  }
  json_string += really_long_string + "\" }\n";

  std::size_t const batch_size = json_string.size() / 3;
  // set smaller batch_size to reduce file size and execution time
  this->set_batch_size(batch_size);

  std::vector<std::uint8_t> cdata;
  if (comptype != cudf::io::compression_type::NONE) {
    cdata = cudf::io::detail::compress(
      comptype,
      cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t const*>(json_string.data()),
                                     json_string.size()));
  } else {
    cdata = std::vector<uint8_t>(
      reinterpret_cast<uint8_t const*>(json_string.data()),
      reinterpret_cast<uint8_t const*>(json_string.data()) + json_string.size());
  }

  constexpr int num_sources = 1;
  std::vector<cudf::host_span<std::byte>> chostbufs(
    num_sources,
    cudf::host_span<std::byte>(reinterpret_cast<std::byte*>(cdata.data()), cdata.size()));

  // Initialize parsing options (reading json lines)
  cudf::io::json_reader_options cjson_lines_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{
        cudf::host_span<cudf::host_span<std::byte>>(chostbufs.data(), chostbufs.size())})
      .lines(true)
      .compression(comptype);

  // Read full test data via existing, nested JSON lines reader
  EXPECT_THROW(cudf::io::read_json(cjson_lines_options), cudf::logic_error);
}
