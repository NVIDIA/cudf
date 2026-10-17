/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "checked_read.hpp"

#include "tests/io/experimental/hybrid_scan_multifile_composer.hpp"
#include "tests/io/parquet_common.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

namespace {

/**
 * @brief Expects `got` to have exactly the columns of `expected`, with no ULP tolerance for
 * floating point values
 */
void expect_tables_equivalent_exactly(cudf::table_view const& expected, cudf::table_view const& got)
{
  ASSERT_EQ(expected.num_columns(), got.num_columns());
  for (cudf::size_type i = 0; i < expected.num_columns(); ++i) {
    SCOPED_TRACE("column " + std::to_string(i));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      expected.column(i), got.column(i), cudf::test::debug_output_level::FIRST_ERROR, 0);
  }
}

}  // namespace

std::string to_string(read_settings const& read)
{
  std::string result;
  switch (read.reader) {
    case parquet_reader_kind::WHOLE: result = "whole read"; break;
    case parquet_reader_kind::CHUNKED: result = "chunked read"; break;
    case parquet_reader_kind::HYBRID_SCAN: result = "hybrid scan read"; break;
  }
  if (read.reader != parquet_reader_kind::WHOLE) {
    result += " chunk_read_limit=" + std::to_string(read.chunk_read_limit) +
              " pass_read_limit=" + std::to_string(read.pass_read_limit);
  }
  if (read.skip_rows > 0) { result += " skip_rows=" + std::to_string(read.skip_rows); }
  if (read.num_rows.has_value()) { result += " num_rows=" + std::to_string(*read.num_rows); }
  return result;
}

void checked_read(read_settings const& read,
                  std::vector<std::string> const& paths,
                  cudf::table_view const& expected)
{
  auto const source = cudf::io::source_info{paths};
  auto options =
    cudf::io::parquet_reader_options::builder(source).skip_rows(read.skip_rows).build();
  if (read.num_rows.has_value()) { options.set_num_rows(*read.num_rows); }

  std::vector<std::unique_ptr<cudf::table>> chunks;
  switch (read.reader) {
    case parquet_reader_kind::WHOLE: chunks.push_back(cudf::io::read_parquet(options).tbl); break;
    case parquet_reader_kind::CHUNKED:
      chunks = read_all_chunks(
        cudf::io::chunked_parquet_reader(read.chunk_read_limit, read.pass_read_limit, options));
      break;
    case parquet_reader_kind::HYBRID_SCAN:
      CUDF_EXPECTS(read.skip_rows == 0 and not read.num_rows.has_value(),
                   "The hybrid scan read does not take row bounds");
      chunks = chunked_hybrid_scan_multifile_passes(source,
                                                    read.chunk_read_limit,
                                                    read.pass_read_limit,
                                                    cudf::get_default_stream(),
                                                    cudf::get_current_device_resource_ref());
      break;
  }

  auto const end_row =
    read.num_rows.has_value() ? read.skip_rows + *read.num_rows : expected.num_rows();
  auto const expected_rows = cudf::slice(expected, {read.skip_rows, end_row})[0];
  auto start_row           = cudf::size_type{0};
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    auto const chunk_end_row = start_row + chunks[i]->num_rows();
    ASSERT_LE(chunk_end_row, expected_rows.num_rows()) << "chunk " << i;
    SCOPED_TRACE("chunk " + std::to_string(i) + " rows [" + std::to_string(start_row) + ", " +
                 std::to_string(chunk_end_row) + ")");
    expect_tables_equivalent_exactly(cudf::slice(expected_rows, {start_row, chunk_end_row})[0],
                                     chunks[i]->view());
    start_row = chunk_end_row;
  }
  EXPECT_EQ(start_row, expected_rows.num_rows());
}
