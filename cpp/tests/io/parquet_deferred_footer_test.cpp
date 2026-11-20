/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/parquet_footer.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Tests that reading a subset of columns or row groups, which defers decoding the footer column
// chunks until the input columns are known, matches reading with fully decoded footers.

namespace {

// Row groups hold whole page fragments, of 5000 rows by default
constexpr cudf::size_type num_rows       = 20000;
constexpr cudf::size_type row_group_rows = 5000;

/**
 * @brief Writes a file with flat, string, struct and list columns and four row groups
 *
 * Leaf columns: a, b, s.x, s.y, l.list.element, d
 */
std::vector<char> write_test_file(int32_t offset = 0)
{
  auto const seq = cuda::counting_iterator<int32_t>{offset};
  auto a         = cudf::test::fixed_width_column_wrapper<int32_t>(seq, seq + num_rows);
  auto strings   = std::vector<std::string>(num_rows);
  std::transform(seq, seq + num_rows, strings.begin(), [](auto i) { return std::to_string(i); });
  auto b                  = cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto sx                 = cudf::test::fixed_width_column_wrapper<int32_t>(seq, seq + num_rows);
  auto sy                 = cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto s                  = cudf::test::structs_column_wrapper{{sx, sy}};
  auto const offsets_iter = cuda::make_transform_iterator(cuda::counting_iterator<int32_t>{0},
                                                          [](auto i) { return 2 * i; });
  auto offsets =
    cudf::test::fixed_width_column_wrapper<int32_t>(offsets_iter, offsets_iter + num_rows + 1);
  auto values = cudf::test::fixed_width_column_wrapper<int32_t>(seq, seq + 2 * num_rows);
  auto l      = cudf::make_lists_column(num_rows,
                                   offsets.release(),
                                   values.release(),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  auto d      = cudf::test::fixed_width_column_wrapper<double>(seq, seq + num_rows);

  auto const table = cudf::table_view{std::vector<cudf::column_view>{cudf::column_view{a},
                                                                     cudf::column_view{b},
                                                                     cudf::column_view{s},
                                                                     l->view(),
                                                                     cudf::column_view{d}}};
  cudf::io::table_input_metadata metadata(table);
  metadata.column_metadata[0].set_name("a");
  metadata.column_metadata[1].set_name("b");
  metadata.column_metadata[2].set_name("s");
  metadata.column_metadata[2].child(0).set_name("x");
  metadata.column_metadata[2].child(1).set_name("y");
  metadata.column_metadata[3].set_name("l");
  metadata.column_metadata[4].set_name("d");

  std::vector<char> buffer;
  auto const options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, table)
      .metadata(std::move(metadata))
      .row_group_size_rows(row_group_rows)
      .max_page_size_rows(row_group_rows / 4)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .build();
  cudf::io::write_parquet(options);
  return buffer;
}

cudf::io::source_info to_source_info(std::vector<std::vector<char>> const& buffers)
{
  std::vector<cudf::host_span<std::byte const>> spans;
  for (auto const& buffer : buffers) {
    spans.emplace_back(reinterpret_cast<std::byte const*>(buffer.data()), buffer.size());
  }
  return cudf::io::source_info{cudf::host_span<cudf::host_span<std::byte const>>{spans}};
}

/**
 * @brief Reads with the given options, deferring the footer column chunks where applicable
 */
cudf::io::table_with_metadata read_deferred(cudf::io::parquet_reader_options const& options)
{
  return cudf::io::read_parquet(options);
}

/**
 * @brief Reads with the given options from fully decoded footers
 */
cudf::io::table_with_metadata read_full_footers(cudf::io::parquet_reader_options const& options)
{
  auto sources   = cudf::io::make_datasources(options.get_source());
  auto metadatas = cudf::io::read_parquet_footers(sources);
  return cudf::io::read_parquet(std::move(sources), std::move(metadatas), options);
}

void expect_same_read(cudf::io::parquet_reader_options const& options)
{
  auto const expected = read_full_footers(options);
  auto const result   = read_deferred(options);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected.tbl->view(), result.tbl->view());
  ASSERT_EQ(expected.metadata.schema_info.size(), result.metadata.schema_info.size());
  for (std::size_t i = 0; i < expected.metadata.schema_info.size(); ++i) {
    EXPECT_EQ(expected.metadata.schema_info[i].name, result.metadata.schema_info[i].name);
  }
  EXPECT_EQ(expected.metadata.num_row_groups_after_stats_filter,
            result.metadata.num_row_groups_after_stats_filter);
}

/**
 * @brief Returns a copy of the file with its footer column chunks reversed in every row group
 *
 * The reversed footer violates the spec's column chunk order, which the reader must still handle by
 * mapping column chunks by their paths.
 */
std::vector<char> reverse_column_chunks(std::vector<char> const& file)
{
  uint32_t footer_size = 0;
  std::memcpy(&footer_size, file.data() + file.size() - 8, sizeof(footer_size));
  auto const footer_start = file.size() - 8 - footer_size;
  auto const footer_bytes = std::span<uint8_t const>{
    reinterpret_cast<uint8_t const*>(file.data()) + footer_start, footer_size};

  auto metadata = cudf::io::parquet::experimental::read_parquet_footer_bytes(footer_bytes);
  for (auto& row_group : metadata.row_groups) {
    std::reverse(row_group.columns.begin(), row_group.columns.end());
  }
  auto const new_footer = cudf::io::parquet::experimental::write_parquet_footer_bytes(metadata);

  std::vector<char> result(file.begin(), file.begin() + footer_start);
  result.insert(result.end(), new_footer.begin(), new_footer.end());
  auto const new_footer_size = static_cast<uint32_t>(new_footer.size());
  auto const* size_bytes     = reinterpret_cast<char const*>(&new_footer_size);
  result.insert(result.end(), size_bytes, size_bytes + sizeof(new_footer_size));
  result.insert(result.end(), {'P', 'A', 'R', '1'});
  return result;
}

}  // namespace

struct ParquetDeferredFooterTest : public cudf::test::BaseFixture {};

TEST_F(ParquetDeferredFooterTest, ColumnSelection)
{
  auto const buffers = std::vector<std::vector<char>>{write_test_file()};
  auto const source  = to_source_info(buffers);

  for (auto const& columns : std::vector<std::vector<std::string>>{
         {"a"}, {"d", "b"}, {"s"}, {"s.y"}, {"l"}, {"l", "s.x", "a"}, {}}) {
    SCOPED_TRACE(columns.size());
    expect_same_read(
      cudf::io::parquet_reader_options::builder(source).column_names(columns).build());
  }
  expect_same_read(
    cudf::io::parquet_reader_options::builder(source).column_indices({4, 2}).build());
}

TEST_F(ParquetDeferredFooterTest, RowGroupSelection)
{
  auto const buffers = std::vector<std::vector<char>>{write_test_file()};
  auto const source  = to_source_info(buffers);

  for (auto const& row_groups :
       std::vector<std::vector<cudf::size_type>>{{0}, {3}, {2, 1}, {1, 3}}) {
    SCOPED_TRACE(row_groups.size());
    expect_same_read(
      cudf::io::parquet_reader_options::builder(source).row_groups({row_groups}).build());
    expect_same_read(cudf::io::parquet_reader_options::builder(source)
                       .row_groups({row_groups})
                       .column_names({"b", "l"})
                       .build());
  }
}

TEST_F(ParquetDeferredFooterTest, RowBounds)
{
  auto const buffers = std::vector<std::vector<char>>{write_test_file()};
  auto const source  = to_source_info(buffers);
  expect_same_read(cudf::io::parquet_reader_options::builder(source)
                     .column_names({"s", "d"})
                     .skip_rows(1500)
                     .num_rows(1200)
                     .build());
}

TEST_F(ParquetDeferredFooterTest, FilterOnUnselectedColumn)
{
  auto const buffers = std::vector<std::vector<char>>{write_test_file()};
  auto const source  = to_source_info(buffers);

  auto literal_value = cudf::numeric_scalar<int32_t>(12500);
  auto const literal = cudf::ast::literal(literal_value);
  auto const col_ref = cudf::ast::column_name_reference("a");
  auto const filter =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref, literal);

  auto const options = cudf::io::parquet_reader_options::builder(source)
                         .column_names({"b", "l"})
                         .filter(filter)
                         .build();
  expect_same_read(options);

  auto const result = read_deferred(options);
  EXPECT_EQ(result.tbl->num_rows(), num_rows - 12500);
  EXPECT_EQ(result.tbl->num_columns(), 2);
}

TEST_F(ParquetDeferredFooterTest, ChunkedRead)
{
  auto const buffers = std::vector<std::vector<char>>{write_test_file()};
  auto const source  = to_source_info(buffers);
  auto const options =
    cudf::io::parquet_reader_options::builder(source).column_names({"b", "s", "l"}).build();

  auto const expected = read_full_footers(options);
  auto reader         = cudf::io::chunked_parquet_reader(64 * 1024, 128 * 1024, options);
  std::vector<std::unique_ptr<cudf::table>> chunks;
  while (reader.has_next()) {
    chunks.push_back(std::move(reader.read_chunk().tbl));
  }
  ASSERT_GT(chunks.size(), 1);
  std::vector<cudf::table_view> views;
  std::transform(chunks.begin(), chunks.end(), std::back_inserter(views), [](auto const& chunk) {
    return chunk->view();
  });
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected.tbl->view(), cudf::concatenate(views)->view());
}

TEST_F(ParquetDeferredFooterTest, MultipleSources)
{
  auto const buffers =
    std::vector<std::vector<char>>{write_test_file(0), write_test_file(num_rows)};
  auto const source = to_source_info(buffers);
  expect_same_read(
    cudf::io::parquet_reader_options::builder(source).column_names({"s.x", "b"}).build());
  expect_same_read(cudf::io::parquet_reader_options::builder(source)
                     .column_names({"l"})
                     .row_groups({{3}, {0, 2}})
                     .build());
}

TEST_F(ParquetDeferredFooterTest, OutOfOrderColumnChunks)
{
  auto const buffers = std::vector<std::vector<char>>{reverse_column_chunks(write_test_file())};
  auto const source  = to_source_info(buffers);

  // Reading all columns maps column chunks by path when decoding the full footer
  auto const all = read_deferred(cudf::io::parquet_reader_options::builder(source).build());
  EXPECT_EQ(all.tbl->num_columns(), 5);

  expect_same_read(
    cudf::io::parquet_reader_options::builder(source).column_names({"d", "s.y"}).build());
  expect_same_read(cudf::io::parquet_reader_options::builder(source)
                     .row_groups({{1}})
                     .column_names({"l"})
                     .build());
}
