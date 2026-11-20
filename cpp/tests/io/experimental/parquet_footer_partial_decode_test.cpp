/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/parquet_footer.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace cudf::io::parquet;
using experimental::footer_decode_options;
using experimental::read_parquet_footer_bytes;
using experimental::write_parquet_footer_bytes;

namespace {

// Builds a flat footer with distinct per-chunk values in every decoded ColumnChunk field
FileMetaData make_footer(int num_leaves, int num_row_groups)
{
  FileMetaData md;
  md.version    = 2;
  md.num_rows   = int64_t{10} * num_row_groups;
  md.created_by = "partial-decode-test";
  md.schema.push_back({.type            = Type::UNDEFINED,
                       .repetition_type = FieldRepetitionType::REQUIRED,
                       .name            = "schema",
                       .num_children    = num_leaves});
  for (int c = 0; c < num_leaves; ++c) {
    md.schema.push_back({.type            = Type::INT64,
                         .repetition_type = FieldRepetitionType::OPTIONAL,
                         .name            = "c" + std::to_string(c)});
  }
  for (int r = 0; r < num_row_groups; ++r) {
    RowGroup rg{.total_byte_size       = 7 * r + 1,
                .num_rows              = 10,
                .file_offset           = int64_t{100} * r,
                .total_compressed_size = 3 * r + 2,
                .ordinal               = static_cast<int16_t>(r)};
    for (int c = 0; c < num_leaves; ++c) {
      auto const v = int64_t{100'000} * r + c;
      ColumnChunk cc{.file_offset         = v,
                     .offset_index_offset = v + 1,
                     .offset_index_length = c + 1,
                     .column_index_offset = v + 2,
                     .column_index_length = c + 2};
      cc.meta_data = {.type                    = Type::INT64,
                      .encodings               = {Encoding::PLAIN, Encoding::RLE},
                      .path_in_schema          = {"c" + std::to_string(c)},
                      .codec                   = Compression::SNAPPY,
                      .num_values              = 10,
                      .total_uncompressed_size = v + 3,
                      .total_compressed_size   = v + 4,
                      .data_page_offset        = v + 5,
                      .statistics              = {.null_count = c,
                                                  .max_value  = std::vector<uint8_t>{uint8_t(r), 9},
                                                  .min_value  = std::vector<uint8_t>{uint8_t(c), 1}},
                      .encoding_stats =
                        std::vector<PageEncodingStats>{{PageType::DATA_PAGE, Encoding::PLAIN, c + 1}},
                      .bloom_filter_offset = v + 6,
                      .size_statistics =
                        SizeStatistics{.unencoded_byte_array_data_bytes = v,
                                       .definition_level_histogram = std::vector<int64_t>{1, v}}};
      rg.columns.push_back(std::move(cc));
    }
    md.row_groups.push_back(std::move(rg));
  }
  md.key_value_metadata = {{"key", "value"}};
  return md;
}

std::vector<cudf::size_type> iota(cudf::size_type count)
{
  std::vector<cudf::size_type> v(count);
  std::iota(v.begin(), v.end(), 0);
  return v;
}

// Expected partial footer: the full footer with unselected chunks dropped and disabled fields reset
FileMetaData expected_partial(FileMetaData md,
                              std::vector<cudf::size_type> const& row_groups,
                              std::vector<cudf::size_type> const& leaf_columns,
                              footer_decode_options const& options)
{
  for (std::size_t r = 0; r < md.row_groups.size(); ++r) {
    auto& columns = md.row_groups[r].columns;
    if (not std::binary_search(
          row_groups.begin(), row_groups.end(), static_cast<cudf::size_type>(r))) {
      columns.clear();
      continue;
    }
    std::vector<ColumnChunk> selected;
    for (auto const c : leaf_columns) {
      selected.push_back(columns[c]);
      auto& meta = selected.back().meta_data;
      if (not options.decode_statistics) { meta.statistics = {}; }
      if (not options.decode_encoding_stats) { meta.encoding_stats.reset(); }
      if (not options.decode_size_statistics) { meta.size_statistics.reset(); }
    }
    columns = std::move(selected);
  }
  return md;
}

// Checks a selective decode of `md` against the filtered full decode, compared as serialized bytes
void check_partial_decode(FileMetaData const& md, footer_decode_options const& options)
{
  auto const bytes = write_parquet_footer_bytes(md);
  auto const full  = read_parquet_footer_bytes(bytes);

  auto const normalized = [](std::optional<std::vector<cudf::size_type>> const& sel,
                             cudf::size_type count) {
    if (not sel.has_value()) { return iota(count); }
    auto v = *sel;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
  };
  auto const num_leaves = static_cast<cudf::size_type>(
    std::count_if(full.schema.begin() + 1, full.schema.end(), [](auto const& e) {
      return e.num_children == 0;
    }));
  auto const row_groups =
    normalized(options.row_groups, static_cast<cudf::size_type>(full.row_groups.size()));
  auto const leaf_columns = normalized(options.leaf_columns, num_leaves);

  auto const partial = read_parquet_footer_bytes(bytes, options);
  EXPECT_EQ(partial.row_groups, row_groups);
  EXPECT_EQ(partial.leaf_columns, leaf_columns);
  ASSERT_EQ(partial.metadata.row_groups.size(), full.row_groups.size());
  EXPECT_EQ(write_parquet_footer_bytes(partial.metadata),
            write_parquet_footer_bytes(expected_partial(full, row_groups, leaf_columns, options)));
}

}  // namespace

struct ParquetFooterPartialDecodeTest : public cudf::test::BaseFixture {};

TEST_F(ParquetFooterPartialDecodeTest, NoSelectionMatchesFullDecode)
{
  auto const md    = make_footer(5, 3);
  auto const bytes = write_parquet_footer_bytes(md);
  auto const full  = read_parquet_footer_bytes(bytes);

  auto const partial = read_parquet_footer_bytes(bytes, footer_decode_options{});
  EXPECT_EQ(partial.row_groups, iota(3));
  EXPECT_EQ(partial.leaf_columns, iota(5));
  EXPECT_EQ(write_parquet_footer_bytes(partial.metadata), write_parquet_footer_bytes(full));
}

TEST_F(ParquetFooterPartialDecodeTest, SelectAllMatchesFullDecode)
{
  check_partial_decode(make_footer(5, 3), {.row_groups = iota(3), .leaf_columns = iota(5)});
}

TEST_F(ParquetFooterPartialDecodeTest, SelectColumns)
{
  check_partial_decode(make_footer(8, 3), {.leaf_columns = std::vector<cudf::size_type>{1, 4, 7}});
  check_partial_decode(make_footer(8, 3), {.leaf_columns = std::vector<cudf::size_type>{0}});
}

TEST_F(ParquetFooterPartialDecodeTest, SelectRowGroups)
{
  check_partial_decode(make_footer(4, 5), {.row_groups = std::vector<cudf::size_type>{0, 3}});
  check_partial_decode(make_footer(4, 5), {.row_groups = std::vector<cudf::size_type>{4}});
}

TEST_F(ParquetFooterPartialDecodeTest, SelectRowGroupsAndColumns)
{
  check_partial_decode(make_footer(6, 4),
                       {.row_groups   = std::vector<cudf::size_type>{1, 2},
                        .leaf_columns = std::vector<cudf::size_type>{0, 5}});
}

TEST_F(ParquetFooterPartialDecodeTest, EmptySelectionDecodesSchemaAndRowGroupScalars)
{
  auto const md    = make_footer(6, 4);
  auto const bytes = write_parquet_footer_bytes(md);
  auto const partial =
    read_parquet_footer_bytes(bytes,
                              {.row_groups   = std::vector<cudf::size_type>{},
                               .leaf_columns = std::vector<cudf::size_type>{}});
  EXPECT_TRUE(partial.row_groups.empty());
  EXPECT_TRUE(partial.leaf_columns.empty());
  EXPECT_EQ(partial.metadata.schema.size(), md.schema.size());
  EXPECT_EQ(partial.metadata.num_rows, md.num_rows);
  EXPECT_EQ(partial.metadata.key_value_metadata.size(), 1);
  ASSERT_EQ(partial.metadata.row_groups.size(), md.row_groups.size());
  for (std::size_t r = 0; r < md.row_groups.size(); ++r) {
    auto const& rg = partial.metadata.row_groups[r];
    EXPECT_TRUE(rg.columns.empty());
    EXPECT_EQ(rg.num_rows, md.row_groups[r].num_rows);
    EXPECT_EQ(rg.total_byte_size, md.row_groups[r].total_byte_size);
    EXPECT_EQ(rg.file_offset, md.row_groups[r].file_offset);
    EXPECT_EQ(rg.ordinal, md.row_groups[r].ordinal);
  }
}

TEST_F(ParquetFooterPartialDecodeTest, UnsortedDuplicateSelectionIsNormalized)
{
  check_partial_decode(make_footer(6, 4),
                       {.row_groups   = std::vector<cudf::size_type>{3, 0, 3},
                        .leaf_columns = std::vector<cudf::size_type>{5, 2, 2, 0}});
}

TEST_F(ParquetFooterPartialDecodeTest, SkipOptionalColumnChunkFields)
{
  auto const md = make_footer(4, 2);
  check_partial_decode(md, {.decode_statistics = false});
  check_partial_decode(md, {.decode_encoding_stats = false});
  check_partial_decode(md, {.decode_size_statistics = false});
  check_partial_decode(md,
                       {.leaf_columns           = std::vector<cudf::size_type>{1, 3},
                        .decode_statistics      = false,
                        .decode_encoding_stats  = false,
                        .decode_size_statistics = false});
}

// 512 or more decoded column chunks per row group take the parallel struct-list path
TEST_F(ParquetFooterPartialDecodeTest, ParallelColumnChunks)
{
  auto const md = make_footer(700, 2);
  check_partial_decode(md, {.leaf_columns = iota(700)});
  // Sparse selection above the threshold: unselected chunks interleave the parallel ranges
  std::vector<cudf::size_type> sparse;
  for (cudf::size_type c = 1; c < 700; ++c) {
    if (c % 7 != 0) { sparse.push_back(c); }
  }
  check_partial_decode(md, {.row_groups = std::vector<cudf::size_type>{1}, .leaf_columns = sparse});
  // Selection below the threshold of a list above it
  check_partial_decode(md, {.leaf_columns = std::vector<cudf::size_type>{3, 699}});
  check_partial_decode(md, {.leaf_columns = iota(700), .decode_statistics = false});
}

// 512 or more row groups take the parallel struct-list path with per-row-group column masks
TEST_F(ParquetFooterPartialDecodeTest, ParallelRowGroups)
{
  auto const md = make_footer(3, 520);
  check_partial_decode(md, {.row_groups = std::vector<cudf::size_type>{0, 255, 519}});
  check_partial_decode(md,
                       {.row_groups   = std::vector<cudf::size_type>{7, 500},
                        .leaf_columns = std::vector<cudf::size_type>{2}});
  check_partial_decode(md, {.leaf_columns = std::vector<cudf::size_type>{1}});
}

TEST_F(ParquetFooterPartialDecodeTest, OutOfRangeSelectionThrows)
{
  auto const bytes = write_parquet_footer_bytes(make_footer(4, 3));
  EXPECT_THROW(std::ignore = read_parquet_footer_bytes(
                 bytes, {.leaf_columns = std::vector<cudf::size_type>{4}}),
               std::out_of_range);
  EXPECT_THROW(std::ignore = read_parquet_footer_bytes(
                 bytes, {.row_groups = std::vector<cudf::size_type>{3}}),
               std::out_of_range);
  EXPECT_THROW(std::ignore = read_parquet_footer_bytes(
                 bytes, {.leaf_columns = std::vector<cudf::size_type>{-1}}),
               std::out_of_range);
  EXPECT_THROW(std::ignore = read_parquet_footer_bytes(
                 bytes, {.row_groups = std::vector<cudf::size_type>{-1}}),
               std::out_of_range);
}

TEST_F(ParquetFooterPartialDecodeTest, ColumnCountMismatchThrows)
{
  auto md = make_footer(4, 2);
  md.row_groups[1].columns.pop_back();
  auto const bytes = write_parquet_footer_bytes(md);
  EXPECT_NO_THROW(std::ignore = read_parquet_footer_bytes(bytes, footer_decode_options{}));
  EXPECT_THROW(std::ignore = read_parquet_footer_bytes(
                 bytes, {.leaf_columns = std::vector<cudf::size_type>{0}}),
               cudf::logic_error);
}

// A footer that encodes row_groups (field 4) before the schema (field 2)
TEST_F(ParquetFooterPartialDecodeTest, RowGroupsBeforeSchema)
{
  std::vector<uint8_t> const bytes = {
    0x49, 0x1C,              // field 4 (delta 4): list<struct> of 1
    0x36, 0x0A, 0x00,        //   RowGroup { 3: num_rows = 5 }
    0x09, 0x04, 0x1C,        // field 2 (long form): list<struct> of 1
    0x48, 0x04, 'r', 'o', 'o', 't', 0x00,  //   SchemaElement { 4: name = "root" }
    0x00};                   // stop
  auto const partial = read_parquet_footer_bytes(bytes, footer_decode_options{});
  ASSERT_EQ(partial.metadata.row_groups.size(), 1);
  EXPECT_EQ(partial.metadata.row_groups[0].num_rows, 5);
  EXPECT_THROW(std::ignore = read_parquet_footer_bytes(
                 bytes, {.leaf_columns = std::vector<cudf::size_type>{}}),
               cudf::logic_error);
}

TEST_F(ParquetFooterPartialDecodeTest, TruncatedFooterThrows)
{
  auto bytes = write_parquet_footer_bytes(make_footer(4, 2));
  bytes.resize(bytes.size() / 2);
  EXPECT_THROW(std::ignore = read_parquet_footer_bytes(
                 bytes, {.leaf_columns = std::vector<cudf::size_type>{1}}),
               cudf::logic_error);
}

// Selective decode of a nested cudf-written footer with statistics and a page index
TEST_F(ParquetFooterPartialDecodeTest, WrittenFooter)
{
  using int_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using str_col  = cudf::test::strings_column_wrapper;
  using list_col = cudf::test::lists_column_wrapper<int32_t>;
  int_col ints{1, 2, 3, 4, 5, 6};
  str_col strs{"a", "bb", "a", "ccc", "dd", "a"};
  int_col child_ints{6, 5, 4, 3, 2, 1};
  str_col child_strs{"x", "y", "zz", "x", "y", "zz"};
  cudf::test::structs_column_wrapper structs({child_ints, child_strs});
  list_col lists{{1, 2}, {3}, {}, {4, 5, 6}, {7}, {8, 9}};
  cudf::table_view const input({ints, strs, structs, lists});

  std::vector<char> buffer;
  auto const options =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{&buffer})
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .build();
  cudf::io::chunked_parquet_writer writer(options);
  for (int i = 0; i < 3; ++i) {
    writer.write(input);
  }
  std::ignore = writer.close();

  auto const source = cudf::io::datasource::create(
    cudf::host_span<std::byte const>{reinterpret_cast<std::byte const*>(buffer.data()),
                                     buffer.size()});
  auto const footer_buf = fetch_footer_to_host(*source);
  auto const bytes      = std::span<uint8_t const>{footer_buf->data(), footer_buf->size()};
  auto const full       = read_parquet_footer_bytes(bytes);
  ASSERT_EQ(full.row_groups.size(), 3);
  // Leaves: ints, strs, structs.child_ints, structs.child_strs, lists.element
  ASSERT_EQ(full.row_groups.front().columns.size(), 5);

  check_partial_decode(full, {.leaf_columns = std::vector<cudf::size_type>{2, 4}});
  check_partial_decode(full,
                       {.row_groups        = std::vector<cudf::size_type>{1},
                        .leaf_columns      = std::vector<cudf::size_type>{1, 3},
                        .decode_statistics = false});
}

CUDF_TEST_PROGRAM_MAIN()
