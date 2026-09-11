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
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <vector>

using namespace cudf::io::parquet;

namespace {

// Fully-populated footer whose values all survive the writer's conditional emission, so a
// write -> read round-trip is exactly recoverable.
FileMetaData const& make_test_footer()
{
  static FileMetaData const meta = [] {
    FileMetaData meta = {
      .version    = 2,
      .num_rows   = 12345,
      .created_by = "cudf-facade-test",
    };

    meta.schema = {
      {.type            = Type::UNDEFINED,
       .repetition_type = FieldRepetitionType::REQUIRED,
       .name            = "schema",
       .num_children    = 3},
      {.type            = Type::INT32,
       .repetition_type = FieldRepetitionType::OPTIONAL,
       .name            = "a",
       .field_id        = 1},
      {.type            = Type::BYTE_ARRAY,
       .repetition_type = FieldRepetitionType::REQUIRED,
       .name            = "b",
       .converted_type  = ConvertedType::UTF8,
       .field_id        = 2},
      // type_length is written only for a typed leaf, so a non-zero value exercises that schema
      // field.
      {.type            = Type::FIXED_LEN_BYTE_ARRAY,
       .type_length     = 16,
       .repetition_type = FieldRepetitionType::REQUIRED,
       .name            = "c",
       .field_id        = 3},
    };

    ColumnChunk const cc[] = {
      {.file_offset = 4,
       .meta_data   = {.type                    = Type::INT32,
                       .encodings               = {Encoding::PLAIN, Encoding::RLE_DICTIONARY},
                       .path_in_schema          = {"a"},
                       .codec                   = Compression::SNAPPY,
                       .num_values              = 12345,
                       .total_uncompressed_size = 1000,
                       .total_compressed_size   = 500,
                       .data_page_offset        = 8,
                       // non-zero exercises the dictionary_page_offset field
                       .dictionary_page_offset = 4}},
      {.file_offset = 504,
       .meta_data   = {.type                    = Type::BYTE_ARRAY,
                       .encodings               = {Encoding::PLAIN},
                       .path_in_schema          = {"b"},
                       .codec                   = Compression::ZSTD,
                       .num_values              = 12345,
                       .total_uncompressed_size = 2000,
                       .total_compressed_size   = 800,
                       .data_page_offset        = 504}},
    };

    meta.row_groups = {{.columns               = {cc[0], cc[1]},
                        .total_byte_size       = 3000,
                        .num_rows              = 12345,
                        .file_offset           = 4,
                        .total_compressed_size = 1300,
                        .ordinal               = static_cast<int16_t>(0)}};

    meta.key_value_metadata = {
      {"pandas", "{\"index\": 1}"},
      // Empty value re-serializes as absent and reads back empty (a documented delta).
      {"empty", ""},
    };
    meta.column_orders = {{{ColumnOrder::TYPE_ORDER}, {ColumnOrder::TYPE_ORDER}}};

    return meta;
  }();
  return meta;
}

void expect_schema_equal(SchemaElement const& e, SchemaElement const& a)
{
  EXPECT_EQ(e.type, a.type);
  EXPECT_EQ(e.name, a.name);
  // repetition_type == UNSPECIFIED is omitted by the writer and reads back as REQUIRED.
  auto const expected_rep = e.repetition_type == FieldRepetitionType::UNSPECIFIED
                              ? FieldRepetitionType::REQUIRED
                              : e.repetition_type;
  EXPECT_EQ(expected_rep, a.repetition_type);
  // type_length is written only for a typed leaf; a group node or a zero length reads back 0.
  auto const expected_len = e.type != Type::UNDEFINED ? e.type_length : 0;
  EXPECT_EQ(expected_len, a.type_length);
  // num_children is written only for group nodes (type == UNDEFINED); leaves read back 0.
  auto const expected_children = e.type == Type::UNDEFINED ? e.num_children : 0;
  EXPECT_EQ(expected_children, a.num_children);
  EXPECT_EQ(e.converted_type, a.converted_type);
  EXPECT_EQ(e.field_id, a.field_id);
}

void expect_column_meta_equal(ColumnChunkMetaData const& e, ColumnChunkMetaData const& a)
{
  EXPECT_EQ(e.type, a.type);
  EXPECT_EQ(e.encodings, a.encodings);
  EXPECT_EQ(e.path_in_schema, a.path_in_schema);
  EXPECT_EQ(e.codec, a.codec);
  EXPECT_EQ(e.num_values, a.num_values);
  EXPECT_EQ(e.total_uncompressed_size, a.total_uncompressed_size);
  EXPECT_EQ(e.total_compressed_size, a.total_compressed_size);
  EXPECT_EQ(e.data_page_offset, a.data_page_offset);
  EXPECT_EQ(e.index_page_offset, a.index_page_offset);
  EXPECT_EQ(e.dictionary_page_offset, a.dictionary_page_offset);
}

void expect_column_chunk_equal(ColumnChunk const& e, ColumnChunk const& a)
{
  EXPECT_EQ(e.file_path, a.file_path);
  EXPECT_EQ(e.file_offset, a.file_offset);
  expect_column_meta_equal(e.meta_data, a.meta_data);
  // The index offsets are written only alongside a non-zero length.
  EXPECT_EQ(e.offset_index_length, a.offset_index_length);
  EXPECT_EQ(e.column_index_length, a.column_index_length);
  if (e.offset_index_length != 0) { EXPECT_EQ(e.offset_index_offset, a.offset_index_offset); }
  if (e.column_index_length != 0) { EXPECT_EQ(e.column_index_offset, a.column_index_offset); }
}

void expect_row_group_equal(RowGroup const& e, RowGroup const& a)
{
  EXPECT_EQ(e.total_byte_size, a.total_byte_size);
  EXPECT_EQ(e.num_rows, a.num_rows);
  EXPECT_EQ(e.file_offset, a.file_offset);
  EXPECT_EQ(e.total_compressed_size, a.total_compressed_size);
  EXPECT_EQ(e.ordinal, a.ordinal);
  ASSERT_EQ(e.columns.size(), a.columns.size());
  for (size_t i = 0; i < e.columns.size(); ++i) {
    SCOPED_TRACE(std::format("column index {}", i));
    expect_column_chunk_equal(e.columns[i], a.columns[i]);
  }
}

// Compares only the fields the Thrift-compact codec serializes; cudf-internal derived fields
// (schema_idx, index blobs) are excluded.
void expect_footer_semantic_equal(FileMetaData const& e, FileMetaData const& a)
{
  EXPECT_EQ(e.version, a.version);
  EXPECT_EQ(e.num_rows, a.num_rows);
  EXPECT_EQ(e.created_by, a.created_by);

  ASSERT_EQ(e.schema.size(), a.schema.size());
  for (size_t i = 0; i < e.schema.size(); ++i) {
    SCOPED_TRACE(std::format("schema index {}", i));
    expect_schema_equal(e.schema[i], a.schema[i]);
  }

  ASSERT_EQ(e.row_groups.size(), a.row_groups.size());
  for (size_t i = 0; i < e.row_groups.size(); ++i) {
    SCOPED_TRACE(std::format("row group index {}", i));
    expect_row_group_equal(e.row_groups[i], a.row_groups[i]);
  }

  ASSERT_EQ(e.key_value_metadata.size(), a.key_value_metadata.size());
  for (size_t i = 0; i < e.key_value_metadata.size(); ++i) {
    SCOPED_TRACE(std::format("key/value index {}", i));
    EXPECT_EQ(e.key_value_metadata[i].key, a.key_value_metadata[i].key);
    EXPECT_EQ(e.key_value_metadata[i].value, a.key_value_metadata[i].value);
  }

  EXPECT_EQ(e.column_orders.has_value(), a.column_orders.has_value());
  if (e.column_orders.has_value() && a.column_orders.has_value()) {
    ASSERT_EQ(e.column_orders->size(), a.column_orders->size());
    for (size_t i = 0; i < e.column_orders->size(); ++i) {
      SCOPED_TRACE(std::format("column order index {}", i));
      EXPECT_EQ(e.column_orders.value()[i].type, a.column_orders.value()[i].type);
    }
  }
}

// Writes a small two-column table to a parquet host buffer and fills `out` with its parsed footer.
void read_written_footer(std::vector<char>& buffer, FileMetaData& out)
{
  auto col0 = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4, 5}};
  auto col1 = cudf::test::strings_column_wrapper{{"a", "bb", "ccc", "dddd", "eeeee"}};
  cudf::table_view const input({col0, col1});

  auto const opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, input).build();
  cudf::io::write_parquet(opts);

  auto const src   = cudf::io::source_info{cudf::host_span<std::byte const>{
    reinterpret_cast<std::byte const*>(buffer.data()), buffer.size()}};
  auto datasources = cudf::io::make_datasources(src);
  auto footers     = cudf::io::read_parquet_footers(datasources);
  // Fail hard before .at(0) so an unexpectedly empty footer reports a clear assertion, not an
  // uncaught std::out_of_range.
  ASSERT_EQ(footers.size(), 1);
  out = footers.at(0);
}

// One past the reader's 512-element threshold where struct-list parsing goes parallel.
constexpr uint32_t parallel_list_count = 513;

// Builds a row_groups LIST of `parallel_list_count` RowGroup structs, each carrying one
// lenient-skippable mismatched field (columns, expected LIST, sent as i32) to reach the parallel
// sub-readers.
std::vector<uint8_t> make_parallel_mismatch_footer()
{
  // clang-format off
  std::vector<uint8_t> footer{
    0x49,         // FileMetaData field 4 (row_groups), type LIST
    0xfc,         // list header: long-form size prefix, STRUCT elements
    0x81, 0x04};  // size varint = 513 (parallel_list_count)
  // clang-format on
  for (uint32_t i = 0; i < parallel_list_count; ++i) {
    // RowGroup field 1 (columns) as i32 = 1 -> mismatched wire type, then RowGroup STOP
    footer.insert(footer.end(), {0x15, 0x02, 0x00});
  }
  footer.push_back(0x00);  // FileMetaData STOP
  return footer;
}

}  // namespace

struct ParquetFooterFacadeTest : public cudf::test::BaseFixture {};

// A constructed footer survives write -> read with semantic equality.
TEST_F(ParquetFooterFacadeTest, RoundTrip)
{
  auto const original = make_test_footer();
  auto const bytes    = experimental::write_parquet_footer_bytes(original);
  ASSERT_FALSE(bytes.empty());
  auto const reparsed = experimental::read_parquet_footer_bytes(bytes);
  expect_footer_semantic_equal(original, reparsed);
}

// A footer parsed from a real cudf-written file survives a facade write -> read round-trip,
// exercising the repetition_type / num_children / meta_data nuances of a genuine schema.
TEST_F(ParquetFooterFacadeTest, RealFooterRoundTrip)
{
  std::vector<char> buffer;
  FileMetaData original;
  read_written_footer(buffer, original);
  ASSERT_FALSE(original.schema.empty());
  ASSERT_FALSE(original.row_groups.empty());

  auto const bytes    = experimental::write_parquet_footer_bytes(original);
  auto const reparsed = experimental::read_parquet_footer_bytes(bytes);
  expect_footer_semantic_equal(original, reparsed);
}

// A footer with no schema or row groups round-trips; an absent column_orders stays absent.
TEST_F(ParquetFooterFacadeTest, EmptyFooterRoundTrip)
{
  FileMetaData meta;
  meta.version  = 1;
  meta.num_rows = 0;

  auto const bytes  = experimental::write_parquet_footer_bytes(meta);
  auto const parsed = experimental::read_parquet_footer_bytes(bytes);
  EXPECT_EQ(parsed.version, 1);
  EXPECT_EQ(parsed.num_rows, 0);
  EXPECT_TRUE(parsed.schema.empty());
  EXPECT_TRUE(parsed.row_groups.empty());
  EXPECT_TRUE(parsed.key_value_metadata.empty());
  EXPECT_TRUE(parsed.created_by.empty());
  EXPECT_FALSE(parsed.column_orders.has_value());
}

// Regression: the facade stops at the struct terminator, so an over-length buffer reparses to the
// same metadata -- padding past the terminator (e.g. spark-rapids' length word) is ignored.
TEST_F(ParquetFooterFacadeTest, TrailingBytesAreTolerated)
{
  auto const original = make_test_footer();
  auto const exact    = experimental::write_parquet_footer_bytes(original);
  ASSERT_FALSE(exact.empty());

  // The appended footer-length word never reaches the reader, so its value here is just a
  // representative payload.
  auto const len = static_cast<uint32_t>(exact.size());
  std::vector<uint8_t> length_word_frame;
  for (int shift = 0; shift < 32; shift += 8) {
    length_word_frame.push_back(static_cast<uint8_t>(len >> shift));
  }
  std::vector<uint8_t> const garbage_tail{0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02};
  // The Parquet end-of-file magic bytes -- another realistic trailing shape.
  std::vector<uint8_t> const magic_frame_tail{'P', 'A', 'R', '1'};

  for (auto const& tail : {length_word_frame, garbage_tail, magic_frame_tail}) {
    SCOPED_TRACE(std::format("trailing tail of {} bytes", tail.size()));
    std::vector<uint8_t> over_length = exact;
    over_length.insert(over_length.end(), tail.begin(), tail.end());
    ASSERT_GT(over_length.size(), exact.size());
    auto const reparsed = experimental::read_parquet_footer_bytes(over_length);
    expect_footer_semantic_equal(original, reparsed);
  }
}

// Count guard: a field 2 (schema) struct-list header declaring 0x7fffffff elements with no
// bytes left is rejected before the resize.
TEST_F(ParquetFooterFacadeTest, OversizedContainerCountThrows)
{
  // clang-format off
  std::vector<uint8_t> const bomb{
    0x29,  // FileMetaData field 2 (schema), type LIST
    0xfc,  // list header: long-form size prefix, STRUCT elements
    0xff, 0xff, 0xff, 0xff, 0x07};  // size varint = 0x7fffffff
  // clang-format on
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(bomb), cudf::logic_error);
}

// Count guard: the I32 `encodings` primitive list under row_groups[0].columns[0].meta_data
// declares 0x7fffffff elements with no bytes left, hitting the primitive parquet_field_list guard
// (distinct from the struct-list guard above).
TEST_F(ParquetFooterFacadeTest, OversizedPrimitiveListCountThrows)
{
  // clang-format off
  std::vector<uint8_t> const bomb{
    0x49,  // FileMetaData field 4 (row_groups), type LIST
    0x1c,  // list header: 1 element, STRUCT
    0x19,  // RowGroup field 1 (columns), type LIST
    0x1c,  // list header: 1 element, STRUCT
    0x3c,  // ColumnChunk field 3 (meta_data), type STRUCT
    0x29,  // ColumnChunkMetaData field 2 (encodings), type LIST
    0xf5,  // list header: long-form size prefix, I32 elements
    0xff, 0xff, 0xff, 0xff, 0x07};  // size varint = 0x7fffffff
  // clang-format on
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(bomb), cudf::logic_error);
}

// Count guard: an unknown top-level LIST field routes to skip_struct_field, whose list branch
// declares 0x7fffffff elements with no bytes left, hitting the skip-path count guard.
TEST_F(ParquetFooterFacadeTest, OversizedSkippedListCountThrows)
{
  // clang-format off
  std::vector<uint8_t> const bomb{
    0x89,  // FileMetaData field 8 (unknown id), type LIST -> skip path
    0xfc,  // list header: long-form size prefix, STRUCT elements
    0xff, 0xff, 0xff, 0xff, 0x07};  // size varint = 0x7fffffff
  // clang-format on
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(bomb), cudf::logic_error);
}

// Count guard: an unknown top-level MAP field declares 0x7fffffff key/value pairs with no bytes
// left, hitting the map-specific count guard in skip_struct_field.
TEST_F(ParquetFooterFacadeTest, OversizedMapCountThrows)
{
  // clang-format off
  std::vector<uint8_t> const bomb{
    0x8b,  // FileMetaData field 8 (unknown id), type MAP -> skip path
    0xff, 0xff, 0xff, 0xff, 0x07};  // map size varint = 0x7fffffff, no pairs follow
  // clang-format on
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(bomb), cudf::logic_error);
}

// An unknown top-level MAP field with one scalar key/value pair is skipped and the parse stays in
// sync. The following known field uses long-form (absolute) field-id encoding, since an unknown
// FileMetaData id exceeds every known id (1-7) and field-id deltas cannot go backwards.
TEST_F(ParquetFooterFacadeTest, UnknownMapFieldIsSkipped)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x8b,        // FileMetaData field 8 (unknown id), type MAP -> skip path
    0x01,        //   map size = 1 key/value pair
    0x55,        //   key type i32 (high nibble) / value type i32 (low nibble)
    0x2a,        //   key varint (value immaterial, skipped)
    0x54,        //   value varint (value immaterial, skipped)
    0x05, 0x02,  // long-form field header: type i32, zigzag field id = 1 (version)
    0x54,        // version i32 = 42
    0x00};       // STOP
  // clang-format on
  auto const parsed = experimental::read_parquet_footer_bytes(footer);
  EXPECT_EQ(parsed.version, 42);
}

// An unknown top-level MAP field with a bool value is skipped without desync: a bool map element is
// a 1-byte i8 (unlike a bool struct field, whose value is in the type nibble), so skipping must
// consume that byte.
TEST_F(ParquetFooterFacadeTest, UnknownMapFieldWithBoolValueIsSkipped)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x8b,        // FileMetaData field 8 (unknown id), type MAP -> skip path
    0x01,        //   map size = 1 key/value pair
    0x51,        //   key type i32 (high nibble) / value type bool (low nibble)
    0x2a,        //   key i32 varint (value immaterial, skipped)
    0x01,        //   value bool: a 1-byte i8 element (must be skipped, not 0 bytes)
    0x05, 0x02,  // long-form field header: type i32, zigzag field id = 1 (version)
    0x54,        // version i32 = 42
    0x00};       // STOP
  // clang-format on
  auto const parsed = experimental::read_parquet_footer_bytes(footer);
  EXPECT_EQ(parsed.version, 42);
}

// An unknown top-level UUID field (16-byte payload) is skipped and the parse stays in sync.
TEST_F(ParquetFooterFacadeTest, UnknownUuidFieldIsSkipped)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x8d,        // FileMetaData field 8 (unknown id), type UUID -> skip path
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 16-byte UUID payload
    0x05, 0x02,  // long-form field header: type i32, zigzag field id = 1 (version)
    0x54,        // version i32 = 42
    0x00};       // STOP
  // clang-format on
  auto const parsed = experimental::read_parquet_footer_bytes(footer);
  EXPECT_EQ(parsed.version, 42);
}

// Lenient mode skips a known field whose wire type mismatches the schema (Thrift forward-compat)
// and keeps parsing; each footer below reproduces one such shape seen in real files.

// A scalar field (FileMetaData.version, id 1, expected i32) arriving as an empty LIST is skipped
// (version keeps its default); the following field is still read.
TEST_F(ParquetFooterFacadeTest, KnownScalarFieldWithWrongTypeIsSkipped)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x19, 0x05,  // field 1 (version) as an empty i32-LIST -> skipped
    0x26, 0x54,  // field 3 (num_rows) i64 = 42
    0x00};       // STOP
  // clang-format on
  auto const parsed =
    experimental::read_parquet_footer_bytes(footer, experimental::thrift_mismatch_policy::COMPAT);
  EXPECT_EQ(parsed.version, 0);
  EXPECT_EQ(parsed.num_rows, 42);
}

// An optional field (FileMetaData.column_orders, id 7, expected list) arriving as an i32 is skipped
// and stays absent (Thrift __isset == false), not set to a garbage value.
TEST_F(ParquetFooterFacadeTest, KnownOptionalFieldWithWrongTypeStaysUnset)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x15, 0x04,        // field 1 (version) i32 = 2
    0x65, 0xc6, 0x01,  // field 7 (column_orders) as i32 = 99 -> skipped
    0x00};             // STOP
  // clang-format on
  auto const parsed =
    experimental::read_parquet_footer_bytes(footer, experimental::thrift_mismatch_policy::COMPAT);
  EXPECT_EQ(parsed.version, 2);
  EXPECT_FALSE(parsed.column_orders.has_value());
}

// Reproduces apache/parquet-testing dict-page-offset-zero.parquet: a nested optional field
// (ColumnChunkMetaData.bloom_filter_length, id 15, expected i32) encoded as LIST<STRUCT> is skipped
// and left unset, preserving the surrounding row-group / column structure.
TEST_F(ParquetFooterFacadeTest, NestedOptionalFieldWithWrongTypeIsSkipped)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x49, 0x1c,  // FileMetaData field 4 (row_groups): LIST of 1 STRUCT
    0x19, 0x1c,  //   RowGroup field 1 (columns): LIST of 1 STRUCT
    0x3c,        //     ColumnChunk field 3 (meta_data): STRUCT
    0x15, 0x00,  //       ColumnChunkMetaData field 1 (type) i32 = 0
    0xe9, 0x0c,  //       field 15 (bloom_filter_length) as an empty LIST<STRUCT> -> skipped
    0x00,        //       ColumnChunkMetaData STOP
    0x00,        //     ColumnChunk STOP
    0x00,        //   RowGroup STOP
    0x00};       // FileMetaData STOP
  // clang-format on
  auto const parsed =
    experimental::read_parquet_footer_bytes(footer, experimental::thrift_mismatch_policy::COMPAT);
  ASSERT_EQ(parsed.row_groups.size(), 1);
  ASSERT_EQ(parsed.row_groups[0].columns.size(), 1);
  EXPECT_FALSE(parsed.row_groups[0].columns[0].meta_data.bloom_filter_length.has_value());
}

// A bool field (SortingColumn.descending, id 2) arrives as an i32. parquet_field_bool has its own
// strict/lenient branch (a bool's value lives in the wire-type nibble, bypassing the generic scalar
// check); the mismatched field is skipped and its neighbours are still read.
TEST_F(ParquetFooterFacadeTest, KnownBoolFieldWithWrongTypeIsSkipped)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x49, 0x1c,  // FileMetaData field 4 (row_groups): LIST of 1 STRUCT
    0x49, 0x1c,  //   RowGroup field 4 (sorting_columns): LIST of 1 STRUCT
    0x15, 0x00,  //     SortingColumn field 1 (column_idx) i32 = 0
    0x15, 0x02,  //     field 2 (descending) as i32 = 1 -> skipped
    0x11,        //     field 3 (nulls_first) bool = true (value lives in the type nibble)
    0x00,        //     SortingColumn STOP
    0x00,        //   RowGroup STOP
    0x00};       // FileMetaData STOP
  // clang-format on
  auto const parsed =
    experimental::read_parquet_footer_bytes(footer, experimental::thrift_mismatch_policy::COMPAT);
  ASSERT_EQ(parsed.row_groups.size(), 1);
  ASSERT_TRUE(parsed.row_groups[0].sorting_columns.has_value());
  ASSERT_EQ(parsed.row_groups[0].sorting_columns->size(), 1);
  auto const& sorting = parsed.row_groups[0].sorting_columns->front();
  EXPECT_EQ(sorting.column_idx, 0);
  EXPECT_FALSE(sorting.descending);  // skipped -> keeps its value-initialized default
  EXPECT_TRUE(sorting.nulls_first);  // the field after the skipped one is still read
}

// A union arm (ColumnOrder.type, id 1, expected STRUCT) arrives as an i32.
// parquet_field_union_enumerator likewise reimplements the strict/lenient decision; the arm is
// treated as absent (the enumerator keeps its value-initialized default) and parsing continues.
TEST_F(ParquetFooterFacadeTest, UnionFieldWithWrongTypeIsTreatedAbsent)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x79, 0x1c,  // FileMetaData field 7 (column_orders): LIST of 1 STRUCT
    0x15, 0x02,  //   ColumnOrder field 1 (type) as i32 = 1 -> arm treated absent
    0x00,        //   ColumnOrder STOP
    0x00};       // FileMetaData STOP
  // clang-format on
  auto const parsed =
    experimental::read_parquet_footer_bytes(footer, experimental::thrift_mismatch_policy::COMPAT);
  ASSERT_TRUE(parsed.column_orders.has_value());
  ASSERT_EQ(parsed.column_orders->size(), 1);
  EXPECT_EQ(parsed.column_orders.value()[0].type, ColumnOrder::UNDEFINED);
}

// Struct-list parsing goes parallel at 512+ elements; every sub-reader must inherit lenient mode.
// All 513 mismatched row groups parse -- a regression to strict sub-readers would throw here.
TEST_F(ParquetFooterFacadeTest, ParallelStructListPropagatesLenientMode)
{
  auto const footer = make_parallel_mismatch_footer();
  auto const parsed =
    experimental::read_parquet_footer_bytes(footer, experimental::thrift_mismatch_policy::COMPAT);
  ASSERT_EQ(parsed.row_groups.size(), parallel_list_count);
  // Every element, not just the boundaries, must have the mismatched field unset -- a
  // task-partitioning off-by-one would leave a silently-wrong middle range untested.
  EXPECT_TRUE(std::all_of(parsed.row_groups.begin(), parsed.row_groups.end(), [](auto const& rg) {
    return rg.columns.empty();
  }));
}

// The same wrong-type footer throws under the default strict mode: cudf's readers keep the
// exact-type contract; only the spark-rapids facade opts into leniency via
// `experimental::thrift_mismatch_policy::COMPAT`.
TEST_F(ParquetFooterFacadeTest, KnownFieldWithWrongTypeThrowsInStrictMode)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x19, 0x05,  // field 1 (version) as an empty i32-LIST -> wrong wire type
    0x26, 0x54,  // field 3 (num_rows) i64 = 42
    0x00};       // STOP
  // clang-format on
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(footer), cudf::logic_error);
}

// The wrong-typed bool field throws under the default strict mode via parquet_field_bool's own
// strict branch.
TEST_F(ParquetFooterFacadeTest, KnownBoolFieldWithWrongTypeThrowsInStrictMode)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x49, 0x1c,  // FileMetaData field 4 (row_groups): LIST of 1 STRUCT
    0x49, 0x1c,  //   RowGroup field 4 (sorting_columns): LIST of 1 STRUCT
    0x15, 0x00,  //     SortingColumn field 1 (column_idx) i32 = 0
    0x15, 0x02,  //     field 2 (descending) as i32 = 1 -> wrong wire type
    0x11,        //     field 3 (nulls_first) bool = true
    0x00,        //     SortingColumn STOP
    0x00,        //   RowGroup STOP
    0x00};       // FileMetaData STOP
  // clang-format on
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(footer), cudf::logic_error);
}

// The wrong-typed union arm throws under the default strict mode via
// parquet_field_union_enumerator's own strict branch.
TEST_F(ParquetFooterFacadeTest, UnionFieldWithWrongTypeThrowsInStrictMode)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x79, 0x1c,  // FileMetaData field 7 (column_orders): LIST of 1 STRUCT
    0x15, 0x02,  //   ColumnOrder field 1 (type) as i32 = 1 -> wrong wire type
    0x00,        //   ColumnOrder STOP
    0x00};       // FileMetaData STOP
  // clang-format on
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(footer), cudf::logic_error);
}

// The parallel sub-readers inherit the default strict mode too: the mismatch is detected in a
// worker task and the exception rethrown to the caller.
TEST_F(ParquetFooterFacadeTest, ParallelStructListThrowsInStrictMode)
{
  auto const footer = make_parallel_mismatch_footer();
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(footer), cudf::logic_error);
}

// A non-empty list with a wrong element-type nibble is skipped wholesale in COMPAT (its
// neighbouring fields still read); here `encodings` (expects I32) arrives as I64 elements.
TEST_F(ParquetFooterFacadeTest, PrimitiveListWithWrongElementTypeIsSkippedInCompat)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x49, 0x1c,  // FileMetaData field 4 (row_groups): LIST of 1 STRUCT
    0x19, 0x1c,  //   RowGroup field 1 (columns): LIST of 1 STRUCT
    0x3c,        //     ColumnChunk field 3 (meta_data): STRUCT
    0x29, 0x16,  //       field 2 (encodings): LIST, 1 element, I64 (not I32)
    0x02,        //         element zigzag varint I64 = 1
    0x36, 0x02,  //       field 5 (num_values) i64 = 1
    0x00,        //       ColumnChunkMetaData STOP
    0x00,        //     ColumnChunk STOP
    0x00,        //   RowGroup STOP
    0x00};       // FileMetaData STOP
  // clang-format on
  auto const parsed =
    experimental::read_parquet_footer_bytes(footer, experimental::thrift_mismatch_policy::COMPAT);
  ASSERT_EQ(parsed.row_groups.size(), 1);
  ASSERT_EQ(parsed.row_groups[0].columns.size(), 1);
  EXPECT_TRUE(parsed.row_groups[0].columns[0].meta_data.encodings.empty());
  EXPECT_EQ(parsed.row_groups[0].columns[0].meta_data.num_values, 1);
}

// The same footer throws under the default strict mode.
TEST_F(ParquetFooterFacadeTest, PrimitiveListWithWrongElementTypeThrowsInStrict)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x49, 0x1c,  // FileMetaData field 4 (row_groups): LIST of 1 STRUCT
    0x19, 0x1c,  //   RowGroup field 1 (columns): LIST of 1 STRUCT
    0x3c,        //     ColumnChunk field 3 (meta_data): STRUCT
    0x29, 0x16,  //       field 2 (encodings): LIST, 1 element, I64 (not I32)
    0x02,        //         element zigzag varint I64 = 1
    0x00,        //       ColumnChunkMetaData STOP
    0x00,        //     ColumnChunk STOP
    0x00,        //   RowGroup STOP
    0x00};       // FileMetaData STOP
  // clang-format on
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(footer), cudf::logic_error);
}

// Same for a STRUCT-element list: `key_value_metadata` (expects STRUCT) arrives as I32 elements.
TEST_F(ParquetFooterFacadeTest, StructListWithWrongElementTypeIsSkippedInCompat)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x15, 0x02,  // FileMetaData field 1 (version) i32 = 1
    0x26, 0x02,  // field 3 (num_rows) i64 = 1
    0x29, 0x15,  // field 5 (key_value_metadata): LIST, 1 element, I32 (not STRUCT)
    0x02,        //   ...zigzag varint I32 = 1
    0x00};       // FileMetaData STOP
  // clang-format on
  auto const parsed =
    experimental::read_parquet_footer_bytes(footer, experimental::thrift_mismatch_policy::COMPAT);
  EXPECT_EQ(parsed.version, 1);
  EXPECT_TRUE(parsed.key_value_metadata.empty());
  EXPECT_EQ(parsed.num_rows, 1);
}

CUDF_TEST_PROGRAM_MAIN()
