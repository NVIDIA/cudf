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
#include <optional>
#include <string>
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
      {.type            = Type::DOUBLE,
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
       .meta_data   = {.type                    = Type::DOUBLE,
                       .encodings               = {Encoding::PLAIN, Encoding::RLE_DICTIONARY},
                       .path_in_schema          = {"a"},
                       .codec                   = Compression::SNAPPY,
                       .num_values              = 12345,
                       .total_uncompressed_size = 1000,
                       .total_compressed_size   = 500,
                       .data_page_offset        = 8,
                       // non-zero exercises the dictionary_page_offset field
                       .dictionary_page_offset = 4,
                       .statistics             = {.null_count = 0, .nan_count = 3}}},
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
      {"empty", std::string("")},
      {"novalue", std::nullopt},
    };
    meta.column_orders = {
      {{ColumnOrder::IEEE_754_TOTAL_ORDER}, {ColumnOrder::TYPE_ORDER}, {ColumnOrder::TYPE_ORDER}}};

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
  EXPECT_EQ(e.statistics.null_count, a.statistics.null_count);
  EXPECT_EQ(e.statistics.nan_count, a.statistics.nan_count);
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
    EXPECT_EQ(e.key_value_metadata[i].value.has_value(), a.key_value_metadata[i].value.has_value());
    if (e.key_value_metadata[i].value.has_value() && a.key_value_metadata[i].value.has_value()) {
      EXPECT_EQ(e.key_value_metadata[i].value.value(), a.key_value_metadata[i].value.value());
    }
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
// `kv_metadata` carries per-sink user key-value metadata forwarded to the writer.
void read_written_footer(std::vector<char>& buffer,
                         FileMetaData& out,
                         std::vector<std::map<std::string, std::string>> kv_metadata = {})
{
  auto col0 = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4, 5}};
  auto col1 = cudf::test::strings_column_wrapper{{"a", "bb", "ccc", "dddd", "eeeee"}};
  cudf::table_view const input({col0, col1});

  auto builder = cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, input);
  if (not kv_metadata.empty()) { builder.key_value_metadata(std::move(kv_metadata)); }
  auto const opts = builder.build();
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

// Spark's empty LEGACY-rebase marker values must remain present after a footer round trip.
TEST_F(ParquetFooterFacadeTest, EmptyKeyValueValueRoundTrip)
{
  FileMetaData meta;
  meta.version            = 2;
  meta.num_rows           = 1;
  meta.key_value_metadata = {
    {"org.apache.spark.legacyDateTime", std::string("")},
    {"org.apache.spark.legacyINT96", std::string("")},
  };

  auto const bytes = experimental::write_parquet_footer_bytes(meta);
  ASSERT_FALSE(bytes.empty());
  auto const reparsed = experimental::read_parquet_footer_bytes(bytes);
  ASSERT_EQ(reparsed.key_value_metadata.size(), 2);
  EXPECT_EQ(reparsed.key_value_metadata[0].key, "org.apache.spark.legacyDateTime");
  ASSERT_TRUE(reparsed.key_value_metadata[0].value.has_value());
  EXPECT_EQ(reparsed.key_value_metadata[0].value.value(), "");
  EXPECT_EQ(reparsed.key_value_metadata[1].key, "org.apache.spark.legacyINT96");
  ASSERT_TRUE(reparsed.key_value_metadata[1].value.has_value());
  EXPECT_EQ(reparsed.key_value_metadata[1].value.value(), "");
}

// A key-value entry written through `write_parquet` with an empty string value survives as a
// present-empty entry: the `map<string, string>` API cannot express absence, so present-empty is
// the intended write-path behavior.
TEST_F(ParquetFooterFacadeTest, EmptyKeyValueValueWrittenByWriter)
{
  std::vector<char> buffer;
  FileMetaData original;
  read_written_footer(buffer, original, {{{"key.empty", ""}, {"key.value", "v"}}});
  ASSERT_EQ(original.key_value_metadata.size(), 2);
  EXPECT_EQ(original.key_value_metadata[0].key, "key.empty");
  ASSERT_TRUE(original.key_value_metadata[0].value.has_value());
  EXPECT_EQ(original.key_value_metadata[0].value.value(), "");
  EXPECT_EQ(original.key_value_metadata[1].key, "key.value");
  ASSERT_TRUE(original.key_value_metadata[1].value.has_value());
  EXPECT_EQ(original.key_value_metadata[1].value.value(), "v");
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

// ---- ColumnOrder / LogicalType union-field regressions ----
// Parquet's ColumnOrder union defines fields 1-3 and LogicalType fields 1-8, 10-19 (9 reserved).
// Parquet-java >= 1.18.1 (Spark master since SPARK-58869) writes
// ColumnOrder.IEEE_754_TOTAL_ORDER for FLOAT/DOUBLE/FLOAT16 columns, which used to crash the
// native read -> prune -> write footer path with CUDF_FAIL on the unmodeled field.

namespace {

// Pairs a logical type with its payload so the two cannot drift apart.
struct LogicalTypeLeaf {
  LogicalType type;
  LogicalTypePayload payload;
};

// Builds a minimal footer whose schema is a root with one leaf per given logical type.
[[nodiscard]] FileMetaData make_logical_type_footer(std::vector<LogicalTypeLeaf> const& leaves)
{
  FileMetaData meta;
  meta.version  = 1;
  meta.num_rows = 0;

  meta.schema.push_back({.type            = Type::UNDEFINED,
                         .repetition_type = FieldRepetitionType::REQUIRED,
                         .name            = "schema",
                         .num_children    = static_cast<int32_t>(leaves.size())});
  for (size_t i = 0; i < leaves.size(); ++i) {
    meta.schema.push_back({.type                 = Type::BYTE_ARRAY,
                           .repetition_type      = FieldRepetitionType::REQUIRED,
                           .name                 = "col" + std::to_string(i),
                           .logical_type         = leaves[i].type,
                           .logical_type_payload = leaves[i].payload});
  }
  return meta;
}

}  // namespace

// All three ColumnOrder fields survive a write -> read -> write round-trip in order.
TEST_F(ParquetFooterFacadeTest, MixedColumnOrdersRoundTrip)
{
  FileMetaData meta;
  meta.version       = 2;
  meta.num_rows      = 4;
  meta.column_orders = std::vector<ColumnOrder>{{ColumnOrder::TYPE_ORDER},
                                                {ColumnOrder::IEEE_754_TOTAL_ORDER},
                                                {ColumnOrder::INT96_TIMESTAMP_ORDER},
                                                {ColumnOrder::TYPE_ORDER}};

  auto const bytes    = experimental::write_parquet_footer_bytes(meta);
  auto const reparsed = experimental::read_parquet_footer_bytes(bytes);
  ASSERT_TRUE(reparsed.column_orders.has_value());
  ASSERT_EQ(reparsed.column_orders->size(), 4);
  EXPECT_EQ(reparsed.column_orders.value()[0].type, ColumnOrder::TYPE_ORDER);
  EXPECT_EQ(reparsed.column_orders.value()[1].type, ColumnOrder::IEEE_754_TOTAL_ORDER);
  EXPECT_EQ(reparsed.column_orders.value()[2].type, ColumnOrder::INT96_TIMESTAMP_ORDER);
  EXPECT_EQ(reparsed.column_orders.value()[3].type, ColumnOrder::TYPE_ORDER);

  EXPECT_EQ(experimental::write_parquet_footer_bytes(reparsed), bytes);
}

// A footer written by parquet-java >= 1.18.1 carries the IEEE_754_TOTAL_ORDER and
// INT96_TIMESTAMP_ORDER ColumnOrder fields (union field ids 2 and 3), which previously parsed to
// UNDEFINED and crashed the writer. Both orders matter independently -- TYPE_ORDER is not
// equivalent to either for min/max comparison -- so one footer carries both and their binding and
// byte identity must hold together. The bytes below are the exact canonical compact-protocol output
// of write_parquet_footer_bytes for this metadata (empty schema/row-groups are emitted
// unconditionally, so the literal includes them).
TEST_F(ParquetFooterFacadeTest, BothColumnOrderBytesRoundTrip)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x15, 0x02,        // field 1 (version) i32 = 1
    0x19, 0x0c,        // field 2 (schema): empty LIST<STRUCT>
    0x16, 0x00,        // field 3 (num_rows) i64 = 0
    0x19, 0x0c,        // field 4 (row_groups): empty LIST<STRUCT>
    0x39, 0x2c,        // field 7 (column_orders): LIST, 2 elements, STRUCT
    0x2c, 0x00, 0x00,  //   ColumnOrder field 2 (IEEE_754_TOTAL_ORDER) empty STRUCT + STOP
    0x3c, 0x00, 0x00,  //   ColumnOrder field 3 (INT96_TIMESTAMP_ORDER) empty STRUCT + STOP
    0x00};             // FileMetaData STOP
  // clang-format on
  auto const parsed = experimental::read_parquet_footer_bytes(footer);
  EXPECT_EQ(parsed.version, 1);
  ASSERT_TRUE(parsed.column_orders.has_value());
  ASSERT_EQ(parsed.column_orders->size(), 2);
  EXPECT_EQ(parsed.column_orders.value()[0].type, ColumnOrder::IEEE_754_TOTAL_ORDER);
  EXPECT_EQ(parsed.column_orders.value()[1].type, ColumnOrder::INT96_TIMESTAMP_ORDER);

  // The canonical writer output for the parsed footer is byte-identical to the input.
  auto const rewritten = experimental::write_parquet_footer_bytes(parsed);
  EXPECT_EQ(rewritten, footer);
}

// An UNDEFINED (or unknown future) ColumnOrder field is a valid parse result in COMPAT but the
// writer must reject it with cudf::logic_error; never emit an empty union for a meaningless field.
TEST_F(ParquetFooterFacadeTest, InvalidColumnOrderFails)
{
  FileMetaData meta;
  meta.version       = 1;
  meta.column_orders = std::vector<ColumnOrder>{{ColumnOrder::UNDEFINED}};
  EXPECT_THROW((void)experimental::write_parquet_footer_bytes(meta), cudf::logic_error);

  // clang-format off
  std::vector<uint8_t> const future_footer{
    0x15, 0x02,        // field 1 (version) i32 = 1
    0x19, 0x0c,        // field 2 (schema): empty LIST<STRUCT>
    0x16, 0x00,        // field 3 (num_rows) i64 = 0
    0x19, 0x0c,        // field 4 (row_groups): empty LIST<STRUCT>
    0x39, 0x1c,        // field 7 (column_orders): LIST, 1 element, STRUCT
    0x4c, 0x00, 0x00,  //   ColumnOrder field 4 (unknown future field) empty STRUCT + STOP -> unbound
    0x00};             // FileMetaData STOP
  // clang-format on
  auto const parsed = experimental::read_parquet_footer_bytes(
    future_footer, experimental::thrift_mismatch_policy::COMPAT);
  ASSERT_TRUE(parsed.column_orders.has_value());
  ASSERT_EQ(parsed.column_orders->size(), 1);
  EXPECT_EQ(parsed.column_orders.value()[0].type, ColumnOrder::UNDEFINED);
  EXPECT_THROW((void)experimental::write_parquet_footer_bytes(parsed), cudf::logic_error);
}

// The payloadless LogicalType fields UUID, FLOAT16, GEOMETRY (tag-only: its string `crs` payload is
// not retained) and FILE round-trip with their tags intact. The GEOMETRY leaf additionally pins the
// engaged-payload path: the writer emits an empty struct for a payloadless GEOMETRY, and the reader
// binds it to an engaged payload with `has_crs` false.
TEST_F(ParquetFooterFacadeTest, EmptyLogicalTypeFieldsRoundTrip)
{
  auto const meta = make_logical_type_footer({{LogicalType{LogicalType::UUID}},
                                              {LogicalType{LogicalType::FLOAT16}},
                                              {LogicalType{LogicalType::GEOMETRY}},
                                              {LogicalType{LogicalType::FILE}}});

  auto const bytes    = experimental::write_parquet_footer_bytes(meta);
  auto const reparsed = experimental::read_parquet_footer_bytes(bytes);
  ASSERT_EQ(reparsed.schema.size(), 5);
  for (size_t i = 1; i < reparsed.schema.size(); ++i) {
    SCOPED_TRACE(std::format("schema index {}", i));
    ASSERT_TRUE(reparsed.schema[i].logical_type.has_value());
    EXPECT_EQ(reparsed.schema[i].logical_type->type, meta.schema[i].logical_type->type);
  }

  // The empty GEOMETRY struct reparses as an engaged payload with no `crs` recorded.
  SCOPED_TRACE("schema index 3");
  ASSERT_TRUE(reparsed.schema[3].logical_type_payload.geometry_type.has_value());
  EXPECT_FALSE(reparsed.schema[3].logical_type_payload.geometry_type->has_crs);

  EXPECT_EQ(experimental::write_parquet_footer_bytes(reparsed), bytes);
}

// The payload-carrying fields keep their payloads: VARIANT's `specification_version` i8 and
// GEOGRAPHY's `algorithm` i32.
TEST_F(ParquetFooterFacadeTest, PayloadLogicalTypeFieldsRoundTrip)
{
  VariantType variant;
  variant.specification_version = 1;
  GeographyType geography;
  geography.algorithm = EdgeInterpolationAlgorithm::KARNEY;

  LogicalType variant_type{LogicalType::VARIANT};
  LogicalType geography_type{LogicalType::GEOGRAPHY};
  LogicalTypePayload variant_payload;
  variant_payload.variant_type = variant;
  LogicalTypePayload geography_payload;
  geography_payload.geography_type = geography;

  auto const meta = make_logical_type_footer(
    {{variant_type, variant_payload}, {geography_type, geography_payload}});
  auto const bytes    = experimental::write_parquet_footer_bytes(meta);
  auto const reparsed = experimental::read_parquet_footer_bytes(bytes);
  ASSERT_EQ(reparsed.schema.size(), 3);
  ASSERT_TRUE(reparsed.schema[1].logical_type.has_value());
  ASSERT_TRUE(reparsed.schema[2].logical_type.has_value());
  EXPECT_EQ(reparsed.schema[1].logical_type->type, LogicalType::VARIANT);
  EXPECT_EQ(reparsed.schema[2].logical_type->type, LogicalType::GEOGRAPHY);
  ASSERT_TRUE(reparsed.schema[1].logical_type_payload.variant_type.has_value());
  ASSERT_TRUE(
    reparsed.schema[1].logical_type_payload.variant_type->specification_version.has_value());
  EXPECT_EQ(reparsed.schema[1].logical_type_payload.variant_type->specification_version.value(), 1);
  ASSERT_TRUE(reparsed.schema[2].logical_type_payload.geography_type.has_value());
  ASSERT_TRUE(reparsed.schema[2].logical_type_payload.geography_type->algorithm.has_value());
  EXPECT_EQ(reparsed.schema[2].logical_type_payload.geography_type->algorithm.value(),
            EdgeInterpolationAlgorithm::KARNEY);

  EXPECT_EQ(experimental::write_parquet_footer_bytes(reparsed), bytes);

  // A conforming external writer emits `crs` (thrift id 1) before `algorithm` (id 2); cudf
  // records only the presence of `crs` on read, preserves `algorithm`, and rejects the rewrite.
  // Hand-built leaf carrying GeographyType field 1 (BINARY "EPSG:4326") then field 2
  // (i32 algorithm 0), as a conforming external writer emits them. Field 18 needs the long-form
  // id encoding (delta 0 + zigzag varint 36) inside LogicalType.
  // clang-format off
  std::vector<uint8_t> const crs_footer{
    0x15, 0x02,        // field 1 (version) i32 = 1
    0x19, 0x1c,        // field 2 (schema): LIST, 1 element, STRUCT
    0x48, 0x01, 0x61,  //   SchemaElement field 4 (name): BINARY len 1 "a"
    0x6c,              //   field 10 (logical_type): STRUCT
    0x0c, 0x24,        //     field 18 (GEOGRAPHY) STRUCT: long-form id, GeographyType body
    0x18, 0x09, 'E', 'P', 'S', 'G', ':', '4', '3', '2', '6',  // field 1 (crs) BINARY len 9
    0x15, 0x00,        // field 2 (algorithm) i32 = 0 (SPHERICAL)
    0x00,              //   GeographyType STOP
    0x00,              //   LogicalType STOP
    0x00,              //   SchemaElement STOP
    0x00};             // FileMetaData STOP
  // clang-format on
  auto const crs_parsed = experimental::read_parquet_footer_bytes(crs_footer);
  ASSERT_TRUE(crs_parsed.schema[0].logical_type.has_value());
  EXPECT_EQ(crs_parsed.schema[0].logical_type->type, LogicalType::GEOGRAPHY);
  ASSERT_TRUE(crs_parsed.schema[0].logical_type_payload.geography_type.has_value());
  ASSERT_TRUE(crs_parsed.schema[0].logical_type_payload.geography_type->algorithm.has_value());
  EXPECT_EQ(crs_parsed.schema[0].logical_type_payload.geography_type->algorithm.value(),
            EdgeInterpolationAlgorithm::SPHERICAL);

  // The dropped `crs` value must not be silently rewritten (absent crs means OGC:CRS84): the
  // writer refuses rather than emitting a footer with altered CRS semantics.
  EXPECT_THROW((void)experimental::write_parquet_footer_bytes(crs_parsed), cudf::logic_error);
}

// Mirror of the GEOGRAPHY crs test for field 17: a conforming external footer carrying a GEOMETRY
// `crs` string parses with `has_crs` set and the writer refuses the rewrite.
TEST_F(ParquetFooterFacadeTest, GeometryCrsRefusesRewrite)
{
  // Hand-built leaf carrying a GeometryType field 1 (BINARY "EPSG:4326"); field 17 uses the
  // long-form id encoding (delta 0 + zigzag varint 34) inside LogicalType.
  // clang-format off
  std::vector<uint8_t> const crs_footer{
    0x15, 0x02,        // field 1 (version) i32 = 1
    0x19, 0x1c,        // field 2 (schema): LIST, 1 element, STRUCT
    0x48, 0x01, 0x61,  //   SchemaElement field 4 (name): BINARY len 1 "a"
    0x6c,              //   field 10 (logical_type): STRUCT
    0x0c, 0x22,        //     field 17 (GEOMETRY) STRUCT: long-form id, GeometryType body
    0x18, 0x09, 'E', 'P', 'S', 'G', ':', '4', '3', '2', '6',  // field 1 (crs) BINARY len 9
    0x00,              //   GeometryType STOP
    0x00,              //   LogicalType STOP
    0x00,              //   SchemaElement STOP
    0x00};             // FileMetaData STOP
  // clang-format on
  auto const crs_parsed = experimental::read_parquet_footer_bytes(crs_footer);
  ASSERT_TRUE(crs_parsed.schema[0].logical_type.has_value());
  EXPECT_EQ(crs_parsed.schema[0].logical_type->type, LogicalType::GEOMETRY);
  ASSERT_TRUE(crs_parsed.schema[0].logical_type_payload.geometry_type.has_value());
  EXPECT_TRUE(crs_parsed.schema[0].logical_type_payload.geometry_type->has_crs);

  EXPECT_THROW((void)experimental::write_parquet_footer_bytes(crs_parsed), cudf::logic_error);
}

// Optional payloads keep presence exactly: an absent field stays absent, and a zero value stays
// present (not conflated with absence). An out-of-range algorithm value round-trips numerically --
// the enum binding reads the raw i32 with no validation, by design.
TEST_F(ParquetFooterFacadeTest, OptionalLogicalTypePayloadsRoundTrip)
{
  // Absent payloads and zero payloads on separate leaves.
  LogicalType variant_absent{LogicalType::VARIANT};
  LogicalType variant_zero{LogicalType::VARIANT};
  LogicalTypePayload variant_zero_payload;
  variant_zero_payload.variant_type                        = VariantType{};
  variant_zero_payload.variant_type->specification_version = 0;
  LogicalType geography_spherical{LogicalType::GEOGRAPHY};
  LogicalTypePayload geography_spherical_payload;
  geography_spherical_payload.geography_type            = GeographyType{};
  geography_spherical_payload.geography_type->algorithm = EdgeInterpolationAlgorithm::SPHERICAL;

  auto const meta     = make_logical_type_footer({{variant_absent, LogicalTypePayload{}},
                                                  {variant_zero, variant_zero_payload},
                                                  {geography_spherical, geography_spherical_payload}});
  auto const bytes    = experimental::write_parquet_footer_bytes(meta);
  auto const reparsed = experimental::read_parquet_footer_bytes(bytes);
  ASSERT_EQ(reparsed.schema.size(), 4);
  // Absent stays absent.
  ASSERT_TRUE(reparsed.schema[1].logical_type_payload.variant_type.has_value());
  EXPECT_FALSE(
    reparsed.schema[1].logical_type_payload.variant_type->specification_version.has_value());
  // Zero version stays present.
  ASSERT_TRUE(
    reparsed.schema[2].logical_type_payload.variant_type->specification_version.has_value());
  EXPECT_EQ(reparsed.schema[2].logical_type_payload.variant_type->specification_version.value(), 0);
  // SPHERICAL (0) stays present.
  ASSERT_TRUE(reparsed.schema[3].logical_type_payload.geography_type->algorithm.has_value());
  EXPECT_EQ(reparsed.schema[3].logical_type_payload.geography_type->algorithm.value(),
            EdgeInterpolationAlgorithm::SPHERICAL);
  EXPECT_EQ(experimental::write_parquet_footer_bytes(reparsed), bytes);

  // An out-of-range algorithm value (7) survives numerically in both strict and COMPAT modes.
  LogicalType geography_unknown{LogicalType::GEOGRAPHY};
  LogicalTypePayload geography_unknown_payload;
  geography_unknown_payload.geography_type            = GeographyType{};
  geography_unknown_payload.geography_type->algorithm = static_cast<EdgeInterpolationAlgorithm>(7);
  auto const unknown_meta =
    make_logical_type_footer({{geography_unknown, geography_unknown_payload}});
  auto const unknown_bytes = experimental::write_parquet_footer_bytes(unknown_meta);
  for (auto const mode : {experimental::thrift_mismatch_policy::THROW,
                          experimental::thrift_mismatch_policy::COMPAT}) {
    SCOPED_TRACE(std::format("mismatch policy {}", static_cast<int>(mode)));
    auto const parsed = experimental::read_parquet_footer_bytes(unknown_bytes, mode);
    ASSERT_TRUE(parsed.schema[1].logical_type.has_value());
    ASSERT_TRUE(parsed.schema[1].logical_type_payload.geography_type.has_value());
    ASSERT_TRUE(parsed.schema[1].logical_type_payload.geography_type->algorithm.has_value());
    EXPECT_EQ(
      static_cast<int32_t>(parsed.schema[1].logical_type_payload.geography_type->algorithm.value()),
      7);
  }
}

// An UNDEFINED, reserved (9), or unknown future LogicalType field is rejected by the writer with
// cudf::logic_error; an absent logical_type stays absent and writable. The bare-constructor forms
// `LogicalType{VARIANT}` / `LogicalType{GEOGRAPHY}` (disengaged payloads) are VALID thrift values
// -- all payload fields are optional -- so the writer emits an empty struct rather than throwing
// bad_optional_access, and they round-trip as the same tag with absent payload.
TEST_F(ParquetFooterFacadeTest, InvalidLogicalTypeFails)
{
  for (auto const bad_type : {
         LogicalType::UNDEFINED,
         static_cast<LogicalType::Type>(9),  // reserved
         static_cast<LogicalType::Type>(20)  // unknown future field
       }) {
    SCOPED_TRACE(std::format("logical type {}", static_cast<int>(bad_type)));
    auto const meta = make_logical_type_footer({{LogicalType{bad_type}}});
    EXPECT_THROW((void)experimental::write_parquet_footer_bytes(meta), cudf::logic_error);
  }

  // A leaf with no logical_type is unaffected.
  FileMetaData meta;
  meta.version = 1;
  meta.schema  = {
    {.type            = Type::UNDEFINED,
      .repetition_type = FieldRepetitionType::REQUIRED,
      .name            = "schema",
      .num_children    = 1},
    {.type = Type::BYTE_ARRAY, .repetition_type = FieldRepetitionType::REQUIRED, .name = "plain"}};
  auto const bytes    = experimental::write_parquet_footer_bytes(meta);
  auto const reparsed = experimental::read_parquet_footer_bytes(bytes);
  EXPECT_FALSE(reparsed.schema[1].logical_type.has_value());
  EXPECT_EQ(experimental::write_parquet_footer_bytes(reparsed), bytes);

  // Disengaged payloads write empty structs and round-trip as the same tag.
  for (auto const tag : {LogicalType::VARIANT, LogicalType::GEOGRAPHY}) {
    SCOPED_TRACE(std::format("logical type {}", static_cast<int>(tag)));
    auto const payload_meta  = make_logical_type_footer({{LogicalType{tag}}});
    auto const payload_bytes = experimental::write_parquet_footer_bytes(payload_meta);
    auto const parsed        = experimental::read_parquet_footer_bytes(payload_bytes);
    ASSERT_TRUE(parsed.schema[1].logical_type.has_value());
    EXPECT_EQ(parsed.schema[1].logical_type->type, tag);
    if (tag == LogicalType::VARIANT) {
      ASSERT_TRUE(parsed.schema[1].logical_type_payload.variant_type.has_value());
      EXPECT_FALSE(
        parsed.schema[1].logical_type_payload.variant_type->specification_version.has_value());
    } else {
      ASSERT_TRUE(parsed.schema[1].logical_type_payload.geography_type.has_value());
      EXPECT_FALSE(parsed.schema[1].logical_type_payload.geography_type->algorithm.has_value());
    }
    EXPECT_EQ(experimental::write_parquet_footer_bytes(parsed), payload_bytes);
  }
}

// A payload member engaged for a union field other than its own would be dropped silently, so the
// writer refuses it; each of the three guards is reached by a distinct mismatch.
TEST_F(ParquetFooterFacadeTest, MismatchedLogicalTypePayloadIsRefused)
{
  LogicalTypePayload variant_payload{.variant_type = VariantType{}};
  LogicalTypePayload geometry_payload{.geometry_type = GeometryType{}};
  LogicalTypePayload geography_payload{.geography_type = GeographyType{}};

  // VARIANT carrying a Geography payload.
  auto const variant_with_geography =
    make_logical_type_footer({{LogicalType{LogicalType::VARIANT}, geography_payload}});
  EXPECT_THROW((void)experimental::write_parquet_footer_bytes(variant_with_geography),
               cudf::logic_error);

  // GEOGRAPHY carrying a Variant payload.
  auto const geography_with_variant =
    make_logical_type_footer({{LogicalType{LogicalType::GEOGRAPHY}, variant_payload}});
  EXPECT_THROW((void)experimental::write_parquet_footer_bytes(geography_with_variant),
               cudf::logic_error);

  // VARIANT carrying a Geometry payload.
  auto const variant_with_geometry =
    make_logical_type_footer({{LogicalType{LogicalType::VARIANT}, geometry_payload}});
  EXPECT_THROW((void)experimental::write_parquet_footer_bytes(variant_with_geometry),
               cudf::logic_error);
}

// A malformed known LogicalType field (GEOMETRY, id 17, expected STRUCT) encoded as i32 keeps the
// mismatch policy: COMPAT skips the field (the inner type stays UNDEFINED and the writer rejects
// it), strict mode throws on parse. Malformed input must not be silently reinterpreted.
TEST_F(ParquetFooterFacadeTest, LogicalTypeWrongWireTypeInCompatRemainsInvalid)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x29, 0x1c,        // FileMetaData field 2 (schema): LIST, 1 element, STRUCT
    0x48, 0x01, 0x61,  //   SchemaElement field 4 (name): BINARY len 1 "a"
    0x6c,              //   field 10 (logical_type): STRUCT
    0x05, 0x22, 0x00,  //     field 17 (GEOMETRY) as i32: long-form id 17, value 0 -> mismatch
    0x00,              //   LogicalType STOP
    0x00,              //   SchemaElement STOP
    0x00};             // FileMetaData STOP
  // clang-format on
  auto const parsed =
    experimental::read_parquet_footer_bytes(footer, experimental::thrift_mismatch_policy::COMPAT);
  ASSERT_EQ(parsed.schema.size(), 1);
  ASSERT_TRUE(parsed.schema[0].logical_type.has_value());
  EXPECT_EQ(parsed.schema[0].logical_type->type, LogicalType::UNDEFINED);
  EXPECT_THROW((void)experimental::write_parquet_footer_bytes(parsed), cudf::logic_error);
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(footer), cudf::logic_error);
}

// A malformed SchemaElement field 10 (logical_type, expected STRUCT) encoded as i32 keeps the
// mismatch policy: COMPAT skips the field (logical_type stays absent), strict mode throws on parse.
TEST_F(ParquetFooterFacadeTest, LogicalTypeFieldWithWrongWireTypeIsSkipped)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x29, 0x1c,        // FileMetaData field 2 (schema): LIST, 1 element, STRUCT
    0x48, 0x01, 0x61,  //   SchemaElement field 4 (name): BINARY len 1 "a"
    0x65, 0x02,        //   field 10 (logical_type) as i32 = 1 -> mismatch, skipped
    0x00,              //   SchemaElement STOP
    0x00};             // FileMetaData STOP
  // clang-format on
  auto const parsed =
    experimental::read_parquet_footer_bytes(footer, experimental::thrift_mismatch_policy::COMPAT);
  ASSERT_EQ(parsed.schema.size(), 1);
  EXPECT_FALSE(parsed.schema[0].logical_type.has_value());
  EXPECT_FALSE(parsed.schema[0].logical_type_payload.variant_type.has_value());
  EXPECT_THROW((void)experimental::read_parquet_footer_bytes(footer), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
