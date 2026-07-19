/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/datasource.hpp>
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
#include <limits>
#include <string>
#include <vector>

namespace pq = cudf::io::parquet;

namespace {

// Wraps a byte vector as the host_span the facade consumes.
cudf::host_span<uint8_t const> as_span(std::vector<uint8_t> const& bytes)
{
  return cudf::host_span<uint8_t const>{bytes.data(), bytes.size()};
}

// Builds a fully-populated footer whose values all survive the writer's conditional emission, so a
// write -> read round-trip is exactly recoverable.
pq::FileMetaData make_test_footer()
{
  pq::FileMetaData meta;
  meta.version    = 2;
  meta.num_rows   = 12345;
  meta.created_by = "cudf-facade-test";

  pq::SchemaElement root;
  root.type            = pq::Type::UNDEFINED;
  root.repetition_type = pq::FieldRepetitionType::REQUIRED;
  root.name            = "schema";
  root.num_children    = 3;
  meta.schema.push_back(root);

  pq::SchemaElement a;
  a.type            = pq::Type::INT32;
  a.repetition_type = pq::FieldRepetitionType::OPTIONAL;
  a.name            = "a";
  a.field_id        = 1;
  meta.schema.push_back(a);

  pq::SchemaElement b;
  b.type            = pq::Type::BYTE_ARRAY;
  b.repetition_type = pq::FieldRepetitionType::REQUIRED;
  b.name            = "b";
  b.converted_type  = pq::ConvertedType::UTF8;
  b.field_id        = 2;
  meta.schema.push_back(b);

  pq::SchemaElement c;
  c.type            = pq::Type::FIXED_LEN_BYTE_ARRAY;
  c.type_length     = 16;  // exercises the schema type_length field (written only for a typed leaf)
  c.repetition_type = pq::FieldRepetitionType::REQUIRED;
  c.name            = "c";
  c.field_id        = 3;
  meta.schema.push_back(c);

  pq::RowGroup rg;
  rg.total_byte_size       = 3000;
  rg.num_rows              = 12345;
  rg.file_offset           = 4;
  rg.total_compressed_size = 1300;
  rg.ordinal               = static_cast<int16_t>(0);

  pq::ColumnChunk cc0;
  cc0.file_offset                       = 4;
  cc0.meta_data.type                    = pq::Type::INT32;
  cc0.meta_data.encodings               = {pq::Encoding::PLAIN, pq::Encoding::RLE_DICTIONARY};
  cc0.meta_data.path_in_schema          = {"a"};
  cc0.meta_data.codec                   = pq::Compression::SNAPPY;
  cc0.meta_data.num_values              = 12345;
  cc0.meta_data.total_uncompressed_size = 1000;
  cc0.meta_data.total_compressed_size   = 500;
  cc0.meta_data.data_page_offset        = 8;
  cc0.meta_data.dictionary_page_offset  = 4;  // non-zero exercises the dictionary_page_offset field
  rg.columns.push_back(cc0);

  pq::ColumnChunk cc1;
  cc1.file_offset                       = 504;
  cc1.meta_data.type                    = pq::Type::BYTE_ARRAY;
  cc1.meta_data.encodings               = {pq::Encoding::PLAIN};
  cc1.meta_data.path_in_schema          = {"b"};
  cc1.meta_data.codec                   = pq::Compression::ZSTD;
  cc1.meta_data.num_values              = 12345;
  cc1.meta_data.total_uncompressed_size = 2000;
  cc1.meta_data.total_compressed_size   = 800;
  cc1.meta_data.data_page_offset        = 504;
  rg.columns.push_back(cc1);

  meta.row_groups.push_back(rg);

  meta.key_value_metadata.push_back(pq::KeyValue{"pandas", "{\"index\": 1}"});
  // Empty value re-serializes as absent and reads back empty (a documented delta).
  meta.key_value_metadata.push_back(pq::KeyValue{"empty", ""});

  meta.column_orders = std::vector<pq::ColumnOrder>{pq::ColumnOrder{pq::ColumnOrder::TYPE_ORDER},
                                                    pq::ColumnOrder{pq::ColumnOrder::TYPE_ORDER}};
  return meta;
}

void expect_schema_equal(pq::SchemaElement const& e, pq::SchemaElement const& a)
{
  EXPECT_EQ(e.type, a.type);
  EXPECT_EQ(e.name, a.name);
  // repetition_type == UNSPECIFIED is omitted by the writer and reads back as REQUIRED.
  auto const expected_rep = e.repetition_type == pq::FieldRepetitionType::UNSPECIFIED
                              ? pq::FieldRepetitionType::REQUIRED
                              : e.repetition_type;
  EXPECT_EQ(expected_rep, a.repetition_type);
  // type_length is written only for a typed leaf; a group node or a zero length reads back 0.
  auto const expected_len = e.type != pq::Type::UNDEFINED ? e.type_length : 0;
  EXPECT_EQ(expected_len, a.type_length);
  // num_children is written only for group nodes (type == UNDEFINED); leaves read back 0.
  auto const expected_children = e.type == pq::Type::UNDEFINED ? e.num_children : 0;
  EXPECT_EQ(expected_children, a.num_children);
  EXPECT_EQ(e.converted_type, a.converted_type);
  EXPECT_EQ(e.field_id, a.field_id);
}

void expect_column_meta_equal(pq::ColumnChunkMetaData const& e, pq::ColumnChunkMetaData const& a)
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

void expect_column_chunk_equal(pq::ColumnChunk const& e, pq::ColumnChunk const& a)
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

void expect_row_group_equal(pq::RowGroup const& e, pq::RowGroup const& a)
{
  EXPECT_EQ(e.total_byte_size, a.total_byte_size);
  EXPECT_EQ(e.num_rows, a.num_rows);
  EXPECT_EQ(e.file_offset, a.file_offset);
  EXPECT_EQ(e.total_compressed_size, a.total_compressed_size);
  EXPECT_EQ(e.ordinal, a.ordinal);
  ASSERT_EQ(e.columns.size(), a.columns.size());
  for (size_t i = 0; i < e.columns.size(); ++i) {
    expect_column_chunk_equal(e.columns[i], a.columns[i]);
  }
}

// Compares only the fields the Thrift-compact codec serializes; cudf-internal derived fields
// (schema_idx, index blobs) are excluded.
void expect_footer_semantic_equal(pq::FileMetaData const& e, pq::FileMetaData const& a)
{
  EXPECT_EQ(e.version, a.version);
  EXPECT_EQ(e.num_rows, a.num_rows);
  EXPECT_EQ(e.created_by, a.created_by);

  ASSERT_EQ(e.schema.size(), a.schema.size());
  for (size_t i = 0; i < e.schema.size(); ++i) {
    expect_schema_equal(e.schema[i], a.schema[i]);
  }

  ASSERT_EQ(e.row_groups.size(), a.row_groups.size());
  for (size_t i = 0; i < e.row_groups.size(); ++i) {
    expect_row_group_equal(e.row_groups[i], a.row_groups[i]);
  }

  ASSERT_EQ(e.key_value_metadata.size(), a.key_value_metadata.size());
  for (size_t i = 0; i < e.key_value_metadata.size(); ++i) {
    EXPECT_EQ(e.key_value_metadata[i].key, a.key_value_metadata[i].key);
    EXPECT_EQ(e.key_value_metadata[i].value, a.key_value_metadata[i].value);
  }

  EXPECT_EQ(e.column_orders.has_value(), a.column_orders.has_value());
  if (e.column_orders.has_value() && a.column_orders.has_value()) {
    ASSERT_EQ(e.column_orders->size(), a.column_orders->size());
    for (size_t i = 0; i < e.column_orders->size(); ++i) {
      EXPECT_EQ(e.column_orders.value()[i].type, a.column_orders.value()[i].type);
    }
  }
}

// Writes a small two-column table to a parquet host buffer and returns its parsed footer.
pq::FileMetaData read_written_footer(std::vector<char>& buffer)
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
  EXPECT_EQ(footers.size(), 1);
  return footers.at(0);
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
  auto const bytes    = cudf::io::write_parquet_footer_bytes(original);
  ASSERT_FALSE(bytes.empty());
  auto const reparsed = cudf::io::read_parquet_footer_bytes(as_span(bytes));
  expect_footer_semantic_equal(original, reparsed);
}

// A footer parsed from a real cudf-written file survives a facade write -> read round-trip,
// exercising the repetition_type / num_children / meta_data nuances of a genuine schema.
TEST_F(ParquetFooterFacadeTest, RealFooterRoundTrip)
{
  std::vector<char> buffer;
  auto const original = read_written_footer(buffer);
  ASSERT_FALSE(original.schema.empty());
  ASSERT_FALSE(original.row_groups.empty());

  auto const bytes    = cudf::io::write_parquet_footer_bytes(original);
  auto const reparsed = cudf::io::read_parquet_footer_bytes(as_span(bytes));
  expect_footer_semantic_equal(original, reparsed);
}

// A footer with no schema or row groups round-trips; an absent column_orders stays absent.
TEST_F(ParquetFooterFacadeTest, EmptyFooterRoundTrip)
{
  pq::FileMetaData meta;
  meta.version  = 1;
  meta.num_rows = 0;

  auto const bytes  = cudf::io::write_parquet_footer_bytes(meta);
  auto const parsed = cudf::io::read_parquet_footer_bytes(as_span(bytes));
  EXPECT_EQ(parsed.version, 1);
  EXPECT_EQ(parsed.num_rows, 0);
  EXPECT_TRUE(parsed.schema.empty());
  EXPECT_TRUE(parsed.row_groups.empty());
  EXPECT_TRUE(parsed.key_value_metadata.empty());
  EXPECT_TRUE(parsed.created_by.empty());
  EXPECT_FALSE(parsed.column_orders.has_value());
}

// A boundary-value num_rows survives the zigzag-varint round-trip.
TEST_F(ParquetFooterFacadeTest, LargeNumRows)
{
  pq::FileMetaData meta;
  meta.version  = 2;
  meta.num_rows = std::numeric_limits<int64_t>::max();

  auto const bytes  = cudf::io::write_parquet_footer_bytes(meta);
  auto const parsed = cudf::io::read_parquet_footer_bytes(as_span(bytes));
  EXPECT_EQ(parsed.num_rows, std::numeric_limits<int64_t>::max());
}

// Regression: the facade stops at the struct terminator, so an over-length buffer reparses to the
// same metadata -- padding past the terminator (e.g. spark-rapids' length word) is ignored.
TEST_F(ParquetFooterFacadeTest, TrailingBytesAreTolerated)
{
  auto const original = make_test_footer();
  auto const exact    = cudf::io::write_parquet_footer_bytes(original);
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
    SCOPED_TRACE("trailing tail of " + std::to_string(tail.size()) + " bytes");
    std::vector<uint8_t> over_length = exact;
    over_length.insert(over_length.end(), tail.begin(), tail.end());
    ASSERT_GT(over_length.size(), exact.size());
    auto const reparsed = cudf::io::read_parquet_footer_bytes(as_span(over_length));
    expect_footer_semantic_equal(original, reparsed);
  }
}

// Overread guard: every proper prefix of a valid footer (including cuts inside the row_groups
// list) must throw cleanly, not return structurally-invalid metadata.
TEST_F(ParquetFooterFacadeTest, TruncatedFooterThrows)
{
  auto const full = cudf::io::write_parquet_footer_bytes(make_test_footer());
  ASSERT_GT(full.size(), 8u);

  for (size_t len : {full.size() / 4, full.size() / 2, full.size() * 3 / 4, full.size() - 1}) {
    std::vector<uint8_t> const truncated(full.begin(), full.begin() + len);
    EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(truncated)), cudf::logic_error)
      << "truncation length " << len << " did not throw";
  }
}

// A mid-struct truncation padded past the cut still throws -- the non-zero padding never forms the
// struct terminator the reader would otherwise stop on.
TEST_F(ParquetFooterFacadeTest, TruncatedFooterWithPaddingThrows)
{
  auto const full = cudf::io::write_parquet_footer_bytes(make_test_footer());
  ASSERT_GT(full.size(), 8u);

  for (size_t len : {full.size() / 4, full.size() / 2, full.size() * 3 / 4, full.size() - 1}) {
    std::vector<uint8_t> padded(full.begin(), full.begin() + len);
    padded.insert(padded.end(), full.size(), 0x5a);  // unrelated bytes past the truncation point
    EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(padded)), cudf::logic_error)
      << "truncation length " << len << " with padding did not throw";
  }
}

// A garbage buffer fails cleanly: 0xff decodes to wire type 0xf, not a valid Thrift type, so
// skip_struct_field's default arm rejects it.
TEST_F(ParquetFooterFacadeTest, GarbageBufferThrows)
{
  std::vector<uint8_t> const garbage(16, 0xff);
  EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(garbage)), cudf::logic_error);
}

// Overread guard: a zero-length buffer trips the sticky overread flag on the first field read
// rather than returning empty metadata.
TEST_F(ParquetFooterFacadeTest, EmptyBufferThrows)
{
  EXPECT_THROW(cudf::io::read_parquet_footer_bytes(cudf::host_span<uint8_t const>{}),
               cudf::logic_error);
}

// Count guard: a field 2 (schema) struct-list header declaring 0x7fffffff elements with no
// bytes left is rejected before the resize.
TEST_F(ParquetFooterFacadeTest, OversizedContainerCountThrows)
{
  std::vector<uint8_t> const bomb{0x29, 0xfc, 0xff, 0xff, 0xff, 0xff, 0x07};
  EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(bomb)), cudf::logic_error);
}

// Count guard: the I32 `encodings` primitive list under row_groups[0].columns[0].meta_data
// declares 0x7fffffff elements with no bytes left, hitting the primitive parquet_field_list guard
// (distinct from the struct-list guard above).
TEST_F(ParquetFooterFacadeTest, OversizedPrimitiveListCountThrows)
{
  std::vector<uint8_t> const bomb{0x49,  // FileMetaData field 4 (row_groups), type LIST
                                  0x1c,  // list header: 1 element, STRUCT
                                  0x19,  // RowGroup field 1 (columns), type LIST
                                  0x1c,  // list header: 1 element, STRUCT
                                  0x3c,  // ColumnChunk field 3 (meta_data), type STRUCT
                                  0x29,  // ColumnChunkMetaData field 2 (encodings), type LIST
                                  0xf5,  // list header: long-form size prefix, I32 elements
                                  0xff,
                                  0xff,
                                  0xff,
                                  0xff,
                                  0x07};  // size varint = 0x7fffffff
  EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(bomb)), cudf::logic_error);
}

// Count guard: an unknown top-level LIST field routes to skip_struct_field, whose list branch
// declares 0x7fffffff elements with no bytes left, hitting the skip-path count guard.
TEST_F(ParquetFooterFacadeTest, OversizedSkippedListCountThrows)
{
  std::vector<uint8_t> const bomb{
    0x89,  // FileMetaData field 8 (unknown id), type LIST -> skip path
    0xfc,  // list header: long-form size prefix, STRUCT elements
    0xff,
    0xff,
    0xff,
    0xff,
    0x07};  // size varint = 0x7fffffff
  EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(bomb)), cudf::logic_error);
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
  EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(bomb)), cudf::logic_error);
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
  auto const parsed = cudf::io::read_parquet_footer_bytes(as_span(footer));
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
  auto const parsed = cudf::io::read_parquet_footer_bytes(as_span(footer));
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
  auto const parsed = cudf::io::read_parquet_footer_bytes(as_span(footer));
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
    cudf::io::read_parquet_footer_bytes(as_span(footer), pq::throw_if_type_mismatch::NO);
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
    cudf::io::read_parquet_footer_bytes(as_span(footer), pq::throw_if_type_mismatch::NO);
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
    cudf::io::read_parquet_footer_bytes(as_span(footer), pq::throw_if_type_mismatch::NO);
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
    cudf::io::read_parquet_footer_bytes(as_span(footer), pq::throw_if_type_mismatch::NO);
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
    cudf::io::read_parquet_footer_bytes(as_span(footer), pq::throw_if_type_mismatch::NO);
  ASSERT_TRUE(parsed.column_orders.has_value());
  ASSERT_EQ(parsed.column_orders->size(), 1);
  EXPECT_EQ(parsed.column_orders.value()[0].type, pq::ColumnOrder::UNDEFINED);
}

// Struct-list parsing goes parallel at 512+ elements; every sub-reader must inherit lenient mode.
// All 513 mismatched row groups parse -- a regression to strict sub-readers would throw here.
TEST_F(ParquetFooterFacadeTest, ParallelStructListPropagatesLenientMode)
{
  auto const footer = make_parallel_mismatch_footer();
  auto const parsed =
    cudf::io::read_parquet_footer_bytes(as_span(footer), pq::throw_if_type_mismatch::NO);
  ASSERT_EQ(parsed.row_groups.size(), parallel_list_count);
  // Every element, not just the boundaries, must have the mismatched field unset -- a
  // task-partitioning off-by-one would leave a silently-wrong middle range untested.
  EXPECT_TRUE(std::all_of(parsed.row_groups.begin(), parsed.row_groups.end(), [](auto const& rg) {
    return rg.columns.empty();
  }));
}

// The same wrong-type footer throws under the default strict mode: cudf's readers keep the
// exact-type contract; only the spark-rapids facade opts into leniency via
// `throw_if_type_mismatch::NO`.
TEST_F(ParquetFooterFacadeTest, KnownFieldWithWrongTypeThrowsInStrictMode)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x19, 0x05,  // field 1 (version) as an empty i32-LIST -> wrong wire type
    0x26, 0x54,  // field 3 (num_rows) i64 = 42
    0x00};       // STOP
  // clang-format on
  EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(footer)), cudf::logic_error);
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
  EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(footer)), cudf::logic_error);
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
  EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(footer)), cudf::logic_error);
}

// The parallel sub-readers inherit the default strict mode too: the mismatch is detected in a
// worker task and the exception rethrown to the caller.
TEST_F(ParquetFooterFacadeTest, ParallelStructListThrowsInStrictMode)
{
  auto const footer = make_parallel_mismatch_footer();
  EXPECT_THROW(cudf::io::read_parquet_footer_bytes(as_span(footer)), cudf::logic_error);
}

// An empty LIST field is accepted regardless of its wire element-type nibble: a zero-length list
// has no elements, so the type is immaterial (some writers, e.g. fastparquet, stamp 0). The reader
// clears the target and keeps parsing.
TEST_F(ParquetFooterFacadeTest, EmptyListWithZeroElementTypeIsAccepted)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x15, 0x04,  // field 1 (version) i32 = 2
    0x19, 0x00,  // field 2 (schema): empty LIST with element-type nibble 0 (not STRUCT)
    0x00};       // STOP
  // clang-format on
  auto const parsed = cudf::io::read_parquet_footer_bytes(as_span(footer));
  EXPECT_EQ(parsed.version, 2);
  EXPECT_TRUE(parsed.schema.empty());
}

// An empty PRIMITIVE-element list is likewise accepted regardless of its wire element-type nibble:
// parquet_field_list<T, ELEM> duplicates the n == 0 short-circuit of parquet_field_struct_list (the
// struct-list case above), so it needs its own coverage.
TEST_F(ParquetFooterFacadeTest, EmptyPrimitiveListWithWrongElementTypeIsAccepted)
{
  // clang-format off
  std::vector<uint8_t> const footer{
    0x49, 0x1c,  // FileMetaData field 4 (row_groups): LIST of 1 STRUCT
    0x19, 0x1c,  //   RowGroup field 1 (columns): LIST of 1 STRUCT
    0x3c,        //     ColumnChunk field 3 (meta_data): STRUCT
    0x29, 0x00,  //       ColumnChunkMetaData field 2 (encodings): empty LIST, nibble 0 (not I32)
    0x00,        //       ColumnChunkMetaData STOP
    0x00,        //     ColumnChunk STOP
    0x00,        //   RowGroup STOP
    0x00};       // FileMetaData STOP
  // clang-format on
  auto const parsed = cudf::io::read_parquet_footer_bytes(as_span(footer));
  ASSERT_EQ(parsed.row_groups.size(), 1);
  ASSERT_EQ(parsed.row_groups[0].columns.size(), 1);
  EXPECT_TRUE(parsed.row_groups[0].columns[0].meta_data.encodings.empty());
}

CUDF_TEST_PROGRAM_MAIN()
