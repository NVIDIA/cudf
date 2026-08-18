/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compact_protocol_writer.hpp"

#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::io::parquet::detail {

/**
 * @brief Parquet CompactProtocolWriter class
 */

size_t CompactProtocolWriter::write(FileMetaData const& f)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, f.version);
  c.field_struct_list(2, f.schema);
  c.field_int(3, f.num_rows);
  c.field_struct_list(4, f.row_groups);
  if (not f.key_value_metadata.empty()) { c.field_struct_list(5, f.key_value_metadata); }
  if (not f.created_by.empty()) { c.field_string(6, f.created_by); }
  if (f.column_orders.has_value()) { c.field_struct_list(7, f.column_orders.value()); }
  return c.value();
}

size_t CompactProtocolWriter::write(DecimalType const& decimal)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, decimal.scale);
  c.field_int(2, decimal.precision);
  return c.value();
}

size_t CompactProtocolWriter::write(TimeUnit const& time_unit)
{
  CompactProtocolFieldWriter c(*this);
  switch (time_unit.type) {
    case TimeUnit::MILLIS:
    case TimeUnit::MICROS:
    case TimeUnit::NANOS: c.field_empty_struct(static_cast<int>(time_unit.type)); break;
    default:
      CUDF_FAIL("Trying to write an invalid TimeUnit " +
                std::to_string(static_cast<int>(time_unit.type)));
  }
  return c.value();
}

size_t CompactProtocolWriter::write(TimeType const& time)
{
  CompactProtocolFieldWriter c(*this);
  c.field_bool(1, time.isAdjustedToUTC);
  c.field_struct(2, time.unit);
  return c.value();
}

size_t CompactProtocolWriter::write(TimestampType const& timestamp)
{
  CompactProtocolFieldWriter c(*this);
  c.field_bool(1, timestamp.isAdjustedToUTC);
  c.field_struct(2, timestamp.unit);
  return c.value();
}

size_t CompactProtocolWriter::write(IntType const& integer)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int8(1, integer.bitWidth);
  c.field_bool(2, integer.isSigned);
  return c.value();
}

size_t CompactProtocolWriter::write(LogicalType const& logical_type)
{
  CompactProtocolFieldWriter c(*this);
  switch (logical_type.type) {
    case LogicalType::STRING:
    case LogicalType::MAP:
    case LogicalType::LIST:
    case LogicalType::ENUM:
    case LogicalType::DATE:
    case LogicalType::UNKNOWN:
    case LogicalType::JSON:
    case LogicalType::BSON: c.field_empty_struct(static_cast<int>(logical_type.type)); break;
    case LogicalType::DECIMAL:
      c.field_struct(static_cast<int>(LogicalType::DECIMAL), logical_type.decimal_type.value());
      break;
    case LogicalType::TIME:
      c.field_struct(static_cast<int>(LogicalType::TIME), logical_type.time_type.value());
      break;
    case LogicalType::TIMESTAMP:
      c.field_struct(static_cast<int>(LogicalType::TIMESTAMP), logical_type.timestamp_type.value());
      break;
    case LogicalType::INTEGER:
      c.field_struct(static_cast<int>(LogicalType::INTEGER), logical_type.int_type.value());
      break;
    default:
      CUDF_FAIL("Trying to write an invalid LogicalType " + std::to_string(logical_type.type));
  }
  return c.value();
}

size_t CompactProtocolWriter::write(SchemaElement const& s)
{
  CompactProtocolFieldWriter c(*this);
  if (s.type != Type::UNDEFINED) {
    c.field_int(1, static_cast<int32_t>(s.type));
    if (s.type_length != 0) { c.field_int(2, s.type_length); }
  }
  if (s.repetition_type != FieldRepetitionType::UNSPECIFIED) {
    c.field_int(3, static_cast<int32_t>(s.repetition_type));
  }
  c.field_string(4, s.name);

  if (s.type == Type::UNDEFINED) { c.field_int(5, s.num_children); }
  if (s.converted_type.has_value()) {
    c.field_int(6, static_cast<int32_t>(s.converted_type.value()));
    if (s.converted_type == ConvertedType::DECIMAL) {
      c.field_int(7, s.decimal_scale);
      c.field_int(8, s.decimal_precision);
    }
  }
  if (s.field_id.has_value()) { c.field_int(9, s.field_id.value()); }
  if (s.logical_type.has_value()) { c.field_struct(10, s.logical_type.value()); }
  return c.value();
}

size_t CompactProtocolWriter::write(RowGroup const& r)
{
  CompactProtocolFieldWriter c(*this);
  c.field_struct_list(1, r.columns);
  c.field_int(2, r.total_byte_size);
  c.field_int(3, r.num_rows);
  if (r.sorting_columns.has_value()) { c.field_struct_list(4, r.sorting_columns.value()); }
  if (r.file_offset.has_value()) { c.field_int(5, r.file_offset.value()); }
  if (r.total_compressed_size.has_value()) { c.field_int(6, r.total_compressed_size.value()); }
  if (r.ordinal.has_value()) { c.field_int16(7, r.ordinal.value()); }
  return c.value();
}

size_t CompactProtocolWriter::write(KeyValue const& k)
{
  CompactProtocolFieldWriter c(*this);
  c.field_string(1, k.key);
  if (not k.value.empty()) { c.field_string(2, k.value); }
  return c.value();
}

size_t CompactProtocolWriter::write(ColumnChunk const& s)
{
  CompactProtocolFieldWriter c(*this);
  if (not s.file_path.empty()) { c.field_string(1, s.file_path); }
  c.field_int(2, s.file_offset);
  c.field_struct(3, s.meta_data);
  if (s.offset_index_length != 0) {
    c.field_int(4, s.offset_index_offset);
    c.field_int(5, s.offset_index_length);
  }
  if (s.column_index_length != 0) {
    c.field_int(6, s.column_index_offset);
    c.field_int(7, s.column_index_length);
  }
  return c.value();
}

size_t CompactProtocolWriter::write(ColumnChunkMetaData const& s)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, static_cast<int32_t>(s.type));
  c.field_int_list(2, s.encodings);
  c.field_string_list(3, s.path_in_schema);
  c.field_int(4, static_cast<int32_t>(s.codec));
  c.field_int(5, s.num_values);
  c.field_int(6, s.total_uncompressed_size);
  c.field_int(7, s.total_compressed_size);
  c.field_int(9, s.data_page_offset);
  if (s.index_page_offset != 0) { c.field_int(10, s.index_page_offset); }
  if (s.dictionary_page_offset != 0) { c.field_int(11, s.dictionary_page_offset); }
  c.field_struct(12, s.statistics);
  if (s.encoding_stats.has_value()) { c.field_struct_list(13, s.encoding_stats.value()); }
  if (s.size_statistics.has_value()) { c.field_struct(16, s.size_statistics.value()); }
  return c.value();
}

size_t CompactProtocolWriter::write(Statistics const& s)
{
  CompactProtocolFieldWriter c(*this);
  if (s.max.has_value()) { c.field_binary(1, s.max.value()); }
  if (s.min.has_value()) { c.field_binary(2, s.min.value()); }
  if (s.null_count.has_value()) { c.field_int(3, s.null_count.value()); }
  if (s.distinct_count.has_value()) { c.field_int(4, s.distinct_count.value()); }
  if (s.max_value.has_value()) { c.field_binary(5, s.max_value.value()); }
  if (s.min_value.has_value()) { c.field_binary(6, s.min_value.value()); }
  if (s.is_max_value_exact.has_value()) { c.field_bool(7, s.is_max_value_exact.value()); }
  if (s.is_min_value_exact.has_value()) { c.field_bool(8, s.is_min_value_exact.value()); }
  return c.value();
}

size_t CompactProtocolWriter::write(PageLocation const& s)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, s.offset);
  c.field_int(2, s.compressed_page_size);
  c.field_int(3, s.first_row_index);
  return c.value();
}

size_t CompactProtocolWriter::write(OffsetIndex const& s)
{
  CompactProtocolFieldWriter c(*this);
  c.field_struct_list(1, s.page_locations);
  if (s.unencoded_byte_array_data_bytes.has_value()) {
    c.field_int_list(2, s.unencoded_byte_array_data_bytes.value());
  }
  return c.value();
}

size_t CompactProtocolWriter::write(SizeStatistics const& s)
{
  CompactProtocolFieldWriter c(*this);
  if (s.unencoded_byte_array_data_bytes.has_value()) {
    c.field_int(1, s.unencoded_byte_array_data_bytes.value());
  }
  if (s.repetition_level_histogram.has_value()) {
    c.field_int_list(2, s.repetition_level_histogram.value());
  }
  if (s.definition_level_histogram.has_value()) {
    c.field_int_list(3, s.definition_level_histogram.value());
  }
  return c.value();
}

size_t CompactProtocolWriter::write(ColumnOrder const& co)
{
  CompactProtocolFieldWriter c(*this);
  switch (co.type) {
    case ColumnOrder::TYPE_ORDER: c.field_empty_struct(static_cast<int>(co.type)); break;
    default:
      CUDF_FAIL("Trying to write an invalid ColumnOrder " +
                std::to_string(static_cast<int>(co.type)));
  }
  return c.value();
}

size_t CompactProtocolWriter::write(PageEncodingStats const& enc)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, static_cast<int32_t>(enc.page_type));
  c.field_int(2, static_cast<int32_t>(enc.encoding));
  c.field_int(3, enc.count);
  return c.value();
}

size_t CompactProtocolWriter::write(SortingColumn const& sc)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, sc.column_idx);
  c.field_bool(2, sc.descending);
  c.field_bool(3, sc.nulls_first);
  return c.value();
}

size_t CompactProtocolWriter::write(DataPageHeader const& pg_hdr)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, pg_hdr.num_values);
  c.field_int(2, static_cast<int32_t>(pg_hdr.encoding));
  c.field_int(3, static_cast<int32_t>(pg_hdr.definition_level_encoding));
  c.field_int(4, static_cast<int32_t>(pg_hdr.repetition_level_encoding));
  return c.value();
}

size_t CompactProtocolWriter::write(DictionaryPageHeader const& dict_pg_hdr)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, dict_pg_hdr.num_values);
  c.field_int(2, static_cast<int32_t>(dict_pg_hdr.encoding));
  return c.value();
}

size_t CompactProtocolWriter::write(DataPageHeaderV2 const& pg_hdr_v2)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, pg_hdr_v2.num_values);
  c.field_int(2, pg_hdr_v2.num_nulls);
  c.field_int(3, pg_hdr_v2.num_rows);
  c.field_int(4, static_cast<int32_t>(pg_hdr_v2.encoding));
  c.field_int(5, pg_hdr_v2.definition_levels_byte_length);
  c.field_int(6, pg_hdr_v2.repetition_levels_byte_length);
  c.field_bool(7, pg_hdr_v2.is_compressed);
  return c.value();
}

size_t CompactProtocolWriter::write(PageHeader const& page_hdr)
{
  CompactProtocolFieldWriter c(*this);
  c.field_int(1, static_cast<int32_t>(page_hdr.type));
  c.field_int(2, page_hdr.uncompressed_page_size);
  c.field_int(3, page_hdr.compressed_page_size);
  // Exactly one page-specific header is set, selected by `type`; field ids (5/7/8) match
  // CompactProtocolReader::read(PageHeader) and the GPU encoder in gpuEncodePageHeaders.
  switch (page_hdr.type) {
    case PageType::DATA_PAGE: c.field_struct(5, page_hdr.data_page_header); break;
    case PageType::DICTIONARY_PAGE: c.field_struct(7, page_hdr.dictionary_page_header); break;
    case PageType::DATA_PAGE_V2: c.field_struct(8, page_hdr.data_page_header_v2); break;
    default:
      CUDF_FAIL("Trying to write an invalid PageType " +
                std::to_string(static_cast<int>(page_hdr.type)));
  }
  return c.value();
}

void CompactProtocolFieldWriter::put_byte(uint8_t v) { writer.m_buf.push_back(v); }

void CompactProtocolFieldWriter::put_byte(uint8_t const* raw, uint32_t len)
{
  for (uint32_t i = 0; i < len; i++)
    writer.m_buf.push_back(raw[i]);
}

uint32_t CompactProtocolFieldWriter::put_uint(uint64_t v)
{
  int l = 1;
  while (v > 0x7f) {
    put_byte(static_cast<uint8_t>(v | 0x80));
    v >>= 7;
    l++;
  }
  put_byte(static_cast<uint8_t>(v));
  return l;
}

uint32_t CompactProtocolFieldWriter::put_int(int64_t v)
{
  int64_t const s = (v < 0);
  return put_uint(((v ^ -s) << 1) + s);
}

void CompactProtocolFieldWriter::put_field_header(int f, int cur, FieldType t)
{
  if (f > cur && f <= cur + 15)
    put_packed_type_byte(f - cur, t);
  else {
    put_byte(static_cast<uint8_t>(t));
    put_int(f);
  }
}

inline void CompactProtocolFieldWriter::field_bool(int field, bool b)
{
  put_field_header(
    field, current_field_value, b ? FieldType::BOOLEAN_TRUE : FieldType::BOOLEAN_FALSE);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_int8(int field, int8_t val)
{
  put_field_header(field, current_field_value, FieldType::I8);
  put_byte(val);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_int16(int field, int16_t val)
{
  put_field_header(field, current_field_value, FieldType::I16);
  put_int(val);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_int(int field, int32_t val)
{
  put_field_header(field, current_field_value, FieldType::I32);
  put_int(val);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_int(int field, int64_t val)
{
  put_field_header(field, current_field_value, FieldType::I64);
  put_int(val);
  current_field_value = field;
}

template <>
inline void CompactProtocolFieldWriter::field_int_list<int64_t>(int field,
                                                                std::vector<int64_t> const& val)
{
  put_field_header(field, current_field_value, FieldType::LIST);
  put_packed_type_byte(val.size(), FieldType::I64);
  if (val.size() >= 0xfUL) { put_uint(val.size()); }
  for (auto const v : val) {
    put_int(v);
  }
  current_field_value = field;
}

template <typename Enum>
inline void CompactProtocolFieldWriter::field_int_list(int field, std::vector<Enum> const& val)
{
  put_field_header(field, current_field_value, FieldType::LIST);
  put_packed_type_byte(val.size(), FieldType::I32);
  if (val.size() >= 0xfUL) { put_uint(val.size()); }
  for (auto const& v : val) {
    put_int(static_cast<int32_t>(v));
  }
  current_field_value = field;
}

template <typename T>
inline void CompactProtocolFieldWriter::field_struct(int field, T const& val)
{
  put_field_header(field, current_field_value, FieldType::STRUCT);
  if constexpr (not std::is_empty_v<T>) {
    writer.write(val);  // write the struct if it's not empty
  } else {
    put_byte(0);  // otherwise, add a stop field
  }
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_empty_struct(int field)
{
  put_field_header(field, current_field_value, FieldType::STRUCT);
  put_byte(0);  // add a stop field
  current_field_value = field;
}

template <typename T>
inline void CompactProtocolFieldWriter::field_struct_list(int field, std::vector<T> const& val)
{
  put_field_header(field, current_field_value, FieldType::LIST);
  put_packed_type_byte(val.size(), FieldType::STRUCT);
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto& v : val) {
    writer.write(v);
  }
  current_field_value = field;
}

inline size_t CompactProtocolFieldWriter::value()
{
  put_byte(0);
  return writer.m_buf.size() - struct_start_pos;
}

inline void CompactProtocolFieldWriter::field_struct_blob(int field,
                                                          std::vector<uint8_t> const& val)
{
  put_field_header(field, current_field_value, FieldType::STRUCT);
  put_byte(val.data(), static_cast<uint32_t>(val.size()));
  put_byte(0);
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_binary(int field, std::vector<uint8_t> const& val)
{
  put_field_header(field, current_field_value, FieldType::BINARY);
  put_uint(val.size());
  put_byte(val.data(), static_cast<uint32_t>(val.size()));
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_string(int field, std::string const& val)
{
  put_field_header(field, current_field_value, FieldType::BINARY);
  put_uint(val.size());
  // FIXME : replace reinterpret_cast
  put_byte(reinterpret_cast<uint8_t const*>(val.data()), static_cast<uint32_t>(val.size()));
  current_field_value = field;
}

inline void CompactProtocolFieldWriter::field_string_list(int field,
                                                          std::vector<std::string> const& val)
{
  put_field_header(field, current_field_value, FieldType::LIST);
  put_packed_type_byte(val.size(), FieldType::BINARY);
  if (val.size() >= 0xf) put_uint(val.size());
  for (auto& v : val) {
    put_uint(v.size());
    // FIXME : replace reinterpret_cast
    put_byte(reinterpret_cast<uint8_t const*>(v.data()), static_cast<uint32_t>(v.size()));
  }
  current_field_value = field;
}

inline int CompactProtocolFieldWriter::current_field() { return current_field_value; }

inline void CompactProtocolFieldWriter::set_current_field(int const& field)
{
  current_field_value = field;
}

}  // namespace cudf::io::parquet::detail
