/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/export.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::parquet::detail {

/**
 * @brief Class for parsing Parquet's Thrift Compact Protocol encoded metadata
 *
 * This class takes in the Parquet structs and outputs a Thrift-encoded binary blob
 */
class CompactProtocolWriter {
 public:
  CompactProtocolWriter(std::vector<uint8_t>* output) : m_buf(*output) {}

  size_t write(FileMetaData const&);
  size_t write(DecimalType const&);
  size_t write(TimeUnit const&);
  size_t write(TimeType const&);
  size_t write(TimestampType const&);
  size_t write(IntType const&);
  size_t write(LogicalType const&);
  size_t write(SchemaElement const&);
  size_t write(RowGroup const&);
  size_t write(KeyValue const&);
  size_t write(ColumnChunk const&);
  size_t write(ColumnChunkMetaData const&);
  size_t write(Statistics const&);
  size_t write(PageLocation const&);
  size_t write(OffsetIndex const&);
  size_t write(SizeStatistics const&);
  size_t write(ColumnOrder const&);
  size_t write(PageEncodingStats const&);
  size_t write(SortingColumn const&);
  size_t write(DataPageHeader const&);
  size_t write(DictionaryPageHeader const&);
  size_t write(DataPageHeaderV2 const&);
  size_t write(PageHeader const&);

 protected:
  std::vector<uint8_t>& m_buf;
  friend class CompactProtocolFieldWriter;
};

class CompactProtocolFieldWriter {
  CompactProtocolWriter& writer;
  size_t struct_start_pos;
  int current_field_value{0};

 public:
  CompactProtocolFieldWriter(CompactProtocolWriter& caller)
    : writer(caller), struct_start_pos(writer.m_buf.size())
  {
  }

  void put_byte(uint8_t v);

  void put_byte(uint8_t const* raw, uint32_t len);

  uint32_t put_uint(uint64_t v);

  uint32_t put_int(int64_t v);

  template <typename T>
  void put_packed_type_byte(T high_bits, FieldType t)
  {
    uint8_t const clamped_high_bits = std::min(std::max(high_bits, T{0}), T{0xf});
    put_byte((clamped_high_bits << 4) | static_cast<uint8_t>(t));
  }

  void put_field_header(int f, int cur, FieldType t);

  inline void field_bool(int field, bool b);

  inline void field_int8(int field, int8_t val);

  inline void field_int16(int field, int16_t val);

  inline void field_int(int field, int32_t val);

  inline void field_int(int field, int64_t val);

  template <typename Enum>
  inline void field_int_list(int field, std::vector<Enum> const& val);

  template <typename T>
  inline void field_struct(int field, T const& val);

  inline void field_empty_struct(int field);

  template <typename T>
  inline void field_struct_list(int field, std::vector<T> const& val);

  inline size_t value();

  inline void field_struct_blob(int field, std::vector<uint8_t> const& val);

  inline void field_binary(int field, std::vector<uint8_t> const& val);

  inline void field_string(int field, std::string const& val);

  inline void field_string_list(int field, std::vector<std::string> const& val);

  inline int current_field();

  inline void set_current_field(int const& field);
};

template <>
inline void CompactProtocolFieldWriter::field_int_list<int64_t>(int field,
                                                                std::vector<int64_t> const& val);

}  // namespace io::parquet::detail
}  // namespace CUDF_EXPORT cudf
