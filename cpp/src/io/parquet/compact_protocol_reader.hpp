/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "parquet_common.hpp"

#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/bit>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace CUDF_EXPORT cudf {
namespace io::parquet::detail {

/**
 * @brief Class for parsing Parquet's Thrift Compact Protocol encoded metadata
 *
 * This class takes in the starting location of the Parquet metadata, and fills
 * out Thrift-derived structs and a schema tree.
 *
 * In a Parquet, the metadata is separated from the data, both conceptually and
 * physically. There may be multiple data files sharing a common metadata file.
 *
 * The parser handles both V1 and V2 Parquet datasets, although not all
 * compression codecs are supported yet.
 */
class CompactProtocolReader {
 public:
  explicit CompactProtocolReader(uint8_t const* base         = nullptr,
                                 size_t len                  = 0,
                                 throw_if_type_mismatch mode = throw_if_type_mismatch::YES)
    : m_throw_if_type_mismatch(mode)
  {
    init(base, len);
  }
  void init(uint8_t const* base, size_t len)
  {
    m_base = m_cur = base;
    // Avoid undefined pointer arithmetic on a null base; a null buffer stays an empty range.
    m_end      = base != nullptr ? base + len : base;
    m_overread = false;
  }
  [[nodiscard]] ptrdiff_t bytecount() const noexcept { return m_cur - m_base; }
  // True if a wire-type/schema-type mismatch must be rejected (default YES); false means skip it
  // per Thrift forward-compat (NO), which the spark-rapids footer facade uses.
  [[nodiscard]] bool should_throw_on_type_mismatch() const noexcept
  {
    return m_throw_if_type_mismatch == throw_if_type_mismatch::YES;
  }
  // A read at end-of-buffer sets the sticky overread flag (checked in read(FileMetaData*)) and
  // yields 0, keeping the hot parse path noexcept.
  unsigned int getb() noexcept
  {
    if (m_cur < m_end) { return *m_cur++; }
    m_overread = true;
    return 0;
  }
  void skip_bytes(size_t bytecnt) noexcept
  {
    size_t const avail = m_end - m_cur;
    if (bytecnt > avail) {
      m_overread = true;
      bytecnt    = avail;
    }
    m_cur += bytecnt;
  }

  // returns a varint encoded integer
  template <typename T>
  T get_varint() noexcept
  {
    T v = 0;
    for (uint32_t l = 0;; l += 7) {
      T c = getb();
      v |= (c & 0x7f) << l;
      if (c < 0x80) { break; }
    }
    return v;
  }

  // returns a zigzag encoded signed integer
  template <typename T>
  T get_zigzag() noexcept
  {
    using U   = std::make_unsigned_t<T>;
    U const u = get_varint<U>();
    return static_cast<T>((u >> 1u) ^ -static_cast<T>(u & 1));
  }

  // thrift spec says to use zigzag i32 for i16 types
  int32_t get_i16() noexcept { return get_zigzag<int32_t>(); }
  int32_t get_i32() noexcept { return get_zigzag<int32_t>(); }
  int64_t get_i64() noexcept { return get_zigzag<int64_t>(); }

  uint32_t get_u32() noexcept { return get_varint<uint32_t>(); }
  uint64_t get_u64() noexcept { return get_varint<uint64_t>(); }

  [[nodiscard]] std::pair<uint8_t, uint32_t> get_listh() noexcept
  {
    uint32_t const c = getb();
    uint32_t sz      = c >> 4;
    uint8_t t        = c & 0xf;
    if (sz == 0xf) { sz = get_u32(); }
    return {t, sz};
  }

  void skip_struct_field(int t, int depth = 0);

  // True if the wire type matches the schema type; on mismatch strict mode throws while lenient
  // mode (`NO`) skips the value and returns false, leaving the field (and any optional) unset.
  [[nodiscard]] bool check_field_type(int type, FieldType expected);

 public:
  // Generate Thrift structure parsing routines
  void read(FileMetaData* f);
  void read(SchemaElement* s);
  void read(LogicalType* l);
  void read(DecimalType* d);
  void read(TimeType* t);
  void read(TimeUnit* u);
  void read(TimestampType* t);
  void read(IntType* t);
  void read(RowGroup* r);
  void read(ColumnChunk* c);
  void read(BloomFilterAlgorithm* bf);
  void read(BloomFilterHash* bf);
  void read(BloomFilterCompression* bf);
  void read(BloomFilterHeader* bf);
  void read(ColumnChunkMetaData* c);
  void read(PageHeader* p);
  void read(DataPageHeader* d);
  void read(DictionaryPageHeader* d);
  void read(DataPageHeaderV2* d);
  void read(KeyValue* k);
  void read(PageLocation* p);
  void read(OffsetIndex* o);
  void read(SizeStatistics* s);
  void read(ColumnIndex* c);
  void read(Statistics* s);
  void read(ColumnOrder* c);
  void read(PageEncodingStats* s);
  void read(SortingColumn* s);

 public:
  static inline constexpr int NumRequiredBits(uint32_t max_level) noexcept
  {
    return 32 - cuda::std::countl_zero(max_level);
  }
  bool InitSchema(FileMetaData* md);

 protected:
  int WalkSchema(FileMetaData* md,
                 int idx           = 0,
                 int parent_idx    = 0,
                 int max_def_level = 0,
                 int max_rep_level = 0);

 protected:
  uint8_t const* m_base = nullptr;
  uint8_t const* m_cur  = nullptr;
  uint8_t const* m_end  = nullptr;
  // Sticky flag: a required read was attempted past end-of-buffer (truncated/corrupt input).
  bool m_overread = false;
  // Reject (`YES`) vs skip (`NO`) a struct field whose wire type mismatches the schema type.
  throw_if_type_mismatch m_throw_if_type_mismatch = throw_if_type_mismatch::YES;

  friend class parquet_field_string;
  friend class parquet_field_string_list;
  friend class parquet_field_binary;
  friend class parquet_field_binary_list;
  friend class parquet_field_struct_blob;
  template <typename T, FieldType EXPECTED_ELEM_TYPE>
  friend class parquet_field_list;
  template <typename T>
  friend class parquet_field_struct_list;
};

}  // namespace io::parquet::detail
}  // namespace CUDF_EXPORT cudf
