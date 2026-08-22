/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "parquet_common.hpp"

#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/bit>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
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
  explicit CompactProtocolReader(uint8_t const* base = nullptr, size_t len = 0) { init(base, len); }
  void init(uint8_t const* base, size_t len)
  {
    // A null base is valid only for an empty buffer; a positive length would then have no backing
    // storage. This keeps every later pointer op defined (the empty state has all-null pointers).
    CUDF_EXPECTS(base != nullptr || len == 0,
                 "CompactProtocolReader requires a non-null buffer when length is non-zero",
                 std::invalid_argument);
    m_base = m_cur = base;
    // Guard against `nullptr + len` (undefined) so a zero-length buffer stays fully defined.
    m_end = base != nullptr ? base + len : base;
  }
  [[nodiscard]] ptrdiff_t bytecount() const noexcept
  {
    // Avoid `nullptr - nullptr` on a null-base reader; it has consumed nothing.
    return m_base != nullptr ? m_cur - m_base : 0;
  }
  unsigned int getb() noexcept { return (m_cur < m_end) ? *m_cur++ : 0; }
  void skip_bytes(size_t bytecnt) noexcept
  {
    bytecnt = std::min(bytecnt, (size_t)(m_end - m_cur));
    m_cur += bytecnt;
  }

  // Returns a varint-encoded integer. `T` is constrained to unsigned so `numeric_limits<T>::digits`
  // is the full value width; a signed `T` would drop the sign bit and misplace the overflow bound.
  template <std::unsigned_integral T>
  T get_varint()
  {
    T v = 0;
    for (uint32_t l = 0;; l += 7) {
      T const c = getb();
      // Reject overlong varints. `l < digits` keeps the shift in range (and guards `max() >> l`);
      // `c <= max() >> l` then bounds the value, since the shift-count check alone would let the
      // top group's bits silently wrap past `T`'s width instead of being rejected. `c` is the raw
      // byte (continuation bit included): comparing it, not the masked payload, only rejects a
      // boundary continuation byte one group early; correct, since the next group overflows anyway.
      CUDF_EXPECTS(l < std::numeric_limits<T>::digits && c <= (std::numeric_limits<T>::max() >> l),
                   "Parquet varint exceeds the width of its target type",
                   std::overflow_error);
      v |= (c & 0x7f) << l;
      if (c < 0x80) { break; }
    }
    return v;
  }

  // returns a zigzag encoded signed integer
  template <typename T>
  T get_zigzag()
  {
    using U   = std::make_unsigned_t<T>;
    U const u = get_varint<U>();
    return static_cast<T>((u >> 1u) ^ -static_cast<T>(u & 1));
  }

  // thrift spec says to use zigzag i32 for i16 types
  int32_t get_i16() { return get_zigzag<int32_t>(); }
  int32_t get_i32() { return get_zigzag<int32_t>(); }
  int64_t get_i64() { return get_zigzag<int64_t>(); }

  uint32_t get_u32() { return get_varint<uint32_t>(); }
  uint64_t get_u64() { return get_varint<uint64_t>(); }

  [[nodiscard]] std::pair<uint8_t, uint32_t> get_listh()
  {
    uint32_t const c = getb();
    uint32_t sz      = c >> 4;
    uint8_t t        = c & 0xf;
    if (sz == 0xf) { sz = get_u32(); }
    return {t, sz};
  }

  void skip_struct_field(int t, int depth = 0);

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

  friend class parquet_field_string;
  friend class parquet_field_string_list;
  friend class parquet_field_binary;
  friend class parquet_field_binary_list;
  friend class parquet_field_struct_blob;
  template <typename T>
  friend class parquet_field_struct_list;
};

}  // namespace io::parquet::detail
}  // namespace CUDF_EXPORT cudf
