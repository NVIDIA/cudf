/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "parquet_common.hpp"

#include <cudf/io/experimental/parquet_footer.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

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
  explicit CompactProtocolReader(
    uint8_t const* base                       = nullptr,
    size_t len                                = 0,
    experimental::thrift_mismatch_policy mode = experimental::thrift_mismatch_policy::THROW)
    : m_mismatch_policy(mode)
  {
    init(base, len);
  }
  void init(uint8_t const* base, size_t len)
  {
    // A null base is valid only for an empty buffer; a positive length would then have no backing
    // storage. This keeps every later pointer op defined (the empty state has all-null pointers).
    CUDF_EXPECTS(base != nullptr || len == 0,
                 "CompactProtocolReader requires a non-null buffer when length is non-zero",
                 std::invalid_argument);
    m_base = m_cur = base;
    // Avoid undefined pointer arithmetic on a null base; a null buffer stays an empty range.
    m_end      = base != nullptr ? base + len : base;
    m_overread = false;
  }
  [[nodiscard]] ptrdiff_t bytecount() const noexcept
  {
    // Avoid `nullptr - nullptr` on a null-base reader; it has consumed nothing.
    return m_base != nullptr ? m_cur - m_base : 0;
  }
  // True if a read went past the end of buffer (set by getb/skip_bytes on overread). Checked by
  // callers after their own schema validation; see read(FileMetaData*).
  [[nodiscard]] bool overread() const noexcept { return m_overread; }
  // True if a wire-type/schema-type mismatch must be rejected (default THROW); false means skip it
  // per Thrift forward-compat (COMPAT), which the spark-rapids footer facade uses.
  [[nodiscard]] bool should_throw_on_type_mismatch() const noexcept
  {
    return m_mismatch_policy == experimental::thrift_mismatch_policy::THROW;
  }
  // A read at end-of-buffer sets the sticky overread flag (queried via overread()) and
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

  // Returns a varint-encoded integer. `T` is constrained to unsigned so `numeric_limits<T>::digits`
  // is the full value width; a signed `T` would drop the sign bit and misplace the overflow bound.
  template <std::unsigned_integral T>
  T get_varint()
  {
    T v = 0;
    for (uint32_t l = 0;; l += 7) {
      T const c = getb();
      // The byte's value, shifted into place, must fit in `T`; `l < digits` also keeps `max() >> l`
      // itself in range. Comparing the raw byte, not the masked payload, is intentional: it also
      // rejects a continuation byte whose successor group could not fit.
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

  // True if the wire type matches the schema type; on mismatch strict mode throws while lenient
  // mode (`COMPAT`) skips the value and returns false.
  [[nodiscard]] bool check_field_type(int type, FieldType expected);

  // True if a non-empty list's wire element type matches `expected`; on mismatch strict mode
  // throws while lenient mode skips all `count` elements and returns false.
  [[nodiscard]] bool check_list_element_type(int type, FieldType expected, uint32_t count);

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
  // Reject (`THROW`) vs skip (`COMPAT`) a struct field whose wire type mismatches the schema type.
  experimental::thrift_mismatch_policy m_mismatch_policy =
    experimental::thrift_mismatch_policy::THROW;

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

/**
 * @brief Decode footer bytes into a `FileMetaData`, rejecting truncation/corruption
 *
 * @param footer_bytes Thrift-compact-encoded footer bytes
 * @param metadata Output metadata
 * @param mode Mismatch policy, see `thrift_mismatch_policy`
 *
 * @throws cudf::logic_error If the footer is truncated or corrupt within the struct
 */
void decode_footer_bytes(
  cudf::host_span<uint8_t const> footer_bytes,
  FileMetaData* metadata,
  experimental::thrift_mismatch_policy mode = experimental::thrift_mismatch_policy::THROW);

/**
 * @brief Decode footer bytes and initialize the schema, rejecting truncation/corruption
 *
 * The schema-init check precedes the overread check so a footer from which no schema can be built
 * reports the specific "Cannot initialize schema" rather than generic overread.
 *
 * @param footer_bytes Thrift-compact-encoded footer bytes
 * @param metadata Output metadata
 *
 * @throws cudf::logic_error If the schema cannot be initialized
 * @throws cudf::logic_error If the footer is truncated or corrupt within the struct
 */
void decode_footer_and_init_schema(cudf::host_span<uint8_t const> footer_bytes,
                                   FileMetaData* metadata);

}  // namespace io::parquet::detail
}  // namespace CUDF_EXPORT cudf
