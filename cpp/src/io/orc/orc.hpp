/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/io/datasource.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/io/orc_types.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/std/optional>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cudf::io::orc::detail {

static constexpr uint32_t block_header_size = 3;
// Seconds from January 1st, 1970 to January 1st, 2015
static constexpr int64_t orc_utc_epoch = 1420070400;
// ORC datasets start with a 3 byte header
static constexpr char const* MAGIC = "ORC";

// Each ORC writer implementation should write its code to the file footer
// the codes are specified in the ORC specification
static constexpr int32_t cudf_writer_code = 5;
// Each ORC writer implementation should write its version to the PostScript
// The version values are based on the ORC Java writer bug fixes and features
// From https://github.com/apache/orc-format/blob/main/src/main/proto/orc_proto.proto:
//   0 = original
//   1 = HIVE-8732 fixed (fixed stripe/file maximum statistics &
//                        string statistics use utf8 for min/max)
//   2 = HIVE-4243 fixed (use real column names from Hive tables)
//   3 = HIVE-12055 added (vectorized writer implementation)
//   4 = HIVE-13083 fixed (decimals write present stream correctly)
//   5 = ORC-101 fixed (bloom filters use utf8 consistently)
//   6 = ORC-135 fixed (timestamp statistics use utc)
//   7 = ORC-517 fixed (decimal64 min/max incorrect)
//   8 = ORC-203 added (trim very long string statistics)
//   9 = ORC-14 added (column encryption)
// Our version should be updated as we implement the features from the list above
static constexpr uint32_t cudf_writer_version = 7;

// Used for the nanosecond remainder in timestamp statistics when the actual nanoseconds of min/max
// are not included. As the timestamp statistics are stored as milliseconds + nanosecond remainder,
// the maximum nanosecond remainder is 999,999 (nanoseconds in a millisecond - 1).
static constexpr int32_t DEFAULT_MIN_NANOS = 0;
static constexpr int32_t DEFAULT_MAX_NANOS = 999'999;

struct PostScript {
  uint64_t footerLength       = 0;        // the length of the footer section in bytes
  CompressionKind compression = NONE;     // the kind of generic compression used
  uint64_t compressionBlockSize{};        // the maximum size of each compression chunk
  std::vector<uint32_t> version;          // the version of the file format [major, minor]
  uint64_t metadataLength = 0;            // the length of the metadata section in bytes
  std::optional<uint32_t> writerVersion;  // The version of the writer that wrote the file
  std::string magic = "";                 // the fixed string "ORC"
};

struct StripeInformation {
  uint64_t offset       = 0;  // the start of the stripe within the file
  uint64_t indexLength  = 0;  // the length of the indexes in bytes
  uint64_t dataLength   = 0;  // the length of the data in bytes
  uint64_t footerLength = 0;  // the length of the footer in bytes
  uint64_t numberOfRows = 0;  // the number of rows in the stripe
};

struct SchemaType {
  TypeKind kind = INVALID_TYPE_KIND;  // the kind of this type
  std::vector<uint32_t> subtypes;  // the type ids of any subcolumns for list, map, struct, or union
  std::vector<std::string> fieldNames;  // the list of field names for struct
  std::optional<uint32_t>
    maximumLength;  // the maximum length of the type for varchar or char in UTF-8 characters
  std::optional<uint32_t> precision;  // the precision for decimal
  std::optional<uint32_t> scale;      // the scale for decimal
};

struct UserMetadataItem {
  std::string name;   // the user defined key
  std::string value;  // the user defined binary value as string
};

using col_stats_blob = std::vector<uint8_t>;  // Column statistics blob

struct Footer {
  uint64_t headerLength  = 0;              // the length of the file header in bytes (always 3)
  uint64_t contentLength = 0;              // the length of the file header and body in bytes
  std::vector<StripeInformation> stripes;  // the information about the stripes
  std::vector<SchemaType> types;           // the schema information
  std::vector<UserMetadataItem> metadata;  // the user metadata that was added
  uint64_t numberOfRows = 0;               // the total number of rows in the file
  std::vector<col_stats_blob> statistics;  // Column statistics blobs
  uint32_t rowIndexStride = 0;             // the maximum number of rows in each index entry
  std::optional<uint32_t> writer;          // Writer code
};

struct Stream {
  StreamKind kind = INVALID_STREAM_KIND;
  std::optional<uint32_t> column_id;  // ORC column id (different from column index in the table!)
  uint64_t length = 0;                // the number of bytes in the stream

  // Returns index of the column in the table, if any
  // Stream of the 'column 0' does not have a corresponding column in the table
  [[nodiscard]] std::optional<uint32_t> column_index() const noexcept
  {
    return column_id.value_or(0) > 0 ? std::optional<uint32_t>{*column_id - 1}
                                     : std::optional<uint32_t>{};
  }
};

struct ColumnEncoding {
  ColumnEncodingKind kind = INVALID_ENCODING_KIND;
  uint32_t dictionarySize = 0;  // for dictionary encodings, record the size of the dictionary
};

struct StripeFooter {
  std::vector<Stream> streams;          // the location of each stream
  std::vector<ColumnEncoding> columns;  // the encoding of each column
  std::string writerTimezone = "";      // time zone of the writer
};

/**
 * @brief Contains per-column ORC statistics.
 *
 * At most one of the `***_statistics` members has a value.
 */
struct column_statistics {
  std::optional<uint64_t> number_of_values;
  std::optional<integer_statistics> int_stats;
  std::optional<double_statistics> double_stats;
  std::optional<string_statistics> string_stats;
  std::optional<bucket_statistics> bucket_stats;
  std::optional<decimal_statistics> decimal_stats;
  std::optional<date_statistics> date_stats;
  std::optional<binary_statistics> binary_stats;
  std::optional<timestamp_statistics> timestamp_stats;
  std::optional<bool> has_null;
};

struct StripeStatistics {
  std::vector<col_stats_blob> colStats;  // Column statistics blobs
};

struct Metadata {
  std::vector<StripeStatistics> stripeStats;
};

int inline constexpr encode_field_number(int field_number, ProtofType field_type) noexcept
{
  return (field_number * 8) + static_cast<int>(field_type);
}

template <typename base_t>
int static constexpr encode_field_number_base(int field_number) noexcept
  requires(!std::is_arithmetic_v<base_t> and !std::is_enum_v<base_t>)
{
  return encode_field_number(field_number, ProtofType::FIXEDLEN);
}

template <typename base_t>
int static constexpr encode_field_number_base(int field_number) noexcept
  requires(std::is_integral_v<base_t> or std::is_enum_v<base_t>)
{
  return encode_field_number(field_number, ProtofType::VARINT);
}

template <typename base_t>
int static constexpr encode_field_number_base(int field_number) noexcept
  requires(std::is_same_v<base_t, float>)
{
  return encode_field_number(field_number, ProtofType::FIXED32);
}

template <typename base_t>
int static constexpr encode_field_number_base(int field_number) noexcept
  requires(std::is_same_v<base_t, double>)
{
  return encode_field_number(field_number, ProtofType::FIXED64);
}

template <typename T>
int constexpr encode_field_number(int field_number) noexcept
  requires(!std::is_class_v<T> or std::is_same_v<T, std::string>)
{
  return encode_field_number_base<T>(field_number);
}

// containers change the field number encoding
template <typename T>
int constexpr encode_field_number(int field_number) noexcept
  requires(std::is_same_v<T, std::vector<typename T::value_type>>)
{
  return encode_field_number_base<T>(field_number);
}

// optional fields don't change the field number encoding
template <typename T>
int constexpr encode_field_number(int field_number) noexcept
  requires(std::is_same_v<T, std::optional<typename T::value_type>>)
{
  return encode_field_number_base<typename T::value_type>(field_number);
}

/**
 * @brief Class for parsing Orc's Protocol Buffers encoded metadata
 */
class protobuf_reader {
 public:
  protobuf_reader(uint8_t const* base, size_t len) : m_base(base), m_cur(base), m_end(base + len) {}

  template <typename T>
  void read(T& s)
  {
    read(s, m_end - m_cur);
  }
  void read(PostScript&, size_t maxlen);
  void read(Footer&, size_t maxlen);
  void read(StripeInformation&, size_t maxlen);
  void read(SchemaType&, size_t maxlen);
  void read(UserMetadataItem&, size_t maxlen);
  void read(StripeFooter&, size_t maxlen);
  void read(Stream&, size_t maxlen);
  void read(ColumnEncoding&, size_t maxlen);
  void read(integer_statistics&, size_t maxlen);
  void read(double_statistics&, size_t maxlen);
  void read(string_statistics&, size_t maxlen);
  void read(bucket_statistics&, size_t maxlen);
  void read(decimal_statistics&, size_t maxlen);
  void read(date_statistics&, size_t maxlen);
  void read(binary_statistics&, size_t maxlen);
  void read(timestamp_statistics&, size_t maxlen);
  void read(column_statistics&, size_t maxlen);
  void read(StripeStatistics&, size_t maxlen);
  void read(Metadata&, size_t maxlen);

 private:
  template <int index>
  friend struct function_switch;

  void skip_bytes(size_t bytecnt)
  {
    bytecnt = std::min(bytecnt, (size_t)(m_end - m_cur));
    m_cur += bytecnt;
  }

  template <typename T>
  T get();

  void skip_struct_field(int t);

  template <typename T, typename... Operator>
  void function_builder(T& s, size_t maxlen, std::tuple<Operator...>& op);

  uint32_t read_field_size(uint8_t const* end);

  template <typename T>
  void read_field(T& value, uint8_t const* end)
    requires(std::is_integral_v<T>)
  {
    value = get<T>();
  }

  template <typename T>
  void read_field(T& value, uint8_t const* end)
    requires(std::is_enum_v<T>)
  {
    value = static_cast<T>(get<uint32_t>());
  }

  template <typename T>
  void read_field(T& value, uint8_t const* end)
    requires(std::is_same_v<T, std::string>)
  {
    auto const size = read_field_size(end);
    value.assign(reinterpret_cast<char const*>(m_cur), size);
    m_cur += size;
  }

  template <typename T>
  void read_field(T& value, uint8_t const* end)
    requires(std::is_same_v<T, std::vector<std::string>>)
  {
    auto const size = read_field_size(end);
    value.emplace_back(reinterpret_cast<char const*>(m_cur), size);
    m_cur += size;
  }

  template <typename T>
  void read_field(T& value, uint8_t const* end)
    requires(std::is_same_v<T, std::vector<typename T::value_type>> and
             !std::is_same_v<std::string, typename T::value_type>)
  {
    auto const size = read_field_size(end);
    value.emplace_back();
    read(value.back(), size);
  }

  template <typename T>
  void read_field(T& value, uint8_t const* end)
    requires(std::is_same_v<T, std::optional<typename T::value_type>>)
  {
    typename T::value_type contained_value;
    read_field(contained_value, end);
    value = std::optional<typename T::value_type>{std::move(contained_value)};
  }

  template <typename T>
  auto read_field(T& value, uint8_t const* end) -> decltype(read(value, 0))
  {
    auto const size = read_field_size(end);
    read(value, size);
  }

  template <typename T>
  void read_field(T& value, uint8_t const* end)
    requires(std::is_floating_point_v<T>)
  {
    CUDF_EXPECTS(std::cmp_less_equal(sizeof(T), end - m_cur), "Protobuf parsing out of bounds");
    memcpy(&value, m_cur, sizeof(T));
    m_cur += sizeof(T);
  }

  template <typename T>
  void read_packed_field(T& value, uint8_t const* end)
  {
    auto const len       = get<uint32_t>();
    auto const field_end = std::min(m_cur + len, end);
    while (m_cur < field_end)
      value.push_back(get<typename T::value_type>());
  }

  template <typename T>
  void read_raw_field(T& value, uint8_t const* end)
  {
    auto const size = read_field_size(end);
    value.emplace_back(m_cur, m_cur + size);
    m_cur += size;
  }

  template <typename T>
  struct field_reader {
    int const encoded_field_number;
    T& output_value;

    field_reader(int field_number, T& field_value)
      : encoded_field_number(encode_field_number<T>(field_number)), output_value(field_value)
    {
    }

    inline void operator()(protobuf_reader* pbr, uint8_t const* end)
    {
      pbr->read_field(output_value, end);
    }
  };

  template <typename T>
  struct packed_field_reader {
    int const encoded_field_number;
    T& output_value;

    packed_field_reader(int field_number, T& field_value)
      : encoded_field_number(encode_field_number<T>(field_number)), output_value(field_value)
    {
    }

    inline void operator()(protobuf_reader* pbr, uint8_t const* end)
    {
      pbr->read_packed_field(output_value, end);
    }
  };

  template <typename T>
  struct raw_field_reader {
    int const encoded_field_number;
    T& output_value;

    raw_field_reader(int field_number, T& field_value)
      : encoded_field_number(encode_field_number<T>(field_number)), output_value(field_value)
    {
    }

    inline void operator()(protobuf_reader* pbr, uint8_t const* end)
    {
      pbr->read_raw_field(output_value, end);
    }
  };

  uint8_t const* const m_base;
  uint8_t const* m_cur;
  uint8_t const* const m_end;
};

template <>
inline uint8_t protobuf_reader::get<uint8_t>()
{
  return (m_cur < m_end) ? *m_cur++ : 0;
};

template <>
inline bool protobuf_reader::get<bool>()
{
  return static_cast<bool>(get<uint8_t>());
};

template <>
inline uint64_t protobuf_reader::get<uint64_t>()
{
  uint64_t v = 0;
  for (uint32_t l = 0; l < sizeof(v) * 8; l += 7) {
    uint64_t c = get<uint8_t>();
    v |= (c & 0x7f) << l;
    if (c < 0x80) return v;
  }
  CUDF_FAIL("Invalid varint: exceeds maximum encoded length");
}

template <>
inline uint32_t protobuf_reader::get<uint32_t>()
{
  // Negative int32 is sign-extended to int64 and encoded as a 10-byte varint, so we decode the
  // full 64-bit varint and truncate to the low 32 bits.
  return static_cast<uint32_t>(get<uint64_t>());
}

template <typename T>
auto decode_zigzag(T u)
{
  using signed_t = std::make_signed_t<T>;
  return static_cast<signed_t>((u >> 1u) ^ -static_cast<signed_t>(u & 1));
}

template <>
inline int32_t protobuf_reader::get<int32_t>()
{
  return decode_zigzag(get<uint32_t>());
}

template <>
inline int64_t protobuf_reader::get<int64_t>()
{
  return decode_zigzag(get<uint64_t>());
}

/**
 * @brief Class for encoding Orc's metadata with Protocol Buffers
 */
class protobuf_writer {
 public:
  protobuf_writer() = default;

  protobuf_writer(std::size_t bytes) : m_buff(bytes) {}

  uint32_t put_byte(uint8_t v)
  {
    m_buff.push_back(v);
    return 1;
  }

  template <typename T>
  uint32_t put_bytes(host_span<T const> values)
  {
    static_assert(sizeof(T) == 1);
    m_buff.reserve(m_buff.size() + values.size());
    m_buff.insert(m_buff.end(), values.begin(), values.end());
    return values.size();
  }

  uint32_t put_uint(uint64_t v)
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

  uint32_t put_int(int64_t v)
  {
    int64_t s = (v < 0);
    return put_uint(((v ^ -s) << 1) + s);
  }

  void put_row_index_entry(int32_t present_blk,
                           int32_t present_ofs,
                           int32_t data_blk,
                           int32_t data_ofs,
                           int32_t data2_blk,
                           int32_t data2_ofs,
                           TypeKind kind,
                           col_stats_blob const* stats);

  [[nodiscard]] std::size_t size() const { return m_buff.size(); }
  uint8_t const* data() { return m_buff.data(); }

  std::vector<uint8_t>& buffer() { return m_buff; }
  std::vector<uint8_t> release() { return std::move(m_buff); }

 public:
  size_t write(PostScript const&);
  size_t write(Footer const&);
  size_t write(StripeInformation const&);
  size_t write(SchemaType const&);
  size_t write(UserMetadataItem const&);
  size_t write(StripeFooter const&);
  size_t write(Stream const&);
  size_t write(ColumnEncoding const&);
  size_t write(StripeStatistics const&);
  size_t write(Metadata const&);

 protected:
  std::vector<uint8_t> m_buff;
  struct protobuf_field_writer;
};

/**
 * @brief Class for decompressing Orc data blocks using the CPU
 */

class orc_decompressor {
 public:
  orc_decompressor(CompressionKind kind, uint64_t blockSize);

  /**
   * @brief ORC block decompression
   *
   * @param src compressed data
   *
   * @return decompressed data
   */
  host_span<uint8_t const> decompress_blocks(host_span<uint8_t const> src);
  [[nodiscard]] uint32_t GetLog2MaxCompressionRatio() const { return m_log2MaxRatio; }
  [[nodiscard]] uint64_t GetMaxUncompressedBlockSize(uint32_t block_len) const
  {
    return std::min(static_cast<uint64_t>(block_len) << m_log2MaxRatio, m_blockSize);
  }
  [[nodiscard]] compression_type compression() const { return _compression; }
  [[nodiscard]] auto GetBlockSize() const { return m_blockSize; }

 protected:
  compression_type _compression;
  uint32_t m_log2MaxRatio = 24;  // log2 of maximum compression ratio
  uint64_t m_blockSize;
  std::vector<uint8_t> m_buf;
};

/**
 * @brief Stores orc id for each column and number of children in that column.
 *
 * @code{.pseudo}
 * Consider following data where a struct has two members and a list column
 * {"struct": [{"a": 1, "b": 2}, {"a":3, "b":5}], "list":[[1, 2], [2, 3]]}
 *
 * `orc_column_meta` for struct column would be
 * id = 0
 * num_children = 2
 *
 * `orc_column_meta` for list column would be
 * id = 3
 * num_children = 1
 * @endcode
 *
 */
struct orc_column_meta {
  size_type id;            // orc id for the column
  size_type num_children;  // number of children at the same level of nesting in case of struct
};

/**
 * @brief Stores column's validity map and null count
 */
struct column_validity_info {
  uint32_t* valid_map_base;
  uint32_t null_count;
};

/**
 * @brief A helper class for ORC file metadata. Provides some additional
 * convenience methods for initializing and accessing metadata.
 */
class metadata {
 public:
  struct orc_stripe_info {
    StripeInformation const* stripe_info;
    StripeFooter const* stripe_footer;
    int source_idx;
  };
  std::vector<orc_stripe_info> stripe_info;

 public:
  explicit metadata(datasource* const src, cuda::stream_ref stream);

  metadata(metadata const&)            = delete;
  metadata& operator=(metadata const&) = delete;
  metadata(metadata&&)                 = default;
  metadata& operator=(metadata&&)      = default;

  [[nodiscard]] auto get_total_rows() const { return ff.numberOfRows; }
  [[nodiscard]] size_type get_num_stripes() const { return ff.stripes.size(); }
  [[nodiscard]] size_type get_num_columns() const { return ff.types.size(); }
  /**
   * @brief Returns the name of the column with the given ID.
   *
   * Name might not be unique in the ORC file, since columns with different parents are allowed to
   * have the same names.
   */
  [[nodiscard]] std::string const& column_name(size_type column_id) const
  {
    CUDF_EXPECTS(column_id < get_num_columns(), "Out of range column id provided");
    return column_names[column_id];
  }
  /**
   * @brief Returns the full name of the column with the given ID - includes the ancestor columns
   * names.
   *
   * Each column in the ORC file has a unique path.
   */
  [[nodiscard]] std::string const& column_path(size_type column_id) const
  {
    CUDF_EXPECTS(column_id < get_num_columns(), "Out of range column id provided");
    return column_paths[column_id];
  }
  [[nodiscard]] auto get_row_index_stride() const { return ff.rowIndexStride; }

  /**
   * @brief Returns the ID of the parent column of the given column.
   */
  [[nodiscard]] size_type parent_id(size_type column_id) const
  {
    return parents.at(column_id).value().id;
  }

  /**
   * @brief Returns the index the given column has in its parent's children list.
   */
  [[nodiscard]] size_type field_index(size_type column_id) const
  {
    return parents.at(column_id).value().field_idx;
  }

  /**
   * @brief Returns whether the given column has a parent.
   */
  [[nodiscard]] size_type column_has_parent(size_type column_id) const
  {
    return parents.at(column_id).has_value();
  }

 public:
  PostScript ps;
  Footer ff;
  Metadata md;
  std::vector<StripeFooter> stripefooters;
  std::unique_ptr<orc_decompressor> decompressor;
  datasource* source;

 private:
  struct column_parent {
    // parent's ID
    size_type id;
    // Index of this column in the parent's list of children
    size_type field_idx;
    column_parent(size_type parent_id, size_type field_idx) : id{parent_id}, field_idx{field_idx} {}
  };
  void init_parent_descriptors();
  std::vector<std::optional<column_parent>> parents;

  void init_column_names();
  std::vector<std::string> column_names;
  std::vector<std::string> column_paths;
};

/**
 * @brief `column_device_view` and additional, ORC specific, information on the column.
 */
struct orc_column_device_view : public column_device_view {
  __device__ orc_column_device_view(column_device_view col,
                                    cuda::std::optional<uint32_t> parent_idx)
    : column_device_view{col}, parent_index{parent_idx}
  {
  }
  cuda::std::optional<uint32_t> parent_index;
  bitmask_type const* pushdown_mask = nullptr;
};

/**
 * @brief Range of rows within a single rowgroup.
 */
struct rowgroup_rows {
  size_type begin;
  size_type end;
  [[nodiscard]] CUDF_HOST_DEVICE constexpr auto size() const noexcept { return end - begin; }
};

}  // namespace cudf::io::orc::detail
