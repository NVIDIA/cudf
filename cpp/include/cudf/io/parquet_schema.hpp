/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file parquet_schema.hpp
 * @brief Parquet footer schema structs
 */

#pragma once

#include <cudf/types.hpp>

#include <cuda/std/optional>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::parquet {
/**
 * @addtogroup io_types
 * @{
 */

/**
 * @brief Basic data types in Parquet, determines how data is physically stored
 */
enum class Type : int8_t {
  UNDEFINED            = -1,  // Undefined for non-leaf nodes
  BOOLEAN              = 0,
  INT32                = 1,
  INT64                = 2,
  INT96                = 3,  // Deprecated
  FLOAT                = 4,
  DOUBLE               = 5,
  BYTE_ARRAY           = 6,
  FIXED_LEN_BYTE_ARRAY = 7,
};

/**
 * @brief High-level data types in Parquet, determines how data is logically interpreted
 */
enum class ConvertedType : int8_t {
  UNKNOWN = -1,  // No type information present
  UTF8    = 0,   // a BYTE_ARRAY may contain UTF8 encoded chars
  MAP     = 1,   // a map is converted as an optional field containing a repeated key/value pair
  MAP_KEY_VALUE = 2,  // a key/value pair is converted into a group of two fields
  LIST =
    3,  // a list is converted into an optional field containing a repeated field for its values
  ENUM    = 4,      // an enum is converted into a binary field
  DECIMAL = 5,      // A decimal value. 10^(-scale) encoded as 2's complement big endian
                    // (precision=number of digits, scale=location of decimal point)
  DATE        = 6,  // A Date, stored as days since Unix epoch, encoded as the INT32 physical type.
  TIME_MILLIS = 7,  // A time. The total number of milliseconds since midnight.The value is stored
                    // as an INT32 physical type.
  TIME_MICROS = 8,  // A time. The total number of microseconds since midnight.  The value is stored
                    // as an INT64 physical type.
  TIMESTAMP_MILLIS = 9,   // A date/time combination, recorded as milliseconds since the Unix epoch
                          // using physical type of INT64.
  TIMESTAMP_MICROS = 10,  // A date/time combination, microseconds since the Unix epoch as INT64
  UINT_8           = 11,  // An unsigned integer 8-bit value as INT32
  UINT_16          = 12,  // An unsigned integer 16-bit value as INT32
  UINT_32          = 13,  // An unsigned integer 32-bit value as INT32
  UINT_64          = 14,  // An unsigned integer 64-bit value as INT64
  INT_8            = 15,  // A signed integer 8-bit value as INT32
  INT_16           = 16,  // A signed integer 16-bit value as INT32
  INT_32           = 17,  // A signed integer 32-bit value as INT32
  INT_64           = 18,  // A signed integer 8-bit value as INT64
  JSON             = 19,  // A JSON document embedded within a single UTF8 column.
  BSON             = 20,  // A BSON document embedded within a single BINARY column.
  INTERVAL = 21,  // This type annotates a time interval stored as a FIXED_LEN_BYTE_ARRAY of length
                  // 12 for 3 integers {months,days,milliseconds}
  NA = 25,        // No Type information, For eg, all-nulls.
};

/**
 * @brief Encoding types for the actual data stream
 */
enum class Encoding : uint8_t {
  PLAIN                   = 0,
  GROUP_VAR_INT           = 1,  // Deprecated, never used
  PLAIN_DICTIONARY        = 2,
  RLE                     = 3,
  BIT_PACKED              = 4,  // Deprecated by parquet-format in 2013, superseded by RLE
  DELTA_BINARY_PACKED     = 5,
  DELTA_LENGTH_BYTE_ARRAY = 6,
  DELTA_BYTE_ARRAY        = 7,
  RLE_DICTIONARY          = 8,
  BYTE_STREAM_SPLIT       = 9,
  NUM_ENCODINGS           = 10,
};

/**
 * @brief Compression codec used for compressed data pages
 */
enum class Compression : uint8_t {
  UNCOMPRESSED = 0,
  SNAPPY       = 1,
  GZIP         = 2,
  LZO          = 3,
  BROTLI       = 4,  // Added in 2.3.2
  LZ4          = 5,  // deprecated; based on LZ4, but with an additional undocumented framing scheme
  ZSTD         = 6,  // Added in 2.3.2
  LZ4_RAW      = 7,  // "standard" LZ4 block format
};

/**
 * @brief Compression codec used for compressed data pages
 */
enum class FieldRepetitionType : int8_t {
  UNSPECIFIED = -1,
  REQUIRED    = 0,  // This field is required (can not be null) and each record has exactly 1 value.
  OPTIONAL    = 1,  // The field is optional (can be null) and each record has 0 or 1 values.
  REPEATED    = 2,  // The field is repeated and can contain 0 or more values
};

/**
 * @brief Types of pages
 */
enum class PageType : uint8_t {
  DATA_PAGE       = 0,
  INDEX_PAGE      = 1,
  DICTIONARY_PAGE = 2,
  DATA_PAGE_V2    = 3,
};

/**
 * @brief Enum to annotate whether lists of min/max elements inside ColumnIndex
 * are ordered and if so, in which direction.
 */
enum class BoundaryOrder : uint8_t {
  UNORDERED  = 0,
  ASCENDING  = 1,
  DESCENDING = 2,
};

/**
 * @brief Thrift compact protocol struct field types
 */
enum class FieldType : uint8_t {
  BOOLEAN_TRUE  = 1,
  BOOLEAN_FALSE = 2,
  I8            = 3,
  I16           = 4,
  I32           = 5,
  I64           = 6,
  DOUBLE        = 7,
  BINARY        = 8,
  LIST          = 9,
  SET           = 10,
  MAP           = 11,
  STRUCT        = 12,
  UUID          = 13,
};

/**
 * @brief Struct that describes the Parquet file data header
 */
struct file_header_s {
  /// Parquet 4-byte magic number "PAR1"
  uint32_t magic;
};

/**
 * @brief Struct that describes the Parquet file data postscript
 */
struct file_ender_s {
  /// Length of the footer
  uint32_t footer_len;
  /// Parquet 4-byte magic number "PAR1"
  uint32_t magic;
};

/**
 * @brief Struct that describes the decimal logical type annotation
 *
 * Allowed for physical types: INT32, INT64, FIXED_LEN_BYTE_ARRAY, and BYTE_ARRAY.
 */
struct DecimalType {
  /// Scale must be zero or a positive integer less than or equal to the precision.
  int32_t scale = 0;
  /// Precision must be a non-zero positive integer.
  int32_t precision = 0;
};

/**
 * @brief Time units for temporal logical types
 */
struct TimeUnit {
  /// Available time units
  enum Type : uint8_t { UNDEFINED, MILLIS, MICROS, NANOS };
  /// Time unit type
  Type type;
};

/**
 * @brief Struct that describes the time logical type annotation
 *
 * Allowed for physical types: INT32 (millis), INT64 (micros, nanos)
 */
struct TimeType {
  /// Default to true because the timestamps are implicitly in UTC.
  /// Writer option overrides this to default
  bool isAdjustedToUTC = true;
  /// Time unit type
  TimeUnit unit = {TimeUnit::Type::MILLIS};
};

/**
 * @brief Struct that describes the timestamp logical type annotation
 *
 * Allowed for physical types: INT64
 */
struct TimestampType {
  /// Default to true because the timestamps are implicitly in UTC.
  /// Writer option overrides this to default
  bool isAdjustedToUTC = true;
  /// Timestamp's time unit
  TimeUnit unit = {TimeUnit::Type::MILLIS};
};

/**
 * @brief Struct that describes the integer logical type annotation
 *
 * Allowed for physical types: INT32, INT64
 */
struct IntType {
  /// bitWidth must be 8, 16, 32, or 64.
  int8_t bitWidth = 0;
  /// Whether the integer is signed
  bool isSigned = false;
};

/**
 * @brief Struct that describes the logical type annotation
 */
struct LogicalType {
  /// Logical type annotations to replace ConvertedType.
  enum Type : uint8_t {
    UNDEFINED,
    STRING,
    MAP,
    LIST,
    ENUM,
    DECIMAL,
    DATE,
    TIME,
    TIMESTAMP,
    // 9 is reserved
    INTEGER = 10,
    UNKNOWN,
    JSON,
    BSON,
    VARIANT = 16,
  };

  /// Logical type
  Type type;
  /// Decimal type
  cuda::std::optional<DecimalType> decimal_type;
  /// Time type
  cuda::std::optional<TimeType> time_type;
  /// Timestamp type
  cuda::std::optional<TimestampType> timestamp_type;
  /// Integer type
  cuda::std::optional<IntType> int_type;

  /**
   * @brief Default constructor
   *
   * @param tp Logical type
   */
  LogicalType(Type tp = Type::UNDEFINED) : type(tp) {}

  /**
   * @brief Constructor for Decimal logical type
   *
   * @param dt Decimal type
   */
  LogicalType(DecimalType&& dt) : type(DECIMAL), decimal_type(dt) {}

  /**
   * @brief Constructor for Time logical type
   *
   * @param tt Time type
   */
  LogicalType(TimeType&& tt) : type(TIME), time_type(tt) {}

  /**
   * @brief Constructor for Timestamp logical type
   *
   * @param tst Timestamp type
   */
  LogicalType(TimestampType&& tst) : type(TIMESTAMP), timestamp_type(tst) {}

  /**
   * @brief Constructor for Integer logical type
   *
   * @param it Integer type
   */
  LogicalType(IntType&& it) : type(INTEGER), int_type(it) {}

  /**
   * @brief Check if the time is in milliseconds
   *
   * @return True if the time is in milliseconds, false otherwise
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool is_time_millis() const
  {
    return type == TIME and time_type->unit.type == TimeUnit::MILLIS;
  }

  /**
   * @brief Check if the time is in microseconds
   *
   * @return True if the time is in microseconds, false otherwise
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool is_time_micros() const
  {
    return type == TIME and time_type->unit.type == TimeUnit::MICROS;
  }

  /**
   * @brief Check if the time is in nanoseconds
   *
   * @return True if the time is in nanoseconds, false otherwise
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool is_time_nanos() const
  {
    return type == TIME and time_type->unit.type == TimeUnit::NANOS;
  }

  /**
   * @brief Check if the timestamp is in milliseconds
   *
   * @return True if the timestamp is in milliseconds, false otherwise
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool is_timestamp_millis() const
  {
    return type == TIMESTAMP and timestamp_type->unit.type == TimeUnit::MILLIS;
  }

  /**
   * @brief Check if the timestamp is in microseconds
   *
   * @return True if the timestamp is in microseconds, false otherwise
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool is_timestamp_micros() const
  {
    return type == TIMESTAMP and timestamp_type->unit.type == TimeUnit::MICROS;
  }

  /**
   * @brief Check if the timestamp is in nanoseconds
   *
   * @return True if the timestamp is in nanoseconds, false otherwise
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr bool is_timestamp_nanos() const
  {
    return type == TIMESTAMP and timestamp_type->unit.type == TimeUnit::NANOS;
  }

  /**
   * @brief Get the bit width of the integer type
   *
   * @return The bit width of the integer type, or -1 if the type is not an integer
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr int8_t bit_width() const
  {
    return type == INTEGER ? int_type->bitWidth : -1;
  }

  /**
   * @brief Check if the integer is signed
   *
   * @return True if the integer is signed, false otherwise
   */
  [[nodiscard]] constexpr bool is_signed() const { return type == INTEGER and int_type->isSigned; }

  /**
   * @brief Get the scale of the decimal type
   *
   * @return The scale of the decimal type, or -1 if the type is not a decimal
   */
  [[nodiscard]] constexpr int32_t scale() const
  {
    return type == DECIMAL ? decimal_type->scale : -1;
  }

  /**
   * @brief Get the precision of the decimal type
   *
   * @return The precision of the decimal type, or -1 if the type is not a decimal
   */
  [[nodiscard]] CUDF_HOST_DEVICE constexpr int32_t precision() const
  {
    return type == DECIMAL ? decimal_type->precision : -1;
  }
};

/**
 * @brief Union to specify the order used for the min_value and max_value fields for a column.
 */
struct ColumnOrder {
  /// Available column order types
  enum Type : uint8_t { UNDEFINED, TYPE_ORDER };
  /// Column order type
  Type type;
};

/**
 * @brief Struct for describing an element/field in the Parquet format schema
 *
 * Parquet is a strongly-typed format so the file layout can be interpreted as
 * as a schema tree.
 */
struct SchemaElement {
  /// 1: parquet physical type for output
  Type type = Type::UNDEFINED;
  /// 2: byte length of FIXED_LENGTH_BYTE_ARRAY elements, or maximum bit length for other types
  int32_t type_length = 0;
  /// 3: repetition of the field
  FieldRepetitionType repetition_type = FieldRepetitionType::REQUIRED;
  /// 4: name of the field
  std::string name = "";
  /// 5: nested fields
  int32_t num_children = 0;
  /// 6: DEPRECATED: record the original type before conversion to parquet type
  std::optional<ConvertedType> converted_type;
  /// 7: DEPRECATED: record the scale for DECIMAL converted type
  int32_t decimal_scale = 0;
  /// 8: DEPRECATED: record the precision for DECIMAL converted type
  int32_t decimal_precision = 0;
  /// 9: save field_id from original schema
  std::optional<int32_t> field_id;
  /// 10: replaces converted type
  cuda::std::optional<LogicalType> logical_type;

  /// extra cudf specific fields
  bool output_as_byte_array = false;

  /// cudf type determined from arrow:schema
  std::optional<type_id> arrow_type;

  // The following fields are filled in later during schema initialization

  /// Maximum definition level
  int max_definition_level = 0;
  /// Maximum repetition level
  int max_repetition_level = 0;
  /// Parent index
  size_type parent_idx = 0;
  /// Children indices
  std::vector<size_type> children_idx;

  /**
   * @brief Check if two schema elements are equal
   *
   * @param other The other schema element to compare to
   * @return True if the two schema elements are equal, false otherwise
   */
  bool operator==(SchemaElement const& other) const
  {
    return type == other.type && converted_type == other.converted_type &&
           type_length == other.type_length && name == other.name &&
           num_children == other.num_children && decimal_scale == other.decimal_scale &&
           decimal_precision == other.decimal_precision && field_id == other.field_id;
  }

  // the parquet format is a little squishy when it comes to interpreting
  // repeated fields. sometimes repeated fields act as "stubs" in the schema
  // that don't represent a true nesting level.
  //
  // this is the case with plain lists:
  //
  // optional group my_list (LIST) {
  //   repeated group element {        <-- not part of the output hierarchy
  //     required binary str (UTF8);
  //   };
  // }
  //
  // However, for backwards compatibility reasons, there are a few special cases, namely
  // List<Struct<>> (which also corresponds to how the map type is specified), where
  // this does not hold true
  //
  // optional group my_list (LIST) {
  //   repeated group element {        <-- part of the hierarchy because it represents a struct
  //     required binary str (UTF8);
  //     required int32 num;
  //  };
  // }

  /**
   * @brief Check if the schema element is a stub
   *
   * A LIST- or MAP-annotated REPEATED group is NOT a stub: in the legacy 2-level Parquet
   * encoding the REPEATED group is the list/map itself (its single child is the element type),
   * not a wrapper that should be collapsed. Treating it as a stub would change
   * `list<list<primitive>>` into `list<primitive>` (and analogously for maps), diverging from
   * parquet-mr / Spark CPU.
   *
   * No file-version or writer gate is needed because the annotation pattern is itself the
   * encoding signal:
   *  - 3-level encoding: only the outer (non-REPEATED) group carries LIST/MAP; the inner
   *    REPEATED group is unannotated, so this test correctly returns true and the wrapper is
   *    collapsed.
   *  - 2-level encoding (Thrift, parquet-avro 1.7, and other legacy writers): the REPEATED
   *    group itself carries LIST/MAP, so this test correctly returns false and the group is
   *    preserved as a nesting level.
   *
   * The check covers both annotation sources -- the deprecated `converted_type` and the newer
   * `logical_type` -- since a Parquet file may carry only one.
   *
   * References:
   *  - Parquet backward-compatibility rules:
   *    https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#backward-compatibility-rules
   *  - cuDF context: https://github.com/NVIDIA/spark-rapids/issues/11589
   *  - parquet-avro variant: https://github.com/NVIDIA/spark-rapids/issues/11592
   *
   * @return True if the schema element is a stub, false otherwise
   */
  [[nodiscard]] bool is_stub() const
  {
    auto const is_list_or_map_logical_type =
      logical_type.has_value() &&
      (logical_type->type == LogicalType::LIST || logical_type->type == LogicalType::MAP);
    return repetition_type == FieldRepetitionType::REPEATED && num_children == 1 &&
           converted_type != ConvertedType::LIST && converted_type != ConvertedType::MAP &&
           !is_list_or_map_logical_type;
  }

  /**
   * @brief Check if the schema element is a one-level list
   *
   * https://github.com/apache/parquet-cpp/blob/642da05/src/parquet/schema.h#L49-L50
   * One-level LIST encoding: Only allows required lists with required cells: repeated value_type
   * name
   *
   * @param parent The parent schema element
   * @return True if the schema element is a one-level list, false otherwise
   */
  [[nodiscard]] bool is_one_level_list(SchemaElement const& parent) const
  {
    return repetition_type == FieldRepetitionType::REPEATED and num_children == 0 and
           not parent.is_list();
  }

  /**
   * @brief Check if the schema element is a list
   *
   * @return True if the schema element is a list, false otherwise
   */
  [[nodiscard]] bool is_list() const { return converted_type == ConvertedType::LIST; }

  /**
   * @brief Check if the schema element is a struct
   *
   * In parquet terms, a group is a level of nesting in the schema. a group can be a struct or a
   * list
   *
   * @return True if the schema element is a struct, false otherwise
   */
  [[nodiscard]] bool is_struct() const
  {
    return type == Type::UNDEFINED &&
           // this assumption might be a little weak.
           ((repetition_type != FieldRepetitionType::REPEATED) ||
            (repetition_type == FieldRepetitionType::REPEATED && num_children > 1));
  }
};

/**
 * @brief Thrift-derived struct describing column chunk statistics
 */
struct Statistics {
  /// deprecated max value in signed comparison order
  std::optional<std::vector<uint8_t>> max;
  /// deprecated min value in signed comparison order
  std::optional<std::vector<uint8_t>> min;
  /// count of null values in the column
  std::optional<int64_t> null_count;
  /// count of distinct values occurring
  std::optional<int64_t> distinct_count;
  /// max value for column determined by ColumnOrder
  std::optional<std::vector<uint8_t>> max_value;
  /// min value for column determined by ColumnOrder
  std::optional<std::vector<uint8_t>> min_value;
  /// If true, max_value is the actual maximum value for a column
  std::optional<bool> is_max_value_exact;
  /// If true, min_value is the actual minimum value for a column
  std::optional<bool> is_min_value_exact;
};

/**
 * @brief Thrift-derived struct containing statistics used to estimate page and column chunk sizes
 */
struct SizeStatistics {
  /// Number of variable-width bytes stored for the page/chunk. Should not be set for anything
  /// but the BYTE_ARRAY physical type.
  std::optional<int64_t> unencoded_byte_array_data_bytes;
  /**
   * When present, there is expected to be one element corresponding to each
   * repetition (i.e. size=max repetition_level+1) where each element
   * represents the number of times the repetition level was observed in the
   * data.
   *
   * This value should not be written if max_repetition_level is 0.
   */
  std::optional<std::vector<int64_t>> repetition_level_histogram;

  /**
   * Same as repetition_level_histogram except for definition levels.
   *
   * This value should not be written if max_definition_level is 0 or 1.
   */
  std::optional<std::vector<int64_t>> definition_level_histogram;
};

/**
 * @brief Thrift-derived struct describing page location information stored
 * in the offsets index.
 */
struct PageLocation {
  /// Offset of the page in the file
  int64_t offset;
  /// Compressed page size in bytes plus the heeader length
  int32_t compressed_page_size;
  /// Index within the column chunk of the first row of the page. reset to 0 at the beginning of
  /// each column chunk
  int64_t first_row_index;
};

/**
 * @brief Thrift-derived struct describing the offset index.
 */
struct OffsetIndex {
  /// Page locations
  std::vector<PageLocation> page_locations;
  /// per-page size info. see description of the same field in SizeStatistics. only present for
  /// columns with a BYTE_ARRAY physical type.
  std::optional<std::vector<int64_t>> unencoded_byte_array_data_bytes;
};

/**
 * @brief Thrift-derived struct describing the column index.
 */
struct ColumnIndex {
  /// Boolean used to determine if a page contains only null values
  std::vector<bool> null_pages;
  /// Lower bound for values in each page
  std::vector<std::vector<uint8_t>> min_values;
  /// Upper bound for values in each page
  std::vector<std::vector<uint8_t>> max_values;
  /// Indicates if min and max values are ordered
  BoundaryOrder boundary_order = BoundaryOrder::UNORDERED;
  /// Optional count of null values per page
  std::optional<std::vector<int64_t>> null_counts;
  /// Repetition level histogram for the column chunk
  std::optional<std::vector<int64_t>> repetition_level_histogram;
  /// Definition level histogram for the column chunk
  std::optional<std::vector<int64_t>> definition_level_histogram;
};

/**
 * @brief Thrift-derived struct describing page encoding statistics
 */
struct PageEncodingStats {
  /// The page type (data/dic/...)
  PageType page_type;
  /// Encoding of the page
  Encoding encoding;
  /// Number of pages of this type with this encoding
  int32_t count;
};

/**
 * @brief Thrift-derived struct describing column sort order
 */
struct SortingColumn {
  /// The column index (in this row group)
  int32_t column_idx;
  /// If true, indicates this column is sorted in descending order
  bool descending;
  /// If true, nulls will come before non-null values
  bool nulls_first;
};

/**
 * @brief Thrift-derived struct describing a column chunk
 */
struct ColumnChunkMetaData {
  /// Type of this column
  Type type = Type::BOOLEAN;
  /// Set of all encodings used for this column. The purpose is to validate whether we can decode
  /// those pages.
  std::vector<Encoding> encodings;
  /// Path in schema
  std::vector<std::string> path_in_schema;
  /// Compression codec
  Compression codec = Compression::UNCOMPRESSED;
  /// Number of values in this column
  int64_t num_values = 0;
  /// Total byte size of all uncompressed pages in this column chunk (including the headers)
  int64_t total_uncompressed_size = 0;
  /// Total byte size of all compressed pages in this column chunk (including the headers)
  int64_t total_compressed_size = 0;
  /// Byte offset from beginning of file to first data page
  int64_t data_page_offset = 0;
  /// Byte offset from beginning of file to root index page
  int64_t index_page_offset = 0;
  /// Byte offset from the beginning of file to first (only) dictionary page
  int64_t dictionary_page_offset = 0;
  /// Optional statistics for this column chunk
  Statistics statistics;
  /// Set of all encodings used for pages in this column chunk. This information can be used to
  /// determine if all data pages are dictionary encoded for example.
  std::optional<std::vector<PageEncodingStats>> encoding_stats;
  /// Byte offset from beginning of file to Bloom filter data.
  std::optional<int64_t> bloom_filter_offset;
  /// Size of Bloom filter data including the serialized header, in bytes. Added in 2.10 so readers
  /// may not read this field from old files and it can be obtained after the BloomFilterHeader has
  /// been deserialized. Writers should write this field so readers can read the bloom filter in a
  /// single I/O.
  std::optional<int32_t> bloom_filter_length;
  /// Optional statistics to help estimate total memory when converted to in-memory representations.
  /// The histograms contained in these statistics can also be useful in some cases for more
  /// fine-grained nullability/list length filter pushdown.
  std::optional<SizeStatistics> size_statistics;
};

/**
 * @brief The algorithm used in bloom filter
 */
struct BloomFilterAlgorithm {
  /// Available bloom filter algorithms
  enum Algorithm : uint8_t { UNDEFINED, SPLIT_BLOCK };
  /// Bloom filter algorithm
  Algorithm algorithm{Algorithm::SPLIT_BLOCK};
};

/**
 * @brief The hash function used in Bloom filter
 */
struct BloomFilterHash {
  /// Available bloom filter hashers
  enum Hash : uint8_t { UNDEFINED, XXHASH };
  /// Bloom filter hasher
  Hash hash{Hash::XXHASH};
};

/**
 * @brief The compression used in the bloom filter
 */
struct BloomFilterCompression {
  /// Available bloom filter compression types
  enum Compression : uint8_t { UNDEFINED, UNCOMPRESSED };
  /// Bloom filter compression type
  Compression compression{Compression::UNCOMPRESSED};
};

/**
 * @brief Bloom filter header struct
 *
 * The bloom filter data of a column chunk stores this header at the beginning
 * following by the filter bitset.
 */
struct BloomFilterHeader {
  /// The size of bitset in bytes
  int32_t num_bytes;
  /// The algorithm for setting bits
  BloomFilterAlgorithm algorithm;
  /// The hash function used for bloom filter
  BloomFilterHash hash;
  /// The compression used in the bloom filter
  BloomFilterCompression compression;
};

/**
 * @brief Thrift-derived struct describing a chunk of data for a particular
 * column
 *
 * Each column chunk lives in a particular row group and are guaranteed to be
 * contiguous in the file. Any missing or corrupted chunks can be skipped during
 * reading.
 */
struct ColumnChunk {
  /// File where column data is stored. If not set, assumed to be same file as metadata. This path
  /// is relative to the current file.
  std::string file_path = "";
  /// Deprecated: Byte offset in file_path to the ColumnMetaData
  int64_t file_offset = 0;
  /// Column metadata for this chunk. Some writers may also replicate this at the location pointed
  /// to by file_path/file_offset.
  ColumnChunkMetaData meta_data;
  /// File offset of ColumnChunk's OffsetIndex
  int64_t offset_index_offset = 0;
  /// Size of ColumnChunk's OffsetIndex, in bytes
  int32_t offset_index_length = 0;
  /// File offset of ColumnChunk's ColumnIndex
  int64_t column_index_offset = 0;
  /// Size of ColumnChunk's ColumnIndex, in bytes
  int32_t column_index_length = 0;

  // Following fields are derived from other fields

  /// Index in flattened schema (derived from path_in_schema)
  int schema_idx = -1;

  // The indexes don't really live here, but it's a convenient place to hang them.

  /// `OffsetIndex` for this column chunk
  std::optional<OffsetIndex> offset_index;
  /// `ColumnIndex` for this column chunk
  std::optional<ColumnIndex> column_index;
};

/**
 * @brief Thrift-derived struct describing a group of row data
 *
 * There may be one or more row groups within a dataset, with each row group
 * consisting of a column chunk for each column.
 */
struct RowGroup {
  /// Metadata for each column chunk in this row group
  std::vector<ColumnChunk> columns;
  /// Total byte size of all the uncompressed column data in this row group
  int64_t total_byte_size = 0;
  /// Number of rows in this row group
  int64_t num_rows = 0;
  /// If set, specifies a sort ordering of the rows in this RowGroup.
  std::optional<std::vector<SortingColumn>> sorting_columns;
  /// Byte offset from beginning of file to first page (data or dictionary) in this row group
  std::optional<int64_t> file_offset;
  /// Total byte size of all compressed (and potentially encrypted) column data in this row group
  std::optional<int64_t> total_compressed_size;
  /// Row group ordinal in the file
  std::optional<int16_t> ordinal;
};

/**
 * @brief Thrift-derived struct describing a key-value pair, for user metadata
 */
struct KeyValue {
  /// string key
  std::string key;
  /// string value
  std::string value;
};

/**
 * @brief Thrift-derived struct describing file-level metadata
 *
 * The additional information stored in the key_value_metadata can be used
 * during reading to reconstruct the output data to the exact original dataset
 * prior to conversion to Parquet.
 */
struct FileMetaData {
  /// Version of this file
  int32_t version = 0;
  /// Parquet schema for this file.  This schema contains metadata for all the columns. The schema
  /// is represented as a tree with a single root. The nodes of the tree are flattened to a list by
  /// doing a depth-first traversal. The column metadata contains the path in the schema for that
  /// column which can be used to map columns to nodes in the schema. The first element is the root
  std::vector<SchemaElement> schema;
  /// Number of rows in this file
  int64_t num_rows = 0;
  /// Row groups in this file
  std::vector<RowGroup> row_groups;
  /// Optional key/value metadata
  std::vector<KeyValue> key_value_metadata;
  /// String for application that wrote this file.
  std::string created_by = "";
  /// Sort order used for the min_value and max_value fields in the Statistics objects and the
  /// min_values and max_values fields in the ColumnIndex objects of each column in this file.
  std::optional<std::vector<ColumnOrder>> column_orders;
};

/**
 * @brief Thrift-derived struct describing the header for a data page
 */
struct DataPageHeader {
  /// Number of values, including NULLs, in this data page.
  int32_t num_values = 0;
  /// Encoding used for this data page
  Encoding encoding = Encoding::PLAIN;
  /// Encoding used for definition levels
  Encoding definition_level_encoding = Encoding::PLAIN;
  /// Encoding used for repetition levels
  Encoding repetition_level_encoding = Encoding::PLAIN;
};

/**
 * @brief Thrift-derived struct describing the header for a V2 data page
 */
struct DataPageHeaderV2 {
  /// Number of values, including NULLs, in this data page.
  int32_t num_values = 0;
  /// Number of NULL values, in this data page.
  int32_t num_nulls = 0;
  /// Number of rows in this data page. which means pages change on record
  /// boundaries (r = 0)
  int32_t num_rows = 0;
  /// Encoding used for this data page
  Encoding encoding = Encoding::PLAIN;
  /// Length of the definition levels
  int32_t definition_levels_byte_length = 0;
  /// Length of the repetition levels
  int32_t repetition_levels_byte_length = 0;
  /// Whether the values are compressed.
  bool is_compressed = true;
};

/**
 * @brief Thrift-derived struct describing the header for a dictionary page
 */
struct DictionaryPageHeader {
  /// Number of values in the dictionary
  int32_t num_values = 0;
  /// Encoding using this dictionary page
  Encoding encoding = Encoding::PLAIN;
};

/**
 * @brief Thrift-derived struct describing the page header
 *
 * Column data are divided into individual chunks, which are subdivided into
 * pages. Each page has an associated header, describing the page type. There
 * can be multiple page types interleaved in a column chunk, and each page is
 * individually compressed and encoded. Any missing or corrupted pages can be
 * skipped during reading.
 */
struct PageHeader {
  /// The type of the page: indicates which of the *_header fields is set
  PageType type = PageType::DATA_PAGE;
  /// Uncompressed page size in bytes (not including the header)
  int32_t uncompressed_page_size = 0;
  /// Compressed page size in bytes (not including the header)
  int32_t compressed_page_size = 0;

  // Headers for page specific data. One only will be set.

  /// Data page header
  DataPageHeader data_page_header;
  /// Dictionary page header
  DictionaryPageHeader dictionary_page_header;
  /// V2 data page header
  DataPageHeaderV2 data_page_header_v2;
};

/** @} */  // end of group
}  // namespace io::parquet
}  // namespace CUDF_EXPORT cudf
