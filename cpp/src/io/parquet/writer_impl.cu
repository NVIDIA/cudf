/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file writer_impl.cu
 * @brief cuDF-IO parquet writer class implementation
 */

#include "arrow_schema_writer.hpp"
#include "compact_protocol_reader.hpp"
#include "compact_protocol_writer.hpp"
#include "io/comp/common.hpp"
#include "io/comp/compression.hpp"
#include "io/parquet/parquet_gpu.hpp"
#include "io/statistics/column_statistics.cuh"
#include "io/utilities/column_utils.cuh"
#include "parquet_common.hpp"
#include "parquet_gpu.cuh"
#include "writer_impl.hpp"
#include "writer_impl_helpers.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/linked_column.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/lists/detail/dremel.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/logger.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuda/iterator>
#include <cuda/numeric>
#include <cuda/stream>
#include <thrust/fill.h>

#include <algorithm>
#include <cstring>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>

#ifndef CUDF_VERSION
#error "CUDF_VERSION is not defined"
#endif

namespace cudf::io::parquet::detail {

using namespace cudf::io::detail;

Compression to_parquet_compression(compression_type compression)
{
  switch (compression) {
    case compression_type::AUTO:
    case compression_type::SNAPPY: return Compression::SNAPPY;
    case compression_type::GZIP: return Compression::GZIP;
    case compression_type::ZSTD: return Compression::ZSTD;
    case compression_type::LZ4:
      // Parquet refers to LZ4 as "LZ4_RAW"; Parquet's "LZ4" is not standard LZ4
      return Compression::LZ4_RAW;
    case compression_type::NONE: return Compression::UNCOMPRESSED;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

struct aggregate_writer_metadata {
  aggregate_writer_metadata(host_span<partition_info const> partitions,
                            host_span<std::map<std::string, std::string> const> kv_md,
                            host_span<SchemaElement const> tbl_schema,
                            size_type num_columns,
                            statistics_freq stats_granularity,
                            std::string const arrow_schema_ipc_message)
    : version(1),
      schema(std::vector<SchemaElement>(tbl_schema.begin(), tbl_schema.end())),
      files(partitions.size())
  {
    for (size_t i = 0; i < partitions.size(); ++i) {
      this->files[i].num_rows = partitions[i].num_rows;
    }

    if (stats_granularity != statistics_freq::STATISTICS_NONE) {
      ColumnOrder default_order = {ColumnOrder::TYPE_ORDER};
      this->column_orders       = std::vector<ColumnOrder>(num_columns, default_order);
    }

    for (size_t p = 0; p < kv_md.size(); ++p) {
      std::transform(kv_md[p].begin(),
                     kv_md[p].end(),
                     std::back_inserter(this->files[p].key_value_metadata),
                     [](auto const& kv) { return KeyValue{kv.first, kv.second}; });
    }

    // Append arrow schema to the key-value metadata
    if (not arrow_schema_ipc_message.empty()) {
      std::for_each(this->files.begin(), this->files.end(), [&](auto& file) {
        file.key_value_metadata.emplace_back(KeyValue{ARROW_SCHEMA_KEY, arrow_schema_ipc_message});
      });
    }
  }

  aggregate_writer_metadata(aggregate_writer_metadata const&) = default;

  void update_files(host_span<partition_info const> partitions)
  {
    CUDF_EXPECTS(partitions.size() == this->files.size(),
                 "New partitions must be same size as previously passed number of partitions");
    for (size_t i = 0; i < partitions.size(); ++i) {
      this->files[i].num_rows += partitions[i].num_rows;
    }
  }

  FileMetaData get_metadata(size_t part)
  {
    CUDF_EXPECTS(part < files.size(), "Invalid part index queried");
    FileMetaData meta{};
    meta.version            = this->version;
    meta.schema             = this->schema;
    meta.num_rows           = this->files[part].num_rows;
    meta.row_groups         = this->files[part].row_groups;
    meta.key_value_metadata = this->files[part].key_value_metadata;
    meta.created_by         = "cudf version " CUDF_STRINGIFY(CUDF_VERSION);
    meta.column_orders      = this->column_orders;
    return meta;
  }

  void set_file_paths(host_span<std::string const> column_chunks_file_path)
  {
    for (size_t p = 0; p < this->files.size(); ++p) {
      auto& file            = this->files[p];
      auto const& file_path = column_chunks_file_path[p];
      for (auto& rowgroup : file.row_groups) {
        for (auto& col : rowgroup.columns) {
          col.file_path = file_path;
        }
      }
    }
  }

  FileMetaData get_merged_metadata()
  {
    FileMetaData merged_md;
    for (size_t p = 0; p < this->files.size(); ++p) {
      auto& file = this->files[p];
      if (p == 0) {
        merged_md = this->get_metadata(0);
      } else {
        merged_md.row_groups.insert(merged_md.row_groups.end(),
                                    std::make_move_iterator(file.row_groups.begin()),
                                    std::make_move_iterator(file.row_groups.end()));
        merged_md.num_rows += file.num_rows;
      }
    }
    return merged_md;
  }

  std::vector<size_t> num_row_groups_per_file()
  {
    std::vector<size_t> global_rowgroup_base;
    std::transform(this->files.begin(),
                   this->files.end(),
                   std::back_inserter(global_rowgroup_base),
                   [](auto const& part) { return part.row_groups.size(); });
    return global_rowgroup_base;
  }

  [[nodiscard]] bool schema_matches(std::vector<SchemaElement> const& schema) const
  {
    return this->schema == schema;
  }
  auto& file(size_t p) { return files[p]; }
  [[nodiscard]] size_t num_files() const { return files.size(); }

 private:
  int32_t version = 0;
  std::vector<SchemaElement> schema;
  struct per_file_metadata {
    int64_t num_rows = 0;
    std::vector<RowGroup> row_groups;
    std::vector<KeyValue> key_value_metadata;
    std::vector<OffsetIndex> offset_indexes;
    std::vector<cudf::detail::host_vector<uint8_t>> column_indexes;
  };
  std::vector<per_file_metadata> files;
  std::optional<std::vector<ColumnOrder>> column_orders = std::nullopt;
};

namespace {

/**
 * @brief Convert a mask of encodings to a vector.
 *
 * @param encodings Vector of `Encoding`s to populate
 * @param enc_mask Mask of encodings used
 */
void update_chunk_encodings(std::vector<Encoding>& encodings, uint32_t enc_mask)
{
  for (uint8_t enc = 0; enc < static_cast<uint8_t>(Encoding::NUM_ENCODINGS); enc++) {
    auto const enc_enum = static_cast<Encoding>(enc);
    if ((enc_mask & encoding_to_mask(enc_enum)) != 0) { encodings.push_back(enc_enum); }
  }
}

/**
 * @brief Update the encoding_stats field in the column chunk metadata.
 *
 * @param chunk_meta The `ColumnChunkMetaData` struct for the column chunk
 * @param ck The column chunk to summarize stats for
 * @param is_v2 True if V2 page headers are used
 */
void update_chunk_encoding_stats(ColumnChunkMetaData& chunk_meta,
                                 EncColumnChunk const& ck,
                                 bool is_v2)
{
  // don't set encoding stats if there are no pages
  if (ck.num_pages == 0) { return; }

  // NOTE: since cudf doesn't use mixed encodings for a chunk, we really only need to account
  // for the dictionary page (if there is one), and the encoding used for the data pages. We can
  // examine the chunk's encodings field to figure out the encodings without having to examine
  // the page data.
  auto const num_data_pages = static_cast<int32_t>(ck.num_data_pages());
  auto const data_page_type = is_v2 ? PageType::DATA_PAGE_V2 : PageType::DATA_PAGE;

  std::vector<PageEncodingStats> result;
  if (ck.use_dictionary) {
    // For dictionary encoding, if V1 then both data and dictionary use PLAIN_DICTIONARY. For V2
    // the dictionary uses PLAIN and the data RLE_DICTIONARY.
    auto const dict_enc = is_v2 ? Encoding::PLAIN : Encoding::PLAIN_DICTIONARY;
    auto const data_enc = is_v2 ? Encoding::RLE_DICTIONARY : Encoding::PLAIN_DICTIONARY;
    result.push_back({PageType::DICTIONARY_PAGE, dict_enc, 1});
    if (num_data_pages > 0) { result.push_back({data_page_type, data_enc, num_data_pages}); }
  } else {
    // No dictionary page, the pages are encoded with something other than RLE (unless it's a
    // boolean column).
    for (auto const enc : chunk_meta.encodings) {
      if (enc != Encoding::RLE) {
        result.push_back({data_page_type, enc, num_data_pages});
        break;
      }
    }
    // if result is empty and we're using V2 headers, then assume the data is RLE as well
    if (result.empty() and is_v2 and (ck.encodings & encoding_to_mask(Encoding::RLE)) != 0) {
      result.push_back({data_page_type, Encoding::RLE, num_data_pages});
    }
  }

  if (not result.empty()) { chunk_meta.encoding_stats = std::move(result); }
}

/**
 * @brief Compute size (in bytes) of the data stored in the given column.
 *
 * @param column The input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The data size of the input
 */
size_t column_size(column_view const& column, cuda::stream_ref stream)
{
  if (column.is_empty()) { return 0; }

  if (is_fixed_width(column.type())) {
    return size_of(column.type()) * column.size();
  } else if (column.type().id() == type_id::STRING) {
    auto const scol = strings_column_view(column);
    return cudf::strings::detail::get_offset_value(
             scol.offsets(), column.size() + column.offset(), stream) -
           cudf::strings::detail::get_offset_value(scol.offsets(), column.offset(), stream);
  } else if (column.type().id() == type_id::STRUCT) {
    auto const scol = structs_column_view(column);
    size_t ret      = 0;
    for (int i = 0; i < scol.num_children(); i++) {
      ret += column_size(scol.get_sliced_child(i, stream), stream);
    }
    return ret;
  } else if (column.type().id() == type_id::LIST) {
    auto const lcol = lists_column_view(column);
    return column_size(lcol.get_sliced_child(stream), stream);
  }

  CUDF_FAIL("Unexpected compound type");
}

/**
 * @brief Extends SchemaElement to add members required in constructing parquet_column_view
 *
 * Added members are:
 * 1. leaf_column: Pointer to leaf linked_column_view which points to the corresponding data stream
 *    of a leaf schema node. For non-leaf struct node, this is nullptr.
 * 2. stats_dtype: datatype for statistics calculation required for the data stream of a leaf node.
 * 3. ts_scale: scale to multiply or divide timestamp by in order to convert timestamp to parquet
 *    supported types
 * 4. requested_encoding: A user provided encoding to use for the column.
 */
struct schema_tree_node : public SchemaElement {
  cudf::detail::LinkedColPtr leaf_column;
  statistics_dtype stats_dtype;
  int32_t ts_scale;
  column_encoding requested_encoding;
  bool skip_compression;

  // TODO(fut): Think about making schema a class that holds a vector of schema_tree_nodes. The
  // function construct_schema_tree could be its constructor. It can have method to get the per
  // column nullability given a schema node index corresponding to a leaf schema. Much easier than
  // that is a method to get path in schema, given a leaf node
};

struct leaf_schema_fn {
  schema_tree_node& col_schema;
  cudf::detail::LinkedColPtr const& col;
  column_in_metadata const& col_meta;
  bool timestamp_is_int96;
  bool timestamp_is_utc;
  bool write_arrow_schema;

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, bool>)
  {
    col_schema.type        = Type::BOOLEAN;
    col_schema.stats_dtype = statistics_dtype::dtype_bool;
    // BOOLEAN needs no converted or logical type
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, int8_t>)
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::INT_8;
    col_schema.stats_dtype    = statistics_dtype::dtype_int8;
    col_schema.logical_type   = LogicalType{IntType{8, true}};
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, int16_t>)
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::INT_16;
    col_schema.stats_dtype    = statistics_dtype::dtype_int16;
    col_schema.logical_type   = LogicalType{IntType{16, true}};
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, int32_t>)
  {
    col_schema.type        = Type::INT32;
    col_schema.stats_dtype = statistics_dtype::dtype_int32;
    // INT32 needs no converted or logical type
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, int64_t>)
  {
    col_schema.type        = Type::INT64;
    col_schema.stats_dtype = statistics_dtype::dtype_int64;
    // INT64 needs no converted or logical type
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, uint8_t>)
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::UINT_8;
    col_schema.stats_dtype    = statistics_dtype::dtype_int8;
    col_schema.logical_type   = LogicalType{IntType{8, false}};
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, uint16_t>)
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::UINT_16;
    col_schema.stats_dtype    = statistics_dtype::dtype_int16;
    col_schema.logical_type   = LogicalType{IntType{16, false}};
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, uint32_t>)
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::UINT_32;
    col_schema.stats_dtype    = statistics_dtype::dtype_int32;
    col_schema.logical_type   = LogicalType{IntType{32, false}};
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, uint64_t>)
  {
    col_schema.type           = Type::INT64;
    col_schema.converted_type = ConvertedType::UINT_64;
    col_schema.stats_dtype    = statistics_dtype::dtype_int64;
    col_schema.logical_type   = LogicalType{IntType{64, false}};
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, float>)
  {
    col_schema.type        = Type::FLOAT;
    col_schema.stats_dtype = statistics_dtype::dtype_float32;
    // FLOAT needs no converted or logical type
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, double>)
  {
    col_schema.type        = Type::DOUBLE;
    col_schema.stats_dtype = statistics_dtype::dtype_float64;
    // DOUBLE needs no converted or logical type
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::string_view>)
  {
    col_schema.type = Type::BYTE_ARRAY;
    if (col_meta.is_enabled_output_as_binary()) {
      col_schema.stats_dtype = statistics_dtype::dtype_byte_array;
      // BYTE_ARRAY needs no converted or logical type
    } else {
      col_schema.converted_type = ConvertedType::UTF8;
      col_schema.stats_dtype    = statistics_dtype::dtype_string;
      col_schema.logical_type   = LogicalType{LogicalType::STRING};
    }
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::timestamp_D>)
  {
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::DATE;
    col_schema.stats_dtype    = statistics_dtype::dtype_int32;
    col_schema.logical_type   = LogicalType{LogicalType::DATE};
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::timestamp_s>)
  {
    col_schema.type        = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.stats_dtype = statistics_dtype::dtype_timestamp64;
    col_schema.ts_scale    = 1000;
    if (not timestamp_is_int96) {
      col_schema.converted_type = ConvertedType::TIMESTAMP_MILLIS;
      col_schema.logical_type   = LogicalType{TimestampType{timestamp_is_utc, TimeUnit::MILLIS}};
    }
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::timestamp_ms>)
  {
    col_schema.type        = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.stats_dtype = statistics_dtype::dtype_timestamp64;
    if (not timestamp_is_int96) {
      col_schema.converted_type = ConvertedType::TIMESTAMP_MILLIS;
      col_schema.logical_type   = LogicalType{TimestampType{timestamp_is_utc, TimeUnit::MILLIS}};
    }
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::timestamp_us>)
  {
    col_schema.type        = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.stats_dtype = statistics_dtype::dtype_timestamp64;
    if (not timestamp_is_int96) {
      col_schema.converted_type = ConvertedType::TIMESTAMP_MICROS;
      col_schema.logical_type   = LogicalType{TimestampType{timestamp_is_utc, TimeUnit::MICROS}};
    }
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::timestamp_ns>)
  {
    col_schema.type           = (timestamp_is_int96) ? Type::INT96 : Type::INT64;
    col_schema.converted_type = std::nullopt;
    col_schema.stats_dtype    = statistics_dtype::dtype_timestamp64;
    if (timestamp_is_int96) {
      col_schema.ts_scale = -1000;  // negative value indicates division by absolute value
    }
    // set logical type if it's not int96
    else {
      col_schema.logical_type = LogicalType{TimestampType{timestamp_is_utc, TimeUnit::NANOS}};
    }
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::duration_D>)
  {
    // duration_D is based on int32_t and not a valid arrow duration type so simply convert to
    // time32(ms).
    col_schema.type           = Type::INT32;
    col_schema.converted_type = ConvertedType::TIME_MILLIS;
    col_schema.stats_dtype    = statistics_dtype::dtype_int32;
    col_schema.ts_scale       = 24 * 60 * 60 * 1000;
    col_schema.logical_type   = LogicalType{TimeType{timestamp_is_utc, TimeUnit::MILLIS}};
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::duration_s>)
  {
    // If writing arrow schema, no logical type nor converted type is necessary
    if (write_arrow_schema) {
      col_schema.type        = Type::INT64;
      col_schema.stats_dtype = statistics_dtype::dtype_int64;
    } else {
      // Write as Time32 logical type otherwise. Parquet TIME_MILLIS annotates INT32
      col_schema.type           = Type::INT32;
      col_schema.stats_dtype    = statistics_dtype::dtype_int32;
      col_schema.converted_type = ConvertedType::TIME_MILLIS;
      col_schema.logical_type   = LogicalType{TimeType{timestamp_is_utc, TimeUnit::MILLIS}};
      col_schema.ts_scale       = 1000;
    }
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::duration_ms>)
  {
    // If writing arrow schema, no logical type nor converted type is necessary
    if (write_arrow_schema) {
      col_schema.type        = Type::INT64;
      col_schema.stats_dtype = statistics_dtype::dtype_int64;
    } else {
      // Write as Time32 logical type otherwise. Parquet TIME_MILLIS annotates INT32
      col_schema.type           = Type::INT32;
      col_schema.stats_dtype    = statistics_dtype::dtype_int32;
      col_schema.converted_type = ConvertedType::TIME_MILLIS;
      col_schema.logical_type   = LogicalType{TimeType{timestamp_is_utc, TimeUnit::MILLIS}};
    }
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::duration_us>)
  {
    col_schema.type        = Type::INT64;
    col_schema.stats_dtype = statistics_dtype::dtype_int64;
    // Only write as time64 logical type if not writing arrow schema
    if (not write_arrow_schema) {
      col_schema.converted_type = ConvertedType::TIME_MICROS;
      col_schema.logical_type   = LogicalType{TimeType{timestamp_is_utc, TimeUnit::MICROS}};
    }
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::duration_ns>)
  {
    col_schema.type        = Type::INT64;
    col_schema.stats_dtype = statistics_dtype::dtype_int64;
    // Only write as time64 logical type if not writing arrow schema
    if (not write_arrow_schema) {
      col_schema.logical_type = LogicalType{TimeType{timestamp_is_utc, TimeUnit::NANOS}};
    }
  }

  template <typename T>
  void operator()()
    requires(cudf::is_fixed_point<T>())
  {
    if constexpr (std::is_same_v<T, numeric::decimal32>) {
      col_schema.type              = Type::INT32;
      col_schema.stats_dtype       = statistics_dtype::dtype_int32;
      col_schema.decimal_precision = MAX_DECIMAL32_PRECISION;
      col_schema.logical_type      = LogicalType{DecimalType{0, MAX_DECIMAL32_PRECISION}};
    } else if constexpr (std::is_same_v<T, numeric::decimal64>) {
      col_schema.type              = Type::INT64;
      col_schema.stats_dtype       = statistics_dtype::dtype_decimal64;
      col_schema.decimal_precision = MAX_DECIMAL64_PRECISION;
      col_schema.logical_type      = LogicalType{DecimalType{0, MAX_DECIMAL64_PRECISION}};
    } else if constexpr (std::is_same_v<T, numeric::decimal128>) {
      col_schema.type              = Type::FIXED_LEN_BYTE_ARRAY;
      col_schema.type_length       = sizeof(__int128_t);
      col_schema.stats_dtype       = statistics_dtype::dtype_decimal128;
      col_schema.decimal_precision = MAX_DECIMAL128_PRECISION;
      col_schema.logical_type      = LogicalType{DecimalType{0, MAX_DECIMAL128_PRECISION}};
    } else {
      CUDF_FAIL("Unsupported fixed point type for parquet writer");
    }

    // Write logical and converted types, decimal scale and precision
    col_schema.converted_type = ConvertedType::DECIMAL;
    col_schema.decimal_scale = -col->type().scale();  // parquet and cudf disagree about scale signs
    col_schema.logical_type->decimal_type->scale = -col->type().scale();
    if (col_meta.is_decimal_precision_set()) {
      CUDF_EXPECTS(col_meta.get_decimal_precision() >= col_schema.decimal_scale,
                   "Precision must be equal to or greater than scale!");
      if (col_schema.type == Type::INT64 and col_meta.get_decimal_precision() < 10) {
        CUDF_LOG_WARN("Parquet writer: writing a decimal column with precision < 10 as int64");
      }
      col_schema.decimal_precision                     = col_meta.get_decimal_precision();
      col_schema.logical_type->decimal_type->precision = col_meta.get_decimal_precision();
    }
  }

  template <typename T>
  void operator()()
    requires(cudf::is_nested<T>())
  {
    CUDF_FAIL("This functor is only meant for physical data types");
  }

  template <typename T>
  void operator()()
    requires(cudf::is_dictionary<T>())
  {
    CUDF_FAIL("Dictionary columns are not supported for writing");
  }
};

/**
 * @brief Construct schema from input columns and per-column input options
 *
 * Recursively traverses through linked_columns and corresponding metadata to construct schema tree.
 * The resulting schema tree is stored in a vector in pre-order traversal order.
 */
std::vector<schema_tree_node> construct_parquet_schema_tree(
  cudf::detail::LinkedColVector const& linked_columns,
  table_input_metadata& metadata,
  single_write_mode write_mode,
  bool int96_timestamps,
  bool utc_timestamps,
  bool write_arrow_schema)
{
  std::vector<schema_tree_node> schema;
  schema_tree_node root{};
  root.type            = Type::UNDEFINED;
  root.repetition_type = FieldRepetitionType::UNSPECIFIED;
  root.name            = "schema";
  root.num_children    = linked_columns.size();
  root.parent_idx      = -1;  // root schema has no parent
  schema.push_back(std::move(root));

  std::function<void(cudf::detail::LinkedColPtr const&, column_in_metadata&, size_t)> add_schema =
    [&](cudf::detail::LinkedColPtr const& col, column_in_metadata& col_meta, size_t parent_idx) {
      bool const col_nullable = is_output_column_nullable(col, col_meta, write_mode);

      auto set_field_id = [](schema_tree_node& s, column_in_metadata const& col_meta) {
        // LIST element nodes still need their own field ids. Only the synthetic repeated "list"
        // group should remain without one.
        if (s.name != "list" and col_meta.is_parquet_field_id_set()) {
          s.field_id = col_meta.get_parquet_field_id();
        }
      };

      auto is_last_list_child = [](cudf::detail::LinkedColPtr col) {
        if (col->type().id() != type_id::LIST) { return false; }
        auto const child_col_type =
          col->children[lists_column_view::child_column_index]->type().id();
        return child_col_type == type_id::UINT8;
      };

      // only call this after col_schema.type has been set
      auto set_encoding = [&schema, parent_idx](schema_tree_node& s,
                                                column_in_metadata const& col_meta) {
        s.requested_encoding = column_encoding::USE_DEFAULT;

        if (s.name != "list" and col_meta.get_encoding() != column_encoding::USE_DEFAULT) {
          // do some validation
          switch (col_meta.get_encoding()) {
            case column_encoding::DELTA_BINARY_PACKED:
              if (s.type != Type::INT32 && s.type != Type::INT64) {
                CUDF_LOG_WARN(
                  "DELTA_BINARY_PACKED encoding is only supported for INT32 and INT64 columns; the "
                  "requested encoding will be ignored");
                return;
              }
              break;

            case column_encoding::DELTA_LENGTH_BYTE_ARRAY:
              if (s.type != Type::BYTE_ARRAY) {
                CUDF_LOG_WARN(
                  "DELTA_LENGTH_BYTE_ARRAY encoding is only supported for BYTE_ARRAY columns; the "
                  "requested encoding will be ignored");
                return;
              }
              // we don't yet allow encoding decimal128 with DELTA_LENGTH_BYTE_ARRAY (nor with
              // the BYTE_ARRAY physical type, but check anyway)
              if (s.converted_type.value_or(ConvertedType::UNKNOWN) == ConvertedType::DECIMAL) {
                CUDF_LOG_WARN(
                  "Decimal types cannot yet be encoded as DELTA_LENGTH_BYTE_ARRAY; the "
                  "requested encoding will be ignored");
                return;
              }
              break;

            case column_encoding::DELTA_BYTE_ARRAY:
              if (s.type != Type::BYTE_ARRAY && s.type != Type::FIXED_LEN_BYTE_ARRAY) {
                CUDF_LOG_WARN(
                  "DELTA_BYTE_ARRAY encoding is only supported for BYTE_ARRAY and "
                  "FIXED_LEN_BYTE_ARRAY columns; the requested encoding will be ignored");
                return;
              }
              // we don't yet allow encoding decimal128 with DELTA_BYTE_ARRAY
              if (s.converted_type.value_or(ConvertedType::UNKNOWN) == ConvertedType::DECIMAL) {
                CUDF_LOG_WARN(
                  "Decimal types cannot yet be encoded as DELTA_BYTE_ARRAY; the "
                  "requested encoding will be ignored");
                return;
              }
              break;

            case column_encoding::BYTE_STREAM_SPLIT:
              if (s.type == Type::BYTE_ARRAY) {
                CUDF_LOG_WARN(
                  "BYTE_STREAM_SPLIT encoding is only supported for fixed width columns; the "
                  "requested encoding will be ignored");
                return;
              }
              if (s.type == Type::INT96) {
                CUDF_LOG_WARN(
                  "BYTE_STREAM_SPLIT encoding is not supported for INT96 columns; the "
                  "requested encoding will be ignored");
                return;
              }
              break;

            // supported parquet encodings
            case column_encoding::PLAIN:
            case column_encoding::DICTIONARY: break;

            // all others
            default:
              CUDF_LOG_WARN(
                "Unsupported page encoding requested: %d; the requested encoding will be ignored",
                static_cast<int>(col_meta.get_encoding()));
              return;
          }

          // requested encoding seems to be ok, set it
          s.requested_encoding = col_meta.get_encoding();
        }
      };

      // There is a special case for a list<int8> column with one byte column child. This column can
      // have a special flag that indicates we write this out as binary instead of a list. This is a
      // more efficient storage mechanism for a single-depth list of bytes, but is a departure from
      // original cuIO behavior so it is locked behind the option. If the option is selected on a
      // column that isn't a single-depth list<int8> the code will throw.
      if (col_meta.is_enabled_output_as_binary() && is_last_list_child(col)) {
        CUDF_EXPECTS(col_meta.num_children() == 2 or col_meta.num_children() == 0,
                     "Binary column's corresponding metadata should have zero or two children");
        if (col_meta.num_children() > 0) {
          CUDF_EXPECTS(col->children[lists_column_view::child_column_index]->children.empty(),
                       "Binary column must not be nested");
        }

        schema_tree_node col_schema{};
        // test if this should be output as FIXED_LEN_BYTE_ARRAY
        if (col_meta.is_type_length_set()) {
          col_schema.type        = Type::FIXED_LEN_BYTE_ARRAY;
          col_schema.type_length = col_meta.get_type_length();
        } else {
          col_schema.type = Type::BYTE_ARRAY;
        }

        col_schema.converted_type = std::nullopt;
        col_schema.stats_dtype    = statistics_dtype::dtype_byte_array;
        col_schema.repetition_type =
          col_nullable ? FieldRepetitionType::OPTIONAL : FieldRepetitionType::REQUIRED;
        col_schema.name = (schema[parent_idx].name == "list") ? "element" : col_meta.get_name();
        col_schema.parent_idx  = parent_idx;
        col_schema.leaf_column = col;
        set_field_id(col_schema, col_meta);
        set_encoding(col_schema, col_meta);
        col_schema.output_as_byte_array = col_meta.is_enabled_output_as_binary();
        col_schema.skip_compression     = col_meta.is_enabled_skip_compression();
        schema.push_back(col_schema);
      } else if (col->type().id() == type_id::STRUCT) {
        // if struct, add current and recursively call for all children
        schema_tree_node struct_schema{};
        struct_schema.repetition_type =
          col_nullable ? FieldRepetitionType::OPTIONAL : FieldRepetitionType::REQUIRED;

        struct_schema.name = (schema[parent_idx].name == "list") ? "element" : col_meta.get_name();
        struct_schema.num_children = col->children.size();
        struct_schema.parent_idx   = parent_idx;
        set_field_id(struct_schema, col_meta);
        schema.push_back(std::move(struct_schema));

        auto struct_node_index = schema.size() - 1;
        // for (auto child_it = col->children.begin(); child_it < col->children.end(); child_it++) {
        //   add_schema(*child_it, struct_node_index);
        // }
        CUDF_EXPECTS(col->children.size() == static_cast<size_t>(col_meta.num_children()),
                     "Mismatch in number of child columns between input table and metadata");
        for (size_t i = 0; i < col->children.size(); ++i) {
          add_schema(col->children[i], col_meta.child(i), struct_node_index);
        }
      } else if (col->type().id() == type_id::LIST && !col_meta.is_map()) {
        // List schema is denoted by two levels for each nesting level and one final level for leaf.
        // The top level is the same name as the column name.
        // So e.g. List<List<int>> is denoted in the schema by
        // "col_name" : { "list" : { "element" : { "list" : { "element" } } } }

        schema_tree_node list_schema_1{};
        list_schema_1.converted_type = ConvertedType::LIST;
        list_schema_1.repetition_type =
          col_nullable ? FieldRepetitionType::OPTIONAL : FieldRepetitionType::REQUIRED;
        list_schema_1.name = (schema[parent_idx].name == "list") ? "element" : col_meta.get_name();
        list_schema_1.num_children = 1;
        list_schema_1.parent_idx   = parent_idx;
        set_field_id(list_schema_1, col_meta);
        schema.push_back(std::move(list_schema_1));

        schema_tree_node list_schema_2{};
        list_schema_2.repetition_type = FieldRepetitionType::REPEATED;
        list_schema_2.name            = "list";
        list_schema_2.num_children    = 1;
        list_schema_2.parent_idx      = schema.size() - 1;  // Parent is list_schema_1, last added.
        schema.push_back(std::move(list_schema_2));

        CUDF_EXPECTS(col_meta.num_children() == 2,
                     "List column's metadata should have exactly two children");

        add_schema(col->children[lists_column_view::child_column_index],
                   col_meta.child(lists_column_view::child_column_index),
                   schema.size() - 1);
      } else if (col->type().id() == type_id::LIST && col_meta.is_map()) {
        // Map schema is denoted by a list of struct
        // e.g. List<Struct<String,String>> will be
        // "col_name" : { "key_value" : { "key", "value" } }

        // verify the List child structure is a struct<left_child, right_child>
        column_view struct_col = *col->children[lists_column_view::child_column_index];
        CUDF_EXPECTS(struct_col.type().id() == type_id::STRUCT, "Map should be a List of struct");
        CUDF_EXPECTS(struct_col.num_children() == 2,
                     "Map should be a List of struct with two children only but found " +
                       std::to_string(struct_col.num_children()));

        schema_tree_node map_schema{};
        map_schema.converted_type = ConvertedType::MAP;
        map_schema.repetition_type =
          col_nullable ? FieldRepetitionType::OPTIONAL : FieldRepetitionType::REQUIRED;
        map_schema.name = col_meta.get_name();
        if (col_meta.is_parquet_field_id_set()) {
          map_schema.field_id = col_meta.get_parquet_field_id();
        }
        map_schema.num_children = 1;
        map_schema.parent_idx   = parent_idx;
        schema.push_back(std::move(map_schema));

        schema_tree_node repeat_group{};
        repeat_group.repetition_type = FieldRepetitionType::REPEATED;
        repeat_group.name            = "key_value";
        repeat_group.num_children    = 2;
        repeat_group.parent_idx      = schema.size() - 1;  // Parent is map_schema, last added.
        schema.push_back(std::move(repeat_group));

        CUDF_EXPECTS(col_meta.num_children() == 2,
                     "List column's metadata should have exactly two children");
        CUDF_EXPECTS(col_meta.child(lists_column_view::child_column_index).num_children() == 2,
                     "Map struct column should have exactly two children");
        // verify the col meta of children of the struct have name key and value
        auto& left_child_meta = col_meta.child(lists_column_view::child_column_index).child(0);
        left_child_meta.set_name("key");
        left_child_meta.set_nullability(false);

        auto& right_child_meta = col_meta.child(lists_column_view::child_column_index).child(1);
        right_child_meta.set_name("value");
        // check the repetition type of key is required i.e. the col should be non-nullable
        auto key_col = col->children[lists_column_view::child_column_index]->children[0];
        CUDF_EXPECTS(!is_output_column_nullable(key_col, left_child_meta, write_mode),
                     "key column cannot be nullable. For chunked writing, explicitly set the "
                     "nullability to false in metadata");
        // process key
        size_type struct_col_index = schema.size() - 1;
        add_schema(key_col, left_child_meta, struct_col_index);
        // process value
        add_schema(col->children[lists_column_view::child_column_index]->children[1],
                   right_child_meta,
                   struct_col_index);

      } else {
        // if leaf, add current
        if (col->type().id() == type_id::STRING) {
          if (col_meta.is_enabled_output_as_binary()) {
            CUDF_EXPECTS(col_meta.num_children() == 2 or col_meta.num_children() == 0,
                         "Binary column's corresponding metadata should have zero or two children");
          } else {
            CUDF_EXPECTS(col_meta.num_children() == 1 or col_meta.num_children() == 0,
                         "String column's corresponding metadata should have zero or one children");
          }
        } else {
          CUDF_EXPECTS(col_meta.num_children() == 0,
                       "Leaf column's corresponding metadata cannot have children");
        }

        schema_tree_node col_schema{};

        bool timestamp_is_int96 = int96_timestamps or col_meta.is_enabled_int96_timestamps();

        cudf::type_dispatcher(
          col->type(),
          leaf_schema_fn{
            col_schema, col, col_meta, timestamp_is_int96, utc_timestamps, write_arrow_schema});

        col_schema.repetition_type =
          col_nullable ? FieldRepetitionType::OPTIONAL : FieldRepetitionType::REQUIRED;
        col_schema.name = (schema[parent_idx].name == "list") ? "element" : col_meta.get_name();
        col_schema.parent_idx  = parent_idx;
        col_schema.leaf_column = col;
        set_field_id(col_schema, col_meta);
        set_encoding(col_schema, col_meta);
        col_schema.skip_compression = col_meta.is_enabled_skip_compression();
        schema.push_back(col_schema);
      }
    };

  CUDF_EXPECTS(metadata.column_metadata.size() == linked_columns.size(),
               "Mismatch in the number of columns and the corresponding metadata elements");
  // Add all linked_columns to schema using parent_idx = 0 (root)
  for (size_t i = 0; i < linked_columns.size(); ++i) {
    add_schema(linked_columns[i], metadata.column_metadata[i], 0);
  }

  return schema;
}

/**
 * @brief Class to store parquet specific information for one data stream.
 *
 * Contains information about a single data stream. In case of struct columns, a data stream is one
 * of the child leaf columns that contains data.
 * e.g. A column Struct<int, List<float>> contains 2 data streams:
 * - Struct<int>
 * - Struct<List<float>>
 *
 */
struct parquet_column_view {
  parquet_column_view(schema_tree_node const& schema_node,
                      std::vector<schema_tree_node> const& schema_tree,
                      cuda::stream_ref stream);

  [[nodiscard]] parquet_column_device_view get_device_view(cuda::stream_ref stream) const;

  [[nodiscard]] column_view cudf_column_view() const { return cudf_col; }
  [[nodiscard]] Type physical_type() const { return schema_node.type; }
  [[nodiscard]] ConvertedType converted_type() const
  {
    return schema_node.converted_type.value_or(ConvertedType::UNKNOWN);
  }

  // Checks to see if the given column has a fixed-width data type. This doesn't
  // check every value, so it assumes string and list columns are not fixed-width, even
  // if each value has the same size.
  [[nodiscard]] bool is_fixed_width() const
  {
    // lists and strings are not fixed width
    return max_rep_level() == 0 and physical_type() != Type::BYTE_ARRAY;
  }

  std::vector<std::string> const& get_path_in_schema() { return path_in_schema; }

  // LIST related member functions
  [[nodiscard]] uint8_t max_def_level() const noexcept { return _max_def_level; }
  [[nodiscard]] uint8_t max_rep_level() const noexcept { return _max_rep_level; }
  [[nodiscard]] bool is_list() const noexcept { return _is_list; }

 private:
  // Schema related members
  schema_tree_node schema_node;
  std::vector<std::string> path_in_schema;
  uint8_t _max_def_level = 0;
  uint8_t _max_rep_level = 0;
  rmm::device_uvector<uint8_t> _d_nullability;

  column_view cudf_col;

  // List-related members
  bool _is_list;
  rmm::device_uvector<size_type>
    _dremel_offsets;  ///< For each row, the absolute offset into the repetition and definition
                      ///< level vectors. O(num rows)
  rmm::device_uvector<uint8_t> _rep_level;
  rmm::device_uvector<uint8_t> _def_level;
  std::vector<uint8_t> _nullability;
  size_type _data_count = 0;
};

parquet_column_view::parquet_column_view(schema_tree_node const& schema_node,
                                         std::vector<schema_tree_node> const& schema_tree,
                                         cuda::stream_ref stream)
  : schema_node(schema_node),
    _d_nullability(0, stream),
    _dremel_offsets(0, stream),
    _rep_level(0, stream),
    _def_level(0, stream)
{
  // Construct single inheritance column_view from linked_column_view
  auto curr_col                           = schema_node.leaf_column.get();
  column_view single_inheritance_cudf_col = *curr_col;
  while (curr_col->parent) {
    auto const& parent = *curr_col->parent;

    // For list columns, we still need to retain the offset child column.
    auto children =
      (parent.type().id() == type_id::LIST)
        ? std::vector<column_view>{*parent.children[lists_column_view::offsets_column_index],
                                   single_inheritance_cudf_col}
        : std::vector<column_view>{single_inheritance_cudf_col};

    single_inheritance_cudf_col = column_view(parent.type(),
                                              parent.size(),
                                              parent.head(),
                                              parent.null_mask(),
                                              parent.null_count(),
                                              parent.offset(),
                                              children);

    curr_col = curr_col->parent;
  }
  cudf_col = single_inheritance_cudf_col;

  // Construct path_in_schema by travelling up in the schema_tree
  std::vector<std::string> path;
  auto curr_schema_node = schema_node;
  do {
    path.push_back(curr_schema_node.name);
    if (curr_schema_node.parent_idx != -1) {
      curr_schema_node = schema_tree[curr_schema_node.parent_idx];
    }
  } while (curr_schema_node.parent_idx != -1);
  path_in_schema = std::vector<std::string>(path.crbegin(), path.crend());

  // Calculate max definition level by counting the number of levels that are optional (nullable)
  // and max repetition level by counting the number of REPEATED levels in this column's hierarchy
  uint16_t max_def_level = 0;
  uint16_t max_rep_level = 0;
  curr_schema_node       = schema_node;
  while (curr_schema_node.parent_idx != -1) {
    if (curr_schema_node.repetition_type == FieldRepetitionType::REPEATED or
        curr_schema_node.repetition_type == FieldRepetitionType::OPTIONAL) {
      ++max_def_level;
    }
    if (curr_schema_node.repetition_type == FieldRepetitionType::REPEATED) { ++max_rep_level; }
    curr_schema_node = schema_tree[curr_schema_node.parent_idx];
  }
  CUDF_EXPECTS(max_def_level < 256, "Definition levels above 255 are not supported");
  CUDF_EXPECTS(max_rep_level < 256, "Definition levels above 255 are not supported");

  _max_def_level = max_def_level;
  _max_rep_level = max_rep_level;

  // Construct nullability vector using repetition_type from schema.
  std::vector<uint8_t> r_nullability;
  curr_schema_node = schema_node;
  while (curr_schema_node.parent_idx != -1) {
    if (not curr_schema_node.is_stub()) {
      r_nullability.push_back(curr_schema_node.repetition_type == FieldRepetitionType::OPTIONAL);
    }
    curr_schema_node = schema_tree[curr_schema_node.parent_idx];
  }
  _nullability = std::vector<uint8_t>(r_nullability.crbegin(), r_nullability.crend());
  // TODO(cp): Explore doing this for all columns in a single go outside this ctor. Maybe using
  // hostdevice_vector. Currently this involves a separate async H2D copy for each column.
  _d_nullability = cudf::detail::make_device_uvector_async(
    _nullability, stream, cudf::get_current_device_resource_ref());

  _is_list = (_max_rep_level > 0);

  if (cudf_col.is_empty()) { return; }

  if (_is_list) {
    // Top level column's offsets are not applied to all children. Get the effective offset and
    // size of the leaf column
    // Calculate row offset into dremel data (repetition/definition values) and the respective
    // definition and repetition levels
    cudf::detail::dremel_data dremel =
      get_dremel_data(cudf_col, _nullability, schema_node.output_as_byte_array, stream);
    _dremel_offsets = std::move(dremel.dremel_offsets);
    _rep_level      = std::move(dremel.rep_level);
    _def_level      = std::move(dremel.def_level);
    _data_count     = dremel.leaf_data_size;  // Needed for knowing what size dictionary to allocate

    stream.sync();
  } else {
    // For non-list struct, the size of the root column is the same as the size of the leaf column
    _data_count = cudf_col.size();
  }
}

parquet_column_device_view parquet_column_view::get_device_view(cuda::stream_ref) const
{
  auto desc        = parquet_column_device_view{};  // Zero out all fields
  desc.stats_dtype = schema_node.stats_dtype;
  desc.ts_scale    = schema_node.ts_scale;
  desc.type_length = schema_node.type_length;

  if (is_list()) {
    desc.level_offsets = _dremel_offsets.data();
    desc.rep_values    = _rep_level.data();
    desc.def_values    = _def_level.data();
  }
  desc.num_rows             = cudf_col.size();
  desc.physical_type        = physical_type();
  desc.converted_type       = converted_type();
  desc.output_as_byte_array = schema_node.output_as_byte_array;

  desc.level_bits = CompactProtocolReader::NumRequiredBits(max_rep_level()) << 4 |
                    CompactProtocolReader::NumRequiredBits(max_def_level());
  desc.nullability        = _d_nullability.data();
  desc.max_def_level      = _max_def_level;
  desc.max_rep_level      = _max_rep_level;
  desc.requested_encoding = schema_node.requested_encoding;
  desc.skip_compression   = schema_node.skip_compression;
  return desc;
}

/**
 * @brief Gather row group fragments
 *
 * This calculates fragments to be used in determining row group boundaries.
 *
 * @param frag Destination row group fragments
 * @param col_desc column description array
 * @param partitions Information about partitioning of table
 * @param part_frag_offset A Partition's offset into fragment array
 * @param fragment_size Number of rows per fragment
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void init_row_group_fragments(cudf::detail::hostdevice_2dvector<PageFragment>& frag,
                              device_span<parquet_column_device_view const> col_desc,
                              host_span<partition_info const> partitions,
                              device_span<int const> part_frag_offset,
                              uint32_t fragment_size,
                              cuda::stream_ref stream)
{
  auto d_partitions = cudf::detail::make_device_uvector_async(
    partitions, stream, cudf::get_current_device_resource_ref());
  InitRowGroupFragments(frag, col_desc, d_partitions, part_frag_offset, fragment_size, stream);
  frag.device_to_host(stream);
}

/**
 * @brief Recalculate page fragments
 *
 * This calculates fragments to be used to determine page boundaries within
 * column chunks.
 *
 * @param frag Destination page fragments
 * @param frag_sizes Array of fragment sizes for each column
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void calculate_page_fragments(device_span<PageFragment> frag,
                              host_span<size_type const> frag_sizes,
                              cuda::stream_ref stream)
{
  auto d_frag_sz = cudf::detail::make_device_uvector_async(
    frag_sizes, stream, cudf::get_current_device_resource_ref());
  CalculatePageFragments(frag, d_frag_sz, stream);
}

/**
 * @brief Gather per-fragment statistics
 *
 * @param frag_stats output statistics
 * @param frags Input page fragments
 * @param int96_timestamps Whether timestamps are written as INT96
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void gather_fragment_statistics(device_span<statistics_chunk> frag_stats,
                                device_span<PageFragment const> frags,
                                bool int96_timestamps,
                                cuda::stream_ref stream)
{
  rmm::device_uvector<statistics_group> frag_stats_group(frag_stats.size(), stream);

  InitFragmentStatistics(frag_stats_group, frags, stream);
  detail::calculate_group_statistics<detail::io_file_format::PARQUET>(
    frag_stats.data(), frag_stats_group.data(), frag_stats.size(), stream, int96_timestamps);
  stream.sync();
}

auto init_page_sizes(hostdevice_2dvector<EncColumnChunk>& chunks,
                     device_span<parquet_column_device_view const> col_desc,
                     uint32_t num_columns,
                     size_t max_page_size_bytes,
                     size_type max_page_size_rows,
                     bool write_v2_headers,
                     bool write_page_stats,
                     compression_type compression,
                     cuda::stream_ref stream)
{
  if (chunks.is_empty()) { return cudf::detail::hostdevice_vector<size_type>{}; }

  kernel_error error_code(stream);

  chunks.host_to_device_async(stream);
  // Calculate number of pages and store in respective chunks
  InitEncoderPages(chunks,
                   {},
                   {},
                   {},
                   col_desc,
                   num_columns,
                   max_page_size_bytes,
                   max_page_size_rows,
                   compress_required_chunk_alignment(compression),
                   write_v2_headers,
                   write_page_stats,
                   nullptr,
                   nullptr,
                   error_code.data(),
                   stream);
  if (auto const error = error_code.value_sync(stream); error != 0) {
    CUDF_FAIL(
      std::format("Parquet encoding failed with code(s) {}", kernel_error::to_string(error)));
  }

  chunks.device_to_host(stream);

  auto num_pages = size_type{0};
  for (auto& chunk : chunks.host_view().flat_view()) {
    chunk.first_page = num_pages;
    num_pages += chunk.num_pages;
  }
  chunks.host_to_device_async(stream);

  // Now that we know the number of pages, allocate an array to hold per page size and get it
  // populated
  cudf::detail::hostdevice_vector<size_type> page_sizes(num_pages, stream);
  InitEncoderPages(chunks,
                   {},
                   page_sizes,
                   {},
                   col_desc,
                   num_columns,
                   max_page_size_bytes,
                   max_page_size_rows,
                   compress_required_chunk_alignment(compression),
                   write_v2_headers,
                   write_page_stats,
                   nullptr,
                   nullptr,
                   error_code.data(),
                   stream);
  if (auto const error = error_code.value_sync(stream); error != 0) {
    CUDF_FAIL(
      std::format("Parquet encoding failed with code(s) {}", kernel_error::to_string(error)));
  }

  page_sizes.device_to_host(stream);

  // Get per-page max compressed size
  cudf::detail::hostdevice_vector<size_type> comp_page_sizes(num_pages, stream);
  std::transform(
    page_sizes.begin(), page_sizes.end(), comp_page_sizes.begin(), [compression](auto page_size) {
      return max_compressed_size(compression, page_size);
    });
  comp_page_sizes.host_to_device_async(stream);

  // Use per-page max compressed size to calculate chunk.compressed_size
  InitEncoderPages(chunks,
                   {},
                   {},
                   comp_page_sizes,
                   col_desc,
                   num_columns,
                   max_page_size_bytes,
                   max_page_size_rows,
                   compress_required_chunk_alignment(compression),
                   write_v2_headers,
                   write_page_stats,
                   nullptr,
                   nullptr,
                   error_code.data(),
                   stream);
  chunks.device_to_host(stream);

  if (auto const error = error_code.value_sync(stream); error != 0) {
    CUDF_FAIL(
      std::format("Parquet encoding failed with code(s) {}", kernel_error::to_string(error)));
  }

  return comp_page_sizes;
}

size_t max_page_bytes(compression_type compression, size_t max_page_size_bytes)
{
  if (compression == compression_type::NONE) { return max_page_size_bytes; }

  auto const comp_limit = compress_max_allowed_chunk_size(compression);

  auto max_size = std::min(comp_limit.value_or(max_page_size_bytes), max_page_size_bytes);
  // page size must fit in a 32-bit signed integer
  return std::min<size_t>(max_size, std::numeric_limits<int32_t>::max());
}

std::pair<std::vector<rmm::device_uvector<size_type>>, std::vector<rmm::device_uvector<size_type>>>
build_chunk_dictionaries(hostdevice_2dvector<EncColumnChunk>& chunks,
                         host_span<parquet_column_device_view const> col_desc,
                         device_2dspan<PageFragment> frags,
                         compression_type compression,
                         dictionary_policy dict_policy,
                         size_t max_dict_size,
                         cuda::stream_ref stream)
{
  // At this point, we know all chunks and their sizes. We want to allocate dictionaries for each
  // chunk that can have dictionary

  auto h_chunks = chunks.host_view().flat_view();

  std::vector<rmm::device_uvector<size_type>> dict_data;
  std::vector<rmm::device_uvector<size_type>> dict_index;

  if (h_chunks.empty()) { return std::pair(std::move(dict_data), std::move(dict_index)); }

  if (dict_policy == dictionary_policy::NEVER) {
    std::for_each(
      h_chunks.begin(), h_chunks.end(), [](auto& chunk) { chunk.use_dictionary = false; });
    chunks.host_to_device_async(stream);
    return std::pair(std::move(dict_data), std::move(dict_index));
  }

  // The static map only needs slots for as many entries as maximum parquet dictionary size.
  auto const max_dict_entries = [&]() -> size_type {
    // Dictionary elements are at least 4 bytes, so cap the entry count at max_dict_data_size / 4.
    if (dict_policy == dictionary_policy::ADAPTIVE) {
      auto const max_dict_data_size = max_page_bytes(compression, max_dict_size);
      return static_cast<size_type>(std::clamp<size_t>(
        max_dict_data_size / sizeof(int32_t), 1, static_cast<size_t>(MAX_DICT_SIZE)));
    }
    return MAX_DICT_SIZE;
  }();

  // Storage slots needed to hold `entries` keys at the target occupancy
  auto const map_extent = [](size_t entries) {
    return static_cast<size_type>(cuco::make_valid_extent<map_cg_size, bucket_size>(
      static_cast<size_type>(occupancy_factor * entries)));
  };

  // Variable to keep track of the current total map storage size
  size_t total_map_storage_size = 0;
  // Populate dict offsets and sizes for each chunk that need to build a dictionary.
  std::for_each(h_chunks.begin(), h_chunks.end(), [&](auto& chunk) {
    auto const& chunk_col_desc = col_desc[chunk.col_desc_id];
    auto const is_requested_non_dict =
      chunk_col_desc.requested_encoding != column_encoding::USE_DEFAULT &&
      chunk_col_desc.requested_encoding != column_encoding::DICTIONARY;
    auto const is_type_non_dict =
      chunk_col_desc.physical_type == Type::BOOLEAN || chunk_col_desc.output_as_byte_array;

    if (is_type_non_dict || is_requested_non_dict) {
      chunk.use_dictionary = false;
    } else {
      chunk.use_dictionary   = true;
      chunk.dict_entry_limit = std::min(chunk.num_values, max_dict_entries);
      // Fragment blocks can insert one extra iteration before observing the limit so overshoot by
      // that amount. In case a thread block inserts two extra iterations, a full cuco static map
      // still safely rejects.
      auto const overshoot = static_cast<size_t>(chunk.num_fragments) * dict_encode_block_size;
      auto const slack =
        static_cast<size_t>(map_extent(chunk.dict_entry_limit) - chunk.dict_entry_limit);
      auto const map_entries = std::min<size_t>(
        chunk.num_values,
        static_cast<size_t>(chunk.dict_entry_limit) + (overshoot > slack ? overshoot - slack : 0));
      chunk.dict_map_size   = map_extent(map_entries);
      chunk.dict_map_offset = total_map_storage_size;
      total_map_storage_size += chunk.dict_map_size;
    }
  });

  // No chunk needs to create a dictionary, exit early
  if (total_map_storage_size == 0) { return {std::move(dict_data), std::move(dict_index)}; }

  // Create a single bulk storage used by all sub-dictionaries
  auto map_storage =
    storage_type{total_map_storage_size, rmm::mr::polymorphic_allocator<char>{}, stream.get()};
  // Create a span of non-const map_storage as map_storage_ref takes in a non-const pointer.
  device_span<slot_type> const map_storage_data{map_storage.data(), total_map_storage_size};

  // Synchronize
  chunks.host_to_device_async(stream);
  // Initialize storage with the given sentinel
  map_storage.initialize_async({KEY_SENTINEL, VALUE_SENTINEL}, {stream.get()});
  // Populate the hash map for each chunk
  populate_chunk_hash_maps(map_storage_data, frags, stream);
  // Synchronize again
  chunks.device_to_host(stream);

  // Make decision about which chunks have dictionary
  bool cannot_honor_request = false;
  for (auto& ck : h_chunks) {
    if (not ck.use_dictionary) { continue; }
    std::tie(ck.use_dictionary, ck.dict_rle_bits) = [&]() -> std::pair<bool, uint8_t> {
      // `populate_chunk_hash_maps` stops inserting once the entry limit is crossed, so a chunk that
      // ended up above the limit has an incomplete dictionary and cannot use one. Such a chunk
      // would have been rejected by the size checks below anyway, but its `uniq_data_size` is
      // truncated and can no longer be trusted to make that call.
      if (ck.num_dict_entries > ck.dict_entry_limit) { return {false, 0}; }

      // calculate size of chunk if dictionary is used

      // If we have N unique values then the idx for the last value is N - 1 and nbits is the number
      // of bits required to encode indices into the dictionary
      auto max_dict_index = (ck.num_dict_entries > 0) ? ck.num_dict_entries - 1 : 0;
      auto nbits          = std::max(CompactProtocolReader::NumRequiredBits(max_dict_index), 1);

      // We don't use dictionary if the indices are > MAX_DICT_BITS bits because that's the maximum
      // bitpacking bitsize we efficiently support
      if (nbits > MAX_DICT_BITS) { return {false, 0}; }

      auto rle_byte_size =
        util::div_rounding_up_safe<size_t>(static_cast<size_t>(ck.num_values) * nbits, 8);
      auto dict_enc_size = ck.uniq_data_size + rle_byte_size;
      if (ck.plain_data_size <= dict_enc_size) { return {false, 0}; }

      // don't use dictionary if it gets too large for the given compression codec
      if (dict_policy == dictionary_policy::ADAPTIVE) {
        auto const unique_size = static_cast<size_t>(ck.uniq_data_size);
        if (unique_size > max_page_bytes(compression, max_dict_size)) { return {false, 0}; }
      }

      return {true, nbits};
    }();
    // If dictionary encoding was requested, but it cannot be used, then print a warning. It will
    // actually be disabled in gpuInitPages.
    if (not ck.use_dictionary) {
      auto const& chunk_col_desc = col_desc[ck.col_desc_id];
      if (chunk_col_desc.requested_encoding == column_encoding::DICTIONARY) {
        cannot_honor_request = true;
      }
    }
  }

  // warn if we have to ignore requested encoding
  if (cannot_honor_request) {
    CUDF_LOG_WARN("DICTIONARY encoding was requested, but resource constraints prevent its use");
  }

  // TODO: (enh) Deallocate hash map storage for chunks that don't use dict and clear pointers.

  dict_data.reserve(h_chunks.size());
  dict_index.reserve(h_chunks.size());
  for (auto& chunk : h_chunks) {
    if (not chunk.use_dictionary) { continue; }

    size_t dict_data_size     = std::min(MAX_DICT_SIZE, chunk.dict_map_size);
    auto& inserted_dict_data  = dict_data.emplace_back(dict_data_size, stream);
    auto& inserted_dict_index = dict_index.emplace_back(chunk.num_values, stream);
    chunk.dict_data           = inserted_dict_data.data();
    chunk.dict_index          = inserted_dict_index.data();
  }
  chunks.host_to_device_async(stream);
  collect_map_entries(map_storage_data, chunks.device_view().flat_view(), frags, stream);
  get_dictionary_indices(map_storage_data, frags, stream);

  return std::pair(std::move(dict_data), std::move(dict_index));
}

/**
 * @brief Initialize encoder pages.
 *
 * @param chunks Column chunk array
 * @param col_desc Column description array
 * @param pages Encoder pages array
 * @param comp_page_sizes Per-page max compressed size
 * @param page_stats Page statistics array
 * @param frag_stats Fragment statistics array
 * @param num_columns Total number of columns
 * @param num_pages Total number of pages
 * @param num_stats_bfr Number of statistics buffers
 * @param alignment Page alignment
 * @param max_page_size_bytes Maximum uncompressed page size, in bytes
 * @param max_page_size_rows Maximum page size, in rows
 * @param write_v2_headers True if version 2 page headers are to be written
 * @param write_page_stats True if statistics are to be written to the data page headers
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void init_encoder_pages(hostdevice_2dvector<EncColumnChunk>& chunks,
                        device_span<parquet_column_device_view const> col_desc,
                        device_span<EncPage> pages,
                        device_span<size_type const> comp_page_sizes,
                        statistics_chunk* page_stats,
                        statistics_chunk* frag_stats,
                        uint32_t num_columns,
                        uint32_t num_pages,
                        uint32_t num_stats_bfr,
                        size_t alignment,
                        size_t max_page_size_bytes,
                        size_type max_page_size_rows,
                        bool write_v2_headers,
                        bool write_page_stats,
                        cuda::stream_ref stream)
{
  rmm::device_uvector<statistics_merge_group> page_stats_mrg(num_stats_bfr, stream);
  kernel_error error_code(stream);
  chunks.host_to_device_async(stream);
  InitEncoderPages(chunks,
                   pages,
                   {},
                   comp_page_sizes,
                   col_desc,
                   num_columns,
                   max_page_size_bytes,
                   max_page_size_rows,
                   alignment,
                   write_v2_headers,
                   write_page_stats,
                   (num_stats_bfr) ? page_stats_mrg.data() : nullptr,
                   (num_stats_bfr > num_pages) ? page_stats_mrg.data() + num_pages : nullptr,
                   error_code.data(),
                   stream);

  if (auto const error = error_code.value_sync(stream); error != 0) {
    CUDF_FAIL(
      std::format("Parquet encoding failed with code(s) {}", kernel_error::to_string(error)));
  }

  if (num_stats_bfr > 0) {
    detail::merge_group_statistics<detail::io_file_format::PARQUET>(
      page_stats, frag_stats, page_stats_mrg.data(), num_pages, stream);
    if (num_stats_bfr > num_pages) {
      detail::merge_group_statistics<detail::io_file_format::PARQUET>(
        page_stats + num_pages,
        page_stats,
        page_stats_mrg.data() + num_pages,
        num_stats_bfr - num_pages,
        stream);
    }
  }
  stream.sync();
}

/**
 * @brief Encode pages.
 *
 * @throws rmm::bad_alloc if there is insufficient space for temporary buffers
 *
 * @param chunks column chunk array
 * @param pages encoder pages array
 * @param page_stats optional page-level statistics (nullptr if none)
 * @param chunk_stats optional chunk-level statistics (nullptr if none)
 * @param column_stats optional page-level statistics for column index (nullptr if none)
 * @param comp_stats optional compression statistics (nullopt if none)
 * @param compression compression format
 * @param column_index_truncate_length maximum length of min or max values in column index, in
 * bytes
 * @param write_v2_headers True if V2 page headers should be written
 * @param page_level_compression True if V2 pages can make per-page compression decisions
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void encode_pages(hostdevice_2dvector<EncColumnChunk>& chunks,
                  device_span<EncPage> pages,
                  statistics_chunk const* page_stats,
                  statistics_chunk const* chunk_stats,
                  statistics_chunk const* column_stats,
                  std::optional<writer_compression_statistics>& comp_stats,
                  compression_type compression,
                  int32_t column_index_truncate_length,
                  bool write_v2_headers,
                  bool page_level_compression,
                  cuda::stream_ref stream)
{
  auto const num_pages = pages.size();
  auto pages_stats     = (page_stats != nullptr)
                           ? device_span<statistics_chunk const>(page_stats, num_pages)
                           : device_span<statistics_chunk const>();

  uint32_t max_comp_pages = (compression != compression_type::NONE) ? num_pages : 0;

  rmm::device_uvector<device_span<uint8_t const>> comp_in(max_comp_pages, stream);
  rmm::device_uvector<device_span<uint8_t>> comp_out(max_comp_pages, stream);
  rmm::device_uvector<codec_exec_result> comp_res(max_comp_pages, stream);
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               comp_res.begin(),
               comp_res.end(),
               codec_exec_result{0, codec_status::FAILURE});

  EncodePages(pages, write_v2_headers, comp_in, comp_out, comp_res, stream);
  compress(compression, comp_in, comp_out, comp_res, stream);

  // TBD: Not clear if the official spec actually allows dynamically turning off compression at the
  // chunk-level

  auto d_chunks = chunks.device_view();
  decide_compression(d_chunks.flat_view(), page_level_compression, stream);
  EncodePageHeaders(pages, comp_res, pages_stats, chunk_stats, stream);
  GatherPages(d_chunks.flat_view(), stream);

  // By now, the var_bytes has been calculated in InitPages, and the histograms in EncodePages.
  // EncodeColumnIndexes can encode the histograms in the ColumnIndex, and also sum up var_bytes
  // and the histograms for inclusion in the chunk's SizeStats.
  if (column_stats != nullptr) {
    EncodeColumnIndexes(
      d_chunks.flat_view(), {column_stats, pages.size()}, column_index_truncate_length, stream);
  }

  chunks.device_to_host_async(stream);

  if (comp_stats.has_value()) {
    comp_stats.value() += collect_compression_statistics(comp_in, comp_res, stream);
  }
  stream.sync();
}

/**
 * @brief Function to calculate the memory needed to encode the column index of the given
 * column chunk.
 *
 * @param ck pointer to column chunk
 * @param col `parquet_column_device_view` for the column
 * @param column_index_truncate_length maximum length of min or max values in column index, in bytes
 * @return Computed buffer size needed to encode the column index
 */
size_t column_index_buffer_size(EncColumnChunk* ck,
                                parquet_column_device_view const& col,
                                int32_t column_index_truncate_length)
{
  // encoding the column index for a given chunk requires:
  //   each list (4 of them) requires 6 bytes of overhead
  //     (1 byte field header, 1 byte type, 4 bytes length)
  //   1 byte overhead for boundary_order
  //   1 byte overhead for termination
  //   sizeof(char) for boundary_order
  //   sizeof(bool) * num_pages for null_pages
  //   (ck_max_stats_len + 4) * num_pages * 2 for min/max values
  //     (each binary requires 4 bytes length + ck_max_stats_len)
  //   sizeof(int64_t) * num_pages for null_counts
  //
  // so 26 bytes overhead + sizeof(char) +
  //    (sizeof(bool) + sizeof(int64_t) + 2 * (4 + ck_max_stats_len)) * num_pages
  //
  // we already have ck->ck_stat_size = 48 + 2 * ck_max_stats_len
  // all of the overhead and non-stats data can fit in under 48 bytes
  //
  // so we can simply use ck_stat_size * num_pages
  //
  // add on some extra padding at the end (plus extra 7 bytes of alignment padding)
  // for scratch space to do stats truncation.

  // additional storage needed for SizeStatistics
  // don't need stats for dictionary pages
  auto const num_pages = ck->num_data_pages();

  // only need variable length size info for BYTE_ARRAY
  // 1 byte for marker, 1 byte vec type, 4 bytes length, 5 bytes per page for values
  // (5 bytes is needed because the varint encoder only encodes 7 bits per byte)
  auto const var_bytes_size = col.physical_type == Type::BYTE_ARRAY ? 6 + 5 * num_pages : 0;

  // for the histograms, need 1 byte for marker, 1 byte vec type, 4 bytes length,
  // (max_level + 1) * 5 bytes per page
  auto const has_def       = col.max_def_level > DEF_LVL_HIST_CUTOFF;
  auto const has_rep       = col.max_def_level > REP_LVL_HIST_CUTOFF;
  auto const def_hist_size = has_def ? 6 + 5 * num_pages * (col.max_def_level + 1) : 0;
  auto const rep_hist_size = has_rep ? 6 + 5 * num_pages * (col.max_rep_level + 1) : 0;

  // total size of SizeStruct is 1 byte marker, 1 byte end-of-struct, plus sizes for components
  auto const size_struct_size = 2 + def_hist_size + rep_hist_size + var_bytes_size;

  // calculating this per-chunk because the sizes can be wildly different.
  constexpr size_t padding = 7;
  return ck->ck_stat_size * num_pages + column_index_truncate_length + padding + size_struct_size;
}

/**
 * @brief Perform the processing steps needed to convert the input table into the output Parquet
 * data for writing, such as compression and encoding.
 *
 * @param[in,out] table_meta The table metadata
 * @param input The input table
 * @param partitions Partitions to divide the table into; if specified then must be same size as
 * number of sinks
 * @param kv_meta Optional user metadata
 * @param curr_agg_meta The current aggregate writer metadata
 * @param max_page_fragment_size_opt Optional maximum number of rows in a page fragment
 * @param max_row_group_size Maximum row group size, in bytes
 * @param max_page_size_bytes Maximum uncompressed page size, in bytes
 * @param max_row_group_rows Maximum row group size, in rows
 * @param max_page_size_rows Maximum page size, in rows
 * @param column_index_truncate_length maximum length of min or max values in column index, in bytes
 * @param stats_granularity Level of statistics requested in output file
 * @param compression Compression format
 * @param collect_compression_statistics Flag to indicate if compression statistics should be
 * collected
 * @param dict_policy Policy for dictionary use
 * @param max_dictionary_size Maximum dictionary size, in bytes
 * @param write_mode Flag to indicate that we are guaranteeing a single table write
 * @param int96_timestamps Flag to indicate if timestamps will be written as INT96
 * @param utc_timestamps Flag to indicate if timestamps are UTC
 * @param write_v2_headers True if V2 page headers are to be written
 * @param page_level_compression True if V2 pages can make per-page compression decisions
 * @param write_arrow_schema True if Arrow schema should be written to the file
 * @param out_sink Sink for checking if device write is supported, should not be used to write any
 *        data in this function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A tuple of the intermediate results containing the processed data
 */
auto convert_table_to_parquet_data(table_input_metadata& table_meta,
                                   table_view const& input,
                                   host_span<partition_info const> partitions,
                                   host_span<std::map<std::string, std::string> const> kv_meta,
                                   std::unique_ptr<aggregate_writer_metadata> const& curr_agg_meta,
                                   std::optional<size_type> max_page_fragment_size_opt,
                                   size_t max_row_group_size,
                                   size_t max_page_size_bytes,
                                   size_type max_row_group_rows,
                                   size_type max_page_size_rows,
                                   int32_t column_index_truncate_length,
                                   statistics_freq stats_granularity,
                                   compression_type compression,
                                   bool collect_compression_statistics,
                                   dictionary_policy dict_policy,
                                   size_t max_dictionary_size,
                                   single_write_mode write_mode,
                                   bool int96_timestamps,
                                   bool utc_timestamps,
                                   bool write_v2_headers,
                                   bool page_level_compression,
                                   bool write_arrow_schema,
                                   host_span<std::unique_ptr<data_sink> const> out_sink,
                                   cuda::stream_ref stream)
{
  // initialize LinkedColVector
  auto vec = table_to_linked_columns(input);

  auto schema_tree = construct_parquet_schema_tree(
    vec, table_meta, write_mode, int96_timestamps, utc_timestamps, write_arrow_schema);
  // Construct parquet_column_views from the schema tree leaf nodes.
  std::vector<parquet_column_view> parquet_columns;

  for (schema_tree_node const& schema_node : schema_tree) {
    if (schema_node.leaf_column) { parquet_columns.emplace_back(schema_node, schema_tree, stream); }
  }

  // Mass allocation of column_device_views for each parquet_column_view
  std::vector<column_view> cudf_cols;
  cudf_cols.reserve(parquet_columns.size());
  for (auto const& parq_col : parquet_columns) {
    cudf_cols.push_back(parq_col.cudf_column_view());
  }
  table_view single_streams_table(cudf_cols);
  size_type num_columns = single_streams_table.num_columns();

  std::vector<SchemaElement> this_table_schema(schema_tree.begin(), schema_tree.end());

  // Initialize column description
  cudf::detail::hostdevice_vector<parquet_column_device_view> col_desc(parquet_columns.size(),
                                                                       stream);
  std::transform(
    parquet_columns.begin(), parquet_columns.end(), col_desc.host_ptr(), [&](auto const& pcol) {
      return pcol.get_device_view(stream);
    });

  // Init page fragments
  // 5000 is good enough for up to ~200-character strings. Longer strings and deeply nested columns
  // will start producing fragments larger than the desired page size, so calculate fragment sizes
  // for each leaf column.  Skip if the fragment size is not the default.
  size_type max_page_fragment_size =
    max_page_fragment_size_opt.value_or(default_max_page_fragment_size);

  auto column_frag_size = cudf::detail::make_host_vector<size_type>(num_columns, stream);
  std::fill(column_frag_size.begin(), column_frag_size.end(), max_page_fragment_size);

  if (input.num_rows() > 0 && not max_page_fragment_size_opt.has_value()) {
    std::vector<size_t> column_sizes;
    std::transform(single_streams_table.begin(),
                   single_streams_table.end(),
                   std::back_inserter(column_sizes),
                   [&](auto const& column) { return column_size(column, stream); });

    // adjust global fragment size if a single fragment will overrun a rowgroup
    auto const table_size  = std::reduce(column_sizes.begin(), column_sizes.end());
    auto const avg_row_len = util::div_rounding_up_safe<size_t>(table_size, input.num_rows());
    if (avg_row_len > 0) {
      // Ensure `rg_frag_size` is not bigger than size_type::max for default max_row_group_size
      // value (=uint64::max) to avoid a sign overflow when comparing
      auto const rg_frag_size =
        std::min<size_t>(std::numeric_limits<size_type>::max(),
                         util::div_rounding_up_safe(max_row_group_size, avg_row_len));
      // Safe comparison as rg_frag_size fits in size_type
      max_page_fragment_size =
        std::min<size_type>(static_cast<size_type>(rg_frag_size), max_page_fragment_size);
    }

    // dividing page size by average row length will tend to overshoot the desired
    // page size when there's high variability in the row lengths. instead, shoot
    // for multiple fragments per page to smooth things out. using 2 was too
    // unbalanced in final page sizes, so using 4 which seems to be a good
    // compromise at smoothing things out without getting fragment sizes too small.
    auto frag_size_fn = [&](auto const& col, size_t col_size) {
      int const target_frags_per_page = col.is_fixed_width() ? 1 : 4;
      auto const avg_len =
        target_frags_per_page * util::div_rounding_up_safe<size_t>(col_size, input.num_rows());
      if (avg_len > 0) {
        auto const frag_size = util::div_rounding_up_safe<size_type>(max_page_size_bytes, avg_len);
        return std::min<size_type>(max_page_fragment_size, frag_size);
      } else {
        return max_page_fragment_size;
      }
    };

    std::transform(parquet_columns.begin(),
                   parquet_columns.end(),
                   column_sizes.begin(),
                   column_frag_size.begin(),
                   frag_size_fn);
  }

  // Create table_device_view so that corresponding column_device_view data
  // can be written into col_desc members
  auto parent_column_table_device_view = table_device_view::create(single_streams_table, stream);
  auto leaf_column_views               = rmm::device_uvector<column_device_view>(0, stream);
  auto d_col_desc =
    device_span<parquet_column_device_view const>(col_desc.device_ptr(), col_desc.size());

  // Fragments are calculated in two phases. Row group fragments use a uniform row span across
  // columns and are re-measured with smaller spans until every fragment fits in a page. After
  // row group (or column chunk) boundaries are known, per-column page fragments are calculated for
  // page boundaries within each column chunk.
  std::vector<int> num_frag_in_part;

  auto part_frag_offset =
    cudf::detail::make_empty_pinned_vector<int>(partitions.size() + 1, stream);
  auto row_group_fragments = cudf::detail::hostdevice_2dvector<PageFragment>(0, 0, stream);

  // A fragment always ends up in a single page, so shrink the fragment row span and re-measure
  // until every fragment fits within the maximum page size.
  while (true) {
    num_frag_in_part.clear();
    std::transform(partitions.begin(),
                   partitions.end(),
                   std::back_inserter(num_frag_in_part),
                   [max_page_fragment_size](auto const& part) {
                     return util::div_rounding_up_safe<size_type>(part.num_rows,
                                                                  max_page_fragment_size);
                   });

    auto const num_fragments = std::reduce(num_frag_in_part.begin(), num_frag_in_part.end());

    // Store the idx of the first fragment in each partition
    part_frag_offset.clear();
    std::exclusive_scan(
      num_frag_in_part.begin(), num_frag_in_part.end(), std::back_inserter(part_frag_offset), 0);
    part_frag_offset.push_back(part_frag_offset.back() + num_frag_in_part.back());

    if (num_fragments == 0) { break; }

    auto d_part_frag_offset = cudf::detail::make_device_uvector_async(
      part_frag_offset, stream, cudf::get_current_device_resource_ref());

    row_group_fragments =
      cudf::detail::hostdevice_2dvector<PageFragment>(num_columns, num_fragments, stream);

    // Move column info to device
    col_desc.host_to_device_async(stream);
    leaf_column_views = create_leaf_column_device_views<parquet_column_device_view>(
      col_desc, *parent_column_table_device_view, stream);

    init_row_group_fragments(row_group_fragments,
                             d_col_desc,
                             partitions,
                             d_part_frag_offset,
                             max_page_fragment_size,
                             stream);

    // Compute a smaller fragment size if any fragment is too large
    auto const smaller_fragment_size =
      compute_smaller_fragment_size(row_group_fragments.host_view(),
                                    {col_desc.host_ptr(), col_desc.size()},
                                    max_page_fragment_size);

    // If no smaller fragment size is found (current size fits), break the loop.
    if (not smaller_fragment_size.has_value()) { break; }

    // Reduce the fragment size and re-measure
    max_page_fragment_size = smaller_fragment_size.value();
  }

  // The per-column fragment sizes used for page boundaries must not exceed the row group
  // fragment size, which the loop above may have reduced.
  std::transform(column_frag_size.begin(),
                 column_frag_size.end(),
                 column_frag_size.begin(),
                 [max_page_fragment_size](auto frag_size) {
                   return std::min(frag_size, max_page_fragment_size);
                 });
  auto const num_fragments = row_group_fragments.size().second;

  std::unique_ptr<aggregate_writer_metadata> agg_meta;
  if (!curr_agg_meta) {
    agg_meta = std::make_unique<aggregate_writer_metadata>(
      partitions,
      kv_meta,
      this_table_schema,
      num_columns,
      stats_granularity,
      (write_arrow_schema)
        ? construct_arrow_schema_ipc_message(vec, table_meta, write_mode, utc_timestamps)
        : "");
  } else {
    agg_meta = std::make_unique<aggregate_writer_metadata>(*curr_agg_meta);

    // verify the user isn't passing mismatched tables
    CUDF_EXPECTS(agg_meta->schema_matches(this_table_schema),
                 "Mismatch in schema between multiple calls to write_chunk");

    agg_meta->update_files(partitions);
  }

  auto global_rowgroup_base = agg_meta->num_row_groups_per_file();

  // Decide row group boundaries based on uncompressed data size
  size_type num_rowgroups = 0;

  std::vector<int> num_rg_in_part(partitions.size());
  // Page buffers of a column chunk are addressed using 32-bit offsets, so each column's
  // contribution to a row group is budgeted separately from the row group's total size.
  std::vector<size_t> chunk_frag_size(num_columns);
  std::vector<size_t> curr_chunk_size(num_columns);

  // Helper to check if the sum of two sizes exceeds a threshold
  auto const sum_exceeds_threshold = [](auto const& lhs, auto const& rhs, auto const& threshold) {
    auto const sum = cuda::add_overflow(lhs, rhs);
    return sum.overflow or sum.value > threshold;
  };

  for (size_t p = 0; p < partitions.size(); ++p) {
    size_type curr_rg_num_rows = 0;
    size_t curr_rg_data_size   = 0;
    int first_frag_in_rg       = part_frag_offset[p];
    int last_frag_in_part      = part_frag_offset[p + 1] - 1;
    std::fill(curr_chunk_size.begin(), curr_chunk_size.end(), 0);

    // Helper to finish a row group and start a new one
    auto finish_row_group = [&] {
      auto& rg    = agg_meta->file(p).row_groups.emplace_back();
      rg.num_rows = curr_rg_num_rows;
      ++num_rowgroups;
      num_rg_in_part[p]++;
    };
    for (auto f = first_frag_in_rg; f <= last_frag_in_part; ++f) {
      size_t fragment_data_size   = 0;
      bool is_chunk_size_exceeded = false;
      for (auto c = 0; c < num_columns; c++) {
        auto const frag = row_group_fragments[c][f];
        fragment_data_size += frag.fragment_data_size;
        chunk_frag_size[c] =
          max_fragment_page_size(frag.fragment_data_size, frag.num_values, col_desc[c]);
        is_chunk_size_exceeded |= sum_exceeds_threshold(
          curr_chunk_size[c], chunk_frag_size[c], EncColumnChunk::max_buffer_size);
      }
      size_type fragment_num_rows = row_group_fragments[0][f].num_rows;

      // If the fragment size gets larger than rg limit then break off a rg
      auto const starts_new_row_group =
        f > first_frag_in_rg and  // There has to be at least one fragment in row group
        (sum_exceeds_threshold(curr_rg_data_size, fragment_data_size, max_row_group_size) or
         sum_exceeds_threshold(curr_rg_num_rows, fragment_num_rows, max_row_group_rows) or
         is_chunk_size_exceeded);
      if (starts_new_row_group) {
        finish_row_group();
        curr_rg_num_rows  = 0;
        curr_rg_data_size = 0;
        first_frag_in_rg  = f;
      }
      curr_rg_num_rows += fragment_num_rows;
      curr_rg_data_size += fragment_data_size;

      if (starts_new_row_group) {
        curr_chunk_size = chunk_frag_size;
      } else {
        std::transform(curr_chunk_size.begin(),
                       curr_chunk_size.end(),
                       chunk_frag_size.begin(),
                       curr_chunk_size.begin(),
                       std::plus{});
      }

      if (f == last_frag_in_part) { finish_row_group(); }
    }
  }

  std::vector<int> first_rg_in_part;
  std::exclusive_scan(
    num_rg_in_part.begin(), num_rg_in_part.end(), std::back_inserter(first_rg_in_part), 0);

  // Initialize row groups and column chunks
  auto const num_chunks = num_rowgroups * num_columns;
  hostdevice_2dvector<EncColumnChunk> chunks(num_rowgroups, num_columns, stream);

  // total fragments per column (in case they are non-uniform)
  std::vector<size_type> frags_per_column(num_columns, 0);

  for (size_t p = 0; p < partitions.size(); ++p) {
    int f               = part_frag_offset[p];
    size_type start_row = partitions[p].start_row;
    for (int r = 0; r < num_rg_in_part[p]; r++) {
      size_t global_r = global_rowgroup_base[p] + r;  // Number of rowgroups already in file/part
      auto& row_group = agg_meta->file(p).row_groups[global_r];
      auto const fragments_in_chunk =
        util::div_rounding_up_safe<uint32_t>(row_group.num_rows, max_page_fragment_size);
      row_group.total_byte_size = 0;
      row_group.columns.resize(num_columns);
      for (int c = 0; c < num_columns; c++) {
        EncColumnChunk& ck = chunks[r + first_rg_in_part[p]][c];

        ck                   = {};
        ck.col_desc          = col_desc.device_ptr() + c;
        ck.col_desc_id       = c;
        ck.fragments         = row_group_fragments.device_view()[c].data() + f;
        ck.stats             = nullptr;
        ck.start_row         = start_row;
        ck.num_rows          = (uint32_t)row_group.num_rows;
        ck.first_fragment    = c * num_fragments + f;
        ck.num_fragments     = fragments_in_chunk;
        ck.encodings         = 0;
        auto chunk_fragments = row_group_fragments[c].subspan(f, fragments_in_chunk);
        // In fragment struct, add a pointer to the chunk it belongs to
        // In each fragment in chunk_fragments, update the chunk pointer here.
        for (auto& frag : chunk_fragments) {
          frag.chunk = chunks.device_view()[r + first_rg_in_part[p]].data() + c;
        }
        ck.num_values = std::accumulate(
          chunk_fragments.begin(), chunk_fragments.end(), 0, [](uint32_t l, auto r) {
            return l + r.num_values;
          });
        ck.plain_data_size = std::accumulate(
          chunk_fragments.begin(),
          chunk_fragments.end(),
          size_t{0},
          [](auto sum, PageFragment const& frag) {
            auto const [result, overflow] =
              cuda::add_overflow<size_t>(sum, frag.fragment_data_size);
            CUDF_EXPECTS(not overflow, "Page fragments data size overflow", std::overflow_error);
            return result;
          });
        auto& column_chunk_meta          = row_group.columns[c].meta_data;
        column_chunk_meta.type           = parquet_columns[c].physical_type();
        column_chunk_meta.path_in_schema = parquet_columns[c].get_path_in_schema();
        column_chunk_meta.codec          = Compression::UNCOMPRESSED;
        column_chunk_meta.num_values     = ck.num_values;

        frags_per_column[c] += util::div_rounding_up_safe<size_type>(
          row_group.num_rows, std::min(column_frag_size[c], max_page_fragment_size));
      }
      f += fragments_in_chunk;
      start_row += (uint32_t)row_group.num_rows;
    }
  }

  row_group_fragments.host_to_device_async(stream);
  [[maybe_unused]] auto dict_info_owner = build_chunk_dictionaries(
    chunks, col_desc, row_group_fragments, compression, dict_policy, max_dictionary_size, stream);

  // The code preceding this used a uniform fragment size for all columns. Now recompute
  // fragments with a (potentially) varying number of fragments per column.

  // first figure out the total number of fragments and calculate the start offset for each column
  std::vector<size_type> frag_offsets(num_columns, 0);
  std::exclusive_scan(frags_per_column.begin(), frags_per_column.end(), frag_offsets.begin(), 0);
  size_type const total_frags =
    frags_per_column.empty() ? 0 : frag_offsets.back() + frags_per_column.back();

  rmm::device_uvector<statistics_chunk> frag_stats(0, stream);
  cudf::detail::hostdevice_vector<PageFragment> page_fragments(total_frags, stream);

  // update fragments and/or prepare for fragment statistics calculation if necessary
  if (total_frags != 0) {
    if (stats_granularity != statistics_freq::STATISTICS_NONE) {
      frag_stats.resize(total_frags, stream);
    }

    for (int c = 0; c < num_columns; c++) {
      auto frag_offset     = frag_offsets[c];
      auto const frag_size = column_frag_size[c];

      for (size_t p = 0; p < partitions.size(); ++p) {
        for (int r = 0; r < num_rg_in_part[p]; r++) {
          auto const global_r   = global_rowgroup_base[p] + r;
          auto const& row_group = agg_meta->file(p).row_groups[global_r];
          auto const fragments_in_chunk =
            util::div_rounding_up_safe<uint32_t>(row_group.num_rows, frag_size);
          EncColumnChunk& ck = chunks[r + first_rg_in_part[p]][c];
          ck.fragments       = page_fragments.device_ptr(frag_offset);
          ck.first_fragment  = frag_offset;
          ck.num_fragments   = fragments_in_chunk;

          // update the chunk pointer here for each fragment in chunk.fragments
          for (uint32_t i = 0; i < fragments_in_chunk; i++) {
            page_fragments[frag_offset + i].chunk =
              chunks.device_view()[r + first_rg_in_part[p]].data() + c;
          }

          if (not frag_stats.is_empty()) { ck.stats = frag_stats.data() + frag_offset; }
          frag_offset += fragments_in_chunk;
        }
      }
    }

    chunks.host_to_device_async(stream);

    // re-initialize page fragments
    page_fragments.host_to_device_async(stream);
    calculate_page_fragments(page_fragments, column_frag_size, stream);

    // and gather fragment statistics
    if (not frag_stats.is_empty()) {
      gather_fragment_statistics(frag_stats,
                                 {page_fragments.device_ptr(), static_cast<size_t>(total_frags)},
                                 int96_timestamps,
                                 stream);
    }
  }

  // Build chunk dictionaries and count pages. Sends chunks to device.
  auto comp_page_sizes = init_page_sizes(chunks,
                                         d_col_desc,
                                         num_columns,
                                         max_page_size_bytes,
                                         max_page_size_rows,
                                         write_v2_headers,
                                         stats_granularity == statistics_freq::STATISTICS_PAGE,
                                         compression,
                                         stream);

  // Find which partition a rg belongs to
  std::vector<int> rg_to_part;
  for (size_t p = 0; p < num_rg_in_part.size(); ++p) {
    std::fill_n(std::back_inserter(rg_to_part), num_rg_in_part[p], p);
  }

  // Initialize rowgroups to encode
  size_type num_pages           = 0;
  size_t max_uncomp_bfr_size    = 0;
  size_t max_comp_bfr_size      = 0;
  size_t column_index_bfr_size  = 0;
  size_t def_histogram_bfr_size = 0;
  size_t rep_histogram_bfr_size = 0;
  for (size_type r = 0; r < num_rowgroups; r++) {
    for (int i = 0; i < num_columns; i++) {
      EncColumnChunk* ck = &chunks[r][i];
      ck->first_page     = num_pages;
      num_pages += ck->num_pages;
      max_uncomp_bfr_size += ck->bfr_size;
      max_comp_bfr_size += ck->compressed_size;
      if (stats_granularity == statistics_freq::STATISTICS_COLUMN) {
        auto const& col = col_desc[ck->col_desc_id];
        column_index_bfr_size += column_index_buffer_size(ck, col, column_index_truncate_length);

        // SizeStatistics are on the ColumnIndex, so only need to allocate the histograms data
        // if we're doing page-level indexes. add 1 to num_pages for per-chunk histograms.
        auto const num_histograms = ck->num_data_pages() + 1;

        if (col.max_def_level > DEF_LVL_HIST_CUTOFF) {
          def_histogram_bfr_size += (col.max_def_level + 1) * num_histograms;
        }
        if (col.max_rep_level > REP_LVL_HIST_CUTOFF) {
          rep_histogram_bfr_size += (col.max_rep_level + 1) * num_histograms;
        }
      }
    }
  }

  // Clear compressed buffer size if compression has been turned off
  if (compression == compression_type::NONE) { max_comp_bfr_size = 0; }

  // Initialize data pointers
  uint32_t const num_stats_bfr =
    (stats_granularity != statistics_freq::STATISTICS_NONE) ? num_pages + num_chunks : 0;

  // Buffers need to be padded.
  // Required by `gpuGatherPages`.
  rmm::device_buffer uncomp_bfr(
    cudf::util::round_up_safe(max_uncomp_bfr_size, BUFFER_PADDING_MULTIPLE), stream);
  rmm::device_buffer comp_bfr(cudf::util::round_up_safe(max_comp_bfr_size, BUFFER_PADDING_MULTIPLE),
                              stream);

  rmm::device_buffer col_idx_bfr(column_index_bfr_size, stream);
  rmm::device_uvector<EncPage> pages(num_pages, stream);
  rmm::device_uvector<uint32_t> def_level_histogram(def_histogram_bfr_size, stream);
  rmm::device_uvector<uint32_t> rep_level_histogram(rep_histogram_bfr_size, stream);

  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    def_level_histogram.begin(),
    def_level_histogram.end(),
    0);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    rep_level_histogram.begin(),
    rep_level_histogram.end(),
    0);

  // This contains stats for both the pages and the rowgroups. TODO: make them separate.
  rmm::device_uvector<statistics_chunk> page_stats(num_stats_bfr, stream);
  auto bfr_i = static_cast<uint8_t*>(col_idx_bfr.data());
  auto bfr_r = rep_level_histogram.data();
  auto bfr_d = def_level_histogram.data();
  if (num_rowgroups != 0) {
    auto bfr   = static_cast<uint8_t*>(uncomp_bfr.data());
    auto bfr_c = static_cast<uint8_t*>(comp_bfr.data());
    for (auto r = 0; r < num_rowgroups; r++) {
      for (auto i = 0; i < num_columns; i++) {
        EncColumnChunk& ck   = chunks[r][i];
        ck.uncompressed_bfr  = bfr;
        ck.compressed_bfr    = bfr_c;
        ck.column_index_blob = bfr_i;
        bfr += ck.bfr_size;
        bfr_c += ck.compressed_size;
        if (stats_granularity == statistics_freq::STATISTICS_COLUMN) {
          auto const& col      = col_desc[ck.col_desc_id];
          ck.column_index_size = column_index_buffer_size(&ck, col, column_index_truncate_length);
          bfr_i += ck.column_index_size;

          auto const num_histograms = ck.num_data_pages() + 1;
          if (col.max_def_level > DEF_LVL_HIST_CUTOFF) {
            ck.def_histogram_data = bfr_d;
            bfr_d += num_histograms * (col.max_def_level + 1);
          }
          if (col.max_rep_level > REP_LVL_HIST_CUTOFF) {
            ck.rep_histogram_data = bfr_r;
            bfr_r += num_histograms * (col.max_rep_level + 1);
          }
        }
      }
    }
  }

  if (num_pages != 0) {
    init_encoder_pages(chunks,
                       d_col_desc,
                       {pages.data(), pages.size()},
                       comp_page_sizes,
                       (num_stats_bfr) ? page_stats.data() : nullptr,
                       (num_stats_bfr) ? frag_stats.data() : nullptr,
                       num_columns,
                       num_pages,
                       num_stats_bfr,
                       compress_required_chunk_alignment(compression),
                       max_page_size_bytes,
                       max_page_size_rows,
                       write_v2_headers,
                       stats_granularity == statistics_freq::STATISTICS_PAGE,
                       stream);

    // Now that page boundaries are finalized and dictionary indices have been materialized, compute
    // minimum required RLE bit width for each data page
    compute_per_page_dict_bits({pages.data(), pages.size()}, stream);
  }

  // Check device write support for all chunks and initialize bounce_buffer.
  bool all_device_write   = true;
  uint32_t max_write_size = 0;
  std::optional<writer_compression_statistics> comp_stats;
  if (collect_compression_statistics) { comp_stats = writer_compression_statistics{}; }

  // Encode row groups
  if (num_rowgroups != 0) {
    encode_pages(
      chunks,
      {pages.data(), pages.size()},
      // Page statistics only go into the data page headers for STATISTICS_PAGE
      (stats_granularity == statistics_freq::STATISTICS_PAGE) ? page_stats.data() : nullptr,
      (stats_granularity != statistics_freq::STATISTICS_NONE) ? page_stats.data() + num_pages
                                                              : nullptr,
      (stats_granularity == statistics_freq::STATISTICS_COLUMN) ? page_stats.data() : nullptr,
      comp_stats,
      compression,
      column_index_truncate_length,
      write_v2_headers,
      page_level_compression,
      stream);

    bool need_sync{false};

    // need to fetch the histogram data from the device
    auto const h_def_histogram = [&]() {
      if (stats_granularity == statistics_freq::STATISTICS_COLUMN && def_histogram_bfr_size > 0) {
        need_sync = true;
        return cudf::detail::make_host_vector_async(def_level_histogram, stream);
      }
      return cudf::detail::make_host_vector<uint32_t>(0, stream);
    }();
    auto const h_rep_histogram = [&]() {
      if (stats_granularity == statistics_freq::STATISTICS_COLUMN && rep_histogram_bfr_size > 0) {
        need_sync = true;
        return cudf::detail::make_host_vector_async(rep_level_histogram, stream);
      }
      return cudf::detail::make_host_vector<uint32_t>(0, stream);
    }();

    for (int r = 0; r < num_rowgroups; r++) {
      int p           = rg_to_part[r];
      int global_r    = global_rowgroup_base[p] + r - first_rg_in_part[p];
      auto& row_group = agg_meta->file(p).row_groups[global_r];

      for (auto i = 0; i < num_columns; i++) {
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
        auto const& ck = chunks[r][i];
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
        auto const dev_bfr      = ck.is_compressed ? ck.compressed_bfr : ck.uncompressed_bfr;
        auto& column_chunk_meta = row_group.columns[i].meta_data;

        if (ck.is_compressed) { column_chunk_meta.codec = to_parquet_compression(compression); }
        if (!out_sink[p]->is_device_write_preferred(ck.compressed_size)) {
          all_device_write = false;
        }
        max_write_size = std::max(max_write_size, ck.compressed_size);

        update_chunk_encodings(column_chunk_meta.encodings, ck.encodings);
        update_chunk_encoding_stats(column_chunk_meta, ck, write_v2_headers);

        if (ck.ck_stat_size != 0) {
          auto const stats_blob = cudf::detail::make_host_vector(
            device_span<uint8_t const>(dev_bfr, ck.ck_stat_size), stream);
          CompactProtocolReader cp(stats_blob.data(), stats_blob.size());
          cp.read(&column_chunk_meta.statistics);
          need_sync = true;
        }

        row_group.total_byte_size += ck.bfr_size;
        row_group.total_compressed_size =
          row_group.total_compressed_size.value_or(0) + ck.compressed_size;
        column_chunk_meta.total_uncompressed_size = ck.bfr_size;
        column_chunk_meta.total_compressed_size   = ck.compressed_size;
      }
    }

    // Sync before calling the next `encode_pages` which may alter the stats data.
    if (need_sync) { stream.sync(); }

    // now add to the column chunk SizeStatistics if necessary
    if (stats_granularity == statistics_freq::STATISTICS_COLUMN) {
      auto h_def_ptr = h_def_histogram.data();
      auto h_rep_ptr = h_rep_histogram.data();

      for (int r = 0; r < num_rowgroups; r++) {
        int const p        = rg_to_part[r];
        int const global_r = global_rowgroup_base[p] + r - first_rg_in_part[p];
        auto& row_group    = agg_meta->file(p).row_groups[global_r];

        for (auto i = 0; i < num_columns; i++) {
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
          auto const& ck = chunks[r][i];
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
          auto const& col         = col_desc[ck.col_desc_id];
          auto& column_chunk_meta = row_group.columns[i].meta_data;

          // Add SizeStatistics for the chunk. For now we're only going to do the column chunk
          // stats if we're also doing them at the page level. There really isn't much value for
          // us in per-chunk stats since everything we do processing wise is at the page level.
          SizeStatistics chunk_stats;

          // var_byte_size will only be non-zero for byte array columns.
          if (ck.var_bytes_size > 0) {
            chunk_stats.unencoded_byte_array_data_bytes = ck.var_bytes_size;
          }

          auto const num_data_pages = ck.num_data_pages();
          if (col.max_def_level > DEF_LVL_HIST_CUTOFF) {
            size_t const hist_size        = col.max_def_level + 1;
            uint32_t const* const ck_hist = h_def_ptr + hist_size * num_data_pages;
            host_span<uint32_t const> ck_def_hist{ck_hist, hist_size};

            chunk_stats.definition_level_histogram = {ck_def_hist.begin(), ck_def_hist.end()};
            h_def_ptr += hist_size * (num_data_pages + 1);
          }

          if (col.max_rep_level > REP_LVL_HIST_CUTOFF) {
            size_t const hist_size        = col.max_rep_level + 1;
            uint32_t const* const ck_hist = h_rep_ptr + hist_size * num_data_pages;
            host_span<uint32_t const> ck_rep_hist{ck_hist, hist_size};

            chunk_stats.repetition_level_histogram = {ck_rep_hist.begin(), ck_rep_hist.end()};
            h_rep_ptr += hist_size * (num_data_pages + 1);
          }

          if (chunk_stats.unencoded_byte_array_data_bytes.has_value() ||
              chunk_stats.definition_level_histogram.has_value() ||
              chunk_stats.repetition_level_histogram.has_value()) {
            column_chunk_meta.size_statistics = std::move(chunk_stats);
          }
        }
      }
    }
  }

  auto bounce_buffer =
    cudf::detail::make_pinned_vector_async<uint8_t>(all_device_write ? 0 : max_write_size, stream);

  return std::tuple{std::move(agg_meta),
                    std::move(pages),
                    std::move(chunks),
                    std::move(global_rowgroup_base),
                    std::move(first_rg_in_part),
                    std::move(rg_to_part),
                    std::move(comp_stats),
                    std::move(uncomp_bfr),
                    std::move(comp_bfr),
                    std::move(col_idx_bfr),
                    std::move(bounce_buffer)};
}

}  // namespace

writer::impl::impl(std::vector<std::unique_ptr<data_sink>> sinks,
                   parquet_writer_options const& options,
                   single_write_mode mode,
                   cuda::stream_ref stream)
  : _stream(stream),
    _compression(options.get_compression()),
    _max_row_group_size{options.get_row_group_size_bytes()},
    _max_row_group_rows{options.get_row_group_size_rows()},
    _max_page_size_bytes(max_page_bytes(_compression, options.get_max_page_size_bytes())),
    _max_page_size_rows(options.get_max_page_size_rows()),
    _stats_granularity(options.get_stats_level()),
    _dict_policy(options.get_dictionary_policy()),
    _max_dictionary_size(options.get_max_dictionary_size()),
    _max_page_fragment_size(options.get_max_page_fragment_size()),
    _int96_timestamps(options.is_enabled_int96_timestamps()),
    _utc_timestamps(options.is_enabled_utc_timestamps()),
    _write_v2_headers(options.is_enabled_write_v2_headers()),
    _page_level_compression(options.is_enabled_page_level_compression()),
    _write_arrow_schema(options.is_enabled_write_arrow_schema()),
    _sorting_columns(options.get_sorting_columns()),
    _column_index_truncate_length(options.get_column_index_truncate_length()),
    _kv_meta(options.get_key_value_metadata()),
    _single_write_mode(mode),
    _out_sink(std::move(sinks)),
    _compression_statistics{options.get_compression_statistics()}
{
  if (options.get_metadata()) {
    _table_meta = std::make_unique<table_input_metadata>(*options.get_metadata());
  }

  CUDF_EXPECTS(is_supported_write_parquet(_compression),
               "Compression type not supported for Parquet writer");

  if (_page_level_compression and not _write_v2_headers) {
    CUDF_LOG_WARN(
      "page_level_compression is only supported with V2 page headers; the option will be ignored");
  }

  init_state();
}

writer::impl::impl(std::vector<std::unique_ptr<data_sink>> sinks,
                   chunked_parquet_writer_options const& options,
                   single_write_mode mode,
                   cuda::stream_ref stream)
  : _stream(stream),
    _compression(options.get_compression()),
    _max_row_group_size{options.get_row_group_size_bytes()},
    _max_row_group_rows{options.get_row_group_size_rows()},
    _max_page_size_bytes(max_page_bytes(_compression, options.get_max_page_size_bytes())),
    _max_page_size_rows(options.get_max_page_size_rows()),
    _stats_granularity(options.get_stats_level()),
    _dict_policy(options.get_dictionary_policy()),
    _max_dictionary_size(options.get_max_dictionary_size()),
    _max_page_fragment_size(options.get_max_page_fragment_size()),
    _int96_timestamps(options.is_enabled_int96_timestamps()),
    _utc_timestamps(options.is_enabled_utc_timestamps()),
    _write_v2_headers(options.is_enabled_write_v2_headers()),
    _page_level_compression(options.is_enabled_page_level_compression()),
    _write_arrow_schema(options.is_enabled_write_arrow_schema()),
    _sorting_columns(options.get_sorting_columns()),
    _column_index_truncate_length(options.get_column_index_truncate_length()),
    _kv_meta(options.get_key_value_metadata()),
    _single_write_mode(mode),
    _out_sink(std::move(sinks)),
    _compression_statistics{options.get_compression_statistics()}
{
  if (options.get_metadata()) {
    _table_meta = std::make_unique<table_input_metadata>(*options.get_metadata());
  }

  CUDF_EXPECTS(is_supported_write_parquet(_compression),
               "Compression type not supported for Parquet writer");

  if (_page_level_compression and not _write_v2_headers) {
    CUDF_LOG_WARN(
      "page_level_compression is only supported with V2 page headers; the option will be ignored");
  }

  init_state();
}

writer::impl::~impl() { close(); }

void writer::impl::init_state()
{
  _current_chunk_offset.resize(_out_sink.size());
  // Write file header
  file_header_s fhdr;
  fhdr.magic = parquet_magic;
  for (auto& sink : _out_sink) {
    sink->host_write(&fhdr, sizeof(fhdr));
  }
  std::fill_n(_current_chunk_offset.begin(), _current_chunk_offset.size(), sizeof(file_header_s));
}

void writer::impl::update_compression_statistics(
  std::optional<writer_compression_statistics> const& compression_stats)
{
  if (compression_stats.has_value() and _compression_statistics != nullptr) {
    *_compression_statistics += compression_stats.value();
  }
}

void writer::impl::write(table_view const& input, std::vector<partition_info> const& partitions)
{
  _last_write_successful = false;
  CUDF_EXPECTS(not _closed, "Data has already been flushed to out and closed");

  if (not _table_meta) { _table_meta = std::make_unique<table_input_metadata>(input); }
  fill_table_meta(*_table_meta);

  // All kinds of memory allocation and data compressions/encoding are performed here.
  // If any error occurs, such as out-of-memory exception, the internal state of the current
  // writer is still intact.
  [[maybe_unused]] auto [updated_agg_meta,
                         pages,
                         chunks,
                         global_rowgroup_base,
                         first_rg_in_part,
                         rg_to_part,
                         comp_stats,
                         uncomp_bfr,   // unused, but contains data for later write to sink
                         comp_bfr,     // unused, but contains data for later write to sink
                         col_idx_bfr,  // unused, but contains data for later write to sink
                         bounce_buffer] =
    convert_table_to_parquet_data(*_table_meta,
                                  input,
                                  partitions,
                                  _kv_meta,
                                  _agg_meta,
                                  _max_page_fragment_size,
                                  _max_row_group_size,
                                  _max_page_size_bytes,
                                  _max_row_group_rows,
                                  _max_page_size_rows,
                                  _column_index_truncate_length,
                                  _stats_granularity,
                                  _compression,
                                  _compression_statistics != nullptr,
                                  _dict_policy,
                                  _max_dictionary_size,
                                  _single_write_mode,
                                  _int96_timestamps,
                                  _utc_timestamps,
                                  _write_v2_headers,
                                  _page_level_compression,
                                  _write_arrow_schema,
                                  _out_sink,
                                  _stream);

  // Compression/encoding were all successful. Now write the intermediate results.
  write_parquet_data_to_sink(updated_agg_meta,
                             pages,
                             chunks,
                             global_rowgroup_base,
                             first_rg_in_part,
                             rg_to_part,
                             bounce_buffer);

  update_compression_statistics(comp_stats);

  _last_write_successful = true;
}

void writer::impl::write_parquet_data_to_sink(
  std::unique_ptr<aggregate_writer_metadata>& updated_agg_meta,
  device_span<EncPage const> pages,
  host_2dspan<EncColumnChunk const> chunks,
  host_span<size_t const> global_rowgroup_base,
  host_span<int const> first_rg_in_part,
  host_span<int const> rg_to_part,
  host_span<uint8_t> bounce_buffer)
{
  _agg_meta                = std::move(updated_agg_meta);
  auto const num_rowgroups = chunks.size().first;
  auto const num_columns   = chunks.size().second;

  if (num_rowgroups != 0) {
    std::vector<std::future<void>> write_tasks;

    for (auto r = 0; r < static_cast<int>(num_rowgroups); r++) {
      int const p        = rg_to_part[r];
      int const global_r = global_rowgroup_base[p] + r - first_rg_in_part[p];
      auto& row_group    = _agg_meta->file(p).row_groups[global_r];

      for (std::size_t i = 0; i < num_columns; i++) {
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
        auto const& ck = chunks[r][i];
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
        auto const dev_bfr = ck.is_compressed ? ck.compressed_bfr : ck.uncompressed_bfr;

        // Skip the range [0, ck.ck_stat_size) since it has already been copied to host
        // and stored in _agg_meta before.
        if (_out_sink[p]->is_device_write_preferred(ck.compressed_size)) {
          write_tasks.push_back(_out_sink[p]->device_write_async(
            dev_bfr + ck.ck_stat_size, ck.compressed_size, _stream));
        } else {
          CUDF_EXPECTS(bounce_buffer.size() >= ck.compressed_size,
                       "Bounce buffer was not properly initialized.");
          cudf::detail::cuda_memcpy(
            host_span{bounce_buffer}.subspan(0, ck.compressed_size),
            device_span<uint8_t const>{dev_bfr + ck.ck_stat_size, ck.compressed_size},
            _stream);

          _out_sink[p]->host_write(bounce_buffer.data(), ck.compressed_size);
        }

        auto const chunk_offset = _current_chunk_offset[p];
        auto& column_chunk_meta = row_group.columns[i].meta_data;
        column_chunk_meta.data_page_offset =
          chunk_offset + ((ck.use_dictionary) ? ck.dictionary_size : 0);
        column_chunk_meta.dictionary_page_offset = (ck.use_dictionary) ? chunk_offset : 0;
        _current_chunk_offset[p] += ck.compressed_size;

        // save location of first page in row group
        if (i == 0) { row_group.file_offset = chunk_offset; }
      }
    }
    for (auto& task : write_tasks) {
      task.get();
    }
  }

  if (_stats_granularity == statistics_freq::STATISTICS_COLUMN) {
    // need pages on host to create offset_indexes
    auto const h_pages = cudf::detail::make_host_vector(pages, _stream);

    // add column and offset indexes to metadata
    if (num_rowgroups != 0) {
      auto curr_page_idx = chunks[0][0].first_page;
      for (auto r = 0; r < static_cast<int>(num_rowgroups); r++) {
        int const p           = rg_to_part[r];
        int const global_r    = global_rowgroup_base[p] + r - first_rg_in_part[p];
        auto const& row_group = _agg_meta->file(p).row_groups[global_r];
        for (std::size_t i = 0; i < num_columns; i++) {
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
          EncColumnChunk const& ck = chunks[r][i];
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
          auto const& column_chunk_meta = row_group.columns[i].meta_data;

          // start transfer of the column index
          auto column_idx = cudf::detail::make_host_vector_async(
            device_span<uint8_t const>{ck.column_index_blob, ck.column_index_size}, _stream);

          // calculate offsets while the column index is transferring
          int64_t curr_pg_offset = column_chunk_meta.data_page_offset;

          OffsetIndex offset_idx;
          std::vector<int64_t> var_bytes;
          auto const is_byte_arr = column_chunk_meta.type == Type::BYTE_ARRAY;

          for (uint32_t pg = 0; pg < ck.num_pages; pg++) {
            auto const& enc_page = h_pages[curr_page_idx++];

            // skip dict pages
            if (enc_page.page_type == PageType::DICTIONARY_PAGE) { continue; }

            int32_t const this_page_size =
              enc_page.hdr_size +
              (enc_page.is_compressed ? enc_page.comp_data_size : enc_page.data_size);
            // first_row_idx is relative to start of row group
            PageLocation loc{curr_pg_offset, this_page_size, enc_page.start_row - ck.start_row};
            if (is_byte_arr) { var_bytes.push_back(enc_page.var_bytes_size); }
            offset_idx.page_locations.push_back(loc);
            curr_pg_offset += this_page_size;
          }

          if (is_byte_arr) { offset_idx.unencoded_byte_array_data_bytes = std::move(var_bytes); }

          _stream.sync();
          _agg_meta->file(p).offset_indexes.emplace_back(std::move(offset_idx));
          _agg_meta->file(p).column_indexes.emplace_back(std::move(column_idx));
        }
      }
    }
  }
}

std::unique_ptr<std::vector<uint8_t>> writer::impl::close(
  std::vector<std::string> const& column_chunks_file_path)
{
  if (_closed) { return nullptr; }
  _closed = true;
  if (not _last_write_successful) { return nullptr; }
  for (size_t p = 0; p < _out_sink.size(); p++) {
    std::vector<uint8_t> buffer;
    CompactProtocolWriter cpw(&buffer);
    auto& fmd = _agg_meta->file(p);

    if (_stats_granularity == statistics_freq::STATISTICS_COLUMN) {
      // write column indices, updating column metadata along the way
      int chunkidx = 0;
      for (auto& r : fmd.row_groups) {
        for (auto& c : r.columns) {
          auto const& index     = fmd.column_indexes[chunkidx++];
          c.column_index_offset = _out_sink[p]->bytes_written();
          c.column_index_length = index.size();
          _out_sink[p]->host_write(index.data(), index.size());
        }
      }

      // write offset indices, updating column metadata along the way
      chunkidx = 0;
      for (auto& r : fmd.row_groups) {
        for (auto& c : r.columns) {
          auto const& offsets = fmd.offset_indexes[chunkidx++];
          buffer.resize(0);
          int32_t len           = cpw.write(offsets);
          c.offset_index_offset = _out_sink[p]->bytes_written();
          c.offset_index_length = len;
          _out_sink[p]->host_write(buffer.data(), buffer.size());
        }
      }
    }

    // set row group ordinals
    auto iter        = cuda::counting_iterator<size_t>{0};
    auto& row_groups = fmd.row_groups;
    std::for_each(
      iter, iter + row_groups.size(), [&row_groups](auto idx) { row_groups[idx].ordinal = idx; });

    // set sorting_columns on row groups
    if (_sorting_columns.has_value()) {
      // convert `sorting_column` to `SortingColumn`
      auto const& sorting_cols = _sorting_columns.value();
      std::vector<SortingColumn> scols;
      std::transform(
        sorting_cols.begin(), sorting_cols.end(), std::back_inserter(scols), [](auto const& sc) {
          return SortingColumn{sc.column_idx, sc.is_descending, sc.is_nulls_first};
        });
      // and copy to each row group
      std::for_each(iter, iter + row_groups.size(), [&row_groups, &scols](auto idx) {
        row_groups[idx].sorting_columns = scols;
      });
    }
    buffer.resize(0);
    file_ender_s fendr{.footer_len = static_cast<uint32_t>(cpw.write(_agg_meta->get_metadata(p))),
                       .magic      = parquet_magic};
    _out_sink[p]->host_write(buffer.data(), buffer.size());
    _out_sink[p]->host_write(&fendr, sizeof(fendr));
    _out_sink[p]->flush();
  }

  // Output raw file metadata
  std::vector<uint8_t> buffer;
  file_header_s fhdr = {.magic = parquet_magic};
  buffer.insert(buffer.end(),
                reinterpret_cast<uint8_t const*>(&fhdr),
                reinterpret_cast<uint8_t const*>(&fhdr) + sizeof(fhdr));

  CUDF_EXPECTS(
    column_chunks_file_path.empty() or column_chunks_file_path.size() == _agg_meta->num_files(),
    "Expected one column chunk path per output file");

  // Only set file paths if column_chunks_file_path is provided
  if (column_chunks_file_path.size() > 0) { _agg_meta->set_file_paths(column_chunks_file_path); }

  CompactProtocolWriter cpw(&buffer);
  file_ender_s fendr{
    .footer_len = static_cast<uint32_t>(cpw.write(_agg_meta->get_merged_metadata())),
    .magic      = parquet_magic};
  buffer.insert(buffer.end(),
                reinterpret_cast<uint8_t const*>(&fendr),
                reinterpret_cast<uint8_t const*>(&fendr) + sizeof(fendr));
  return std::make_unique<std::vector<uint8_t>>(std::move(buffer));
}

// Forward to implementation
writer::writer(std::vector<std::unique_ptr<data_sink>> sinks,
               parquet_writer_options const& options,
               single_write_mode mode,
               cuda::stream_ref stream)
  : _impl(std::make_unique<impl>(std::move(sinks), options, mode, stream))
{
}

writer::writer(std::vector<std::unique_ptr<data_sink>> sinks,
               chunked_parquet_writer_options const& options,
               single_write_mode mode,
               cuda::stream_ref stream)
  : _impl(std::make_unique<impl>(std::move(sinks), options, mode, stream))
{
}

// Destructor within this translation unit
writer::~writer() = default;

// Forward to implementation
void writer::write(table_view const& table, std::vector<partition_info> const& partitions)
{
  _impl->write(
    table, partitions.empty() ? std::vector<partition_info>{{0, table.num_rows()}} : partitions);
}

// Forward to implementation
std::unique_ptr<std::vector<uint8_t>> writer::close(
  std::vector<std::string> const& column_chunks_file_path)
{
  return _impl->close(column_chunks_file_path);
}

std::unique_ptr<std::vector<uint8_t>> writer::merge_row_group_metadata(
  std::vector<std::unique_ptr<std::vector<uint8_t>>> const& metadata_list)
{
  std::vector<uint8_t> output;
  CompactProtocolWriter cpw(&output);
  FileMetaData md;

  md.row_groups.reserve(metadata_list.size());
  for (auto const& blob : metadata_list) {
    CompactProtocolReader cpreader(
      blob.get()->data(),
      std::max<size_t>(blob.get()->size(), sizeof(file_ender_s)) - sizeof(file_ender_s));
    cpreader.skip_bytes(sizeof(file_header_s));  // Skip over file header
    if (md.num_rows == 0) {
      cpreader.read(&md);
    } else {
      FileMetaData tmp;
      cpreader.read(&tmp);
      md.row_groups.insert(md.row_groups.end(),
                           std::make_move_iterator(tmp.row_groups.begin()),
                           std::make_move_iterator(tmp.row_groups.end()));
      md.num_rows += tmp.num_rows;
    }
  }

  // Remove any LogicalType::UNKNOWN annotations that were passed in as they can confuse
  // column type inferencing.
  // See https://github.com/NVIDIA/cudf/pull/14264#issuecomment-1778311615
  for (auto& se : md.schema) {
    if (se.logical_type.has_value() && se.logical_type.value().type == LogicalType::UNKNOWN) {
      se.logical_type = cuda::std::nullopt;
    }
  }

  // Thrift-encode the resulting output
  file_header_s fhdr;
  file_ender_s fendr;
  fhdr.magic = parquet_magic;
  output.insert(output.end(),
                reinterpret_cast<uint8_t const*>(&fhdr),
                reinterpret_cast<uint8_t const*>(&fhdr) + sizeof(fhdr));
  fendr.footer_len = static_cast<uint32_t>(cpw.write(md));
  fendr.magic      = parquet_magic;
  output.insert(output.end(),
                reinterpret_cast<uint8_t const*>(&fendr),
                reinterpret_cast<uint8_t const*>(&fendr) + sizeof(fendr));
  return std::make_unique<std::vector<uint8_t>>(std::move(output));
}

}  // namespace cudf::io::parquet::detail
