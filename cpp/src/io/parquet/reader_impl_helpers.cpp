/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl_helpers.hpp"

#include "column_path_helpers.hpp"
#include "compact_protocol_reader.hpp"
#include "io/utilities/base64_utilities.hpp"
#include "io/utilities/row_selection.hpp"
#include "ipc/Message_generated.h"
#include "ipc/Schema_generated.h"
#include "parquet_common.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/host_memory.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/logger.hpp>

#include <cuda/iterator>
#include <cuda/numeric>
#include <cuda/std/tuple>
#include <thrust/iterator/zip_iterator.h>

#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <future>
#include <numeric>
#include <optional>
#include <regex>
#include <string_view>
#include <utility>

namespace cudf::io::parquet::detail {

std::size_t derive_pass_read_limit(std::size_t chunk_read_limit)
{
  if (chunk_read_limit == 0) { return 0; }

  // Derive a heuristic pass limit (1.5x the chunk_read_limit) to reduce surprising OOMs
  auto const sum             = cuda::add_overflow(chunk_read_limit, chunk_read_limit / 2);
  auto const pass_read_limit = sum.overflow ? 0 : sum.value;

  CUDF_LOG_WARN(std::format(
    "Chunked Parquet reader: a chunk_read_limit ({} bytes) was provided without a "
    "pass_read_limit; defaulting pass_read_limit to {} bytes to bound input and decompression "
    "memory and reduce the risk of out-of-memory errors on large files. Use a constructor overload "
    "that accepts pass_read_limit to control this explicitly.",
    chunk_read_limit,
    pass_read_limit));

  return pass_read_limit;
}

size_type find_colchunk_iter_offset(RowGroup const& row_group,
                                    size_type schema_idx,
                                    std::optional<size_type> cached_offset)
{
  if (cached_offset.has_value() and cached_offset.value() >= 0 and
      std::cmp_less(cached_offset.value(), row_group.columns.size()) and
      row_group.columns[cached_offset.value()].schema_idx == schema_idx) {
    return cached_offset.value();
  }
  auto const& colchunk_iter =
    std::find_if(row_group.columns.begin(), row_group.columns.end(), [schema_idx](auto const& col) {
      return col.schema_idx == schema_idx;
    });
  CUDF_EXPECTS(colchunk_iter != row_group.columns.end(),
               std::format("Column chunk with schema index {} not found in row group", schema_idx),
               std::invalid_argument);
  return std::distance(row_group.columns.begin(), colchunk_iter);
}

namespace flatbuf = cudf::io::parquet::flatbuf;

namespace {

/**
 * @brief Computes the total number of row groups in input span of row group indices
 */
[[nodiscard]] size_type compute_total_row_groups(
  host_span<std::vector<size_type> const> row_group_indices)
{
  auto const total_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    size_t{0},
                    [](auto sum, auto const& row_groups) { return sum + row_groups.size(); });

  // Check if we have less than 2B total row groups.
  CUDF_EXPECTS(total_row_groups <= std::numeric_limits<size_type>::max(),
               "Total number of row groups exceed the column size limit",
               std::overflow_error);
  return static_cast<size_type>(total_row_groups);
}

cuda::std::optional<LogicalType> converted_to_logical_type(SchemaElement const& schema)
{
  if (schema.converted_type.has_value()) {
    switch (schema.converted_type.value()) {
      case ConvertedType::ENUM:  // treat ENUM as UTF8 string
      case ConvertedType::UTF8: return LogicalType{LogicalType::STRING};
      case ConvertedType::MAP: return LogicalType{LogicalType::MAP};
      case ConvertedType::LIST: return LogicalType{LogicalType::LIST};
      case ConvertedType::DECIMAL:
        return LogicalType{DecimalType{schema.decimal_scale, schema.decimal_precision}};
      case ConvertedType::DATE: return LogicalType{LogicalType::DATE};
      case ConvertedType::TIME_MILLIS: return LogicalType{TimeType{true, {TimeUnit::MILLIS}}};
      case ConvertedType::TIME_MICROS: return LogicalType{TimeType{true, {TimeUnit::MICROS}}};
      case ConvertedType::TIMESTAMP_MILLIS:
        return LogicalType{TimestampType{true, {TimeUnit::MILLIS}}};
      case ConvertedType::TIMESTAMP_MICROS:
        return LogicalType{TimestampType{true, {TimeUnit::MICROS}}};
      case ConvertedType::UINT_8: return LogicalType{IntType{8, false}};
      case ConvertedType::UINT_16: return LogicalType{IntType{16, false}};
      case ConvertedType::UINT_32: return LogicalType{IntType{32, false}};
      case ConvertedType::UINT_64: return LogicalType{IntType{64, false}};
      case ConvertedType::INT_8: return LogicalType{IntType{8, true}};
      case ConvertedType::INT_16: return LogicalType{IntType{16, true}};
      case ConvertedType::INT_32: return LogicalType{IntType{32, true}};
      case ConvertedType::INT_64: return LogicalType{IntType{64, true}};
      case ConvertedType::JSON: return LogicalType{LogicalType::JSON};
      case ConvertedType::BSON: return LogicalType{LogicalType::BSON};
      case ConvertedType::INTERVAL:  // there is no logical type for INTERVAL yet
      default: return LogicalType{LogicalType::UNDEFINED};
    }
  }
  return cuda::std::nullopt;
}

/**
 * @brief Lookup schema children by name or field ID across multiple sources.
 */
struct schema_child_lookup {
  using get_schema_fn = std::function<SchemaElement const&(int, int)>;

  schema_child_lookup(get_schema_fn&& schema_fn,
                      bool case_sensitive_names,
                      column_selection_mode selection_mode)
    : schema_fn{std::move(schema_fn)},
      case_sensitive_names{case_sensitive_names},
      match_schema_by_field_id{selection_mode == column_selection_mode::BY_FIELD_ID}
  {
  }

  /**
   * @brief Cache of schema indices of all children of a schema element in the specified source
   */
  struct schema_idx_cache {
    column_path_map<size_type> by_name;
    std::unordered_map<int32_t, size_type> by_field_id;

    explicit schema_idx_cache(get_schema_fn const& schema_fn,
                              bool case_sensitive_names,
                              int const schema_idx,
                              int const src_idx)
      : by_name{make_column_path_map<size_type>(
          case_sensitive_names, schema_fn(schema_idx, src_idx).children_idx.size())},
        by_field_id{schema_fn(schema_idx, src_idx).children_idx.size()}
    {
      // Cache all children schema indices of the specified schema element by name and field ID
      auto const& schema_elem = schema_fn(schema_idx, src_idx);
      for (auto const child_schema_idx : schema_elem.children_idx) {
        auto const& child_schema = schema_fn(child_schema_idx, src_idx);
        by_name.emplace(child_schema.name, static_cast<size_type>(child_schema_idx));
        if (child_schema.field_id.has_value()) {
          by_field_id.emplace(child_schema.field_id.value(),
                              static_cast<size_type>(child_schema_idx));
        }
      }
    }
  };

  /**
   * @brief Find child schema index by name in the specified source
   */
  [[nodiscard]] size_type find_schema_child_by_name(int const schema_idx,
                                                    std::string_view name,
                                                    int const src_idx = 0)
  {
    auto const& cache     = get_schema_idx_cache(schema_idx, src_idx);
    auto const child_iter = cache.by_name.find(name);
    return child_iter != cache.by_name.end() ? child_iter->second : -1;
  }

  /**
   * @brief Find matching child schema index in the target source by field ID or name
   */
  [[nodiscard]] size_type find_target_schema_child(int const src_schema_idx,
                                                   int const dst_schema_idx,
                                                   std::string_view name,
                                                   int const src_idx)
  {
    if (match_schema_by_field_id) {
      auto const src_child_idx = find_schema_child_by_name(src_schema_idx, name);
      CUDF_EXPECTS(src_child_idx != -1,
                   "Column name not found in the source schema tree",
                   std::invalid_argument);

      auto const& src_child = schema_fn(src_child_idx, 0);
      CUDF_EXPECTS(src_child.field_id.has_value(),
                   "Column field ID not found in the source schema tree",
                   std::invalid_argument);

      return find_schema_child_by_field_id(dst_schema_idx, src_child.field_id.value(), src_idx);
    } else {
      return find_schema_child_by_name(dst_schema_idx, name, src_idx);
    }
  }

 private:
  /**
   * @brief Find child schema index by field ID in the specified source
   */
  [[nodiscard]] size_type find_schema_child_by_field_id(int const schema_idx,
                                                        int32_t field_id,
                                                        int const src_idx = 0)
  {
    auto const& cache     = get_schema_idx_cache(schema_idx, src_idx);
    auto const child_iter = cache.by_field_id.find(field_id);
    return child_iter != cache.by_field_id.end() ? child_iter->second : -1;
  }

  /**
   * @brief Get or build cached child lookup maps for the specified schema element
   */
  [[nodiscard]] schema_idx_cache const& get_schema_idx_cache(int const schema_idx,
                                                             int const src_idx = 0)
  {
    auto const cache_key =
      (static_cast<std::uint64_t>(src_idx) << 32) | static_cast<std::uint32_t>(schema_idx);
    auto [cache_iter, inserted] = schema_idx_caches.try_emplace(
      cache_key, schema_fn, case_sensitive_names, schema_idx, src_idx);
    return cache_iter->second;
  }

  get_schema_fn schema_fn;
  bool case_sensitive_names;
  bool match_schema_by_field_id;
  std::unordered_map<std::uint64_t, schema_idx_cache> schema_idx_caches;
};

}  // namespace

type_id to_type_id(SchemaElement const& schema,
                   bool strings_to_categorical,
                   type_id timestamp_type_id,
                   type_id decimal_type_id)
{
  auto const physical_type = schema.type;
  auto const arrow_type    = schema.arrow_type;
  auto logical_type        = schema.logical_type;

  // sanity check, but not worth failing over
  if (schema.converted_type.has_value() and not logical_type.has_value()) {
    CUDF_LOG_WARN("ConvertedType is specified but not LogicalType");
    logical_type = converted_to_logical_type(schema);
  }

  // check if have set the type through arrow schema?
  if (arrow_type.has_value()) {
    // is it duration type? i.e. phyical_type == INT64 and no logical/converted types
    if (physical_type == Type::INT64 and not logical_type.has_value()) {
      return arrow_type.value();
    }
    // should warn but not fail.
    CUDF_LOG_WARN("Indeterminable arrow type encountered");
  }

  if (logical_type.has_value()) {
    switch (logical_type->type) {
      case LogicalType::INTEGER: {
        auto const is_signed = logical_type->is_signed();
        switch (logical_type->bit_width()) {
          case 8: return is_signed ? type_id::INT8 : type_id::UINT8;
          case 16: return is_signed ? type_id::INT16 : type_id::UINT16;
          case 32: return is_signed ? type_id::INT32 : type_id::UINT32;
          case 64: return is_signed ? type_id::INT64 : type_id::UINT64;
          default: CUDF_FAIL("Invalid integer bitwidth");
        }
      } break;

      case LogicalType::DATE: return type_id::TIMESTAMP_DAYS;

      case LogicalType::TIME:
        if (logical_type->is_time_millis()) {
          return type_id::DURATION_MILLISECONDS;
        } else if (logical_type->is_time_micros()) {
          return type_id::DURATION_MICROSECONDS;
        } else if (logical_type->is_time_nanos()) {
          return type_id::DURATION_NANOSECONDS;
        }
        break;

      case LogicalType::TIMESTAMP:
        if (timestamp_type_id != type_id::EMPTY) {
          return timestamp_type_id;
        } else if (logical_type->is_timestamp_millis()) {
          return type_id::TIMESTAMP_MILLISECONDS;
        } else if (logical_type->is_timestamp_micros()) {
          return type_id::TIMESTAMP_MICROSECONDS;
        } else if (logical_type->is_timestamp_nanos()) {
          return type_id::TIMESTAMP_NANOSECONDS;
        }

      case LogicalType::DECIMAL: {
        int32_t const decimal_precision = logical_type->precision();
        if (physical_type == Type::INT32) {
          return type_id::DECIMAL32;
        } else if (physical_type == Type::INT64) {
          return type_id::DECIMAL64;
        } else if (physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
          if (std::cmp_less_equal(schema.type_length, sizeof(int32_t))) {
            return type_id::DECIMAL32;
          } else if (std::cmp_less_equal(schema.type_length, sizeof(int64_t))) {
            return type_id::DECIMAL64;
          } else if (std::cmp_less_equal(schema.type_length, sizeof(__int128_t))) {
            return type_id::DECIMAL128;
          }
        } else if (physical_type == Type::BYTE_ARRAY) {
          CUDF_EXPECTS(decimal_precision <= MAX_DECIMAL128_PRECISION, "Invalid decimal precision");
          if (decimal_precision <= MAX_DECIMAL32_PRECISION) {
            return type_id::DECIMAL32;
          } else if (decimal_precision <= MAX_DECIMAL64_PRECISION) {
            return type_id::DECIMAL64;
          } else {
            return type_id::DECIMAL128;
          }
        } else {
          CUDF_FAIL("Invalid representation of decimal type");
        }
      } break;

      // maps are just List<Struct<>>.
      case LogicalType::MAP:
      case LogicalType::LIST: return type_id::LIST;

      // All null column that can't have its type deduced.
      // Note: originally LogicalType::UNKNOWN was converted to ConvertedType::NA, and
      // NA then became type_id::STRING, but with the following TODO:
      // return type_id::EMPTY; //TODO(kn): enable after Null/Empty column support
      case LogicalType::UNKNOWN: return type_id::STRING;

      case LogicalType::VARIANT: return type_id::STRUCT;

      default: break;
    }
  }

  // is it simply a struct?
  if (schema.is_struct()) { return type_id::STRUCT; }

  // Physical storage type supported by Parquet; controls the on-disk storage
  // format in combination with the encoding type.
  switch (physical_type) {
    case Type::BOOLEAN: return type_id::BOOL8;
    case Type::INT32: return type_id::INT32;
    case Type::INT64: return type_id::INT64;
    case Type::FLOAT: return type_id::FLOAT32;
    case Type::DOUBLE: return type_id::FLOAT64;
    case Type::BYTE_ARRAY:
      // strings can be mapped to a 32-bit hash
      if (strings_to_categorical) { return type_id::INT32; }
      [[fallthrough]];
    case Type::FIXED_LEN_BYTE_ARRAY: return type_id::STRING;
    case Type::INT96:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_NANOSECONDS;
    default: break;
  }

  return type_id::EMPTY;
}

void metadata::sanitize_schema()
{
  // Parquet isn't very strict about incoming metadata. Lots of things can and should be inferred.
  // There are also a lot of rules that simply aren't followed and are expected to be worked around.
  // This step sanitizes the metadata to something that isn't ambiguous.
  //
  // Take, for example, the following schema:
  //
  //  required group field_id=-1 user {
  //    required int32 field_id=-1 id;
  //    optional group field_id=-1 phoneNumbers {
  //      repeated group field_id=-1 phone {
  //        required int64 field_id=-1 number;
  //        optional binary field_id=-1 kind (String);
  //      }
  //    }
  //  }
  //
  // This real-world example has no annotations telling us what is a list or a struct. On the
  // surface this looks like a column of id's and a column of list<struct<int64, string>>, but this
  // actually should be interpreted as a struct<list<struct<int64, string>>>. The phoneNumbers field
  // has to be a struct because it is a group with no repeated tag and we have no annotation. The
  // repeated group is actually BOTH a struct due to the multiple children and a list due to
  // repeated.
  //
  // This code attempts to make this less messy for the code that follows.

  std::function<void(size_t)> process = [&](size_t schema_idx) -> void {
    auto& schema_elem = schema[schema_idx];
    if (schema_idx != 0 && schema_elem.type == Type::UNDEFINED) {
      auto const& parent_schema    = schema[schema_elem.parent_idx];
      auto const is_parent_variant = parent_schema.logical_type.has_value() &&
                                     parent_schema.logical_type->type == LogicalType::VARIANT;
      auto const parent_type = parent_schema.converted_type;
      if (not is_parent_variant && schema_elem.repetition_type == FieldRepetitionType::REPEATED &&
          schema_elem.num_children >= 1 && parent_type != ConvertedType::LIST &&
          parent_type != ConvertedType::MAP) {
        // Parquet backward-compatibility rule: when the repeated field is a group,
        // the element type is the group itself, regardless of its field count. Rewrite
        // as `list<struct<...>>` for any number of children >= 1.
        // Spec:
        // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#backward-compatibility-rules
        auto const struct_node_idx = static_cast<size_type>(schema.size());

        SchemaElement struct_elem;
        struct_elem.name                 = "struct_node";
        struct_elem.repetition_type      = FieldRepetitionType::REQUIRED;
        struct_elem.num_children         = schema_elem.num_children;
        struct_elem.type                 = Type::UNDEFINED;
        struct_elem.converted_type       = std::nullopt;
        struct_elem.parent_idx           = schema_idx;
        struct_elem.max_definition_level = schema_elem.max_definition_level;
        struct_elem.max_repetition_level = schema_elem.max_repetition_level;
        struct_elem.children_idx         = std::move(schema_elem.children_idx);
        for (auto const child_idx : struct_elem.children_idx) {
          schema[child_idx].parent_idx = struct_node_idx;
        }

        schema_elem.converted_type  = ConvertedType::LIST;
        schema_elem.logical_type    = LogicalType::LIST;
        schema_elem.repetition_type = FieldRepetitionType::OPTIONAL;
        schema_elem.max_definition_level--;
        schema_elem.max_repetition_level = schema[schema_elem.parent_idx].max_repetition_level;

        // push_back may reallocate; do not use `schema_elem` past this point.
        schema.push_back(std::move(struct_elem));
        schema[schema_idx].children_idx = {struct_node_idx};
        schema[schema_idx].num_children = 1;
      }
    }

    // convert ConvertedType to LogicalType for older files
    if (schema[schema_idx].converted_type.has_value() and
        not schema[schema_idx].logical_type.has_value()) {
      schema[schema_idx].logical_type = converted_to_logical_type(schema[schema_idx]);
    }

    // Recursive `process` may append to `schema`; index into it each iteration
    // instead of holding a reference.
    auto const num_children = schema[schema_idx].num_children;
    for (size_type i = 0; i < num_children; ++i) {
      process(schema[schema_idx].children_idx[i]);
    }
  };

  process(0);
}

metadata::metadata(FileMetaData&& other) : FileMetaData(std::move(other)) {}

metadata::metadata(datasource* source, bool read_page_indexes)
{
  auto const buffer = cudf::io::parquet::fetch_footer_to_host(*source);
  CompactProtocolReader cp(buffer->data(), buffer->size());
  cp.read(this);
  auto const is_schema_initialized = cp.InitSchema(this);
  CUDF_EXPECTS(is_schema_initialized, "Cannot initialize schema");

  // Reading the page indexes is somewhat expensive, so skip if there are no byte array columns.
  // Currently the indexes are only used for the string size calculations.
  // Could also just read indexes for string columns, but that would require changes elsewhere
  // where we're trying to determine if we have the indexes or not.
  // Note: This will have to be modified if there are other uses in the future (e.g. calculating
  // chunk/pass boundaries).
  auto const has_strings = std::any_of(
    schema.begin(), schema.end(), [](auto const& elem) { return elem.type == Type::BYTE_ARRAY; });

  if (read_page_indexes and has_strings and not row_groups.empty() and
      not row_groups.front().columns.empty()) {
    // column index and offset index are encoded back to back.
    // the first column of the first row group will have the first column index, the last
    // column of the last row group will have the final offset index.
    int64_t const min_offset = row_groups.front().columns.front().column_index_offset;
    auto const& last_col     = row_groups.back().columns.back();
    int64_t const max_offset = last_col.offset_index_offset + last_col.offset_index_length;

    if (max_offset > min_offset) {
      size_t const length     = max_offset - min_offset;
      auto const page_idx_buf = source->host_read(min_offset, length);
      setup_page_index({page_idx_buf->data(), length}, min_offset);
    }
  }

  sanitize_schema();
}

void metadata::setup_page_index(cudf::host_span<uint8_t const> page_index_bytes, int64_t min_offset)
{
  CUDF_FUNC_RANGE();

  // Flatten all columns into a single vector for easier task distribution
  std::vector<std::reference_wrapper<ColumnChunk>> all_column_chunks;
  all_column_chunks.reserve(row_groups.size() * row_groups.front().columns.size());
  for (auto& rg : row_groups) {
    for (auto& col : rg.columns) {
      all_column_chunks.emplace_back(std::ref(col));
    }
  }

  auto read_column_indexes = [page_index_bytes, min_offset](CompactProtocolReader& reader,
                                                            ColumnChunk& col) {
    if (col.column_index_length > 0 && col.column_index_offset > 0) {
      int64_t const offset = col.column_index_offset - min_offset;
      reader.init(page_index_bytes.data() + offset, col.column_index_length);
      col.column_index = ColumnIndex();
      reader.read(&col.column_index.value());
    }
    if (col.offset_index_length > 0 && col.offset_index_offset > 0) {
      int64_t const offset = col.offset_index_offset - min_offset;
      reader.init(page_index_bytes.data() + offset, col.offset_index_length);
      col.offset_index = OffsetIndex();
      reader.read(&col.offset_index.value());
    }
  };

  // Use parallel processing only if we have enough columns to justify the overhead
  constexpr std::size_t parallel_threshold = 256;
  auto const total_column_chunks           = all_column_chunks.size();
  if (total_column_chunks >= parallel_threshold) {
    // Dynamically calculate number of tasks based the number or row groups and columns
    constexpr std::size_t min_tasks = 4;
    constexpr std::size_t max_tasks = 32;
    auto const ratio                = static_cast<double>(total_column_chunks) / parallel_threshold;
    // Scale the number of tasks and task size evenly (e.g. quadrupling the number of elements
    // doubles both the number of tasks and the task size)
    auto const multiplier = std::size_t(1) << (static_cast<size_t>(std::log2(ratio)) / 2);

    auto const num_tasks              = std::clamp(min_tasks * multiplier, min_tasks, max_tasks);
    auto const column_chunks_per_task = total_column_chunks / num_tasks;
    auto const remainder              = total_column_chunks % num_tasks;

    std::vector<std::future<void>> tasks;
    tasks.reserve(num_tasks);

    std::size_t start_idx = 0;
    for (std::size_t task_id = 0; task_id < num_tasks; ++task_id) {
      auto const task_size = column_chunks_per_task + (task_id < remainder ? 1 : 0);
      auto const end_idx   = start_idx + task_size;

      if (start_idx >= total_column_chunks) break;

      tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(
        [&all_column_chunks, &read_column_indexes, start_idx, end_idx]() {
          CompactProtocolReader local_cp;

          for (size_t i = start_idx; i < end_idx && i < all_column_chunks.size(); ++i) {
            read_column_indexes(local_cp, all_column_chunks[i].get());
          }
        }));

      start_idx = end_idx;
    }

    for (auto& task : tasks) {
      task.get();
    }
  } else {
    CompactProtocolReader cp(page_index_bytes.data(), page_index_bytes.size());
    // For small numbers of columns, use sequential processing to avoid overhead
    for (auto& col_ref : all_column_chunks) {
      read_column_indexes(cp, col_ref.get());
    }
  }
}

metadata::~metadata()
{
  constexpr auto defer_threshold = 512;
  if (row_groups.size() > 0 and
      row_groups.front().columns.size() * row_groups.size() > defer_threshold) {
    // Defer destruction to the worker pool when there are many vectors to destroy
    cudf::detail::host_worker_pool().submit_task(
      [schema_to_destroy        = std::move(schema),
       row_groups_to_destroy    = std::move(row_groups),
       key_value_to_destroy     = std::move(key_value_metadata),
       created_by_to_destroy    = std::move(created_by),
       column_orders_to_destroy = std::move(column_orders)] {});
  }
}

std::vector<metadata> aggregate_reader_metadata::metadatas_from_sources(
  host_span<std::unique_ptr<datasource> const> sources, bool read_page_indexes)
{
  // Avoid using the thread pool for a single source
  if (sources.size() == 1) {
    std::vector<metadata> result;
    result.emplace_back(sources[0].get(), read_page_indexes);
    return result;
  }

  std::vector<std::future<metadata>> metadata_ctor_tasks;
  metadata_ctor_tasks.reserve(sources.size());
  for (auto const& source : sources) {
    metadata_ctor_tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(
      [source = source.get(), read_page_indexes] { return metadata{source, read_page_indexes}; }));
  }
  std::vector<metadata> metadatas;
  metadatas.reserve(sources.size());
  std::transform(metadata_ctor_tasks.begin(),
                 metadata_ctor_tasks.end(),
                 std::back_inserter(metadatas),
                 [](std::future<metadata>& task) { return std::move(task).get(); });
  return metadatas;
}

std::vector<std::unordered_map<std::string, std::string>>
aggregate_reader_metadata::collect_keyval_metadata() const
{
  std::vector<std::unordered_map<std::string, std::string>> kv_maps;
  std::transform(per_file_metadata.cbegin(),
                 per_file_metadata.cend(),
                 std::back_inserter(kv_maps),
                 [](auto const& pfm) {
                   std::unordered_map<std::string, std::string> kv_map;
                   std::transform(pfm.key_value_metadata.cbegin(),
                                  pfm.key_value_metadata.cend(),
                                  std::inserter(kv_map, kv_map.end()),
                                  [](auto const& kv) { return std::pair{kv.key, kv.value}; });
                   return kv_map;
                 });

  return kv_maps;
}

std::vector<std::unordered_map<int32_t, int32_t>> aggregate_reader_metadata::init_schema_idx_maps(
  bool const has_cols_from_mismatched_srcs) const
{
  // Only initialize if more than 1 data sources and has select columns from mismatched data sources
  if (has_cols_from_mismatched_srcs and per_file_metadata.size() > 1) {
    return std::vector<std::unordered_map<int32_t, int32_t>>{per_file_metadata.size() - 1};
  }

  return {};
}

int64_t aggregate_reader_metadata::calc_num_rows() const
{
  return std::accumulate(
    per_file_metadata.cbegin(), per_file_metadata.cend(), int64_t{0}, [](auto sum, auto& pfm) {
      auto const rowgroup_rows = std::accumulate(
        pfm.row_groups.cbegin(), pfm.row_groups.cend(), int64_t{0}, [](auto rg_sum, auto& rg) {
          return rg_sum + rg.num_rows;
        });
      CUDF_EXPECTS(pfm.num_rows == 0 || pfm.num_rows == rowgroup_rows,
                   "Header and row groups disagree about number of rows in file!");
      return sum + (pfm.num_rows == 0 && rowgroup_rows > 0 ? rowgroup_rows : pfm.num_rows);
    });
}

size_type aggregate_reader_metadata::calc_num_row_groups() const
{
  auto const total_row_groups = std::accumulate(
    per_file_metadata.cbegin(), per_file_metadata.cend(), size_t{0}, [](size_t sum, auto& pfm) {
      return sum + pfm.row_groups.size();
    });

  // Check if we have less than 2B total row groups.
  CUDF_EXPECTS(total_row_groups <= std::numeric_limits<cudf::size_type>::max(),
               "Total number of row groups exceed the size_type's limit");

  return static_cast<size_type>(total_row_groups);
}

std::vector<size_type> aggregate_reader_metadata::get_num_row_groups_per_file() const
{
  std::vector<size_type> per_file_num_row_groups(per_file_metadata.size());
  std::transform(per_file_metadata.begin(),
                 per_file_metadata.end(),
                 per_file_num_row_groups.begin(),
                 [](auto const& pfm) { return pfm.row_groups.size(); });
  return per_file_num_row_groups;
}

std::vector<size_t> aggregate_reader_metadata::compute_source_row_group_offsets(
  size_type src_idx) const
{
  CUDF_EXPECTS(src_idx >= 0 && std::cmp_less(src_idx, per_file_metadata.size()),
               "invalid source index");
  auto const& row_groups = per_file_metadata[src_idx].row_groups;
  std::vector<size_t> source_row_offsets(row_groups.size());
  std::transform_exclusive_scan(row_groups.cbegin(),
                                row_groups.cend(),
                                source_row_offsets.begin(),
                                size_t{0},
                                std::plus<size_t>{},
                                [](auto const& rg) { return rg.num_rows; });
  return source_row_offsets;
}

// Copies info from the column and offset indexes into the passed in row_group_info.
void aggregate_reader_metadata::column_info_for_row_group(row_group_info& rg_info,
                                                          size_t chunk_start_row) const
{
  auto const& fmd = per_file_metadata[rg_info.source_index];
  auto const& rg  = fmd.row_groups[rg_info.index];

  std::vector<column_chunk_info> chunks(rg.columns.size());

  for (size_t col_idx = 0; col_idx < rg.columns.size(); col_idx++) {
    auto const& col_chunk = rg.columns[col_idx];
    auto const is_schema_idx_mapped =
      is_schema_index_mapped(col_chunk.schema_idx, rg_info.source_index);
    auto const mapped_schema_idx = is_schema_idx_mapped
                                     ? map_schema_index(col_chunk.schema_idx, rg_info.source_index)
                                     : col_chunk.schema_idx;
    auto& schema = get_schema(mapped_schema_idx, is_schema_idx_mapped ? rg_info.source_index : 0);
    auto const max_def_level = schema.max_definition_level;
    auto const max_rep_level = schema.max_repetition_level;

    // Skip this column chunk if it does not have an offset index. This is because the decode
    // paths can use column-index-derived information only together with offset index data.
    if (not col_chunk.offset_index.has_value()) { continue; }

    auto const& offset_index = col_chunk.offset_index.value();

    auto& chunk_info     = chunks[col_idx];
    auto const num_pages = offset_index.page_locations.size();

    // There is a bug in older versions of parquet-mr where the first data page offset
    // really points to the dictionary page. The first possible offset in a file is 4 (after
    // the "PAR1" header), so check to see if the dictionary_page_offset is > 0. If it is, then
    // we haven't encountered the bug.
    if (col_chunk.meta_data.dictionary_page_offset > 0) {
      chunk_info.dictionary_offset = col_chunk.meta_data.dictionary_page_offset;
      chunk_info.dictionary_size =
        col_chunk.meta_data.data_page_offset - chunk_info.dictionary_offset.value();
    } else {
      // dictionary_page_offset is 0, so check to see if the data_page_offset does not match
      // the first offset in the offset index.  If they don't match, then data_page_offset points
      // to the dictionary page.
      if (num_pages > 0 &&
          col_chunk.meta_data.data_page_offset < offset_index.page_locations[0].offset) {
        chunk_info.dictionary_offset = col_chunk.meta_data.data_page_offset;
        chunk_info.dictionary_size =
          offset_index.page_locations[0].offset - col_chunk.meta_data.data_page_offset;
      }
    }

    // Populate additional value-count metadata if column index is also available.
    auto const* column_index =
      col_chunk.column_index.has_value() ? &col_chunk.column_index.value() : nullptr;

    // Use the definition_level_histogram to get num_valid and num_null. For now, these are
    // only ever used for byte array columns. The repetition_level_histogram might be
    // necessary to determine the total number of values in the page if the
    // definition_level_histogram is absent.
    //
    // In the future we might want the full histograms saved in the `column_info` struct.
    int64_t const* const def_hist =
      column_index != nullptr and column_index->definition_level_histogram.has_value()
        ? column_index->definition_level_histogram.value().data()
        : nullptr;
    int64_t const* const rep_hist =
      column_index != nullptr and column_index->repetition_level_histogram.has_value()
        ? column_index->repetition_level_histogram.value().data()
        : nullptr;

    for (size_t pg_idx = 0; pg_idx < num_pages; pg_idx++) {
      auto const& page_loc = offset_index.page_locations[pg_idx];
      // translate chunk-relative row nums to absolute within the file
      auto const pg_start_row = chunk_start_row + page_loc.first_row_index;
      auto const pg_end_row =
        chunk_start_row + (pg_idx == (num_pages - 1)
                             ? rg.num_rows
                             : offset_index.page_locations[pg_idx + 1].first_row_index);

      auto const num_rows = static_cast<int64_t>(pg_end_row - pg_start_row);
      page_info pg_info{.location = page_loc, .num_rows = num_rows};

      // check to see if we already have null counts for each page
      if (column_index != nullptr and column_index->null_counts.has_value()) {
        pg_info.num_nulls = column_index->null_counts.value()[pg_idx];
      }

      // save variable length byte info if present
      if (offset_index.unencoded_byte_array_data_bytes.has_value()) {
        pg_info.var_bytes_size = offset_index.unencoded_byte_array_data_bytes.value()[pg_idx];
      }

      // if def histogram is present, then use it to calculate num_valid and num_nulls
      if (def_hist != nullptr) {
        auto const h      = &def_hist[pg_idx * (max_def_level + 1)];
        pg_info.num_valid = h[max_def_level];

        // calculate num_nulls if not available from column index
        if (not pg_info.num_nulls.has_value()) {
          pg_info.num_nulls = std::reduce(h, h + max_def_level);
        }
      }
      // there is no def histogram.
      // if there is no repetition (no lists), then num_values == num_rows, and num_nulls can be
      // obtained from the column index
      else if (max_rep_level == 0) {
        // if we already have num_nulls from column index
        if (pg_info.num_nulls.has_value()) {
          pg_info.num_valid = pg_info.num_rows - pg_info.num_nulls.value();
        }
        // if max_def is 0, there are no nulls
        else if (max_def_level == 0) {
          pg_info.num_nulls = 0;
          pg_info.num_valid = pg_info.num_rows;
        }
      }
      // if the rep level histogram is present, we can get the total number of values
      // from that
      else if (rep_hist != nullptr) {
        if (pg_info.num_nulls.has_value()) {
          auto const h          = &rep_hist[pg_idx * (max_rep_level + 1)];
          auto const num_values = std::reduce(h, h + max_rep_level + 1);
          pg_info.num_valid     = num_values - pg_info.num_nulls.value();
        }
      }

      // If column-index metadata is insufficient to derive all value information, leave those
      // fields unset. Later decoding derives the missing values from page headers and levels, and
      // scans string data when its byte size is unavailable.

      chunk_info.pages.push_back(std::move(pg_info));
    }
  }

  rg_info.column_chunks = std::move(chunks);
}

bool aggregate_reader_metadata::has_offset_index(
  std::span<row_group_info const> row_groups,
  std::span<input_column_info const> input_columns) const
{
  for (auto const& rg_info : row_groups) {
    auto const& row_group = per_file_metadata[rg_info.source_index].row_groups[rg_info.index];
    for (auto const& input_column : input_columns) {
      auto const schema_idx      = map_schema_index(input_column.schema_idx, rg_info.source_index);
      auto const colchunk_offset = find_colchunk_iter_offset(row_group, schema_idx);
      if (not row_group.columns[colchunk_offset].offset_index.has_value()) { return false; }
    }
  }
  return true;
}

void aggregate_reader_metadata::initialize_internals(bool use_arrow_schema,
                                                     bool has_cols_from_mismatched_srcs)
{
  if (per_file_metadata.size() > 1) {
    auto& first_meta = per_file_metadata.front();
    auto const first_nonempty_source_iter =
      std::find_if(per_file_metadata.begin(), per_file_metadata.end(), [&](auto const& pfm) {
        return pfm.row_groups.size() > 0;
      });
    auto const num_cols = first_nonempty_source_iter != per_file_metadata.end()
                            ? first_nonempty_source_iter->row_groups.front().columns.size()
                            : 0;
    auto& schema        = first_meta.schema;

    // Validate that all sources have the same schema unless we are reading select columns
    // from mismatched sources, in which case, we will only check the projected columns later.
    if (not has_cols_from_mismatched_srcs) {
      // Verify that the input files have matching numbers of columns and schema.
      for (auto const& pfm : per_file_metadata) {
        if (pfm.row_groups.size() > 0) {
          CUDF_EXPECTS(pfm.row_groups.empty() or num_cols == pfm.row_groups.front().columns.size(),
                       "All non-empty sources must have the same number of columns");
        }
        CUDF_EXPECTS(schema == pfm.schema, "All sources must have the same schema");
      }
    }

    // Mark the column schema in the first (default) source as nullable if it is nullable in any of
    // the input sources. This avoids recomputing this within build_column() and
    // populate_metadata().
    std::for_each(
      cuda::counting_iterator{static_cast<size_t>(1)},
      cuda::counting_iterator{schema.size()},
      [&](auto const schema_idx) {
        if (schema[schema_idx].repetition_type == FieldRepetitionType::REQUIRED and
            std::any_of(
              per_file_metadata.begin() + 1, per_file_metadata.end(), [&](auto const& pfm) {
                return pfm.schema[schema_idx].repetition_type != FieldRepetitionType::REQUIRED;
              })) {
          schema[schema_idx].repetition_type = FieldRepetitionType::OPTIONAL;
        }
      });
  }

  // Collect and apply arrow:schema from Parquet's key value metadata section
  if (use_arrow_schema) { apply_arrow_schema(); }

  // Erase ARROW_SCHEMA_KEY from the output pfm if exists
  std::for_each(
    keyval_maps.begin(), keyval_maps.end(), [](auto& pfm) { pfm.erase(ARROW_SCHEMA_KEY); });
}

aggregate_reader_metadata::aggregate_reader_metadata(std::vector<FileMetaData>&& parquet_metadatas,
                                                     bool use_arrow_schema,
                                                     bool has_cols_from_mismatched_srcs)
{
  per_file_metadata.reserve(parquet_metadatas.size());
  std::transform(std::make_move_iterator(parquet_metadatas.begin()),
                 std::make_move_iterator(parquet_metadatas.end()),
                 std::back_inserter(per_file_metadata),
                 [](FileMetaData&& meta) { return metadata{std::move(meta)}; });

  keyval_maps     = collect_keyval_metadata();
  schema_idx_maps = init_schema_idx_maps(has_cols_from_mismatched_srcs);
  num_rows        = calc_num_rows();
  num_row_groups  = calc_num_row_groups();

  initialize_internals(use_arrow_schema, has_cols_from_mismatched_srcs);
}

aggregate_reader_metadata::aggregate_reader_metadata(
  host_span<std::unique_ptr<datasource> const> sources,
  bool use_arrow_schema,
  bool has_cols_from_mismatched_srcs,
  bool read_page_indexes)
  : per_file_metadata(metadatas_from_sources(sources, read_page_indexes)),
    keyval_maps(collect_keyval_metadata()),
    schema_idx_maps(init_schema_idx_maps(has_cols_from_mismatched_srcs)),
    num_rows(calc_num_rows()),
    num_row_groups(calc_num_row_groups())
{
  initialize_internals(use_arrow_schema, has_cols_from_mismatched_srcs);
}

arrow_schema_data_types aggregate_reader_metadata::collect_arrow_schema() const
{
  // Check the key_value metadata for arrow schema, decode and walk it
  // Function to convert from flatbuf::duration type to cudf::type_id
  auto const duration_from_flatbuffer = [](flatbuf::Duration const* duration) {
    // TODO: we only need this for arrow::DurationType for now. Else, we can take in a
    // void ptr and typecast it to the corresponding type based on the type_id.
    auto fb_unit = duration->unit();
    switch (fb_unit) {
      case flatbuf::TimeUnit::TimeUnit_SECOND:
        return cudf::data_type{cudf::type_id::DURATION_SECONDS};
      case flatbuf::TimeUnit::TimeUnit_MILLISECOND:
        return cudf::data_type{cudf::type_id::DURATION_MILLISECONDS};
      case flatbuf::TimeUnit::TimeUnit_MICROSECOND:
        return cudf::data_type{cudf::type_id::DURATION_MICROSECONDS};
      case flatbuf::TimeUnit::TimeUnit_NANOSECOND:
        return cudf::data_type{cudf::type_id::DURATION_NANOSECONDS};
      default: return cudf::data_type{};
    }
  };

  // variable that tracks if an arrow_type specific column is seen
  // in the walk
  bool arrow_type_col_seen = false;

  // Lambda function to walk a field and its children in DFS manner and
  // return boolean walk success status
  std::function<bool(flatbuf::Field const* const, arrow_schema_data_types&)> walk_field =
    [&walk_field, &duration_from_flatbuffer, &arrow_type_col_seen](
      flatbuf::Field const* const field, arrow_schema_data_types& schema_elem) {
      // DFS: recursively walk over the children first
      auto const field_children = field->children();

      if (field_children != nullptr) {
        auto schema_children = std::vector<arrow_schema_data_types>(field->children()->size());

        if (not std::all_of(cuda::counting_iterator<int32_t>{0},
                            cuda::counting_iterator{static_cast<int32_t>(field_children->size())},
                            [&](auto const& idx) {
                              return walk_field((*field_children)[idx], schema_children[idx]);
                            })) {
          return false;
        }
        // arrow and parquet schemas are structured slightly differently for list type fields. list
        // type fields in arrow are structured as: "field:list<element>" vs structured as:
        // "field:list.element" in Parquet. To handle this, whenever we encounter a list type field,
        // we add a dummy node "field.list" to the end of current children and move the current
        // children (".element") to it.
        switch (field->type_type()) {
          case flatbuf::Type::Type_List:
          case flatbuf::Type::Type_LargeList:
          case flatbuf::Type::Type_FixedSizeList:
            schema_elem.children.emplace_back(arrow_schema_data_types{std::move(schema_children)});
            break;
          default: schema_elem.children = std::move(schema_children); break;
        }
      }

      // Walk the field itself
      if (field->type_type() == flatbuf::Type::Type_Duration) {
        auto type_data = field->type_as_Duration();
        if (type_data != nullptr) {
          auto name = field->name() ? field->name()->str() : "";
          // set the schema_elem type to duration type
          schema_elem.type = duration_from_flatbuffer(type_data);
          arrow_type_col_seen |= (schema_elem.type.id() != type_id::EMPTY);
        } else {
          CUDF_LOG_ERROR("Parquet reader encountered an invalid type_data pointer.",
                         "arrow:schema not processed.");
          return false;
        }
      }
      return true;
    };

  auto const it = keyval_maps[0].find(ARROW_SCHEMA_KEY);
  if (it == keyval_maps[0].end()) { return {}; }

  // Decode the base64 encoded ipc message string
  // Note: Store the output from base64_decode in the lvalue here and then pass
  // it to decode_ipc_message. Directly passing rvalue from base64_decode to
  // decode_ipc_message can lead to unintended nullptr dereferences.
  auto const decoded_message = cudf::io::detail::base64_decode(it->second);

  // Decode the ipc message to get an optional string_view of the ipc:Message flatbuffer
  auto const metadata_buf = decode_ipc_message(decoded_message);

  // Check if the string_view exists
  if (not metadata_buf.has_value()) {
    // No need to re-log error here as already logged inside decode_ipc_message
    return {};
  }

  // Check if the decoded Message flatbuffer is valid
  if (flatbuf::GetMessage(metadata_buf.value().data()) == nullptr) {
    CUDF_LOG_ERROR("Parquet reader encountered an invalid ipc:Message flatbuffer pointer.",
                   "arrow:schema not processed.");
    return {};
  }

  // Check if the Message flatbuffer has a valid arrow:schema in its header
  if (flatbuf::GetMessage(metadata_buf.value().data())->header_as_Schema() == nullptr) {
    CUDF_LOG_ERROR("Parquet reader encountered an invalid arrow:schema flatbuffer pointer.",
                   "arrow:schema not processed.");
    return {};
  }

  // Get the vector of fields from arrow:schema flatbuffer object
  auto const fields =
    flatbuf::GetMessage(metadata_buf.value().data())->header_as_Schema()->fields();
  if (fields == nullptr) {
    CUDF_LOG_ERROR("Parquet reader encountered an invalid fields pointer.",
                   "arrow:schema not processed.");
    return {};
  }

  // arrow schema structure to return
  arrow_schema_data_types schema;

  // Recursively walk the arrow schema and set cudf::data_type for all duration columns
  if (fields->size() > 0) {
    schema.children = std::vector<arrow_schema_data_types>(fields->size());

    if (not std::all_of(
          cuda::counting_iterator<int32_t>{0},
          cuda::counting_iterator{static_cast<int32_t>(fields->size())},
          [&](auto const& idx) { return walk_field((*fields)[idx], schema.children[idx]); })) {
      return {};
    }

    // if no arrow type column seen, return nullopt.
    if (not arrow_type_col_seen) { return {}; }
  }

  return schema;
}

void aggregate_reader_metadata::apply_arrow_schema()
{
  // Collect the arrow schema from the key value section of Parquet metadata
  auto arrow_schema_root = collect_arrow_schema();

  // Check if empty arrow schema collected
  if (arrow_schema_root.type.id() == type_id::EMPTY and arrow_schema_root.children.size() == 0) {
    return;
  }

  // Function to verify equal num_children at each level in Parquet and arrow schemas.
  std::function<bool(arrow_schema_data_types const&, int const)> validate_schemas =
    [&](arrow_schema_data_types const& arrow_schema, int const schema_idx) {
      auto& pq_schema_elem = per_file_metadata[0].schema[schema_idx];

      // ensure equal number of children first to avoid any segfaults in children
      if (std::cmp_equal(pq_schema_elem.num_children, arrow_schema.children.size())) {
        // true if and only if true for all children as well
        return std::all_of(
          thrust::make_zip_iterator(cuda::std::make_tuple(arrow_schema.children.begin(),
                                                          pq_schema_elem.children_idx.begin())),
          thrust::make_zip_iterator(
            cuda::std::make_tuple(arrow_schema.children.end(), pq_schema_elem.children_idx.end())),
          [&](auto const& elem) {
            return validate_schemas(cuda::std::get<0>(elem), cuda::std::get<1>(elem));
          });
      } else {
        return false;
      }
    };

  // Function to co-walk arrow and parquet schemas
  std::function<void(arrow_schema_data_types const&, int const)> co_walk_schemas =
    [&](arrow_schema_data_types const& arrow_schema, int const schema_idx) {
      auto& pq_schema_elem = per_file_metadata[0].schema[schema_idx];
      std::for_each(thrust::make_zip_iterator(cuda::std::make_tuple(
                      arrow_schema.children.begin(), pq_schema_elem.children_idx.begin())),
                    thrust::make_zip_iterator(cuda::std::make_tuple(
                      arrow_schema.children.end(), pq_schema_elem.children_idx.end())),
                    [&](auto const& elem) {
                      co_walk_schemas(cuda::std::get<0>(elem), cuda::std::get<1>(elem));
                    });

      // true for DurationType columns only for now.
      if (arrow_schema.type.id() != type_id::EMPTY) {
        pq_schema_elem.arrow_type = arrow_schema.type.id();
      }
    };

  // Get Parquet schema root
  auto pq_schema_root = get_schema(0);

  // verify equal number of children for both schemas at root level
  if (std::cmp_not_equal(pq_schema_root.num_children, arrow_schema_root.children.size())) {
    CUDF_LOG_ERROR("Parquet reader encountered a mismatch between Parquet and arrow schema.",
                   "arrow:schema not processed.");
    return;
  }

  // zip iterator to validate and co-walk the two schemas
  auto schemas = thrust::make_zip_iterator(
    cuda::std::make_tuple(arrow_schema_root.children.begin(), pq_schema_root.children_idx.begin()));

  // Verify equal number of children at all sub-levels
  if (not std::all_of(schemas, schemas + pq_schema_root.num_children, [&](auto const& elem) {
        return validate_schemas(cuda::std::get<0>(elem), cuda::std::get<1>(elem));
      })) {
    CUDF_LOG_ERROR("Parquet reader encountered a mismatch between Parquet and arrow schema.",
                   "arrow:schema not processed.");
    return;
  }

  // All good, now co-walk schemas
  std::for_each(schemas, schemas + pq_schema_root.num_children, [&](auto const& elem) {
    co_walk_schemas(cuda::std::get<0>(elem), cuda::std::get<1>(elem));
  });
}

std::optional<std::string_view> aggregate_reader_metadata::decode_ipc_message(
  std::string_view const serialized_message) const
{
  // message buffer
  auto message_buf = serialized_message.data();
  // current message (buffer) size
  auto message_size = static_cast<int32_t>(serialized_message.size());

  // Lambda function to read and return 4 bytes as int32_t from the ipc message buffer and update
  // buffer pointer and size
  auto read_int32_from_ipc_message = [&]() {
    int32_t bytes = 0;
    std::memcpy(&bytes, message_buf, sizeof(int32_t));
    // Offset the message buf and reduce remaining size
    message_buf += sizeof(int32_t);
    message_size -= sizeof(int32_t);
    return bytes;
  };

  // Check for empty message
  if (message_size == 0) {
    CUDF_LOG_ERROR("Parquet reader encountered zero length arrow:schema.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // Check for improper message size.
  if (message_size < MESSAGE_DECODER_NEXT_REQUIRED_SIZE_INITIAL) {
    CUDF_LOG_ERROR("Parquet reader encountered unexpected arrow:schema message length.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // Get the first 4 bytes (continuation) of the ipc message
  // and check if it matches the expected token
  if (read_int32_from_ipc_message() != IPC_CONTINUATION_TOKEN) {
    CUDF_LOG_ERROR("Parquet reader encountered unexpected IPC continuation token.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // Check for improper message size after the continuation bytes.
  if (message_size < MESSAGE_DECODER_NEXT_REQUIRED_SIZE_METADATA_LENGTH) {
    CUDF_LOG_ERROR("Parquet reader encountered unexpected arrow:schema message length.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // Get the next 4 bytes (metadata_len) of the ipc message
  // and check if invalid metadata length read
  auto const metadata_len = read_int32_from_ipc_message();

  // Check if the read metadata (header) length is > zero
  if (metadata_len <= 0) {
    CUDF_LOG_ERROR("Parquet reader encountered unexpected metadata length.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // Check if the remaining message size is smaller than the expected metadata length
  // TODO: Since the arrow:schema message doesn't have a body,
  // the following check may be made tighter from < to ==
  if (message_size < metadata_len) {
    CUDF_LOG_ERROR("Parquet reader encountered unexpected arrow:schema message length.",
                   "arrow:schema not processed.");
    return std::nullopt;
  }

  // All good, return the current message_buf as string_view
  return std::string_view{message_buf,
                          static_cast<std::basic_string_view<char>::size_type>(message_size)};
}

RowGroup const& aggregate_reader_metadata::get_row_group(size_type row_group_index,
                                                         size_type src_idx) const
{
  CUDF_EXPECTS(src_idx >= 0 && std::cmp_less(src_idx, per_file_metadata.size()),
               "invalid source index");
  return per_file_metadata[src_idx].row_groups[row_group_index];
}

row_group_size_info aggregate_reader_metadata::get_row_group_size_info(
  size_type row_group_index,
  size_type src_idx,
  std::optional<std::span<input_column_info const>> input_columns) const
{
  auto const& row_group = get_row_group(row_group_index, src_idx);

  CUDF_EXPECTS(row_group.num_rows >= 0 && std::in_range<size_t>(row_group.num_rows),
               "Row group has an invalid number of rows",
               std::invalid_argument);
  auto size_info =
    row_group_size_info{.unadjusted_num_rows = static_cast<size_t>(row_group.num_rows)};

  // Helper function to overflow-safeadd compressed sizes
  auto const add_compressed_size = [&](auto&& current_size, auto&& colchunk_compressed_size) {
    auto const sum = cuda::add_overflow<size_t>(current_size, colchunk_compressed_size);
    CUDF_EXPECTS(not sum.overflow,
                 "Row group compressed size exceeds the supported range",
                 std::overflow_error);
    return sum.value;
  };

  if (input_columns.has_value()) {
    for (auto const& column : *input_columns) {
      auto const& column_metadata =
        get_column_metadata(row_group_index, src_idx, column.schema_idx);
      size_info.compressed_size =
        add_compressed_size(size_info.compressed_size, column_metadata.total_compressed_size);
      size_info.max_leaf_values =
        std::max<size_t>(size_info.max_leaf_values, column_metadata.num_values);
    }
  } else {
    for (auto const& column_chunk : row_group.columns) {
      size_info.compressed_size = add_compressed_size(size_info.compressed_size,
                                                      column_chunk.meta_data.total_compressed_size);
      size_info.max_leaf_values =
        std::max<size_t>(size_info.max_leaf_values, column_chunk.meta_data.num_values);
    }
  }

  return size_info;
}

ColumnChunkMetaData const& aggregate_reader_metadata::get_column_metadata(size_type row_group_index,
                                                                          size_type src_idx,
                                                                          int schema_idx) const
{
  // Map schema index to the provided source file index
  schema_idx = map_schema_index(schema_idx, src_idx);

  auto const& row_group      = per_file_metadata[src_idx].row_groups[row_group_index];
  auto const colchunk_offset = find_colchunk_iter_offset(row_group, schema_idx);
  return row_group.columns[colchunk_offset].meta_data;
}

std::vector<std::unordered_map<std::string, int64_t>>
aggregate_reader_metadata::get_rowgroup_metadata() const
{
  std::vector<std::unordered_map<std::string, int64_t>> rg_metadata;

  std::for_each(
    per_file_metadata.cbegin(), per_file_metadata.cend(), [&rg_metadata](auto const& pfm) {
      std::transform(pfm.row_groups.cbegin(),
                     pfm.row_groups.cend(),
                     std::back_inserter(rg_metadata),
                     [](auto const& rg) {
                       std::unordered_map<std::string, int64_t> rg_meta_map;
                       rg_meta_map["num_rows"]        = rg.num_rows;
                       rg_meta_map["total_byte_size"] = rg.total_byte_size;
                       return rg_meta_map;
                     });
    });
  return rg_metadata;
}

std::unordered_map<std::string, std::vector<int64_t>>
aggregate_reader_metadata::get_column_chunk_metadata() const
{
  std::unordered_map<std::string, std::vector<int64_t>> column_chunk_metadata;

  std::function<void(std::string const&, int)> const populate_column_chunk_metadata =
    [&](std::string const& path_till_now, int schema_idx) {
      auto const& schema = get_schema(schema_idx);
      auto const path_in_schema =
        path_till_now.empty() ? schema.name : path_till_now + "." + schema.name;

      // If this is not a leaf column, keep traversing the schema tree
      if (not schema.children_idx.empty()) {
        for (auto const& child_idx : schema.children_idx) {
          populate_column_chunk_metadata(path_in_schema, child_idx);
        }
        return;
      }

      // Otherwise, if this is a leaf column, collect its `total_uncompressed_size` fields from
      // all column chunks and add to the `column_chunk_metadata` map.
      auto total_uncompressed_sizes = std::vector<int64_t>{};
      total_uncompressed_sizes.reserve(num_row_groups);
      // For each input source
      std::for_each(cuda::counting_iterator<size_t>{0},
                    cuda::counting_iterator{per_file_metadata.size()},
                    [&](auto const& src_idx) {
                      auto const& file_metadata = per_file_metadata[src_idx];
                      // For each row group in this source
                      std::transform(cuda::counting_iterator<size_t>{0},
                                     cuda::counting_iterator{file_metadata.row_groups.size()},
                                     std::back_inserter(total_uncompressed_sizes),
                                     [&](auto const& row_group_idx) {
                                       // Return `total_uncompressed_size` of this column's chunk
                                       return get_column_metadata(
                                                row_group_idx, src_idx, schema_idx)
                                         .total_uncompressed_size;
                                     });
                    });
      // Check if this column already exists in the map
      CUDF_EXPECTS(column_chunk_metadata.find(path_in_schema) == column_chunk_metadata.end(),
                   "Encountered an already mapped leaf column in the schema tree" + path_in_schema,
                   std::invalid_argument);
      // Map the collected metadata fields for this column to its named path
      column_chunk_metadata[path_in_schema] = total_uncompressed_sizes;
    };

  // Traverse the schema tree and populate `ColumnChunkMetaData`s for all leaf columns.
  auto const& root = get_schema(0);
  for (auto const& child_idx : root.children_idx) {
    populate_column_chunk_metadata("", child_idx);
  }

  return column_chunk_metadata;
}

bool aggregate_reader_metadata::is_schema_index_mapped(int schema_idx, int src_idx) const
{
  // Check if schema_idx or src_idx is invalid
  CUDF_EXPECTS(
    schema_idx >= 0 and src_idx >= 0 and std::cmp_less(src_idx, per_file_metadata.size()),
    "Parquet reader encountered an invalid schema_idx or src_idx",
    std::out_of_range);

  // True if root index requested or zeroth file index or schema_idx maps doesn't exist. (i.e.
  // schemas are identical).
  if (schema_idx == 0 or src_idx == 0 or schema_idx_maps.empty()) { return true; }

  // Check if mapped
  auto const& schema_idx_map = schema_idx_maps[src_idx - 1];
  return schema_idx_map.find(schema_idx) != schema_idx_map.end();
}

int aggregate_reader_metadata::map_schema_index(int schema_idx, int src_idx) const
{
  // Check if schema_idx or src_idx is invalid
  CUDF_EXPECTS(
    schema_idx >= 0 and src_idx >= 0 and std::cmp_less(src_idx, per_file_metadata.size()),
    "Parquet reader encountered an invalid schema_idx or src_idx",
    std::out_of_range);

  // Check if src_idx is zero or root index requested or schema_idx_maps doesn't exist (i.e.
  // schemas are identical).
  if (schema_idx == 0 or src_idx == 0 or schema_idx_maps.empty()) { return schema_idx; }

  // schema_idx_maps will only have > 0 size when we are reading matching column projection from
  // mismatched Parquet sources.
  auto const& schema_idx_map = schema_idx_maps[src_idx - 1];
  CUDF_EXPECTS(schema_idx_map.find(schema_idx) != schema_idx_map.end(),
               "Unmapped schema index encountered in the specified source tree",
               std::out_of_range);

  // Return the mapped schema idx.
  return schema_idx_map.at(schema_idx);
}

std::string aggregate_reader_metadata::get_pandas_index() const
{
  // Assumes that all input files have the same metadata
  // TODO: verify this assumption
  auto it = keyval_maps[0].find("pandas");
  if (it != keyval_maps[0].end()) {
    // Captures a list of quoted strings found inside square brackets after `"index_columns":`
    // Inside quotes supports newlines, brackets, escaped quotes, etc.
    // One-liner regex:
    // "index_columns"\s*:\s*\[\s*((?:"(?:|(?:.*?(?![^\\]")).?)[^\\]?",?\s*)*)\]
    // Documented below.
    std::regex const index_columns_expr{
      R"("index_columns"\s*:\s*\[\s*)"  // match preamble, opening square bracket, whitespace
      R"(()"                            // Open first capturing group
      R"((?:")"                         // Open non-capturing group match opening quote
      R"((?:|(?:.*?(?![^\\]")).?))"     // match empty string or anything between quotes
      R"([^\\]?")"                      // Match closing non-escaped quote
      R"(,?\s*)"                        // Match optional comma and whitespace
      R"()*)"                           // Close non-capturing group and repeat 0 or more times
      R"())"                            // Close first capturing group
      R"(\])"                           // Match closing square brackets
    };
    std::smatch sm;
    if (std::regex_search(it->second, sm, index_columns_expr)) { return sm[1].str(); }
  }
  return "";
}

std::vector<std::string> aggregate_reader_metadata::get_pandas_index_names() const
{
  std::vector<std::string> names;
  auto str = get_pandas_index();
  if (str.length() != 0) {
    std::regex const index_name_expr{R"(\"((?:\\.|[^\"])*)\")"};
    std::smatch sm;
    while (std::regex_search(str, sm, index_name_expr)) {
      if (sm.size() == 2) {  // 2 = whole match, first item
        if (std::find(names.begin(), names.end(), sm[1].str()) == names.end()) {
          std::regex const esc_quote{R"(\\")"};
          names.emplace_back(std::regex_replace(sm[1].str(), esc_quote, R"(")"));
        }
      }
      str = sm.suffix();
    }
  }
  return names;
}

std::tuple<int64_t,
           std::vector<std::vector<size_type>>,
           std::vector<std::vector<size_t>>,
           std::vector<std::vector<size_t>>>
aggregate_reader_metadata::apply_row_bounds_filter(
  host_span<std::vector<size_type> const> input_row_group_indices,
  int64_t rows_to_skip,
  int64_t rows_to_read) const
{
  auto const num_sources = input_row_group_indices.size();

  auto filtered_row_group_indices = std::vector<std::vector<size_type>>(num_sources);
  auto row_group_row_counts       = std::vector<std::vector<size_t>>{};
  row_group_row_counts.reserve(num_sources);
  auto row_group_row_offsets = std::vector<std::vector<size_t>>{};
  row_group_row_offsets.reserve(num_sources);

  size_t current_row_index = 0;

  // Rows to skip relative to the row offset of the first surviving row group index
  int64_t first_row_group_relative_rows_to_skip = 0;

  // For each data source
  std::for_each(
    cuda::counting_iterator<size_t>{0},
    cuda::counting_iterator{num_sources},
    [&](auto const& src_idx) {
      auto const& file_metadata = per_file_metadata[src_idx];
      auto const num_row_groups = input_row_group_indices[src_idx].size();

      // Fill row group rows counts and row offsets with zeros
      row_group_row_counts.emplace_back(num_row_groups, size_type{0});
      row_group_row_offsets.emplace_back(num_row_groups, current_row_index);

      // Return early if we have reached the maximum number of rows
      if (std::cmp_greater_equal(current_row_index, rows_to_skip + rows_to_read)) { return; }

      // For each row group in this data source
      std::for_each(
        input_row_group_indices[src_idx].begin(),
        input_row_group_indices[src_idx].end(),
        [&](auto const& rg_idx) {
          // Return early if we have reached the maximum number of rows to read
          if (std::cmp_greater_equal(current_row_index, rows_to_skip + rows_to_read)) { return; }

          auto const& rg             = file_metadata.row_groups[rg_idx];
          auto const chunk_start_row = current_row_index;
          current_row_index += rg.num_rows;
          row_group_row_offsets[src_idx][rg_idx] = chunk_start_row;
          if (std::cmp_greater(current_row_index, rows_to_skip) or current_row_index == 0) {
            // Check if this is the first surviving row group
            if (std::cmp_less_equal(chunk_start_row, rows_to_skip)) {
              // Set the relative number of rows to skip relative to this row group's
              // offset
              first_row_group_relative_rows_to_skip = rows_to_skip - chunk_start_row;
              row_group_row_counts[src_idx][rg_idx] = current_row_index - rows_to_skip;
            } else {
              row_group_row_counts[src_idx][rg_idx] = rg.num_rows;
            }

            filtered_row_group_indices[src_idx].emplace_back(rg_idx);
          }
          // Adjust the number of rows for the last source file.
          if (std::cmp_greater_equal(current_row_index, rows_to_skip + rows_to_read)) {
            row_group_row_counts[src_idx][rg_idx] -=
              current_row_index - rows_to_skip - rows_to_read;
          }
        });
    });

  return std::tuple{first_row_group_relative_rows_to_skip,
                    std::move(filtered_row_group_indices),
                    std::move(row_group_row_counts),
                    std::move(row_group_row_offsets)};
}

std::vector<std::vector<size_type>> aggregate_reader_metadata::apply_byte_bounds_filter(
  host_span<std::vector<size_type> const> input_row_group_indices,
  size_t bytes_to_skip,
  std::optional<size_t> const& bytes_to_read) const
{
  CUDF_EXPECTS(input_row_group_indices.size() == 1,
               "Byte bounds filter can only be applied to a single source",
               std::invalid_argument);

  auto filtered_row_group_indices =
    std::vector<std::vector<size_type>>(input_row_group_indices.size());

  std::for_each(input_row_group_indices.front().begin(),
                input_row_group_indices.front().end(),
                [&](auto const& rg_idx) {
                  // Get the file offset of this row group
                  auto const row_group_file_offset = [&]() {
                    auto const& rg = per_file_metadata.front().row_groups[rg_idx];
                    if (rg.file_offset.has_value()) {
                      return rg.file_offset.value();
                    } else if (rg.columns.front().file_offset != 0) {
                      return rg.columns.front().file_offset;
                    } else {
                      auto const& col_meta = rg.columns.front().meta_data;
                      return col_meta.dictionary_page_offset != 0
                               ? std::min(col_meta.dictionary_page_offset,
                                          col_meta.data_page_offset)
                               : col_meta.data_page_offset;
                    }
                  }();

                  // Check if the row group starts within the byte range: row group file offset is
                  // >= bytes_to_skip AND (bytes_to_read is not specified OR the max byte offset
                  // overflows size_t OR row group file offset is < bytes_to_skip + bytes_to_read)
                  auto const is_within_byte_range =
                    std::cmp_greater_equal(row_group_file_offset, bytes_to_skip) and
                    (not bytes_to_read.has_value() or
                     cuda::add_overflow<size_t>(bytes_to_skip, bytes_to_read.value()).overflow or
                     std::cmp_less(row_group_file_offset, bytes_to_skip + bytes_to_read.value()));

                  if (is_within_byte_range) {
                    filtered_row_group_indices.front().emplace_back(rg_idx);
                  }
                });

  return filtered_row_group_indices;
}

std::tuple<int64_t,
           size_t,
           std::vector<row_group_info>,
           std::vector<size_t>,
           size_type,
           surviving_row_group_metrics>
aggregate_reader_metadata::select_row_groups(
  host_span<std::unique_ptr<datasource> const> sources,
  host_span<std::vector<size_type> const> row_group_indices,
  int64_t skip_rows_opt,
  std::optional<int64_t> const& num_rows_opt,
  size_t skip_bytes_opt,
  std::optional<size_t> const& byte_count_opt,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter,
  rmm::cuda_stream_view stream) const
{
  // Input row group indices must be either empty or equal to the number of data sources
  CUDF_EXPECTS(row_group_indices.empty() or row_group_indices.size() == per_file_metadata.size(),
               "Encountered unexpected number of input row group indices",
               std::invalid_argument);

  // Maximum number of rows in all row groups
  auto const max_num_rows = get_num_rows();

  // Compute the row bounds if no row group indices are provided, else return zeros
  auto [rows_to_skip, rows_to_read] = [&]() {
    if (not row_group_indices.empty()) { return std::pair<int64_t, size_t>{}; }
    auto const from_opts =
      cudf::io::detail::skip_rows_num_rows_from_options(skip_rows_opt, num_rows_opt, max_num_rows);
    return std::pair{static_cast<int64_t>(from_opts.first), static_cast<size_t>(from_opts.second)};
  }();

  // If there are no input row groups specified and zero number of rows to read, return empty
  // selection
  if (row_group_indices.empty() and rows_to_read == 0) {
    auto constexpr total_row_bounds_filtered_row_groups = 0;
    // If filter is not empty, then set the number of row groups after filters to 0, else nullopt
    auto const num_filtered_row_groups =
      filter.has_value() ? std::make_optional(total_row_bounds_filtered_row_groups) : std::nullopt;
    return {rows_to_skip,
            rows_to_read,
            {},
            std::vector<size_t>(per_file_metadata.size(), 0),
            total_row_bounds_filtered_row_groups,
            {num_filtered_row_groups, num_filtered_row_groups}};
  }

  // Variable to rows to skip relative to the row offset of the first surviving row group index
  auto first_row_group_relative_rows_to_skip = rows_to_skip;

  // Vector to store all row group indices across all sources
  std::vector<std::vector<size_type>> all_row_group_indices;

  // Span to track the current row group indices
  host_span<std::vector<size_type> const> current_row_group_indices;

  // If input row group indices are not specified, populate the vector of all row group indices
  if (row_group_indices.empty()) {
    all_row_group_indices.reserve(per_file_metadata.size());
    std::transform(per_file_metadata.cbegin(),
                   per_file_metadata.cend(),
                   std::back_inserter(all_row_group_indices),
                   [](auto const& file_meta) {
                     std::vector<size_type> rg_idx(file_meta.row_groups.size());
                     std::iota(rg_idx.begin(), rg_idx.end(), 0);
                     return rg_idx;
                   });

    // Set the current span of row group indices to the vector of all row group indices
    current_row_group_indices = host_span<std::vector<size_type> const>(all_row_group_indices);
  }
  // Otherwise, set the current span of row group indices to the specified input row group indices
  else {
    current_row_group_indices = row_group_indices;
  }

  // Compute total number of input row groups
  size_type const total_input_row_groups = row_group_indices.empty()
                                             ? num_row_groups  // Use pre-computed member variable
                                             : compute_total_row_groups(current_row_group_indices);

  // Flag to check if the row groups will be filtered using row bounds
  bool const is_row_bounded_row_groups =
    row_group_indices.empty() and (rows_to_skip > 0 or std::cmp_less(rows_to_read, max_num_rows));

  // Use row bounds to filter row group indices if possible
  std::vector<std::vector<size_type>> trimmed_row_group_indices;
  std::vector<std::vector<size_t>> row_group_row_counts;
  std::vector<std::vector<size_t>> row_group_row_offsets;

  if (is_row_bounded_row_groups) {
    std::tie(first_row_group_relative_rows_to_skip,
             trimmed_row_group_indices,
             row_group_row_counts,
             row_group_row_offsets) =
      apply_row_bounds_filter(current_row_group_indices, rows_to_skip, rows_to_read);

    // Update the current span of row group indices
    current_row_group_indices = host_span<std::vector<size_type> const>(trimmed_row_group_indices);
  }

  // Flag to check if the row groups will be filtered using byte bounds
  bool const is_byte_bounded_row_groups =
    row_group_indices.empty() and (skip_bytes_opt > 0 or byte_count_opt.has_value());

  // We can't filter with both row bounds and byte bounds
  CUDF_EXPECTS(not(is_row_bounded_row_groups and is_byte_bounded_row_groups),
               "Byte bounds filter can only be applied to a single source",
               std::invalid_argument);

  if (is_byte_bounded_row_groups) {
    // Byte bounds don't skip rows within row groups
    first_row_group_relative_rows_to_skip = 0;
    trimmed_row_group_indices =
      apply_byte_bounds_filter(current_row_group_indices, skip_bytes_opt, byte_count_opt);

    // Update the current span of row group indices
    current_row_group_indices = host_span<std::vector<size_type> const>(trimmed_row_group_indices);
  }

  // Compute number of input row groups after row or byte bounds trimming
  auto const total_trimmed_row_groups = (is_row_bounded_row_groups or is_byte_bounded_row_groups)
                                          ? compute_total_row_groups(current_row_group_indices)
                                          : total_input_row_groups;

  // Struct to store the number of row groups after filtering using stats and bloom filters
  surviving_row_group_metrics num_row_groups_after_filters{};

  // Vector to store the stats and bloom filtered row group indices
  std::optional<std::vector<std::vector<size_type>>> filtered_row_group_indices;

  if (filter.has_value()) {
    std::tie(filtered_row_group_indices, num_row_groups_after_filters) =
      filter_row_groups(sources,
                        current_row_group_indices,
                        total_trimmed_row_groups,
                        output_dtypes,
                        output_column_schemas,
                        filter.value(),
                        stream);

    // If row groups were filtered, update the current span of row group indices as well as the
    // rows to skip relative to the first surviving row group
    if (filtered_row_group_indices.has_value()) {
      // Update the current span of row group indices
      current_row_group_indices =
        host_span<std::vector<size_type> const>(filtered_row_group_indices.value());

      // Only need to update the rows to skip relative to the first surviving row group
      // if row bounds were previously applied
      if (is_row_bounded_row_groups) {
        // Find the source index of the first non-empty row group
        auto const first_non_empty_source_idx =
          std::distance(current_row_group_indices.begin(),
                        std::find_if(current_row_group_indices.begin(),
                                     current_row_group_indices.end(),
                                     [](auto const& indices) { return not indices.empty(); }));

        // Update the rows to skip relative to the first surviving row group
        if (std::cmp_less(first_non_empty_source_idx, current_row_group_indices.size())) {
          // Row group index of the first element in the first non-empty source
          auto const first_non_empty_row_group_idx =
            current_row_group_indices[first_non_empty_source_idx].front();

          first_row_group_relative_rows_to_skip = std::max<int64_t>(
            0,
            rows_to_skip -
              row_group_row_offsets[first_non_empty_source_idx][first_non_empty_row_group_idx]);
        }
      }
    }
  }

  // Vector to hold the `row_group_info`s of the final selection of row groups
  std::vector<row_group_info> selection;

  // Vector to store the number of rows read from each data source
  std::vector<size_t> num_rows_per_source(per_file_metadata.size(), 0);

  // Track the starting row index of the current row group
  size_t current_row_index = 0;

  // Track the number of rows read so far (may be different from the current row index if row bounds
  // were applied)
  size_t total_selected_rows = 0;

  // For each data source
  std::for_each(
    cuda::counting_iterator<size_t>{0},
    cuda::counting_iterator{current_row_group_indices.size()},
    [&](auto const& src_idx) {
      auto const& file_metadata = per_file_metadata[src_idx];

      // File-local row group row offsets
      auto const source_row_offsets =
        compute_source_row_group_offsets(static_cast<size_type>(src_idx));

      // For each row group in this data source
      std::for_each(
        current_row_group_indices[src_idx].begin(),
        current_row_group_indices[src_idx].end(),
        [&](auto const& rg_idx) {
          // Validate the row group index
          CUDF_EXPECTS(rg_idx >= 0 and std::cmp_less(rg_idx, file_metadata.row_groups.size()),
                       "Invalid row group index",
                       std::invalid_argument);

          // Get the row group
          auto const& rg = file_metadata.row_groups[rg_idx];

          // Use the effective start index of this row group if row bounds filter was
          // applied, else use the unadjusted start index
          auto const row_group_start_row = current_row_index;

          // Update the starting row index of the next row group
          current_row_index += rg.num_rows;

          // Get the number of rows in this row group
          auto const num_rows_this_row_group =
            is_row_bounded_row_groups ? row_group_row_counts[src_idx][rg_idx] : rg.num_rows;

          // Update the total number of rows selected
          total_selected_rows += num_rows_this_row_group;

          // Update the number of rows read from this data source
          num_rows_per_source[src_idx] += num_rows_this_row_group;

          // We need the unadjusted start index of this row group to correctly
          // initialize ColumnChunkDesc for this row group in
          // create_global_chunk_info() and calculate the row offset for the first
          // pass in compute_input_passes().
          selection.emplace_back(
            row_group_info{.index               = rg_idx,
                           .start_row           = row_group_start_row,
                           .source_start_row    = source_row_offsets[rg_idx],
                           .unadjusted_num_rows = static_cast<size_t>(rg.num_rows),
                           .source_index        = static_cast<cudf::size_type>(src_idx)});

          // If page-level indexes are present, then collect extra chunk and page
          // info. The page indexes rely on absolute row numbers - not adjusted for
          // skip_rows.
          column_info_for_row_group(selection.back(), row_group_start_row);
        });
    });

  // Set rows_to_read to the total number of rows selected and rows_to_skip to the number of rows
  // to skip relative to the first surviving row group
  rows_to_read = total_selected_rows;
  rows_to_skip = first_row_group_relative_rows_to_skip;

  return {rows_to_skip,
          rows_to_read,
          std::move(selection),
          std::move(num_rows_per_source),
          total_trimmed_row_groups,
          std::move(num_row_groups_after_filters)};
}

std::tuple<std::vector<input_column_info>,
           std::vector<cudf::io::detail::inline_column_buffer>,
           std::vector<int>>
aggregate_reader_metadata::select_columns(
  std::optional<std::vector<std::string>> const& use_names,
  std::optional<std::vector<std::string>> const& filter_columns_names,
  column_selection_options const& selection_options)
{
  auto const include_index          = selection_options.include_index;
  auto const strings_to_categorical = selection_options.strings_to_categorical;
  auto const ignore_missing_columns = selection_options.ignore_missing_columns;
  auto const timestamp_type_id      = selection_options.timestamp_type_id;
  auto const decimal_type_id        = selection_options.decimal_type_id;
  auto const case_sensitive_names   = selection_options.case_sensitive_names;
  auto const selection_mode         = selection_options.selection_mode;

  // Setup schema lookup helper
  auto schema_lookup =
    schema_child_lookup{[&](int const schema_idx, int const src_idx) -> SchemaElement const& {
                          return get_schema(schema_idx, src_idx);
                        },
                        case_sensitive_names,
                        selection_mode};

  std::vector<cudf::io::detail::inline_column_buffer> output_columns;
  std::vector<input_column_info> input_columns;
  std::vector<int> nesting;

  // Return true if column path is valid. e.g. if the path is {"struct1", "child1"}, then it is
  // valid if "struct1.child1" exists in this file's schema. If "struct1" exists but "child1" is
  // not a child of "struct1" then the function will return false for "struct1"
  std::function<bool(
    column_name_info const*, int, std::vector<cudf::io::detail::inline_column_buffer>&, bool)>
    build_column = [&](column_name_info const* col_name_info,
                       int schema_idx,
                       std::vector<cudf::io::detail::inline_column_buffer>& out_col_array,
                       bool has_list_parent) {
      auto const& schema_elem = get_schema(schema_idx);

      // if schema_elem is a stub then it does not exist in the column_name_info and column_buffer
      // hierarchy. So continue on
      if (schema_elem.is_stub()) {
        // is this legit?
        CUDF_EXPECTS(schema_elem.num_children == 1, "Unexpected number of children for stub");
        auto const child_col_name_info = col_name_info ? &col_name_info->children[0] : nullptr;
        return build_column(
          child_col_name_info, schema_elem.children_idx[0], out_col_array, has_list_parent);
      }

      auto const& parent_schema = get_schema(schema_elem.parent_idx);
      auto const one_level_list = schema_elem.is_one_level_list(parent_schema);

      // if we're at the root, this is a new output column
      auto const col_type =
        one_level_list
          ? type_id::LIST
          : to_type_id(schema_elem, strings_to_categorical, timestamp_type_id, decimal_type_id);
      auto const dtype = to_data_type(col_type, schema_elem);

      cudf::io::detail::inline_column_buffer output_col(
        dtype, schema_elem.repetition_type == FieldRepetitionType::OPTIONAL);
      if (has_list_parent) { output_col.user_data |= PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT; }
      // store the index of this element if inserted in out_col_array
      nesting.push_back(static_cast<int>(out_col_array.size()));
      output_col.name = schema_elem.name;

      // build each child
      bool path_is_valid = false;
      if (col_name_info == nullptr or col_name_info->children.empty()) {
        // add all children of schema_elem.
        // At this point, we can no longer pass a col_name_info to build_column
        for (int idx = 0; idx < schema_elem.num_children; idx++) {
          path_is_valid |= build_column(nullptr,
                                        schema_elem.children_idx[idx],
                                        output_col.children,
                                        has_list_parent || col_type == type_id::LIST);
        }
      } else {
        for (auto const& idx : col_name_info->children) {
          path_is_valid |=
            build_column(&idx,
                         schema_lookup.find_schema_child_by_name(schema_idx, idx.name),
                         output_col.children,
                         has_list_parent || col_type == type_id::LIST);
        }
      }

      // if I have no children, we're at a leaf and I'm an input column (that is, one with actual
      // data stored) so add me to the list.
      if (schema_elem.num_children == 0) {
        input_column_info& input_col = input_columns.emplace_back(
          schema_idx, schema_elem.name, schema_elem.max_repetition_level > 0);

        // set up child output column for one-level encoding list
        if (one_level_list) {
          // determine the element data type
          auto const element_type =
            to_type_id(schema_elem, strings_to_categorical, timestamp_type_id, decimal_type_id);
          auto const element_dtype = to_data_type(element_type, schema_elem);

          cudf::io::detail::inline_column_buffer element_col(
            element_dtype, schema_elem.repetition_type == FieldRepetitionType::OPTIONAL);
          if (has_list_parent || col_type == type_id::LIST) {
            element_col.user_data |= PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT;
          }
          // store the index of this element
          nesting.push_back(static_cast<int>(output_col.children.size()));
          // TODO: not sure if we should assign a name or leave it blank
          element_col.name = "element";

          output_col.children.push_back(std::move(element_col));
        }

        std::copy(nesting.cbegin(), nesting.cend(), std::back_inserter(input_col.nesting));

        // pop off the extra nesting element.
        if (one_level_list) { nesting.pop_back(); }

        // Flag the `metadata` / `value` BYTE_ARRAY children of a VARIANT group so that
        // `make_column` materializes them as `list<uint8>` instead of strings.
        if (schema_elem.type == Type::BYTE_ARRAY && parent_schema.logical_type.has_value() &&
            parent_schema.logical_type->type == LogicalType::VARIANT &&
            (schema_elem.name == "metadata" || schema_elem.name == "value")) {
          output_col.string_as_binary = true;
        }

        path_is_valid = true;  // If we're able to reach leaf then path is valid
      }

      if (path_is_valid) { out_col_array.push_back(std::move(output_col)); }

      nesting.pop_back();
      return path_is_valid;
    };

  // Compares two schema elements to be equal except their number of children
  auto const equal_to_except_num_children = [selection_mode](SchemaElement const& lhs,
                                                             SchemaElement const& rhs) {
    // Match by field ID if enabled, otherwise match by name
    auto const match_schema_by_field_id = selection_mode == column_selection_mode::BY_FIELD_ID;
    auto const names_match =
      (match_schema_by_field_id and lhs.field_id.has_value() and rhs.field_id.has_value())
        ? lhs.field_id == rhs.field_id
        : lhs.name == rhs.name;
    return lhs.type == rhs.type and lhs.converted_type == rhs.converted_type and
           lhs.type_length == rhs.type_length and names_match and
           lhs.decimal_scale == rhs.decimal_scale and
           lhs.decimal_precision == rhs.decimal_precision and lhs.field_id == rhs.field_id;
  };

  // Maps a projected column's schema_idx in the zeroth per_file_metadata (source) to the
  // corresponding schema_idx in src_idx'th per_file_metadata (destination). The projected
  // column's path must match across sources, else an appropriate exception is thrown.
  std::function<void(column_name_info const*, int const, int const, int const)> map_column =
    [&](column_name_info const* col_name_info,
        int const src_schema_idx,
        int const dst_schema_idx,
        int const src_idx) {
      auto const& src_schema_elem = get_schema(src_schema_idx);
      auto const& dst_schema_elem = get_schema(dst_schema_idx, src_idx);

      // Check the schema elements to be equal except their number of children as we only care about
      // the specific column paths in the schema trees. Raise an invalid_argument error if the
      // schema elements don't match.
      CUDF_EXPECTS(
        equal_to_except_num_children(src_schema_elem, dst_schema_elem),
        std::format(
          "Encountered mismatching data type or schema across the Parquet sources for column '{}'",
          src_schema_elem.name),
        std::invalid_argument);

      // Get the schema_idx_map for this data source (pfm)
      auto& schema_idx_map = schema_idx_maps[src_idx - 1];
      // Map the schema index from 0th tree (src) to the one in the current (dst) tree.
      schema_idx_map[src_schema_idx] = dst_schema_idx;

      // If src_schema_elem is a stub, it does not exist in the column_name_info and column_buffer
      // hierarchy. So continue on with mapping.
      if (src_schema_elem.is_stub()) {
        // Check if dst_schema_elem is also a stub i.e. has num_children == 1 that we didn't
        // previously check. Raise an invalid_argument error if dst_schema_elem is not a stub.
        CUDF_EXPECTS(
          dst_schema_elem.is_stub(),
          std::format(
            "Encountered mismatching nested structure across the Parquet sources for column '{}'",
            src_schema_elem.name),
          std::invalid_argument);
        auto const child_col_name_info = col_name_info ? &col_name_info->children[0] : nullptr;
        return map_column(child_col_name_info,
                          src_schema_elem.children_idx[0],
                          dst_schema_elem.children_idx[0],
                          src_idx);
      }

      // The path ends here. If this is a list/struct col (has children), then map all its children
      // which must be identical.
      if (col_name_info == nullptr or col_name_info->children.empty()) {
        // Check the number of children to be equal to be mapped.
        CUDF_EXPECTS(
          src_schema_elem.num_children == dst_schema_elem.num_children,
          std::format(
            "Encountered mismatching number of children across Parquet sources for column '{}'",
            src_schema_elem.name),
          std::invalid_argument);

        for (auto const& child_idx : src_schema_elem.children_idx) {
          auto const dst_child_idx = schema_lookup.find_target_schema_child(
            src_schema_idx, dst_schema_idx, get_schema(child_idx).name, src_idx);
          CUDF_EXPECTS(
            dst_child_idx != -1,
            std::format(
              "Encountered missing nested column '{}' across Parquet sources for column '{}'",
              get_schema(child_idx).name,
              src_schema_elem.name),
            std::invalid_argument);
          map_column(nullptr, child_idx, dst_child_idx, src_idx);
        }
      }
      // The path goes further down to specific child(ren) of this column so map only those
      // children.
      else {
        for (auto const& child_col_name_info : col_name_info->children) {
          // Ensure that each named child column exists in the destination schema tree for the
          // paths to align up.
          auto const src_child_idx =
            schema_lookup.find_schema_child_by_name(src_schema_idx, child_col_name_info.name);
          auto const dst_child_idx = schema_lookup.find_target_schema_child(
            src_schema_idx, dst_schema_idx, child_col_name_info.name, src_idx);
          CUDF_EXPECTS(
            dst_child_idx != -1,
            std::format(
              "Encountered missing nested column '{}' across Parquet sources for column '{}'",
              child_col_name_info.name,
              src_schema_elem.name),
            std::invalid_argument);
          map_column(&child_col_name_info, src_child_idx, dst_child_idx, src_idx);
        }
      }
    };

  std::vector<int> output_column_schemas;

  //
  // there is not necessarily a 1:1 mapping between input columns and output columns.
  // For example, parquet does not explicitly store a ColumnChunkDesc for struct columns.
  // The "structiness" is simply implied by the schema.  For example, this schema:
  //  required group field_id=1 name {
  //    required binary field_id=2 firstname (String);
  //    required binary field_id=3 middlename (String);
  //    required binary field_id=4 lastname (String);
  // }
  // will only contain 3 internal columns of data (firstname, middlename, lastname).  But of
  // course "name" is ultimately the struct column we want to return.
  //
  // "firstname", "middlename" and "lastname" represent the input columns in the file that we
  // process to produce the final cudf "name" column.
  //
  // A user can ask for a single field out of the struct e.g. firstname.
  // In this case they'll pass a fully qualified name to the schema element like
  // ["name", "firstname"]
  //
  auto const& root = get_schema(0);
  if (not use_names.has_value()) {
    for (auto const& schema_idx : root.children_idx) {
      build_column(nullptr, schema_idx, output_columns, false);
      output_column_schemas.push_back(schema_idx);
    }
  } else {
    struct path_info {
      std::string full_path;
      int schema_idx;
    };

    // Convert schema into a vector of every possible path
    std::vector<path_info> all_paths;
    std::function<void(std::string, int)> add_path = [&](std::string path_till_now,
                                                         int schema_idx) {
      auto const& schema_elem     = get_schema(schema_idx);
      std::string const curr_path = path_till_now + schema_elem.name;
      all_paths.push_back({curr_path, schema_idx});
      for (auto const& child_idx : schema_elem.children_idx) {
        add_path(curr_path + ".", child_idx);
      }
    };
    for (auto const& child_idx : get_schema(0).children_idx) {
      add_path("", child_idx);
    }

    // Find which of the selected paths are valid and get their schema index
    std::vector<path_info> valid_selected_paths;
    auto const empty_names = std::vector<std::string>{};
    // vector reference pushback for select column and/or filter names passed. Use empty names
    // reference if the no filter column names provided.
    std::vector<std::reference_wrapper<std::vector<std::string> const>> const column_names{
      *use_names, filter_columns_names.has_value() ? *filter_columns_names : empty_names};
    for (auto const& used_column_names : column_names) {
      for (auto const& selected_path : used_column_names.get()) {
        auto found_path =
          std::find_if(all_paths.begin(), all_paths.end(), [&](path_info& valid_path) {
            return are_column_paths_equal(
              valid_path.full_path, selected_path, case_sensitive_names);
          });
        // Ensure that selected path matches a path in all_paths
        CUDF_EXPECTS(
          found_path != all_paths.end() or ignore_missing_columns,
          std::format("Encountered non-existent column '{}' in the selected path", selected_path),
          std::invalid_argument);
        if (found_path != all_paths.end()) {
          // Use the file's actual path (preserving original case) for the valid_selected_paths
          valid_selected_paths.push_back({found_path->full_path, found_path->schema_idx});
        }
      }
    }

    // Now construct paths as vector of strings for further consumption
    std::vector<std::vector<std::string>> use_names3;
    std::transform(valid_selected_paths.cbegin(),
                   valid_selected_paths.cend(),
                   std::back_inserter(use_names3),
                   [&](path_info const& valid_path) {
                     auto schema_idx = valid_path.schema_idx;
                     std::vector<std::string> result_path;
                     do {
                       SchemaElement const& elem = get_schema(schema_idx);
                       result_path.push_back(elem.name);
                       schema_idx = elem.parent_idx;
                     } while (schema_idx > 0);
                     return std::vector<std::string>(result_path.rbegin(), result_path.rend());
                   });

    std::vector<column_name_info> selected_columns;
    if (include_index) {
      std::vector<std::string> const index_names = get_pandas_index_names();
      std::transform(index_names.cbegin(),
                     index_names.cend(),
                     std::back_inserter(selected_columns),
                     [](std::string const& name) { return column_name_info{.name = name}; });
    }
    // Merge the vector use_names into a set of hierarchical column_name_info objects
    /* This is because if we have columns like this:
     *     col1
     *      / \
     *    s3   f4
     *   / \
     * f5   f6
     *
     * there may be common paths in use_names like:
     * {"col1", "s3", "f5"}, {"col1", "f4"}
     * which means we want the output to contain
     *     col1
     *      / \
     *    s3   f4
     *   /
     * f5
     *
     * rather than
     *  col1   col1
     *   |      |
     *   s3     f4
     *   |
     *   f5
     */
    for (auto const& path : use_names3) {
      auto array_to_find_in = &selected_columns;
      for (auto const& name_to_find : path) {
        // Check if the path exists in our selected_columns and if not, add it.
        auto found_col = std::find_if(
          array_to_find_in->begin(),
          array_to_find_in->end(),
          [&name_to_find](column_name_info const& col) { return col.name == name_to_find; });
        if (found_col == array_to_find_in->end()) {
          auto& col        = array_to_find_in->emplace_back(name_to_find);
          array_to_find_in = &col.children;
        } else {
          // Path exists. go down further.
          array_to_find_in = &found_col->children;
        }
      }
    }
    for (auto& col : selected_columns) {
      auto constexpr root_idx = 0;
      auto const& top_level_col_schema_idx =
        schema_lookup.find_schema_child_by_name(root_idx, col.name);
      bool const valid_column = build_column(&col, top_level_col_schema_idx, output_columns, false);
      if (valid_column) {
        output_column_schemas.push_back(top_level_col_schema_idx);

        // Map the column's schema_idx across the rest of the data sources if required.
        if (per_file_metadata.size() > 1 and not schema_idx_maps.empty()) {
          std::for_each(
            cuda::counting_iterator{static_cast<size_t>(1)},
            cuda::counting_iterator{per_file_metadata.size()},
            [&](auto const src_idx) {
              // Ensure that each top level column exists in the destination schema tree.
              auto const dst_col_schema_idx =
                schema_lookup.find_target_schema_child(root_idx, root_idx, col.name, src_idx);
              CUDF_EXPECTS(
                dst_col_schema_idx != -1,
                std::format("Encountered missing top-level column '{}' across Parquet sources",
                            col.name),
                std::invalid_argument);
              map_column(&col, top_level_col_schema_idx, dst_col_schema_idx, src_idx);
            });
        }
      }
    }
  }

  return std::make_tuple(
    std::move(input_columns), std::move(output_columns), std::move(output_column_schemas));
}

std::vector<Type> aggregate_reader_metadata::get_parquet_types(
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<int const> column_schemas) const
{
  std::vector<Type> parquet_types(column_schemas.size());
  // Find a source with at least one row group
  auto const src_iter = std::find_if(row_group_indices.begin(),
                                     row_group_indices.end(),
                                     [](auto const& rg) { return rg.size() > 0; });
  CUDF_EXPECTS(src_iter != row_group_indices.end(),
               "Cannot determine Parquet types as no source has any selected row groups.",
               std::invalid_argument);

  // Source index
  auto const src_index = std::distance(row_group_indices.begin(), src_iter);
  // Use the first row group in this source
  auto const first_row_group_index = row_group_indices[src_index].front();
  std::transform(column_schemas.begin(),
                 column_schemas.end(),
                 parquet_types.begin(),
                 [&](auto const schema_idx) {
                   return get_column_metadata(first_row_group_index, src_index, schema_idx).type;
                 });

  return parquet_types;
}

}  // namespace cudf::io::parquet::detail
