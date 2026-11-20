/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

/**
 * @file
 * @brief Freeform APIs for reading and writing Parquet footer bytes
 */

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental {

/**
 * @brief Footer-reader policy on a wire-type/schema-type mismatch: reject (`THROW`, the historical
 * exact-type contract) or skip per Thrift forward-compat (`COMPAT`)
 *
 * @ingroup io_readers
 */
enum class thrift_mismatch_policy : bool { THROW, COMPAT };

/**
 * @brief Deserialize a Parquet footer (`FileMetaData`) from Thrift-compact-encoded bytes
 *
 * @ingroup io_readers
 *
 * @note Stops at the footer struct terminator, so trailing frame bytes (e.g. a footer-length word
 * or magic) are accepted and ignored rather than validated
 *
 * @throws cudf::logic_error If the footer is truncated or corrupt within the struct, caught by the
 * reader's overread guard and per-field bounds checks
 * @throws cudf::logic_error If `mode == thrift_mismatch_policy::THROW` and a field's wire type does
 * not match the expected schema type
 *
 * @param footer_bytes Thrift-compact-encoded Parquet `FileMetaData` (footer) bytes
 * @param mode Mismatch policy, see `thrift_mismatch_policy`
 *
 * @return The deserialized `FileMetaData`
 */
[[nodiscard]] parquet::FileMetaData read_parquet_footer_bytes(
  std::span<uint8_t const> footer_bytes,
  thrift_mismatch_policy mode = thrift_mismatch_policy::THROW);

/**
 * @brief Options to selectively decode a Parquet footer
 *
 * @ingroup io_readers
 *
 * Column chunks are decoded only for the selected row groups and leaf columns; all others are
 * skipped on the wire without materializing them. Leaf columns are indexed in schema (depth-first)
 * order, which is also the order of `RowGroup::columns`.
 */
struct footer_decode_options {
  /// Row groups whose column chunks to decode. `std::nullopt` selects all row groups
  std::optional<std::vector<cudf::size_type>> row_groups;
  /// Leaf columns whose column chunks to decode. `std::nullopt` selects all leaf columns
  std::optional<std::vector<cudf::size_type>> leaf_columns;
  /// Whether to decode `ColumnChunkMetaData::statistics` of the selected column chunks
  bool decode_statistics = true;
  /// Whether to decode `ColumnChunkMetaData::encoding_stats` of the selected column chunks
  bool decode_encoding_stats = true;
  /// Whether to decode `ColumnChunkMetaData::size_statistics` of the selected column chunks
  bool decode_size_statistics = true;
  /// Mismatch policy, see `thrift_mismatch_policy`
  thrift_mismatch_policy mismatch_policy = thrift_mismatch_policy::THROW;
};

/**
 * @brief A selectively decoded Parquet footer
 *
 * @ingroup io_readers
 *
 * `metadata` holds every file-level field and every row group, but `RowGroup::columns` is populated
 * only for the selected row groups and holds only the selected leaf columns:
 * `metadata.row_groups[row_groups[r]].columns[c]` is the chunk of leaf column `leaf_columns[c]`.
 * `RowGroup::columns` of an unselected row group is empty.
 *
 * @warning `metadata` is not a complete footer and must not be passed to APIs that expect one
 */
struct partial_file_metadata {
  FileMetaData metadata;  ///< Partially decoded footer
  std::vector<cudf::size_type> row_groups;    ///< Sorted indices of the decoded row groups
  std::vector<cudf::size_type> leaf_columns;  ///< Sorted indices of the decoded leaf columns
};

/**
 * @brief Selectively deserialize a Parquet footer (`FileMetaData`) from Thrift-compact-encoded
 * bytes
 *
 * @ingroup io_readers
 *
 * Decodes all file-level fields (schema, key-value metadata, etc.) and the scalar fields of every
 * row group, but decodes column chunks only for the selected row groups and leaf columns.
 * Decoding with no row groups and no leaf columns selected is a cheap way to obtain the schema.
 *
 * @throws std::out_of_range If a selected row group or leaf column index is out of range
 * @throws cudf::logic_error If a selection is given but the footer encodes the row groups before
 * the schema, or a row group's column count does not match the schema's leaf count
 * @throws cudf::logic_error On a truncated or corrupt footer, as `read_parquet_footer_bytes`
 *
 * @param footer_bytes Thrift-compact-encoded Parquet `FileMetaData` (footer) bytes
 * @param options Selection and decode options, see `footer_decode_options`
 *
 * @return The partially decoded footer, see `partial_file_metadata`
 */
[[nodiscard]] partial_file_metadata read_parquet_footer_bytes(
  std::span<uint8_t const> footer_bytes, footer_decode_options const& options);

/**
 * @brief Serialize a Parquet footer (`FileMetaData`) to Thrift-compact-encoded bytes
 *
 * @ingroup io_writers
 *
 * @param metadata The `FileMetaData` (footer) to serialize
 *
 * @note Fields that the writer omits when default-valued cannot distinguish the default from
 * absent on re-read (e.g., `created_by` and `ColumnChunk::file_path` read back as `""`). Fields
 * modeled with presence (`std::optional`) round-trip faithfully. Unmodeled fields are dropped, and
 * the footer signature is not recomputed.
 *
 * @return The Thrift-compact-encoded bytes
 */
[[nodiscard]] std::vector<uint8_t> write_parquet_footer_bytes(
  parquet::FileMetaData const& metadata);

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
