/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <cstdint>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental {

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
 * @throws cudf::logic_error If `mode == throw_if_type_mismatch::YES` and a field's wire type does
 * not match the expected schema type
 *
 * @param footer_bytes Thrift-compact-encoded Parquet `FileMetaData` (footer) bytes
 * @param mode `throw_if_type_mismatch::YES` (default) rejects a field whose wire type mismatches
 * the schema type; `throw_if_type_mismatch::NO` skips it (Thrift forward-compat)
 *
 * @return The deserialized `FileMetaData`
 */
[[nodiscard]] FileMetaData read_parquet_footer_bytes(
  host_span<uint8_t const> footer_bytes, throw_if_type_mismatch mode = throw_if_type_mismatch::YES);

/**
 * @brief Serialize a Parquet footer (`FileMetaData`) to Thrift-compact-encoded bytes
 *
 * @ingroup io_writers
 *
 * @param metadata The `FileMetaData` (footer) to serialize
 *
 * @return The Thrift-compact-encoded bytes
 */
[[nodiscard]] std::vector<uint8_t> write_parquet_footer_bytes(FileMetaData const& metadata);

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
