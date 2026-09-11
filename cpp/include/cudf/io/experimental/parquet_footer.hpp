/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <vector>

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
 * @brief Serialize a Parquet footer (`FileMetaData`) to Thrift-compact-encoded bytes
 *
 * @ingroup io_writers
 *
 * @param metadata The `FileMetaData` (footer) to serialize
 *
 * @return The Thrift-compact-encoded bytes
 */
[[nodiscard]] std::vector<uint8_t> write_parquet_footer_bytes(
  parquet::FileMetaData const& metadata);

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
