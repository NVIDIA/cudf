/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <memory>
#include <optional>
#include <string>
#include <string_view>

/**
 * @file
 * @brief Utilities for Parquet VARIANT columns read as `struct` of `list<uint8>` children.
 */

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental {
/**
 * @addtogroup io_readers
 * @{
 */

/**
 * @brief Extract the raw VARIANT-encoded bytes of a nested field by JSONPath-like path.
 *
 * Path grammar:
 *   path  := "$"? first_step step*
 *   first := name | "." name | "[" index "]"
 *   step  := "." name | "[" index "]"
 *   name  := any sequence of bytes other than '.' or '['
 *   index := non-negative base-10 integer (leading zeros are allowed, e.g. "[01]" == "[1]")
 *
 * Examples:
 *   "x"            -> top-level field "x" (leading $ optional)
 *   "$.foo"        -> top-level field "foo"
 *   "$.foo.bar"    -> object descent foo -> bar
 *   "$[0]"         -> first element of a top-level array
 *   "$.a[0].b"     -> object key "a" -> first array element -> object key "b"
 *
 * @param variant_column Struct column (VARIANT materialization) with `list<uint8>` children
 *                       (`metadata`, `value`), plus optional shredded siblings
 * @param path JSONPath-like path string identifying the target field
 * @param status Optional. When provided, filled with `variant_operation_status` values, one per
 *               row. Must be non-nullable, `UINT8`, and have the same row count as
 *               `variant_column`
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return `list<uint8>` column with the extracted value's encoded bytes. A row is null when the
 *         input row is null, a name is absent, an index is out of bounds, or a step does not match
 *         the current value.
 *
 * @throws std::invalid_argument on empty path or malformed syntax (`[*]` wildcards, negative
 *         indices, out-of-range indices, and quoted names inside `[...]` are not supported); or if
 *         `status` is provided but is nullable, not `UINT8`, or has a different row count than
 *         `variant_column`
 */
[[nodiscard]] std::unique_ptr<column> get_variant_field(
  column_view const& variant_column,
  std::string_view path,
  std::optional<mutable_column_view> status = std::nullopt,
  cuda::stream_ref stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr         = cudf::get_current_device_resource_ref());

/**
 * @brief Decode a VARIANT value column's blobs into a typed cuDF column.
 *
 * A null value is produced when the input row is null or the encoded type does not match
 * `desired_type`.
 *
 * For a decimal `desired_type`, every encoded width is accepted and each value is rescaled from its
 * own encoded scale to `desired_type.scale()`, truncating toward zero; a value that no longer fits
 * produces a null row with `variant_operation_status::OVERFLOW`.
 *
 * @param values `list<uint8>` column of VARIANT-encoded value bytes
 * @param desired_type Target cuDF type (`STRING`, `INT8`/`INT16`/`INT32`/`INT64`,
 *        `FLOAT32`/`FLOAT64`, `BOOL8`, or `DECIMAL32`/`DECIMAL64`/`DECIMAL128`)
 * @param status Optional in-out parameter, `variant_operation_status` values, one per row. Must be
 *        non-nullable, `UINT8`, and have the same row count as `values`. On input, its existing
 *        values are treated as status from a prior `get_variant_field` call: rows already marked
 *        non-success are propagated directly to the output without decoding. It is then
 *        overwritten in place with the final per-row status. Callers with no prior status to
 *        propagate must initialize every row to `variant_operation_status::SUCCESS` before calling
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Typed column decoded from the VARIANT value blobs
 *
 * @throws std::invalid_argument if `values` is not a `list<uint8>` column; if `desired_type`
 *         is not one of the supported types (`STRING`, `INT8`/`INT16`/`INT32`/`INT64`,
 *         `FLOAT32`/`FLOAT64`, `BOOL8`, or `DECIMAL32`/`DECIMAL64`/`DECIMAL128`); or if `status`
 *         is provided but is nullable, not `UINT8`, or has a different row count than `values`
 */
[[nodiscard]] std::unique_ptr<column> cast_variant(
  column_view const& values,
  data_type desired_type,
  std::optional<mutable_column_view> status = std::nullopt,
  cuda::stream_ref stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr         = cudf::get_current_device_resource_ref());

/**
 * @brief Convenience wrapper: extract a nested object value by path and decode into a typed column.
 *
 * Semantically equivalent to extracting the field with `get_variant_field` and then decoding
 * the extracted `list<uint8>` values with `cast_variant`.
 *
 * @param variant_column Struct column (VARIANT materialization)
 * @param path JSONPath-like path string (see `get_variant_field` for syntax)
 * @param desired_type Target type: `STRING`, `INT8`/`INT16`/`INT32`/`INT64`,
 *        `FLOAT32`/`FLOAT64`, `BOOL8`, or `DECIMAL32`/`DECIMAL64`/`DECIMAL128`
 *        (see `cast_variant` for decimal rescaling)
 * @param status Optional. When provided, filled with `variant_operation_status` values, one per
 *               row. Must be non-nullable, `UINT8`, and have the same row count as
 *               `variant_column`
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Column of `desired_type`
 *
 * @throws std::invalid_argument on empty path or malformed syntax; or if `status` is provided but
 *         is nullable, not `UINT8`, or has a different row count than `variant_column`
 */
[[nodiscard]] std::unique_ptr<column> extract_variant_field(
  column_view const& variant_column,
  std::string_view path,
  data_type desired_type,
  std::optional<mutable_column_view> status = std::nullopt,
  cuda::stream_ref stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr         = cudf::get_current_device_resource_ref());

/**
 * @brief Return the logical type of each VARIANT value blob in a `list<uint8>` column.
 *
 * Classifies only the value_metadata header byte; does not validate the remaining payload.
 * A recognized header returns its logical type even when the payload is truncated. A null output
 * row is produced when the input row is null, the blob is empty, or the header carries an
 * unrecognized type. An encoded Variant null (NULLVAL) produces a valid `NULL_VALUE` row.
 *
 * @param values `list<uint8>` column of VARIANT-encoded value bytes
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return `UINT8` column of `variant_logical_type` values cast to `uint8_t`
 *
 * @throws std::invalid_argument if `values` is not a `list<uint8>` column
 */
[[nodiscard]] std::unique_ptr<column> get_variant_type_id(
  column_view const& values,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * `@brief` Encode a strings column of flat JSON objects as Parquet VARIANT.
 *
 * Each row of `@p` input must be a non-nested JSON object string (e.g. `{"a":1,"b":"hi"}`).
 * The function extracts the scalar fields named in `@p` column_names and encodes the result
 * as a Parquet VARIANT column: a `struct<list<uint8> metadata, list<uint8> value>`.
 *
 * Supported JSON value types per field:
 *   - `null`              → VARIANT null primitive
 *   - `true` / `false`    → VARIANT boolean primitive
 *   - Integer literals    → VARIANT INT64 primitive
 *   - Floating-point      → VARIANT FLOAT64 primitive
 *   - Quoted strings      → VARIANT short-string or long-string
 *
 * Fields absent from a row (or whose input row is null) are omitted from that row's
 * VARIANT object; they do not appear in the value blob.
 *
 * `@param` input Strings column where each non-null row is a flat JSON object
 * `@param` column_names Field names to encode; ordering need not be sorted. At most 255 names
 * `@param` stream CUDA stream
 * `@param` mr Device memory resource
 * `@return` `struct<list<uint8> metadata, list<uint8> value>` VARIANT column
 *
 * `@throws` std::invalid_argument if `column_names` contains more than 255 names
 * `@throws` std::invalid_argument if `column_names` contains duplicate entries
 * `@throws` std::invalid_argument if any name in `column_names` contains '.' or '['
 */
[[nodiscard]] std::unique_ptr<column> encode_strings_to_variant(
  cudf::strings_column_view const& input,
  cudf::host_span<std::string const> column_names,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
