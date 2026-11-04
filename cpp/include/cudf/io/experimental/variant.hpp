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

#include <cuda/stream>

#include <memory>
#include <optional>
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
 * `TIMESTAMP_MICROSECONDS`/`TIMESTAMP_NANOSECONDS` accept both the timezone-adjusted and the NTZ
 * encoding of the same unit, unchanged; no unit conversion is performed.
 *
 * @param values `list<uint8>` column of VARIANT-encoded value bytes
 * @param desired_type Target cuDF type (`STRING`, `INT8`/`INT16`/`INT32`/`INT64`,
 *        `FLOAT32`/`FLOAT64`, `TIMESTAMP_MICROSECONDS`/`TIMESTAMP_NANOSECONDS`, `BOOL8`, or
 *        `DECIMAL32`/`DECIMAL64`/`DECIMAL128`)
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
 *         is not one of the supported types listed above; or if `status`
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
 * @param desired_type Target type (see `cast_variant` for the supported types)
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
 * @brief Parse each row of a strings column as one JSON document and encode it as a VARIANT.
 *
 * The output is byte-identical to Spark's `parse_json` (Spark 4.0 `VariantBuilder`):
 *   - Integers use the narrowest of INT8/INT16/INT32/INT64. Numbers that do not fit, or that have
 *     a fraction, become DECIMAL4/8/16 when written without an exponent and with precision and
 *     scale of at most 38; all other numbers become correctly rounded FLOAT64 values.
 *   - Strings of up to 63 bytes are short strings; longer strings are LONG_STRING primitives.
 *   - Each row has its own metadata dictionary, in order of first use. Object fields are sorted by
 *     key in UTF-16 code unit order.
 *   - Content after the first complete JSON value in a row is ignored.
 *
 * Rows that do not convert are null in the output, and `status` records why:
 *   - `ROW_NULL`: the input row is null.
 *   - `INVALID_JSON`: the row is not valid JSON under Jackson's default rules, including its limits
 *     on nesting depth (1000), number length (1000), string length, and field name length.
 *   - `DUPLICATE_KEY`: an object repeats a key and `allow_duplicate_keys` is false. When it is
 * true, the last occurrence of the key is kept.
 *   - `SIZE_LIMIT`: the encoded value or metadata of the row exceeds 128 MiB.
 *   - `UNSUPPORTED_INPUT`: the row is valid, but cannot be converted on the GPU. This happens when
 *     arrays and objects nest more than 127 levels deep, counting each object twice, or when a
 *     top-level `true`, `false`, or `null` is followed directly by a non-ASCII character. Callers
 *     that need a result for such rows must convert them on the CPU.
 *
 * Input strings are assumed to be valid UTF-8.
 *
 * @param input Strings column with one JSON document per row
 * @param allow_duplicate_keys Whether objects may repeat a key, keeping the last occurrence
 * @param status Optional. When provided, filled with `variant_operation_status` values, one per
 *               row. Must be non-nullable, `UINT8`, and have the same row count as `input`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Struct column with `metadata` and `value` children, each `list<uint8>`
 *
 * @throws std::invalid_argument if `status` is provided but is nullable, not `UINT8`, or has a
 *         different row count than `input`
 * @throws std::overflow_error if the encoded metadata or values of a batch of rows exceed the
 *         `size_type` limit
 */
[[nodiscard]] std::unique_ptr<column> parse_json_to_variant(
  strings_column_view const& input,
  bool allow_duplicate_keys,
  std::optional<mutable_column_view> status = std::nullopt,
  cuda::stream_ref stream                   = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr         = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
