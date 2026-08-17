/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
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
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return `list<uint8>` column with the extracted value's encoded bytes. A row is null when the
 *         input row is null, a name is absent, an index is out of bounds, or a step does not match
 *         the current value.
 *
 * @throws std::invalid_argument on empty path or malformed syntax (`[*]` wildcards, negative
 *         indices, out-of-range indices, and quoted names inside `[...]` are not supported)
 */
[[nodiscard]] std::unique_ptr<column> get_variant_field(
  column_view const& variant_column,
  std::string_view path,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Decode a VARIANT value column's blobs into a typed cuDF column.
 *
 * A null value is produced when the input row is null or the encoded type does not match
 * `desired_type`.
 *
 * @param values `list<uint8>` column of VARIANT-encoded value bytes
 * @param desired_type Target cuDF type (`STRING`, `INT8`/`INT16`/`INT32`/`INT64`,
 *        `FLOAT32`/`FLOAT64`, or `BOOL8`)
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Typed column decoded from the VARIANT value blobs
 *
 * @throws std::invalid_argument if `values` is not a `list<uint8>` column, or if `desired_type`
 *         is not one of the supported types (`STRING`, `INT8`/`INT16`/`INT32`/`INT64`,
 *         `FLOAT32`/`FLOAT64`, or `BOOL8`)
 */
[[nodiscard]] std::unique_ptr<column> cast_variant(
  column_view const& values,
  data_type desired_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convenience wrapper: extract a nested object value by path and decode into a typed column.
 *
 * Semantically equivalent to extracting the field with `get_variant_field` and then decoding
 * the extracted `list<uint8>` values with `cast_variant`.
 *
 * @param variant_column Struct column (VARIANT materialization)
 * @param path JSONPath-like path string (see `get_variant_field` for syntax)
 * @param desired_type Target type: `STRING`, `INT8`/`INT16`/`INT32`/`INT64`,
 *        `FLOAT32`/`FLOAT64`, or `BOOL8`
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Column of `desired_type`
 *
 * @throws std::invalid_argument on empty path or malformed syntax
 */
[[nodiscard]] std::unique_ptr<column> extract_variant_field(
  column_view const& variant_column,
  std::string_view path,
  data_type desired_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

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
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */
}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
