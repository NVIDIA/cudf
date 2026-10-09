/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::protobuf {

/**
 * @addtogroup io_readers
 * @{
 * @file
 */

/**
 * @brief Protobuf field encoding types.
 */
enum class proto_encoding : int {
  DEFAULT     = 0,  ///< Standard varint, fixed-width float, or length-delimited encoding
  FIXED       = 1,  ///< Fixed-width integer encoding (fixed32/fixed64/sfixed32/sfixed64)
  ZIGZAG      = 2,  ///< ZigZag varint encoding for signed integers (sint32/sint64)
  ENUM_STRING = 3,  ///< Enum field decoded as its name string
};

/// Maximum protobuf field number (29-bit).
constexpr int max_field_number = (1 << 29) - 1;

/**
 * @brief Protobuf wire types.
 */
enum class proto_wire_type : uint32_t {
  VARINT = 0,  ///< Variable-length integer
  I64BIT = 1,  ///< 64-bit fixed
  LEN    = 2,  ///< Length-delimited
  SGROUP = 3,  ///< Start group (deprecated)
  EGROUP = 4,  ///< End group (deprecated)
  I32BIT = 5,  ///< 32-bit fixed
};

/// Maximum supported nesting depth for nested protobuf messages.
constexpr int max_nesting_depth = 10;

/**
 * @brief Descriptor for a single field in a flattened protobuf schema.
 *
 * Parent-child relationships are expressed via @p parent_idx. Top-level fields have
 * `parent_idx == -1`. For repeated fields, @p output_type is the element type.
 */
struct nested_field_descriptor {
  int field_number;           ///< Protobuf field number
  int parent_idx;             ///< Index of parent field in schema (-1 for top-level)
  int depth;                  ///< Nesting depth (0 for top-level)
  proto_wire_type wire_type;  ///< Expected wire type
  cudf::type_id output_type;  ///< Output cudf type
  proto_encoding encoding;    ///< Encoding type
  bool is_repeated;           ///< Whether this field is repeated (array)
  bool is_required;           ///< Whether this field is required (proto2)
  bool has_default_value;     ///< Whether this field has a default value
};

class decode_protobuf_options_builder;

/**
 * @brief Schema, default values, and enum metadata needed to decode protobuf messages.
 *
 * All per-field vectors are indexed by schema position. The options are validated by
 * `decode_protobuf`.
 */
struct decode_protobuf_options {
  decode_protobuf_options() = default;

  /**
   * @brief Construct options from a schema, initializing per-field metadata to defaults.
   *
   * @param schema Flattened field descriptors; a parent must precede its children
   */
  explicit decode_protobuf_options(std::vector<nested_field_descriptor> schema)
    : schema(std::move(schema)),
      default_ints(this->schema.size()),
      default_floats(this->schema.size()),
      default_bools(this->schema.size()),
      default_strings(this->schema.size()),
      enum_valid_values(this->schema.size()),
      enum_names(this->schema.size())
  {
  }

  /**
   * @brief Creates a builder for decode_protobuf_options.
   *
   * @param schema Flattened field descriptors; a parent must precede its children
   * @return A builder initialized with the provided schema
   */
  static decode_protobuf_options_builder builder(std::vector<nested_field_descriptor> schema);

  std::vector<nested_field_descriptor> schema;                ///< Flattened field descriptors
  std::vector<int64_t> default_ints;                          ///< Default integer values
  std::vector<double> default_floats;                         ///< Default floating-point values
  std::vector<bool> default_bools;                            ///< Default boolean values
  std::vector<std::vector<uint8_t>> default_strings;          ///< Default string/bytes values
  std::vector<std::vector<int32_t>> enum_valid_values;        ///< Sorted valid enum numbers
  std::vector<std::vector<std::vector<uint8_t>>> enum_names;  ///< UTF-8 enum names
  bool fail_on_errors = true;  ///< If true, throw on malformed messages; otherwise null the row
  /// Hidden fields are still decoded so required/enum/wire validation runs. An empty vector means
  /// all fields are included in the returned struct.
  std::vector<bool> output_fields;
};

/**
 * @brief Builder for decode_protobuf_options.
 */
class decode_protobuf_options_builder {
 public:
  /**
   * @brief Construct a builder for decode_protobuf_options.
   *
   * @param schema Flattened field descriptors; a parent must precede its children
   */
  explicit decode_protobuf_options_builder(std::vector<nested_field_descriptor> schema)
    : _options(std::move(schema))
  {
  }

  /**
   * @brief Set default integer values.
   *
   * @param values Default integer values per field
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& default_ints(std::vector<int64_t> values)
  {
    _options.default_ints = std::move(values);
    return *this;
  }

  /**
   * @brief Set default floating-point values.
   *
   * @param values Default floating-point values per field
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& default_floats(std::vector<double> values)
  {
    _options.default_floats = std::move(values);
    return *this;
  }

  /**
   * @brief Set default boolean values.
   *
   * @param values Default boolean values per field
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& default_bools(std::vector<bool> values)
  {
    _options.default_bools = std::move(values);
    return *this;
  }

  /**
   * @brief Set default string/bytes values.
   *
   * @param values Default string/bytes values per field
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& default_strings(std::vector<std::vector<uint8_t>> values)
  {
    _options.default_strings = std::move(values);
    return *this;
  }

  /**
   * @brief Set valid enum numbers.
   *
   * @param values Sorted valid enum numbers per field, empty for non-enum fields
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& enum_valid_values(std::vector<std::vector<int32_t>> values)
  {
    _options.enum_valid_values = std::move(values);
    return *this;
  }

  /**
   * @brief Set enum names.
   *
   * @param values UTF-8 enum names per field, parallel to the valid enum numbers
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& enum_names(std::vector<std::vector<std::vector<uint8_t>>> values)
  {
    _options.enum_names = std::move(values);
    return *this;
  }

  /**
   * @brief Set error handling behavior.
   *
   * @param value If true, throw on malformed messages; otherwise null the row
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& fail_on_errors(bool value)
  {
    _options.fail_on_errors = value;
    return *this;
  }

  /**
   * @brief Set which fields appear in the returned struct.
   *
   * A nested field must use the same flag as its parent.
   *
   * @param values Per-field output flags, or empty to return every field
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& output_fields(std::vector<bool> values)
  {
    _options.output_fields = std::move(values);
    return *this;
  }

  /**
   * @brief Build decode_protobuf_options.
   *
   * @return The built options
   */
  [[nodiscard]] decode_protobuf_options build() { return std::move(_options); }

 private:
  decode_protobuf_options _options;
};

/**
 * @brief Decode serialized protobuf messages into a STRUCT column.
 *
 * Each row of @p binary_input holds one serialized message. The result has one child per
 * top-level output field; nested messages become STRUCT columns and repeated fields become LIST
 * columns. Null input rows produce null output rows.
 *
 * @note Field extraction is not implemented yet: every returned field is all-null.
 *
 * @throws std::invalid_argument if @p options is inconsistent
 * @throws cudf::logic_error if @p binary_input is not a LIST<INT8/UINT8> column
 *
 * @param binary_input LIST<INT8/UINT8> column of serialized protobuf messages
 * @param options Decoding schema, defaults, and enum metadata
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return STRUCT column of the decoded output fields
 */
[[nodiscard]] std::unique_ptr<cudf::column> decode_protobuf(
  cudf::column_view const& binary_input,
  decode_protobuf_options const& options,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace io::protobuf
}  // namespace CUDF_EXPORT cudf
