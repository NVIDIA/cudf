/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io::protobuf {

class decode_protobuf_options_builder;

/**
 * @brief Protobuf field encoding types.
 */
enum class proto_encoding : int {
  DEFAULT     = 0,  ///< Standard varint encoding
  FIXED       = 1,  ///< Fixed-width encoding (32-bit or 64-bit)
  ZIGZAG      = 2,  ///< ZigZag encoding for signed integers
  ENUM_STRING = 3,  ///< Enum field decoded as string
};

/**
 * @brief Get the integer value of a proto_encoding.
 */
CUDF_HOST_DEVICE constexpr int encoding_value(proto_encoding encoding)
{
  return static_cast<int>(encoding);
}

/// Maximum protobuf field number (29-bit).
constexpr int max_field_number = (1 << 29) - 1;

/**
 * @brief Protobuf wire types.
 */
enum class proto_wire_type : int {
  VARINT = 0,  ///< Variable-length integer
  I64BIT = 1,  ///< 64-bit fixed
  LEN    = 2,  ///< Length-delimited
  SGROUP = 3,  ///< Start group (deprecated)
  EGROUP = 4,  ///< End group (deprecated)
  I32BIT = 5,  ///< 32-bit fixed
};

/**
 * @brief Get the integer value of a proto_wire_type.
 */
CUDF_HOST_DEVICE constexpr int wire_type_value(proto_wire_type wire_type)
{
  return static_cast<int>(wire_type);
}

/// Maximum supported nesting depth for nested protobuf messages.
constexpr int max_nesting_depth = 10;

/**
 * @brief Descriptor for a single field in a (possibly nested) protobuf schema.
 *
 * Fields are organized in a flat array where parent-child relationships are
 * expressed via @p parent_idx. Top-level fields have `parent_idx == -1`.
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

/**
 * @brief Context for decoding protobuf messages.
 *
 * Contains the schema (as a flat array of field descriptors), default values,
 * and enum metadata needed for decoding.
 */
struct decode_protobuf_options {
  using byte_vector       = std::vector<uint8_t>;  ///< Raw byte vector for string or enum name data
  using enum_value_vector = std::vector<int32_t>;  ///< Valid enum numbers for one field
  using enum_name_vector  = std::vector<byte_vector>;  ///< UTF-8 enum names for one field

  decode_protobuf_options() = default;

  /**
   * @brief Construct options from a schema, initializing per-field metadata to defaults.
   *
   * @param schema Flat array of field descriptors
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
   * @param schema Flat array of field descriptors
   * @return A builder initialized with the provided schema
   */
  static decode_protobuf_options_builder builder(std::vector<nested_field_descriptor> schema);

  std::vector<nested_field_descriptor> schema;       ///< Flat array of field descriptors
  std::vector<int64_t> default_ints;                 ///< Default integer values per field
  std::vector<double> default_floats;                ///< Default float values per field
  std::vector<uint8_t> default_bools;                ///< Default boolean values per field (0 or 1)
  std::vector<byte_vector> default_strings;          ///< Default string values per field
  std::vector<enum_value_vector> enum_valid_values;  ///< Valid enum numbers per field
  std::vector<enum_name_vector> enum_names;          ///< UTF-8 enum names per field
  bool fail_on_errors = true;  ///< If true, throw on malformed messages; otherwise return nulls
};

/**
 * @brief Builder for decode_protobuf_options.
 */
class decode_protobuf_options_builder {
 public:
  /**
   * @brief Construct a builder for decode_protobuf_options.
   *
   * @param schema Flat array of field descriptors
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
   * @brief Set default float values.
   *
   * @param values Default float values per field
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
  decode_protobuf_options_builder& default_bools(std::vector<uint8_t> values)
  {
    _options.default_bools = std::move(values);
    return *this;
  }

  /**
   * @brief Set default string values.
   *
   * @param values Default string values per field
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& default_strings(
    std::vector<decode_protobuf_options::byte_vector> values)
  {
    _options.default_strings = std::move(values);
    return *this;
  }

  /**
   * @brief Set valid enum numeric values.
   *
   * @param values Valid enum numbers per field
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& enum_valid_values(
    std::vector<decode_protobuf_options::enum_value_vector> values)
  {
    _options.enum_valid_values = std::move(values);
    return *this;
  }

  /**
   * @brief Set enum names.
   *
   * @param values UTF-8 enum names per field
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& enum_names(
    std::vector<decode_protobuf_options::enum_name_vector> values)
  {
    _options.enum_names = std::move(values);
    return *this;
  }

  /**
   * @brief Set error handling behavior.
   *
   * @param value Whether malformed messages should raise an error
   * @return Reference to this builder
   */
  decode_protobuf_options_builder& fail_on_errors(bool value)
  {
    _options.fail_on_errors = value;
    return *this;
  }

  /**
   * @brief Build decode_protobuf_options.
   *
   * @return Completed decode_protobuf_options
   */
  [[nodiscard]] decode_protobuf_options build() { return std::move(_options); }

 private:
  decode_protobuf_options _options;
};

inline decode_protobuf_options_builder decode_protobuf_options::builder(
  std::vector<nested_field_descriptor> schema)
{
  return decode_protobuf_options_builder{std::move(schema)};
}

/**
 * @brief Decode serialized protobuf messages into a struct column.
 *
 * Takes a LIST<INT8> or LIST<UINT8> column where each row contains a serialized
 * protobuf message, and decodes it into a STRUCT column according to the provided schema.
 *
 * Supports nested messages (up to max_nesting_depth levels), repeated fields (as LIST columns),
 * enum-as-string conversion, default values, and required field checking.
 *
 * @param binary_input LIST<INT8> or LIST<UINT8> column of serialized protobuf messages
 * @param options Decoding options including schema, defaults, and enum metadata
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A STRUCT column containing the decoded protobuf fields
 *
 * @throws cudf::logic_error if the schema is invalid
 * @throws cudf::logic_error if fail_on_errors is true and a message cannot be decoded
 */
[[nodiscard]] std::unique_ptr<cudf::column> decode_protobuf(
  cudf::column_view const& binary_input,
  decode_protobuf_options const& options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace io::protobuf
}  // namespace CUDF_EXPORT cudf
