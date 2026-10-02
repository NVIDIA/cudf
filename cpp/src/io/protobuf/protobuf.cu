/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/protobuf/protobuf_host_helpers.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <utility>

namespace cudf::io::protobuf {

decode_protobuf_options_builder decode_protobuf_options::builder(
  std::vector<nested_field_descriptor> schema)
{
  return decode_protobuf_options_builder{std::move(schema)};
}

namespace detail {

std::unique_ptr<cudf::column> make_null_column_with_schema(protobuf_schema const& schema,
                                                           int schema_idx,
                                                           cudf::size_type num_rows,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  auto const& field = schema[schema_idx];
  auto const dtype  = cudf::data_type{field.output_type};

  if (field.is_repeated) {
    std::unique_ptr<cudf::column> empty_child;
    if (dtype.id() == cudf::type_id::STRUCT) {
      empty_child = make_empty_struct_column_with_schema(schema, schema_idx, stream, mr);
    } else {
      empty_child = make_empty_column_safe(dtype, stream, mr);
    }
    return make_null_list_column_with_child(std::move(empty_child), num_rows, stream, mr);
  }

  if (dtype.id() == cudf::type_id::STRUCT) {
    auto const& child_indices = schema.children(schema_idx);
    std::vector<std::unique_ptr<cudf::column>> children;
    for (auto const child_idx : child_indices) {
      children.push_back(make_null_column_with_schema(schema, child_idx, num_rows, stream, mr));
    }
    auto null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    return cudf::make_structs_column(
      num_rows, std::move(children), num_rows, std::move(null_mask), stream, mr);
  }

  return make_null_column(dtype, num_rows, stream, mr);
}

bool is_encoding_compatible(nested_field_descriptor const& field, cudf::data_type const& type)
{
  switch (field.encoding) {
    case proto_encoding::DEFAULT:
      switch (type.id()) {
        case cudf::type_id::BOOL8:
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64: return field.wire_type == proto_wire_type::VARINT;
        case cudf::type_id::FLOAT32: return field.wire_type == proto_wire_type::I32BIT;
        case cudf::type_id::FLOAT64: return field.wire_type == proto_wire_type::I64BIT;
        case cudf::type_id::STRING:
        case cudf::type_id::LIST:
        case cudf::type_id::STRUCT: return field.wire_type == proto_wire_type::LEN;
        default: return false;
      }
    case proto_encoding::FIXED:
      switch (type.id()) {
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::FLOAT32: return field.wire_type == proto_wire_type::I32BIT;
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64:
        case cudf::type_id::FLOAT64: return field.wire_type == proto_wire_type::I64BIT;
        default: return false;
      }
    case proto_encoding::ZIGZAG:
      return field.wire_type == proto_wire_type::VARINT &&
             (type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::INT64);
    case proto_encoding::ENUM_STRING:
      return field.wire_type == proto_wire_type::VARINT && type.id() == cudf::type_id::STRING;
    default: return false;
  }
}

void validate_decode_options(decode_protobuf_options const& options)
{
  auto const& schema    = options.schema;
  auto const num_fields = schema.size();
  CUDF_EXPECTS(options.default_ints.size() == num_fields,
               "protobuf decode options: default_ints size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(options.default_floats.size() == num_fields,
               "protobuf decode options: default_floats size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(options.default_bools.size() == num_fields,
               "protobuf decode options: default_bools size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(options.default_strings.size() == num_fields,
               "protobuf decode options: default_strings size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(options.enum_valid_values.size() == num_fields,
               "protobuf decode options: enum_valid_values size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(options.enum_names.size() == num_fields,
               "protobuf decode options: enum_names size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(options.output_fields.empty() || options.output_fields.size() == num_fields,
               "protobuf decode options: output_fields size mismatch",
               std::invalid_argument);

  std::set<std::pair<int, int>> seen_field_numbers;
  for (size_t i = 0; i < num_fields; ++i) {
    auto const& field = schema[i];
    auto const type   = cudf::data_type{field.output_type};
    CUDF_EXPECTS(field.field_number > 0 && field.field_number <= max_field_number,
                 "protobuf decode options: invalid field number at field " + std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(field.depth >= 0 && field.depth < max_nesting_depth,
                 "protobuf decode options: field depth exceeds limit at field " + std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(field.parent_idx >= -1 && field.parent_idx < static_cast<int>(i),
                 "protobuf decode options: invalid parent index at field " + std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(seen_field_numbers.emplace(field.parent_idx, field.field_number).second,
                 "protobuf decode options: duplicate field number under same parent at field " +
                   std::to_string(i),
                 std::invalid_argument);

    if (field.parent_idx == -1) {
      CUDF_EXPECTS(
        field.depth == 0,
        "protobuf decode options: top-level field must have depth 0 at field " + std::to_string(i),
        std::invalid_argument);
    } else {
      auto const& parent = schema[field.parent_idx];
      CUDF_EXPECTS(field.depth == parent.depth + 1,
                   "protobuf decode options: child depth mismatch at field " + std::to_string(i),
                   std::invalid_argument);
      CUDF_EXPECTS(schema[field.parent_idx].output_type == cudf::type_id::STRUCT,
                   "protobuf decode options: parent must be STRUCT at field " + std::to_string(i),
                   std::invalid_argument);
      if (!options.output_fields.empty()) {
        // A field and its parent must share the same output flag: a hidden STRUCT cannot have
        // visible descendants (the parent would have to be materialized anyway), and a visible
        // STRUCT cannot have hidden children. Forbid the mismatch up front.
        CUDF_EXPECTS(
          options.output_fields[i] == options.output_fields[field.parent_idx],
          "protobuf decode options: child output flag mismatch at field " + std::to_string(i),
          std::invalid_argument);
      }
    }

    CUDF_EXPECTS(
      field.wire_type == proto_wire_type::VARINT || field.wire_type == proto_wire_type::I64BIT ||
        field.wire_type == proto_wire_type::LEN || field.wire_type == proto_wire_type::I32BIT,
      "protobuf decode options: invalid wire type at field " + std::to_string(i),
      std::invalid_argument);
    CUDF_EXPECTS(
      field.encoding >= proto_encoding::DEFAULT && field.encoding <= proto_encoding::ENUM_STRING,
      "protobuf decode options: invalid encoding at field " + std::to_string(i),
      std::invalid_argument);
    CUDF_EXPECTS(!(field.is_repeated && field.is_required),
                 "protobuf decode options: field cannot be both repeated and required at field " +
                   std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(!(field.is_repeated && field.has_default_value),
                 "protobuf decode options: repeated field cannot carry default value at field " +
                   std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(!(field.has_default_value &&
                   (type.id() == cudf::type_id::STRUCT || type.id() == cudf::type_id::LIST)),
                 "protobuf decode options: STRUCT/LIST field cannot carry default value at field " +
                   std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(is_encoding_compatible(field, type),
                 "protobuf decode options: incompatible wire type/encoding/output type at field " +
                   std::to_string(i),
                 std::invalid_argument);

    auto const has_enum_metadata = !options.enum_valid_values[i].empty();
    auto const is_numeric_enum =
      type.id() == cudf::type_id::INT32 && field.encoding == proto_encoding::DEFAULT;
    auto const is_string_enum =
      type.id() == cudf::type_id::STRING && field.encoding == proto_encoding::ENUM_STRING;
    CUDF_EXPECTS(!has_enum_metadata || is_numeric_enum || is_string_enum,
                 "protobuf decode options: enum metadata requires INT32/DEFAULT or "
                 "STRING/ENUM_STRING at field " +
                   std::to_string(i) + " (output_type=" + cudf::type_to_name(type) +
                   ", encoding=" + std::to_string(static_cast<int>(field.encoding)) + ")",
                 std::invalid_argument);

    auto const& enum_values_for_field = options.enum_valid_values[i];
    CUDF_EXPECTS(std::ranges::is_sorted(enum_values_for_field, std::less_equal{}),
                 "protobuf decode options: enum_valid_values must be strictly sorted at field " +
                   std::to_string(i),
                 std::invalid_argument);
    if (!enum_values_for_field.empty() && field.has_default_value) {
      auto const default_value = options.default_ints[i];
      CUDF_EXPECTS(
        std::in_range<int32_t>(default_value) &&
          std::ranges::binary_search(enum_values_for_field, static_cast<int32_t>(default_value)),
        "protobuf decode options: enum default must be present in enum_valid_values "
        "at field " +
          std::to_string(i),
        std::invalid_argument);
    }

    if (field.encoding == proto_encoding::ENUM_STRING) {
      CUDF_EXPECTS(
        !(enum_values_for_field.empty() || options.enum_names[i].empty()),
        "protobuf decode options: enum-as-string field requires non-empty metadata at field " +
          std::to_string(i),
        std::invalid_argument);
      CUDF_EXPECTS(
        enum_values_for_field.size() == options.enum_names[i].size(),
        "protobuf decode options: enum-as-string metadata mismatch at field " + std::to_string(i),
        std::invalid_argument);
    }
  }
}

protobuf_schema::protobuf_schema(decode_protobuf_options const& context) : context_(context)
{
  children_by_parent_.resize(context.schema.size() + 1);
  std::vector<size_t> child_counts(children_by_parent_.size());
  for (auto const& field : context.schema) {
    ++child_counts[field.parent_idx + 1];
  }
  for (size_t parent_idx = 0; parent_idx < children_by_parent_.size(); ++parent_idx) {
    children_by_parent_[parent_idx].reserve(child_counts[parent_idx]);
  }
  for (int schema_idx = 0; schema_idx < static_cast<int>(context.schema.size()); ++schema_idx) {
    children_by_parent_[context.schema[schema_idx].parent_idx + 1].push_back(schema_idx);
  }
}

protobuf_field_meta_view protobuf_schema::field(int schema_idx) const
{
  auto const idx = static_cast<size_t>(schema_idx);
  return {context_.schema.at(idx),
          cudf::data_type{context_.schema.at(idx).output_type},
          context_.default_ints.at(idx),
          context_.default_floats.at(idx),
          context_.default_bools.at(idx),
          context_.default_strings.at(idx),
          context_.enum_valid_values.at(idx),
          context_.enum_names.at(idx)};
}

std::vector<int> const& protobuf_schema::children(int parent_schema_idx) const
{
  CUDF_EXPECTS(parent_schema_idx >= -1 && parent_schema_idx < static_cast<int>(size()),
               "protobuf schema parent index is out of bounds");
  return children_by_parent_[parent_schema_idx + 1];
}

bool protobuf_schema::is_output(int schema_idx) const
{
  auto const idx = static_cast<size_t>(schema_idx);
  return context_.output_fields.empty() || context_.output_fields.at(idx);
}

std::unique_ptr<cudf::column> decode_protobuf(cudf::column_view const& binary_input,
                                              decode_protobuf_options const& options,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  validate_decode_options(options);
  protobuf_schema schema_context{options};
  auto const& schema = schema_context.fields();
  CUDF_EXPECTS(binary_input.type().id() == cudf::type_id::LIST,
               "binary_input must be a LIST<INT8/UINT8> column");
  cudf::lists_column_view const in_list(binary_input);
  auto const child_type = in_list.child().type().id();
  CUDF_EXPECTS(child_type == cudf::type_id::INT8 || child_type == cudf::type_id::UINT8,
               "binary_input must be a LIST<INT8/UINT8> column");

  auto const num_rows   = binary_input.size();
  auto const num_fields = static_cast<int>(schema.size());

  if (num_rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_children;
    for (int i = 0; i < num_fields; i++) {
      if (schema[i].parent_idx != -1 || !schema_context.is_output(i)) { continue; }
      auto field_type  = cudf::data_type{schema[i].output_type};
      auto empty_child = (field_type.id() == cudf::type_id::STRUCT)
                           ? make_empty_struct_column_with_schema(schema_context, i, stream, mr)
                           : make_empty_column_safe(field_type, stream, mr);
      if (schema[i].is_repeated) {
        empty_child = make_empty_list_column(std::move(empty_child), stream, mr);
      }
      empty_children.push_back(std::move(empty_child));
    }
    return cudf::make_structs_column(0,
                                     std::move(empty_children),
                                     0,
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                     stream,
                                     mr);
  }

  // Field extraction is added in follow-up changes; until then every output field is all-null.
  std::vector<std::unique_ptr<cudf::column>> top_level_children;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx != -1 || !schema_context.is_output(i)) { continue; }
    top_level_children.push_back(
      make_null_column_with_schema(schema_context, i, num_rows, stream, mr));
  }

  auto const input_null_count = binary_input.null_count();
  auto struct_mask            = input_null_count > 0
                                  ? cudf::copy_bitmask(binary_input, stream, mr)
                                  : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED);
  return cudf::make_structs_column(
    num_rows, std::move(top_level_children), input_null_count, std::move(struct_mask), stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> decode_protobuf(cudf::column_view const& binary_input,
                                              decode_protobuf_options const& options,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::decode_protobuf(binary_input, options, stream, mr);
}

}  // namespace cudf::io::protobuf
