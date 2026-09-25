/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/protobuf/kernels.cuh"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <string>
#include <unordered_set>

namespace cudf::io::protobuf {

namespace detail {

/**
 * @brief View into a single field's metadata from a decode_protobuf_options.
 */
struct protobuf_field_meta_view {
  nested_field_descriptor const& schema;
  cudf::data_type const output_type;
  int64_t default_int;
  double default_float;
  bool default_bool;
  decode_protobuf_options::byte_vector const& default_string;
  decode_protobuf_options::enum_value_vector const& enum_valid_values;
  decode_protobuf_options::enum_name_vector const& enum_names;
};

namespace {

std::unique_ptr<cudf::column> make_null_column_with_schema(
  std::vector<nested_field_descriptor> const& schema,
  int schema_idx,
  int num_fields,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const& field = schema[schema_idx];
  auto const dtype  = cudf::data_type{schema[schema_idx].output_type};

  if (field.is_repeated) {
    std::unique_ptr<cudf::column> empty_child;
    if (dtype.id() == cudf::type_id::STRUCT) {
      empty_child =
        make_empty_struct_column_with_schema(schema, schema_idx, num_fields, stream, mr);
    } else {
      empty_child = make_empty_column_safe(dtype, stream, mr);
    }
    return make_null_list_column_with_child(std::move(empty_child), num_rows, stream, mr);
  }

  if (dtype.id() == cudf::type_id::STRUCT) {
    auto child_indices = find_child_field_indices(schema, num_fields, schema_idx);
    std::vector<std::unique_ptr<cudf::column>> children;
    for (auto const child_idx : child_indices) {
      children.push_back(
        make_null_column_with_schema(schema, child_idx, num_fields, num_rows, stream, mr));
    }
    auto null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    return cudf::make_structs_column(
      num_rows, std::move(children), num_rows, std::move(null_mask), stream, mr);
  }

  return make_null_column(dtype, num_rows, stream, mr);
}

}  // namespace

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

void validate_decode_options(decode_protobuf_options const& context)
{
  auto const num_fields = context.schema.size();
  if (context.default_ints.size() != num_fields) {
    CUDF_FAIL("protobuf decode context: default_ints size mismatch with schema (" +
                std::to_string(context.default_ints.size()) + " vs " + std::to_string(num_fields) +
                ")",
              std::invalid_argument);
  }
  if (context.default_floats.size() != num_fields) {
    CUDF_FAIL("protobuf decode context: default_floats size mismatch with schema (" +
                std::to_string(context.default_floats.size()) + " vs " +
                std::to_string(num_fields) + ")",
              std::invalid_argument);
  }
  if (context.default_bools.size() != num_fields) {
    CUDF_FAIL("protobuf decode context: default_bools size mismatch with schema (" +
                std::to_string(context.default_bools.size()) + " vs " + std::to_string(num_fields) +
                ")",
              std::invalid_argument);
  }
  if (context.default_strings.size() != num_fields) {
    CUDF_FAIL("protobuf decode context: default_strings size mismatch with schema (" +
                std::to_string(context.default_strings.size()) + " vs " +
                std::to_string(num_fields) + ")",
              std::invalid_argument);
  }
  if (context.enum_valid_values.size() != num_fields) {
    CUDF_FAIL("protobuf decode context: enum_valid_values size mismatch with schema (" +
                std::to_string(context.enum_valid_values.size()) + " vs " +
                std::to_string(num_fields) + ")",
              std::invalid_argument);
  }
  if (context.enum_names.size() != num_fields) {
    CUDF_FAIL("protobuf decode context: enum_names size mismatch with schema (" +
                std::to_string(context.enum_names.size()) + " vs " + std::to_string(num_fields) +
                ")",
              std::invalid_argument);
  }

  std::unordered_set<uint64_t> seen_field_numbers;
  for (size_t i = 0; i < num_fields; ++i) {
    auto const& field = context.schema[i];
    auto const type   = cudf::data_type{field.output_type};
    if (field.field_number <= 0 || field.field_number > max_field_number) {
      CUDF_FAIL("protobuf decode context: invalid field number at field " + std::to_string(i),
                std::invalid_argument);
    }
    if (field.depth < 0 || field.depth >= max_nesting_depth) {
      CUDF_FAIL("protobuf decode context: field depth exceeds supported limit at field " +
                  std::to_string(i),
                std::invalid_argument);
    }
    if (field.parent_idx < -1 || field.parent_idx >= static_cast<int>(i)) {
      CUDF_FAIL("protobuf decode context: invalid parent index at field " + std::to_string(i),
                std::invalid_argument);
    }
    auto const key = (static_cast<uint64_t>(static_cast<uint32_t>(field.parent_idx)) << 32) |
                     static_cast<uint64_t>(field.field_number);
    if (!seen_field_numbers.insert(key).second) {
      CUDF_FAIL("protobuf decode context: duplicate field number under same parent at field " +
                  std::to_string(i),
                std::invalid_argument);
    }

    if (field.parent_idx == -1) {
      if (field.depth != 0) {
        CUDF_FAIL("protobuf decode context: top-level field must have depth 0 at field " +
                    std::to_string(i),
                  std::invalid_argument);
      }
    } else {
      auto const& parent = context.schema[field.parent_idx];
      if (field.depth != parent.depth + 1) {
        CUDF_FAIL("protobuf decode context: child depth mismatch at field " + std::to_string(i),
                  std::invalid_argument);
      }
      if (cudf::data_type{context.schema[field.parent_idx].output_type}.id() !=
          cudf::type_id::STRUCT) {
        CUDF_FAIL("protobuf decode context: parent must be STRUCT at field " + std::to_string(i),
                  std::invalid_argument);
      }
    }

    if (field.wire_type != proto_wire_type::VARINT && field.wire_type != proto_wire_type::I64BIT &&
        field.wire_type != proto_wire_type::LEN && field.wire_type != proto_wire_type::I32BIT) {
      CUDF_FAIL("protobuf decode context: invalid wire type at field " + std::to_string(i),
                std::invalid_argument);
    }
    if (field.encoding < proto_encoding::DEFAULT || field.encoding > proto_encoding::ENUM_STRING) {
      CUDF_FAIL("protobuf decode context: invalid encoding at field " + std::to_string(i),
                std::invalid_argument);
    }
    if (field.is_repeated && field.is_required) {
      CUDF_FAIL("protobuf decode context: field cannot be both repeated and required at field " +
                  std::to_string(i),
                std::invalid_argument);
    }
    if (context.default_bools[i] > 1) {
      CUDF_FAIL("protobuf decode context: default_bools must contain only 0 or 1 at field " +
                  std::to_string(i),
                std::invalid_argument);
    }
    if (field.is_repeated && field.has_default_value) {
      CUDF_FAIL("protobuf decode context: repeated field cannot carry default value at field " +
                  std::to_string(i),
                std::invalid_argument);
    }
    if (field.has_default_value &&
        (type.id() == cudf::type_id::STRUCT || type.id() == cudf::type_id::LIST)) {
      CUDF_FAIL("protobuf decode context: STRUCT/LIST field cannot carry default value at field " +
                  std::to_string(i),
                std::invalid_argument);
    }
    if (!is_encoding_compatible(field, type)) {
      CUDF_FAIL("protobuf decode context: incompatible wire type/encoding/output type at field " +
                  std::to_string(i),
                std::invalid_argument);
    }

    if (field.encoding == proto_encoding::ENUM_STRING) {
      if (context.enum_valid_values[i].empty() || context.enum_names[i].empty()) {
        CUDF_FAIL(
          "protobuf decode context: enum-as-string field requires non-empty metadata at field " +
            std::to_string(i),
          std::invalid_argument);
      }
      if (context.enum_valid_values[i].size() != context.enum_names[i].size()) {
        CUDF_FAIL(
          "protobuf decode context: enum-as-string metadata mismatch at field " + std::to_string(i),
          std::invalid_argument);
      }
      auto const& ev = context.enum_valid_values[i];
      for (size_t j = 1; j < ev.size(); ++j) {
        if (ev[j] <= ev[j - 1]) {
          CUDF_FAIL("protobuf decode context: enum_valid_values must be strictly sorted at field " +
                      std::to_string(i),
                    std::invalid_argument);
        }
      }
    }
  }
}

protobuf_field_meta_view make_field_meta_view(decode_protobuf_options const& context,
                                              int schema_idx)
{
  auto const idx = static_cast<size_t>(schema_idx);
  return protobuf_field_meta_view{context.schema.at(idx),
                                  cudf::data_type{context.schema.at(idx).output_type},
                                  context.default_ints.at(idx),
                                  context.default_floats.at(idx),
                                  context.default_bools.at(idx) != 0,
                                  context.default_strings.at(idx),
                                  context.enum_valid_values.at(idx),
                                  context.enum_names.at(idx)};
}

std::unique_ptr<cudf::column> decode_protobuf(cudf::column_view const& binary_input,
                                              decode_protobuf_options const& context,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  validate_decode_options(context);
  auto const& schema = context.schema;
  CUDF_EXPECTS(binary_input.type().id() == cudf::type_id::LIST,
               "binary_input must be a LIST<INT8/UINT8> column");
  cudf::lists_column_view const in_list(binary_input);
  auto const child_type = in_list.child().type().id();
  CUDF_EXPECTS(child_type == cudf::type_id::INT8 || child_type == cudf::type_id::UINT8,
               "binary_input must be a LIST<INT8/UINT8> column");

  auto const num_rows   = binary_input.size();
  auto const num_fields = static_cast<int>(schema.size());

  if (num_fields == 0) {
    auto const input_null_count = binary_input.null_count();
    if (input_null_count > 0) {
      auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
      return cudf::make_structs_column(
        num_rows, {}, input_null_count, std::move(null_mask), stream, mr);
    }
    return cudf::make_structs_column(num_rows, {}, 0, rmm::device_buffer{}, stream, mr);
  }

  if (num_rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_children;
    for (int i = 0; i < num_fields; i++) {
      if (schema[i].parent_idx == -1) {
        auto field_type = cudf::data_type{schema[i].output_type};
        if (schema[i].is_repeated && field_type.id() == cudf::type_id::STRUCT) {
          auto empty_struct =
            make_empty_struct_column_with_schema(schema, i, num_fields, stream, mr);
          empty_children.push_back(make_empty_list_column(std::move(empty_struct), stream, mr));
        } else if (schema[i].is_repeated) {
          auto empty_child = make_empty_column_safe(field_type, stream, mr);
          empty_children.push_back(make_empty_list_column(std::move(empty_child), stream, mr));
        } else if (field_type.id() == cudf::type_id::STRUCT) {
          empty_children.push_back(
            make_empty_struct_column_with_schema(schema, i, num_fields, stream, mr));
        } else {
          empty_children.push_back(make_empty_column_safe(field_type, stream, mr));
        }
      }
    }
    return cudf::make_structs_column(
      0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
  }

  std::vector<std::unique_ptr<cudf::column>> top_level_children;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == -1) {
      top_level_children.push_back(
        make_null_column_with_schema(schema, i, num_fields, num_rows, stream, mr));
    }
  }

  auto const input_null_count = binary_input.null_count();
  if (input_null_count > 0) {
    auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
    return cudf::make_structs_column(
      num_rows, std::move(top_level_children), input_null_count, std::move(null_mask), stream, mr);
  }

  return cudf::make_structs_column(
    num_rows, std::move(top_level_children), 0, rmm::device_buffer{}, stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> decode_protobuf(cudf::column_view const& binary_input,
                                              decode_protobuf_options const& context,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::decode_protobuf(binary_input, context, stream, mr);
}

}  // namespace cudf::io::protobuf
