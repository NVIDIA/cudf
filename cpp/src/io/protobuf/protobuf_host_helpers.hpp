/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/io/protobuf.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cudf::io::protobuf::detail {

struct protobuf_field_meta_view {
  nested_field_descriptor const& schema;
  cudf::data_type const output_type;
  int64_t default_int;
  double default_float;
  bool default_bool;
  std::vector<uint8_t> const& default_string;
  std::vector<int32_t> const& enum_valid_values;
  std::vector<std::vector<uint8_t>> const& enum_names;
};

bool is_encoding_compatible(nested_field_descriptor const& field, cudf::data_type const& type);

void validate_decode_options(decode_protobuf_options const& options);

class protobuf_schema {
 public:
  explicit protobuf_schema(decode_protobuf_options const& context);
  protobuf_schema(decode_protobuf_options&&)       = delete;
  protobuf_schema(decode_protobuf_options const&&) = delete;

  protobuf_schema(protobuf_schema const&)            = delete;
  protobuf_schema& operator=(protobuf_schema const&) = delete;
  protobuf_schema(protobuf_schema&&)                 = delete;
  protobuf_schema& operator=(protobuf_schema&&)      = delete;

  [[nodiscard]] std::vector<nested_field_descriptor> const& fields() const
  {
    return context_.schema;
  }

  [[nodiscard]] nested_field_descriptor const& operator[](int schema_idx) const
  {
    return context_.schema.at(static_cast<size_t>(schema_idx));
  }

  [[nodiscard]] size_t size() const { return context_.schema.size(); }

  [[nodiscard]] protobuf_field_meta_view field(int schema_idx) const;
  [[nodiscard]] std::vector<int> const& children(int parent_schema_idx) const;
  [[nodiscard]] bool is_output(int schema_idx) const;

 private:
  // The decode options outlive this stack-scoped facade.
  decode_protobuf_options const& context_;
  std::vector<std::vector<int>> children_by_parent_;
};

inline std::unique_ptr<cudf::column> make_offsets_column(cudf::size_type num_rows,
                                                         rmm::device_uvector<int32_t>&& offsets)
{
  CUDF_EXPECTS(offsets.size() == static_cast<size_t>(num_rows) + 1,
               std::string{__func__} + ": offsets size must match row count");
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                        num_rows + 1,
                                        offsets.release(),
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                        0);
}

inline std::vector<int> const& find_child_field_indices(protobuf_schema const& schema,
                                                        int parent_idx)
{
  return schema.children(parent_idx);
}

// Forward declarations needed by make_empty_struct_column_with_schema
std::unique_ptr<cudf::column> make_empty_column_safe(cudf::data_type dtype,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> make_empty_list_column(std::unique_ptr<cudf::column> element_col,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr);

// Forward declaration for the mutual recursion with make_empty_struct_column_from_children.
template <typename SchemaT>
std::unique_ptr<cudf::column> make_empty_struct_column_with_schema(
  SchemaT const& schema,
  int parent_idx,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

// Build an empty (0-row) STRUCT column from an explicit child-index list, recursing into
// STRUCT children and wrapping repeated fields in an empty LIST.
template <typename SchemaT>
std::unique_ptr<cudf::column> make_empty_struct_column_from_children(
  SchemaT const& schema,
  std::vector<int> const& child_indices,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> children;
  for (int child_idx : child_indices) {
    auto child_type = cudf::data_type{schema[child_idx].output_type};

    std::unique_ptr<cudf::column> child_col;
    if (child_type.id() == cudf::type_id::STRUCT) {
      child_col = make_empty_struct_column_with_schema(schema, child_idx, stream, mr);
    } else {
      child_col = make_empty_column_safe(child_type, stream, mr);
    }

    if (schema[child_idx].is_repeated) {
      child_col = make_empty_list_column(std::move(child_col), stream, mr);
    }

    children.push_back(std::move(child_col));
  }

  return cudf::make_structs_column(0,
                                   std::move(children),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                   stream,
                                   mr);
}

template <typename SchemaT>
std::unique_ptr<cudf::column> make_empty_struct_column_with_schema(
  SchemaT const& schema, int parent_idx, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto const& child_indices = find_child_field_indices(schema, parent_idx);
  return make_empty_struct_column_from_children(schema, child_indices, stream, mr);
}

std::unique_ptr<cudf::column> make_null_column(cudf::data_type dtype,
                                               cudf::size_type num_rows,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr);

// Schema-aware all-null builder: recurses into STRUCT children and wraps repeated fields
// in a null-list, mirroring the shape `make_empty_struct_column_with_schema` would produce
// but with `num_rows` all-null rows.
std::unique_ptr<cudf::column> make_null_column_with_schema(protobuf_schema const& schema,
                                                           int schema_idx,
                                                           cudf::size_type num_rows,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> make_null_list_column_with_child(
  std::unique_ptr<cudf::column> child_col,
  cudf::size_type num_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::io::protobuf::detail
