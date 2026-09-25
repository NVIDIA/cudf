/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/protobuf/types.cuh"

#include <cudf/column/column_factories.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace cudf::io::protobuf::detail {

// ============================================================================
// Field number lookup table helpers
// ============================================================================

/**
 * Build a host-side direct-mapped lookup table: field_number -> index.
 * @param get_field_number Callable: (int i) -> field_number for the i-th entry.
 * @param num_entries Number of entries.
 * @return Empty vector if the max field number exceeds the threshold.
 */
template <typename FieldNumberFn>
inline std::vector<int> build_lookup_table(FieldNumberFn get_field_number, int num_entries)
{
  int max_fn = 0;
  for (int i = 0; i < num_entries; i++) {
    max_fn = std::max(max_fn, get_field_number(i));
  }
  if (max_fn > FIELD_LOOKUP_TABLE_MAX) { return {}; }
  std::vector<int> table(max_fn + 1, -1);
  for (int i = 0; i < num_entries; i++) {
    table[get_field_number(i)] = i;
  }
  return table;
}

inline std::vector<int> build_index_lookup_table(nested_field_descriptor const* schema,
                                                 int const* field_indices,
                                                 int num_indices)
{
  return build_lookup_table([&](int i) { return schema[field_indices[i]].field_number; },
                            num_indices);
}

inline std::vector<int> build_field_lookup_table(field_descriptor const* descs, int num_fields)
{
  return build_lookup_table([&](int i) { return descs[i].field_number; }, num_fields);
}

/**
 * Find all child field indices for a given parent index in the schema.
 * This is a commonly used pattern throughout the codebase.
 *
 * @param schema The schema vector (either nested_field_descriptor or
 * device_nested_field_descriptor)
 * @param num_fields Number of fields in the schema
 * @param parent_idx The parent index to search for
 * @return Vector of child field indices
 */
template <typename SchemaT>
std::vector<int> find_child_field_indices(SchemaT const& schema, int num_fields, int parent_idx)
{
  std::vector<int> child_indices;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == parent_idx) { child_indices.push_back(i); }
  }
  return child_indices;
}

// Forward declarations needed by make_empty_struct_column_with_schema
std::unique_ptr<cudf::column> make_empty_column_safe(cudf::data_type dtype,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> make_empty_list_column(std::unique_ptr<cudf::column> element_col,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

template <typename SchemaT>
std::unique_ptr<cudf::column> make_empty_struct_column_with_schema(
  SchemaT const& schema,
  int parent_idx,
  int num_fields,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto child_indices = find_child_field_indices(schema, num_fields, parent_idx);

  std::vector<std::unique_ptr<cudf::column>> children;
  for (int child_idx : child_indices) {
    auto child_type = cudf::data_type{schema[child_idx].output_type};

    std::unique_ptr<cudf::column> child_col;
    if (child_type.id() == cudf::type_id::STRUCT) {
      child_col = make_empty_struct_column_with_schema(schema, child_idx, num_fields, stream, mr);
    } else {
      child_col = make_empty_column_safe(child_type, stream, mr);
    }

    if (schema[child_idx].is_repeated) {
      child_col = make_empty_list_column(std::move(child_col), stream, mr);
    }

    children.push_back(std::move(child_col));
  }

  return cudf::make_structs_column(0, std::move(children), 0, rmm::device_buffer{}, stream, mr);
}

void maybe_check_required_fields(field_location const* locations,
                                 std::vector<int> const& field_indices,
                                 std::vector<nested_field_descriptor> const& schema,
                                 int num_rows,
                                 cudf::bitmask_type const* input_null_mask,
                                 cudf::size_type input_offset,
                                 field_location const* parent_locs,
                                 bool* row_force_null,
                                 int32_t const* top_row_indices,
                                 int* error_flag,
                                 rmm::cuda_stream_view stream);

void propagate_invalid_enum_flags_to_rows(rmm::device_uvector<bool> const& item_invalid,
                                          rmm::device_uvector<bool>& row_invalid,
                                          int num_items,
                                          int32_t const* top_row_indices,
                                          bool propagate_to_rows,
                                          rmm::cuda_stream_view stream);

void validate_enum_and_propagate_rows(rmm::device_uvector<int32_t> const& values,
                                      rmm::device_uvector<bool>& valid,
                                      decode_protobuf_options::enum_value_vector const& valid_enums,
                                      rmm::device_uvector<bool>& row_invalid,
                                      int num_items,
                                      int32_t const* top_row_indices,
                                      bool propagate_to_rows,
                                      rmm::cuda_stream_view stream);

// ============================================================================
// Forward declarations of builder/utility functions
// ============================================================================

std::unique_ptr<cudf::column> make_null_column(cudf::data_type dtype,
                                               cudf::size_type num_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> make_null_list_column_with_child(
  std::unique_ptr<cudf::column> child_col,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_enum_string_column(
  rmm::device_uvector<int32_t>& enum_values,
  rmm::device_uvector<bool>& valid,
  decode_protobuf_options::enum_value_vector const& valid_enums,
  decode_protobuf_options::enum_name_vector const& enum_name_bytes,
  rmm::device_uvector<bool>& d_row_force_null,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices = nullptr,
  bool propagate_invalid_rows    = true);

// Complex builder forward declarations
std::unique_ptr<cudf::column> build_repeated_enum_string_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  rmm::device_uvector<int32_t> const& d_field_counts,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  decode_protobuf_options::enum_value_vector const& valid_enums,
  decode_protobuf_options::enum_name_vector const& enum_name_bytes,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_repeated_string_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  device_nested_field_descriptor const& field_desc,
  rmm::device_uvector<int32_t> const& d_field_counts,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  bool is_bytes,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_nested_struct_column(
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  rmm::device_uvector<field_location> const& d_parent_locs,
  std::vector<int> const& child_field_indices,
  std::vector<nested_field_descriptor> const& schema,
  int num_fields,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<uint8_t> const& default_bools,
  std::vector<decode_protobuf_options::byte_vector> const& default_strings,
  std::vector<decode_protobuf_options::enum_value_vector> const& enum_valid_values,
  std::vector<decode_protobuf_options::enum_name_vector> const& enum_names,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<int>& d_error,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices,
  int depth,
  bool propagate_invalid_rows = true);

std::unique_ptr<cudf::column> build_repeated_child_list_column(
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  field_location const* parent_locs,
  int num_parent_rows,
  int child_schema_idx,
  std::vector<nested_field_descriptor> const& schema,
  int num_fields,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<uint8_t> const& default_bools,
  std::vector<decode_protobuf_options::byte_vector> const& default_strings,
  std::vector<decode_protobuf_options::enum_value_vector> const& enum_valid_values,
  std::vector<decode_protobuf_options::enum_name_vector> const& enum_names,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices,
  int depth,
  bool propagate_invalid_rows = true);

std::unique_ptr<cudf::column> build_repeated_struct_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  device_nested_field_descriptor const& field_desc,
  rmm::device_uvector<int32_t> const& d_field_counts,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  std::vector<device_nested_field_descriptor> const& h_device_schema,
  std::vector<int> const& child_field_indices,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<uint8_t> const& default_bools,
  std::vector<decode_protobuf_options::byte_vector> const& default_strings,
  std::vector<nested_field_descriptor> const& schema,
  std::vector<decode_protobuf_options::enum_value_vector> const& enum_valid_values,
  std::vector<decode_protobuf_options::enum_name_vector> const& enum_names,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<int>& d_error_top,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::io::protobuf::detail
