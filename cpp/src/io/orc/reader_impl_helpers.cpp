/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl_helpers.hpp"

#include <cudf/utilities/memory_resource.hpp>

namespace cudf::io::orc::detail {

std::unique_ptr<column> create_empty_column(size_type orc_col_id,
                                            aggregate_orc_metadata const& metadata,
                                            host_span<std::string const> decimal128_columns,
                                            bool use_np_dtypes,
                                            data_type timestamp_type,
                                            column_name_info& schema_info,
                                            cuda::stream_ref stream)
{
  schema_info.name = metadata.column_name(0, orc_col_id);
  auto const kind  = metadata.get_col_type(orc_col_id).kind;
  auto const type  = to_cudf_type(kind,
                                 use_np_dtypes,
                                 timestamp_type.id(),
                                 to_cudf_decimal_type(decimal128_columns, metadata, orc_col_id));

  switch (kind) {
    case LIST: {
      schema_info.children.emplace_back("offsets");
      schema_info.children.emplace_back("");
      return make_lists_column(0,
                               make_empty_column(type_id::INT32),
                               create_empty_column(metadata.get_col_type(orc_col_id).subtypes[0],
                                                   metadata,
                                                   decimal128_columns,
                                                   use_np_dtypes,
                                                   timestamp_type,
                                                   schema_info.children.back(),
                                                   stream),
                               0,
                               rmm::device_buffer{0, stream});
    }
    case MAP: {
      schema_info.children.emplace_back("offsets");
      schema_info.children.emplace_back("struct");
      auto const child_column_ids = metadata.get_col_type(orc_col_id).subtypes;
      auto& children_schema       = schema_info.children.back().children;
      std::vector<std::unique_ptr<column>> child_columns;
      for (std::size_t idx = 0; idx < metadata.get_col_type(orc_col_id).subtypes.size(); idx++) {
        children_schema.emplace_back("");
        child_columns.push_back(create_empty_column(child_column_ids[idx],
                                                    metadata,
                                                    decimal128_columns,
                                                    use_np_dtypes,
                                                    timestamp_type,
                                                    schema_info.children.back().children.back(),
                                                    stream));
        children_schema[idx].name = get_map_child_col_name(idx);
      }
      return make_lists_column(0,
                               make_empty_column(type_id::INT32),
                               make_structs_column(0,
                                                   std::move(child_columns),
                                                   0,
                                                   rmm::device_buffer{0, stream},
                                                   stream,
                                                   cudf::get_current_device_resource_ref()),
                               0,
                               rmm::device_buffer{0, stream});
    }

    case STRUCT: {
      std::vector<std::unique_ptr<column>> child_columns;
      for (auto const col : metadata.get_col_type(orc_col_id).subtypes) {
        schema_info.children.emplace_back("");
        child_columns.push_back(create_empty_column(col,
                                                    metadata,
                                                    decimal128_columns,
                                                    use_np_dtypes,
                                                    timestamp_type,
                                                    schema_info.children.back(),
                                                    stream));
      }
      return make_structs_column(0,
                                 std::move(child_columns),
                                 0,
                                 rmm::device_buffer{0, stream},
                                 stream,
                                 cudf::get_current_device_resource_ref());
    }

    case DECIMAL: {
      int32_t scale = 0;
      if (type == type_id::DECIMAL32 or type == type_id::DECIMAL64 or type == type_id::DECIMAL128) {
        scale = -static_cast<int32_t>(metadata.get_types()[orc_col_id].scale.value_or(0));
      }
      return make_empty_column(data_type(type, scale));
    }

    default: return make_empty_column(type);
  }
}

column_buffer assemble_buffer(size_type orc_col_id,
                              std::size_t level,
                              reader_column_meta const& col_meta,
                              aggregate_orc_metadata const& metadata,
                              column_hierarchy const& selected_columns,
                              std::vector<std::vector<column_buffer>>& col_buffers,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  auto const col_id = col_meta.orc_col_map[level][orc_col_id];
  auto& col_buffer  = col_buffers[level][col_id];

  col_buffer.name = metadata.column_name(0, orc_col_id);
  auto kind       = metadata.get_col_type(orc_col_id).kind;
  switch (kind) {
    case LIST:
    case STRUCT: {
      auto const& children_indices = selected_columns.children.at(orc_col_id);
      for (auto const child_id : children_indices) {
        col_buffer.children.emplace_back(assemble_buffer(
          child_id, level + 1, col_meta, metadata, selected_columns, col_buffers, stream, mr));
      }
    } break;

    case MAP: {
      std::vector<column_buffer> child_col_buffers;
      // Get child buffers
      auto const& children_indices = selected_columns.children.at(orc_col_id);
      for (std::size_t idx = 0; idx < children_indices.size(); idx++) {
        auto const col = children_indices[idx];
        child_col_buffers.emplace_back(assemble_buffer(
          col, level + 1, col_meta, metadata, selected_columns, col_buffers, stream, mr));
        child_col_buffers.back().name = get_map_child_col_name(idx);
      }
      // Create a struct buffer
      auto num_rows = child_col_buffers[0].size;
      auto struct_buffer =
        column_buffer(cudf::data_type(type_id::STRUCT), num_rows, false, stream, mr);
      struct_buffer.children = std::move(child_col_buffers);
      struct_buffer.name     = "struct";

      col_buffer.children.emplace_back(std::move(struct_buffer));
    } break;

    default: break;
  }

  return std::move(col_buffer);
}

}  // namespace cudf::io::orc::detail
