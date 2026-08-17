/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file writer_impl_helpers.cpp
 * @brief Helper function implementation for Parquet writer
 */

#include "writer_impl_helpers.hpp"

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <functional>
#include <string>

namespace cudf::io::parquet::detail {

using namespace cudf::io::detail;

void fill_table_meta(table_input_metadata& table_meta)
{
  // Fill unnamed columns' names in table_meta
  std::function<void(column_in_metadata&, std::string)> add_default_name =
    [&](column_in_metadata& col_meta, std::string default_name) {
      if (col_meta.get_name().empty()) col_meta.set_name(default_name);
      for (size_type i = 0; i < col_meta.num_children(); ++i) {
        add_default_name(col_meta.child(i), col_meta.get_name() + "_" + std::to_string(i));
      }
    };
  for (size_t i = 0; i < table_meta.column_metadata.size(); ++i) {
    add_default_name(table_meta.column_metadata[i], "_col" + std::to_string(i));
  }
}

[[nodiscard]] size_t column_size(column_view const& column, cuda::stream_ref stream)
{
  if (column.is_empty()) { return 0; }

  if (is_fixed_width(column.type())) {
    return size_of(column.type()) * column.size();
  } else if (column.type().id() == type_id::STRING) {
    auto const scol = strings_column_view(column);
    return cudf::strings::detail::get_offset_value(
             scol.offsets(), column.size() + column.offset(), stream) -
           cudf::strings::detail::get_offset_value(scol.offsets(), column.offset(), stream);
  } else if (column.type().id() == type_id::STRUCT) {
    auto const scol = structs_column_view(column);
    size_t ret      = 0;
    for (int i = 0; i < scol.num_children(); i++) {
      ret += column_size(scol.get_sliced_child(i, stream), stream);
    }
    return ret;
  } else if (column.type().id() == type_id::LIST) {
    auto const lcol = lists_column_view(column);
    return column_size(lcol.get_sliced_child(stream), stream);
  }

  CUDF_FAIL("Unexpected compound type");
}

[[nodiscard]] bool is_output_column_nullable(cudf::detail::LinkedColPtr const& column,
                                             column_in_metadata const& column_metadata,
                                             single_write_mode write_mode)
{
  if (column_metadata.is_nullability_defined()) {
    CUDF_EXPECTS(column_metadata.nullable() or column->null_count() == 0,
                 "Mismatch in metadata prescribed nullability and input column. "
                 "Metadata for input column with nulls cannot prescribe nullability = false");
    return column_metadata.nullable();
  }
  // For chunked write, when not provided nullability, we assume the worst case scenario
  // that all columns are nullable.
  return write_mode == single_write_mode::NO or column->nullable();
}

}  // namespace cudf::io::parquet::detail
