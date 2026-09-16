/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file writer_impl_helpers.cpp
 * @brief Helper function implementation for Parquet writer
 */

#include "writer_impl_helpers.hpp"

#include "io/parquet/parquet_gpu.hpp"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <algorithm>
#include <format>
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

std::optional<size_type> compute_smaller_fragment_size(
  cudf::detail::host_2dspan<PageFragment const> fragments,
  host_span<parquet_column_device_view const> col_desc,
  size_type input_fragment_size)
{
  auto fragment_size     = input_fragment_size;
  auto const num_columns = fragments.size().first;

  for (auto col_idx = 0; std::cmp_less(col_idx, num_columns); ++col_idx) {
    for (auto const& frag : fragments[col_idx]) {
      auto const page_size =
        max_fragment_page_size(frag.fragment_data_size, frag.num_values, col_desc[col_idx]);
      if (page_size <= MAX_PARQUET_PAGE_SIZE) { continue; }

      CUDF_EXPECTS(frag.num_rows > 1,
                   std::format("A single row of column {} needs {} bytes, which exceeds the "
                               "maximum Parquet page size of {} bytes",
                               col_idx,
                               page_size,
                               MAX_PARQUET_PAGE_SIZE),
                   std::overflow_error);

      // Scale the row span down by the overshoot, then halve it again so that columns with
      // uneven row lengths are less likely to need another pass. Halving also bounds the number
      // of passes to the width of `size_type`.
      auto const scaled_row_span = util::div_rounding_up_safe<size_t>(
        static_cast<size_t>(frag.num_rows) * MAX_PARQUET_PAGE_SIZE, page_size * 2);
      fragment_size = std::min<size_type>(fragment_size, std::max<size_t>(1, scaled_row_span));
    }
  }

  return fragment_size < input_fragment_size ? std::optional{fragment_size} : std::nullopt;
}

}  // namespace cudf::io::parquet::detail
