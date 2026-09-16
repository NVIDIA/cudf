/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/orc/aggregate_orc_metadata.hpp"
#include "io/orc/orc.hpp"
#include "io/utilities/column_buffer.hpp"

#include <cuda/stream>

#include <memory>
#include <vector>

namespace cudf::io::orc::detail {
using namespace cudf::io::detail;

/**
 * @brief Keeps track of orc mapping and child column details.
 */
struct reader_column_meta {
  // Mapping between column id in orc to processing order.
  std::vector<std::vector<size_type>> orc_col_map;

  // Number of rows in child columns.
  std::vector<int64_t> num_child_rows;

  // Consists of parent column valid_map and null count.
  std::vector<column_validity_info> parent_column_data;

  std::vector<size_type> parent_column_index;

  // Start row of child columns [stripe][column].
  std::vector<int64_t> child_start_row;

  // Number of rows of child columns [stripe][column].
  std::vector<int64_t> num_child_rows_per_stripe;

  struct row_group_meta {
    size_type num_rows;  // number of rows in a column in a row group
    int64_t start_row;   // start row in a column in a row group
  };

  // Row group metadata [rowgroup][column].
  std::vector<row_group_meta> rwgrp_meta;
};

/**
 * @brief Function that translates ORC data kind to cuDF type enum
 */
inline constexpr type_id to_cudf_type(TypeKind kind,
                                      bool use_np_dtypes,
                                      type_id timestamp_type_id,
                                      type_id decimal_type_id)
{
  switch (kind) {
    case BOOLEAN: return type_id::BOOL8;
    case BYTE: return type_id::INT8;
    case SHORT: return type_id::INT16;
    case INT: return type_id::INT32;
    case LONG: return type_id::INT64;
    case FLOAT: return type_id::FLOAT32;
    case DOUBLE: return type_id::FLOAT64;
    case STRING:
    case BINARY:
    case VARCHAR:
    case CHAR:
      // Variable-length types can all be mapped to STRING
      return type_id::STRING;
    case TIMESTAMP:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_NANOSECONDS;
    case DATE:
      // There isn't a (DAYS -> np.dtype) mapping
      return (use_np_dtypes) ? type_id::TIMESTAMP_MILLISECONDS : type_id::TIMESTAMP_DAYS;
    case DECIMAL: return decimal_type_id;
    // Need to update once cuDF plans to support map type
    case MAP:
    case LIST: return type_id::LIST;
    case STRUCT: return type_id::STRUCT;
    default: break;
  }

  return type_id::EMPTY;
}

/**
 * @brief Determines cuDF type of an ORC Decimal column.
 */
inline type_id to_cudf_decimal_type(host_span<std::string const> decimal128_columns,
                                    aggregate_orc_metadata const& metadata,
                                    int column_index)
{
  if (metadata.get_col_type(column_index).kind != DECIMAL) { return type_id::EMPTY; }

  if (std::find(decimal128_columns.begin(),
                decimal128_columns.end(),
                metadata.column_path(0, column_index)) != decimal128_columns.end()) {
    return type_id::DECIMAL128;
  }

  auto const precision = metadata.get_col_type(column_index)
                           .precision.value_or(cuda::std::numeric_limits<int64_t>::digits10);
  if (precision <= cuda::std::numeric_limits<int32_t>::digits10) { return type_id::DECIMAL32; }
  if (precision <= cuda::std::numeric_limits<int64_t>::digits10) { return type_id::DECIMAL64; }
  return type_id::DECIMAL128;
}

inline std::string get_map_child_col_name(std::size_t const idx)
{
  return (idx == 0) ? "key" : "value";
}

/**
 * @brief Create empty columns and respective schema information from the buffer.
 */
std::unique_ptr<column> create_empty_column(size_type orc_col_id,
                                            aggregate_orc_metadata const& metadata,
                                            host_span<std::string const> decimal128_columns,
                                            bool use_np_dtypes,
                                            data_type timestamp_type,
                                            column_name_info& schema_info,
                                            cuda::stream_ref stream);

/**
 * @brief Assemble the buffer with child columns.
 */
column_buffer assemble_buffer(size_type orc_col_id,
                              std::size_t level,
                              reader_column_meta const& col_meta,
                              aggregate_orc_metadata const& metadata,
                              column_hierarchy const& selected_columns,
                              std::vector<std::vector<column_buffer>>& col_buffers,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

}  // namespace cudf::io::orc::detail
