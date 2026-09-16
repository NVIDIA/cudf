/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/json.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf {
class column_device_view;
class lists_column_view;
class string_scalar;
class string_view;
class table_view;

namespace io::json::detail {

std::unique_ptr<column> make_escaped_json_strings(column_device_view const& d_column,
                                                  size_type size,
                                                  size_type null_count,
                                                  rmm::device_buffer null_mask,
                                                  bool append_colon,
                                                  bool escaped_utf8,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr);

std::unique_ptr<column> string_to_strings(column_view const& column,
                                          bool escaped_utf8,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr);

std::unique_ptr<column> timestamp_to_strings(column_view const& column,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr);

std::unique_ptr<column> duration_to_strings(column_view const& column,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

std::unique_ptr<column> struct_to_strings(table_view const& strings_columns,
                                          column_view const& column_names,
                                          size_type num_rows,
                                          string_view row_prefix,
                                          string_view row_suffix,
                                          string_view value_separator,
                                          string_scalar const& narep,
                                          bool include_nulls,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr);

std::unique_ptr<column> join_list_of_strings(lists_column_view const& lists_strings,
                                             string_view list_prefix,
                                             string_view list_suffix,
                                             string_view element_separator,
                                             string_view element_narep,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr);

std::unique_ptr<column> leaf_column_to_strings(column_view const& column,
                                               json_writer_options const& options,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr);

}  // namespace io::json::detail
}  // namespace cudf
