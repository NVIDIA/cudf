/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file write_json.cu
 * @brief cuDF-IO JSON writer implementation
 */

#include "io/json/write_json.hpp"
#include "io/utilities/parsing_utils.cuh"
#include "lists/utilities.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/detail/utils.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <memory>
namespace cudf::io::json::detail {
namespace {

/**
 * @brief Functor to modify a string column for JSON format.
 *
 * This will convert escape characters and wrap quotes around strings.
 */
struct escape_strings_fn {
  column_device_view const d_column;
  bool const append_colon{false};
  bool const escaped_utf8{true};
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void write_char(char_utf8 chr, char*& d_buffer, size_type& bytes)
  {
    if (d_buffer)
      d_buffer += cudf::strings::detail::from_char_utf8(chr, d_buffer);
    else
      bytes += cudf::strings::detail::bytes_in_char_utf8(chr);
  }

  __device__ inline char nibble_to_hex(uint8_t nibble) const
  {
    return nibble < 10 ? '0' + nibble : 'a' + nibble - 10;
  }

  __device__ void write_utf8_codepoint(uint16_t codepoint, char*& d_buffer, size_type& bytes)
  {
    if (d_buffer) {
      d_buffer[0] = '\\';
      d_buffer[1] = 'u';
      d_buffer[2] = nibble_to_hex((codepoint >> 12) & 0x0F);
      d_buffer[3] = nibble_to_hex((codepoint >> 8) & 0x0F);
      d_buffer[4] = nibble_to_hex((codepoint >> 4) & 0x0F);
      d_buffer[5] = nibble_to_hex((codepoint) & 0x0F);
      d_buffer += 6;
    } else {
      bytes += 6;
    }
  }

  __device__ void write_utf16_codepoint(uint32_t codepoint, char*& d_buffer, size_type& bytes)
  {
    constexpr uint16_t UTF16_HIGH_SURROGATE_BEGIN = 0xD800;
    constexpr uint16_t UTF16_LOW_SURROGATE_BEGIN  = 0xDC00;
    codepoint -= 0x1'0000;
    uint16_t hex_high = ((codepoint >> 10) & 0x3FF) + UTF16_HIGH_SURROGATE_BEGIN;
    uint16_t hex_low  = (codepoint & 0x3FF) + UTF16_LOW_SURROGATE_BEGIN;
    write_utf8_codepoint(hex_high, d_buffer, bytes);
    write_utf8_codepoint(hex_low, d_buffer, bytes);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    auto const d_str = d_column.element<string_view>(idx);

    // entire string must be double-quoted.
    constexpr char_utf8 const quote = '\"';  // wrap quotes
    bool constexpr quote_row        = true;

    char* d_buffer  = d_chars ? d_chars + d_offsets[idx] : nullptr;
    size_type bytes = 0;

    if (quote_row) write_char(quote, d_buffer, bytes);
    for (auto utf8_char : d_str) {
      if (utf8_char > 0x0000'00FF) {
        if (!escaped_utf8) {
          // write original utf8 character if unescaping is enabled
          write_char(utf8_char, d_buffer, bytes);
          continue;
        }
        // multi-byte char
        uint32_t codepoint = cudf::strings::detail::utf8_to_codepoint(utf8_char);
        if (codepoint <= 0x0000'FFFF) {
          // write \uXXXX utf-8 codepoint
          write_utf8_codepoint(codepoint, d_buffer, bytes);
        } else {
          // write \uXXXX\uXXXX utf-16 surrogate pair
          // codepoint > 0xFFFF && codepoint <= 0x10FFFF
          write_utf16_codepoint(codepoint, d_buffer, bytes);
        }
        continue;
      }
      auto escaped_chars = get_escaped_char(utf8_char);
      if (escaped_chars.first == '\0') {
        write_char(escaped_chars.second, d_buffer, bytes);
      } else {
        write_char(escaped_chars.first, d_buffer, bytes);
        write_char(escaped_chars.second, d_buffer, bytes);
      }
    }
    if (quote_row) write_char(quote, d_buffer, bytes);
    constexpr char_utf8 const colon = ':';  // append colon
    if (append_colon) write_char(colon, d_buffer, bytes);

    if (!d_chars) { d_sizes[idx] = bytes; }
  }

  std::unique_ptr<column> make_strings_column(size_type size,
                                              size_type null_count,
                                              rmm::device_buffer null_mask,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
  {
    if (size == 0) {  // empty begets empty
      return make_empty_column(type_id::STRING);
    }
    auto [offsets_column, chars] =
      cudf::strings::detail::make_strings_children(*this, size, stream, mr);

    return cudf::make_strings_column(
      size, std::move(offsets_column), chars.release(), null_count, std::move(null_mask));
  }
};

}  // namespace

std::unique_ptr<column> make_escaped_json_strings(column_device_view const& d_column,
                                                  size_type size,
                                                  size_type null_count,
                                                  rmm::device_buffer null_mask,
                                                  bool append_colon,
                                                  bool escaped_utf8,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  return escape_strings_fn{d_column, append_colon, escaped_utf8}.make_strings_column(
    size, null_count, std::move(null_mask), stream, mr);
}

namespace {

// Struct - scatter string_views of each element in a struct column
struct struct_scatter_strings_fn {
  table_device_view const tbl;
  column_device_view const col_names;
  size_type const strviews_per_column;
  size_type const num_strviews_per_row;
  string_view const row_prefix;       // "{"
  string_view const row_suffix;       // "}" or "}\n" for json-lines
  string_view const value_separator;  // ","
  string_view const narep;            // null entry replacement
  bool const include_nulls;
  string_view* d_strviews;

  /**
   * @brief Scatters string_views for each element in a struct column
   *
   * @param idx Column-major index of the element to scatter
   */
  __device__ void operator()(size_type idx)
  {
    auto const row        = idx / tbl.num_columns();
    auto const col        = idx % tbl.num_columns();
    auto const d_str_null = tbl.column(col).is_null(row);
    auto const this_index = row * num_strviews_per_row + col * strviews_per_column + 1;
    // prefix
    if (col == 0) d_strviews[this_index - 1] = row_prefix;
    if (col != 0) d_strviews[this_index - 1] = include_nulls ? value_separator : string_view{};
    if (!include_nulls && d_str_null) {
      d_strviews[this_index]     = string_view{};
      d_strviews[this_index + 1] = string_view{};
    } else {
      auto const d_col_name = col_names.element<string_view>(col);
      auto const d_str = d_str_null ? narep : tbl.column(col).template element<string_view>(row);
      // column_name: value
      d_strviews[this_index]     = d_col_name;
      d_strviews[this_index + 1] = d_str;
    }
    // suffix
    if (col == tbl.num_columns() - 1) { d_strviews[this_index + 2] = row_suffix; }
  }
};

struct validity_fn {
  table_device_view const tbl;
  __device__ bool operator()(size_type idx) const
  {
    auto const row = idx / tbl.num_columns();
    auto const col = idx % tbl.num_columns();
    return tbl.column(col).is_valid(row);
  }
};

}  // namespace

/**
 * @brief Concatenate the strings from each row of the given table as structs in JSON string
 *
 * Each row will be struct with field name as column names and values from each column in the table.
 *
 * @param strings_columns Table of strings columns
 * @param column_names Column of names for each column in the table
 * @param num_rows Number of rows in the table
 * @param row_prefix Prepend this string to each row
 * @param row_suffix  Append this string to each row
 * @param value_separator Separator between values
 * @param narep Null-String replacement
 * @param include_nulls Include null string entries in the output
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource to use for device memory allocation.
 * @return New strings column of JSON structs in each row
 */
std::unique_ptr<column> struct_to_strings(table_view const& strings_columns,
                                          column_view const& column_names,
                                          size_type const num_rows,
                                          string_view const row_prefix,
                                          string_view const row_suffix,
                                          string_view const value_separator,
                                          string_scalar const& narep,
                                          bool include_nulls,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(column_names.type().id() == type_id::STRING, "Column names must be of type string");
  auto const num_columns = strings_columns.num_columns();
  CUDF_EXPECTS(num_columns == column_names.size(),
               "Number of column names should be equal to number of columns in the table");
  if (num_rows == 0)  // empty begets empty
    return make_empty_column(type_id::STRING);
  // check all columns are of type string
  CUDF_EXPECTS(std::all_of(strings_columns.begin(),
                           strings_columns.end(),
                           [](auto const& c) { return c.type().id() == type_id::STRING; }),
               "All columns must be of type string");
  auto constexpr strviews_per_column = 3;  // (for each "column_name:", "value",  "separator")
  auto const num_strviews_per_row    = strings_columns.num_columns() == 0
                                         ? 2
                                         : (1 + strings_columns.num_columns() * strviews_per_column);
  // e.g. {col1: value, col2: value, col3: value} = 1 + 3 + 3 + (3-1) + 1 = 10

  auto tbl_device_view = cudf::table_device_view::create(strings_columns, stream);
  auto d_column_names  = column_device_view::create(column_names, stream);

  // Note for future: chunk it but maximize parallelism, if memory usage is high.
  auto const total_strings = num_strviews_per_row * num_rows;
  auto const total_rows    = num_rows * strings_columns.num_columns();
  rmm::device_uvector<string_view> d_strviews(total_strings, stream);
  if (strings_columns.num_columns() > 0) {
    struct_scatter_strings_fn scatter_fn{*tbl_device_view,
                                         *d_column_names,
                                         strviews_per_column,
                                         num_strviews_per_row,
                                         row_prefix,
                                         row_suffix,
                                         value_separator,
                                         narep.value(stream),
                                         include_nulls,
                                         d_strviews.begin()};
    // scatter row_prefix, row_suffix, column_name:, value, value_separator as string_views
    thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     cuda::counting_iterator<size_type>{total_rows},
                     scatter_fn);
  } else {
    thrust::for_each(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator<size_type>{num_rows},
      [d_strviews = d_strviews.begin(), row_prefix, row_suffix, num_strviews_per_row] __device__(
        auto idx) {
        auto const this_index                             = idx * num_strviews_per_row;
        d_strviews[this_index]                            = row_prefix;
        d_strviews[this_index + num_strviews_per_row - 1] = row_suffix;
      });
  }
  if (!include_nulls) {
    // if previous column was null, then we skip the value separator
    rmm::device_uvector<bool> d_str_separator(total_rows, stream);
    auto row_num = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<size_type>([tbl = *tbl_device_view] __device__(auto idx)
                                              -> size_type { return idx / tbl.num_columns(); }));
    auto validity_iterator =
      cudf::detail::make_counting_transform_iterator(0, validity_fn{*tbl_device_view});
    thrust::exclusive_scan_by_key(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      row_num,
      row_num + total_rows,
      validity_iterator,
      d_str_separator.begin(),
      false,
      cuda::std::equal_to<size_type>{},
      cuda::std::logical_or<bool>{});
    thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     cuda::counting_iterator<size_type>{total_rows},
                     [write_separator = d_str_separator.begin(),
                      d_strviews      = d_strviews.begin(),
                      value_separator,
                      tbl = *tbl_device_view,
                      strviews_per_column,
                      num_strviews_per_row] __device__(auto idx) {
                       auto const row = idx / tbl.num_columns();
                       auto const col = idx % tbl.num_columns();
                       auto const this_index =
                         row * num_strviews_per_row + col * strviews_per_column + 1;
                       if (write_separator[idx] && tbl.column(col).is_valid(row)) {
                         d_strviews[this_index - 1] = value_separator;
                       }
                     });
  }
  auto joined_col = make_strings_column(d_strviews, string_view{nullptr, 0}, stream, mr);

  // gather from offset and create a new string column
  auto old_offsets = strings_column_view(joined_col->view()).offsets();
  rmm::device_uvector<size_type> row_string_offsets(num_rows + 1, stream, mr);
  auto const d_strview_offsets = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_type>([num_strviews_per_row] __device__(size_type const i) {
      return i * num_strviews_per_row;
    }));
  thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 d_strview_offsets,
                 d_strview_offsets + row_string_offsets.size(),
                 old_offsets.begin<size_type>(),
                 row_string_offsets.begin());
  auto chars_data = joined_col->release().data;
  return make_strings_column(
    num_rows,
    std::make_unique<cudf::column>(std::move(row_string_offsets), rmm::device_buffer{}, 0),
    std::move(chars_data.release()[0]),
    0,
    {});
}

struct scatter_fn {
  column_device_view _col;
  size_type* _d_strview_offsets;
  string_view* _d_strviews;
  size_type const* _labels;
  size_type const* _list_offsets;
  column_device_view _d_strings_children;
  string_view _element_seperator;
  string_view _element_narep;

  scatter_fn(column_device_view col,
             size_type* d_strview_offsets,
             string_view* d_strviews,
             size_type const* labels,
             size_type const* list_offsets,
             column_device_view d_strings_children,
             string_view const element_separator,
             string_view const element_narep) noexcept
    : _col{col},
      _d_strview_offsets{d_strview_offsets},
      _d_strviews{d_strviews},
      _labels{labels},
      _list_offsets{list_offsets},
      _d_strings_children{d_strings_children},
      _element_seperator{element_separator},
      _element_narep{element_narep}
  {
  }

  __device__ void operator()(size_type idx) const
  {
    auto const label         = _labels[idx];
    auto const sublist_index = idx - _list_offsets[label];
    auto const strview_index = _d_strview_offsets[label] + sublist_index * 2 + 1;
    // value or na_rep
    auto const strview         = _d_strings_children.element<cudf::string_view>(idx);
    _d_strviews[strview_index] = _d_strings_children.is_null(idx) ? _element_narep : strview;
    // separator
    if (sublist_index != 0) { _d_strviews[strview_index - 1] = _element_seperator; }
  }
};

/**
 * @brief Concatenates a list of strings columns into a single strings column.
 *
 * @param lists_strings Column containing lists of strings to concatenate.
 * @param list_prefix String to place before each list. (typically [)
 * @param list_suffix String to place after each list. (typically ])
 * @param element_separator String that should inserted between strings of each list row.
 * @param element_narep String that should be used in place of any null strings.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with concatenated results.
 */
std::unique_ptr<column> join_list_of_strings(lists_column_view const& lists_strings,
                                             string_view const list_prefix,
                                             string_view const list_suffix,
                                             string_view const element_separator,
                                             string_view const element_narep,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  /*
  create string_views of the list elements, and the list separators and list prefix/suffix.
  then concatenates them all together.
  gather offset of first string_view of each row as offsets for output string column.
  Algorithm:
    calculate #strviews per list using null mask, and list_offsets.
    scan #strviews to get strviews_offset
    create label segments.
    sublist_index = index - offsets[label]
    strviews_offset[label] + sublist_index = string_view index +1, +2
    use above 2 to scatter element, element_seperator
    scatter list_prefix, list_suffix to the right place using list_offsets
    make_strings_column() and gather offsets, based on strviews_offset.
  */
  auto const offsets          = lists_strings.offsets();
  auto const strings_children = lists_strings.get_sliced_child(stream);
  auto const num_lists        = lists_strings.size();
  auto const num_strings      = strings_children.size();
  auto const num_offsets      = offsets.size();

  rmm::device_uvector<size_type> d_strview_offsets(num_offsets, stream);
  auto num_strings_per_list = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>(
      [offsets = offsets.begin<size_type>(), num_offsets] __device__(size_type idx) {
        if (idx + 1 >= num_offsets) return 0;
        auto const length = offsets[idx + 1] - offsets[idx];
        return length == 0 ? 2 : (2 + length + length - 1);
      }));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         num_strings_per_list,
                         num_strings_per_list + num_offsets,
                         d_strview_offsets.begin());
  auto const total_strings = d_strview_offsets.back_element(stream);

  rmm::device_uvector<string_view> d_strviews(total_strings, stream);
  // scatter null_list and list_prefix, list_suffix
  auto col_device_view = cudf::column_device_view::create(lists_strings.parent(), stream);
  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   cuda::counting_iterator<size_type>{0},
                   cuda::counting_iterator<size_type>{num_lists},
                   [col = *col_device_view,
                    list_prefix,
                    list_suffix,
                    d_strview_offsets = d_strview_offsets.begin(),
                    d_strviews        = d_strviews.begin()] __device__(auto idx) {
                     if (col.is_null(idx)) {
                       d_strviews[d_strview_offsets[idx]]     = string_view{};
                       d_strviews[d_strview_offsets[idx] + 1] = string_view{};
                     } else {
                       // [ ]
                       d_strviews[d_strview_offsets[idx]]         = list_prefix;
                       d_strviews[d_strview_offsets[idx + 1] - 1] = list_suffix;
                     }
                   });

  // scatter string and separator
  auto labels = cudf::lists::detail::generate_labels(
    lists_strings, num_strings, stream, cudf::get_current_device_resource_ref());
  auto d_strings_children = cudf::column_device_view::create(strings_children, stream);
  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   cuda::counting_iterator<size_type>{0},
                   cuda::counting_iterator<size_type>{num_strings},
                   scatter_fn{*col_device_view,
                              d_strview_offsets.data(),
                              d_strviews.data(),
                              labels->view().data<size_type>(),
                              offsets.data<size_type>(),
                              *d_strings_children,
                              element_separator,
                              element_narep});

  auto joined_col = make_strings_column(d_strviews, string_view{nullptr, 0}, stream, mr);

  // gather from offset and create a new string column
  auto old_offsets = strings_column_view(joined_col->view()).offsets();
  rmm::device_uvector<size_type> row_string_offsets(num_offsets, stream, mr);
  thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 d_strview_offsets.begin(),
                 d_strview_offsets.end(),
                 old_offsets.begin<size_type>(),
                 row_string_offsets.begin());
  auto chars_data = joined_col->release().data;
  return make_strings_column(
    num_lists,
    std::make_unique<cudf::column>(std::move(row_string_offsets), rmm::device_buffer{}, 0),
    std::move(chars_data.release()[0]),
    lists_strings.null_count(),
    cudf::detail::copy_bitmask(lists_strings.parent(), stream, mr));
}

}  // namespace cudf::io::json::detail
