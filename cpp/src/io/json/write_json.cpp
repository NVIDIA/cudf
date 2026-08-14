/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "write_json.hpp"

#include "io/comp/compression.hpp"
#include "io/csv/durations.hpp"
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
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/tabulate.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace cudf::io::json::detail {

std::unique_ptr<column> make_strings_column_from_host(host_span<std::string const> host_strings,
                                                      rmm::cuda_stream_view stream)
{
  std::string const host_chars =
    std::accumulate(host_strings.begin(), host_strings.end(), std::string(""));
  auto d_chars = cudf::detail::make_device_uvector_async(
    host_span<char const>{host_chars}, stream, cudf::get_current_device_resource_ref());
  std::vector<cudf::size_type> offsets(host_strings.size() + 1, 0);
  std::transform_inclusive_scan(host_strings.begin(),
                                host_strings.end(),
                                offsets.begin() + 1,
                                std::plus<cudf::size_type>{},
                                [](auto& str) { return str.size(); });
  auto d_offsets = std::make_unique<cudf::column>(
    cudf::detail::make_device_uvector(offsets, stream, cudf::get_current_device_resource_ref()),
    rmm::device_buffer{},
    0);
  return cudf::make_strings_column(
    host_strings.size(), std::move(d_offsets), d_chars.release(), 0, {});
}

std::unique_ptr<column> make_column_names_column(host_span<column_name_info const> column_names,
                                                 size_type num_columns,
                                                 rmm::cuda_stream_view stream)
{
  std::vector<std::string> unescaped_column_names;
  if (column_names.empty()) {
    std::generate_n(std::back_inserter(unescaped_column_names), num_columns, [v = 0]() mutable {
      return std::to_string(v++);
    });
  } else {
    std::transform(column_names.begin(),
                   column_names.end(),
                   std::back_inserter(unescaped_column_names),
                   [](column_name_info const& name_info) { return name_info.name; });
  }
  auto unescaped_string_col = make_strings_column_from_host(unescaped_column_names, stream);
  auto d_column             = column_device_view::create(*unescaped_string_col, stream);
  return make_escaped_json_strings(
    *d_column,
    unescaped_string_col->size(),
    unescaped_string_col->null_count(),
    cudf::detail::copy_bitmask(
      *unescaped_string_col, stream, cudf::get_current_device_resource_ref()),
    true,
    true,
    stream,
    cudf::get_current_device_resource_ref());
}

std::unique_ptr<column> timestamp_to_strings(column_view const& column,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  auto format = [&]() {
    switch (column.type().id()) {
      case type_id::TIMESTAMP_SECONDS: return std::string{"%Y-%m-%dT%H:%M:%SZ"};
      case type_id::TIMESTAMP_MILLISECONDS: return std::string{"%Y-%m-%dT%H:%M:%S.%3fZ"};
      case type_id::TIMESTAMP_MICROSECONDS: return std::string{"%Y-%m-%dT%H:%M:%S.%6fZ"};
      case type_id::TIMESTAMP_NANOSECONDS: return std::string{"%Y-%m-%dT%H:%M:%S.%9fZ"};
      default: return std::string{"%Y-%m-%d"};
    }
  }();

  format = "\"" + format + "\"";

  auto empty_strings = make_empty_column(type_id::STRING);
  return cudf::strings::detail::from_timestamps(
    column, format, strings_column_view(empty_strings->view()), stream, mr);
}

std::unique_ptr<column> duration_to_strings(column_view const& column,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto duration_string = cudf::io::detail::csv::pandas_format_durations(column, stream, mr);
  auto d_column        = column_device_view::create(duration_string->view(), stream);
  return make_escaped_json_strings(*d_column,
                                   duration_string->size(),
                                   duration_string->null_count(),
                                   cudf::detail::copy_bitmask(duration_string->view(), stream, mr),
                                   false,
                                   true,
                                   stream,
                                   mr);
}

std::unique_ptr<column> string_to_strings(column_view const& column,
                                          bool escaped_utf8,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto d_column = column_device_view::create(column, stream);
  return make_escaped_json_strings(*d_column,
                                   column.size(),
                                   column.null_count(),
                                   cudf::detail::copy_bitmask(column, stream, mr),
                                   false,
                                   escaped_utf8,
                                   stream,
                                   mr);
}

std::unique_ptr<column> leaf_column_to_strings(column_view const& column,
                                               json_writer_options const& options,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (column.type().id() == type_id::STRING) {
    return string_to_strings(column, options.is_enabled_utf8_escaped(), stream, mr);
  }
  if (cudf::is_timestamp(column.type())) { return timestamp_to_strings(column, stream, mr); }
  if (cudf::is_duration(column.type())) { return duration_to_strings(column, stream, mr); }
  if (cudf::is_boolean(column.type())) {
    string_scalar const true_value(
      options.get_true_value(), true, stream, cudf::get_current_device_resource_ref());
    string_scalar const false_value(
      options.get_false_value(), true, stream, cudf::get_current_device_resource_ref());
    return cudf::strings::detail::from_booleans(column, true_value, false_value, stream, mr);
  }
  if (cudf::is_integral(column.type())) {
    return cudf::strings::detail::from_integers(column, stream, mr);
  }
  if (cudf::is_floating_point(column.type())) {
    return cudf::strings::detail::from_floats(column, stream, mr);
  }
  if (cudf::is_fixed_point(column.type())) {
    return cudf::strings::detail::from_fixed_point(column, stream, mr);
  }

  CUDF_FAIL("Unsupported column type.");
}

namespace {

host_span<column_name_info const> child_column_names(
  host_span<column_name_info const> children_names, size_t child_index)
{
  return children_names.size() > child_index
           ? host_span<column_name_info const>{children_names[child_index].children}
           : host_span<column_name_info const>{};
}

struct column_to_strings_fn {
  explicit column_to_strings_fn(json_writer_options const& options,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
    : options_(options),
      stream_(stream),
      mr_(std::move(mr)),
      narep(options.get_na_rep(), true, stream, cudf::get_current_device_resource_ref()),
      struct_value_separator(",", true, stream, cudf::get_current_device_resource_ref()),
      struct_row_begin_wrap("{", true, stream, cudf::get_current_device_resource_ref()),
      struct_row_end_wrap("}", true, stream, cudf::get_current_device_resource_ref()),
      list_value_separator(",", true, stream, cudf::get_current_device_resource_ref()),
      list_row_begin_wrap("[", true, stream, cudf::get_current_device_resource_ref()),
      list_row_end_wrap("]", true, stream, cudf::get_current_device_resource_ref())
  {
  }

  column_to_strings_fn(column_to_strings_fn const&)            = delete;
  column_to_strings_fn& operator=(column_to_strings_fn const&) = delete;
  column_to_strings_fn(column_to_strings_fn&&)                 = delete;
  column_to_strings_fn& operator=(column_to_strings_fn&&)      = delete;

  template <typename column_type>
  std::unique_ptr<column> operator()(column_view const&, host_span<column_name_info const>) const
    requires(!std::is_same_v<column_type, cudf::list_view> &&
             !std::is_same_v<column_type, cudf::struct_view>)
  {
    CUDF_FAIL("Only nested columns should use this overload.");
  }

  template <typename column_type>
  std::unique_ptr<column> operator()(column_view const& column,
                                     host_span<column_name_info const> children_names) const
    requires(std::is_same_v<column_type, cudf::list_view>)
  {
    auto child_view            = lists_column_view(column).get_sliced_child(stream_);
    auto constexpr child_index = lists_column_view::child_column_index;

    auto child_string_with_null = [&]() {
      if (child_view.type().id() == type_id::STRUCT) {
        return this->template operator()<cudf::struct_view>(
          child_view, child_column_names(children_names, child_index));
      } else if (child_view.type().id() == type_id::LIST) {
        return this->template operator()<cudf::list_view>(
          child_view, child_column_names(children_names, child_index));
      } else {
        return leaf_column_to_strings(child_view, options_, stream_, mr_);
      }
    };

    auto new_offsets = cudf::lists::detail::get_normalized_offsets(
      lists_column_view(column), stream_, cudf::get_current_device_resource_ref());
    auto const list_child_string = make_lists_column(
      column.size(),
      std::move(new_offsets),
      child_string_with_null(),
      column.null_count(),
      cudf::detail::copy_bitmask(column, stream_, cudf::get_current_device_resource_ref()));
    return join_list_of_strings(lists_column_view(*list_child_string),
                                list_row_begin_wrap.value(stream_),
                                list_row_end_wrap.value(stream_),
                                list_value_separator.value(stream_),
                                narep.value(stream_),
                                stream_,
                                mr_);
  }

  template <typename column_type>
  std::unique_ptr<column> operator()(column_view const& column,
                                     host_span<column_name_info const> children_names) const
    requires(std::is_same_v<column_type, cudf::struct_view>)
  {
    auto structs_view   = structs_column_view{column};
    auto const child_it = cudf::detail::make_counting_transform_iterator(
      0, [&stream = stream_, &s_v = structs_view](auto const child_idx) {
        return s_v.get_sliced_child(child_idx, stream);
      });
    auto col_string = operator()(child_it,
                                 child_it + column.num_children(),
                                 children_names,
                                 column.size(),
                                 struct_row_end_wrap.value(stream_));
    col_string->set_null_mask(cudf::detail::copy_bitmask(column, stream_, mr_),
                              column.null_count());
    return col_string;
  }

  template <typename column_iterator>
  std::unique_ptr<column> operator()(column_iterator column_begin,
                                     column_iterator column_end,
                                     host_span<column_name_info const> children_names,
                                     size_type num_rows,
                                     cudf::string_view row_end_wrap_value) const
  {
    auto const num_columns = std::distance(column_begin, column_end);
    auto column_names      = make_column_names_column(children_names, num_columns, stream_);
    auto column_names_view = column_names->view();
    std::vector<std::unique_ptr<cudf::column>> str_column_vec;

    auto i_col_begin = cuda::make_zip_iterator(cuda::counting_iterator<size_t>(0), column_begin);
    std::transform(i_col_begin,
                   i_col_begin + num_columns,
                   std::back_inserter(str_column_vec),
                   [this, &children_names](auto const& i_current_col) {
                     auto const i            = cuda::std::get<0>(i_current_col);
                     auto const& current_col = cuda::std::get<1>(i_current_col);
                     if (current_col.type().id() == type_id::STRUCT) {
                       return this->template operator()<cudf::struct_view>(
                         current_col, child_column_names(children_names, i));
                     } else if (current_col.type().id() == type_id::LIST) {
                       return this->template operator()<cudf::list_view>(
                         current_col, child_column_names(children_names, i));
                     } else {
                       return leaf_column_to_strings(current_col, options_, stream_, mr_);
                     }
                   });

    auto str_table_ptr  = std::make_unique<cudf::table>(std::move(str_column_vec));
    auto str_table_view = str_table_ptr->view();

    return struct_to_strings(str_table_view,
                             column_names_view,
                             num_rows,
                             struct_row_begin_wrap.value(stream_),
                             row_end_wrap_value,
                             struct_value_separator.value(stream_),
                             narep,
                             options_.is_enabled_include_nulls(),
                             stream_,
                             cudf::get_current_device_resource_ref());
  }

 private:
  json_writer_options const& options_;
  rmm::cuda_stream_view stream_;
  rmm::device_async_resource_ref mr_;
  string_scalar const narep;
  string_scalar const struct_value_separator;
  string_scalar const struct_row_begin_wrap;
  string_scalar const struct_row_end_wrap;
  string_scalar const list_value_separator;
  string_scalar const list_row_begin_wrap;
  string_scalar const list_row_end_wrap;
};

void write_chunked(data_sink* out_sink,
                   strings_column_view const& str_column_view,
                   int skip_last_chars,
                   json_writer_options const&,
                   rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(str_column_view.size() > 0, "Unexpected empty strings column.");

  auto const total_num_bytes = str_column_view.chars_size(stream) - skip_last_chars;
  char const* ptr_all_bytes  = str_column_view.chars_begin(stream);

  if (out_sink->is_device_write_preferred(total_num_bytes)) {
    out_sink->device_write(ptr_all_bytes, total_num_bytes, stream);
  } else {
    auto const h_bytes = cudf::detail::make_host_vector(
      device_span<char const>(ptr_all_bytes, total_num_bytes), stream);
    out_sink->host_write(h_bytes.data(), total_num_bytes);
  }
}

}  // namespace

void write_json_uncompressed(data_sink* out_sink,
                             table_view const& table,
                             json_writer_options const& options,
                             rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  std::vector<column_name_info> user_column_names = [&]() {
    auto const& metadata = options.get_metadata();
    if (metadata.has_value() and not metadata->schema_info.empty()) {
      return metadata->schema_info;
    }

    std::vector<column_name_info> names;
    std::transform(cuda::counting_iterator<cudf::size_type>{0},
                   cuda::counting_iterator{table.num_columns()},
                   std::back_inserter(names),
                   [](auto i) { return column_name_info{std::to_string(i)}; });
    return names;
  }();

  auto const line_terminator = std::string(options.is_enabled_lines() ? "\n" : ",");
  string_scalar const d_line_terminator_with_row_end{
    "}" + line_terminator, true, stream, cudf::get_current_device_resource_ref()};
  string_scalar const d_line_terminator{
    line_terminator, true, stream, cudf::get_current_device_resource_ref()};
  std::string const list_braces{"[]"};
  string_scalar const d_list_braces{
    list_braces, true, stream, cudf::get_current_device_resource_ref()};

  if (!options.is_enabled_lines()) {
    if (out_sink->is_device_write_preferred(1)) {
      out_sink->device_write(d_list_braces.data(), 1, stream);
    } else {
      out_sink->host_write(list_braces.data(), 1);
    }
  }

  if (table.num_rows() > 0) {
    auto n_rows_per_chunk = options.get_rows_per_chunk();
    if (n_rows_per_chunk % 8) { n_rows_per_chunk += 8 - (n_rows_per_chunk % 8); }

    CUDF_EXPECTS(n_rows_per_chunk >= 8, "write_json: invalid chunk_rows; must be at least 8");

    auto num_rows = table.num_rows();
    std::vector<table_view> vector_views;

    if (num_rows <= n_rows_per_chunk) {
      vector_views.push_back(table);
    } else {
      auto const n_chunks = num_rows / n_rows_per_chunk;
      std::vector<size_type> splits(n_chunks);
      thrust::tabulate(splits.begin(), splits.end(), [n_rows_per_chunk](auto idx) {
        return (idx + 1) * n_rows_per_chunk;
      });
      vector_views = cudf::detail::split(table, splits, stream);
    }

    column_to_strings_fn converter{options, stream, cudf::get_current_device_resource_ref()};

    for (auto&& sub_view : vector_views) {
      if (sub_view.num_rows() == 0) continue;

      auto str_concat_col = converter(sub_view.begin(),
                                      sub_view.end(),
                                      user_column_names,
                                      sub_view.num_rows(),
                                      d_line_terminator_with_row_end.value(stream));

      bool const include_line_terminator =
        (&sub_view != &vector_views.back()) or options.is_enabled_lines();
      auto const skip_last_chars = include_line_terminator ? 0 : line_terminator.size();
      write_chunked(out_sink, str_concat_col->view(), skip_last_chars, options, stream);
    }
  } else if (options.is_enabled_lines()) {
    if (out_sink->is_device_write_preferred(1)) {
      out_sink->device_write(d_line_terminator.data(), d_line_terminator.size(), stream);
    } else {
      out_sink->host_write(line_terminator.data(), line_terminator.size());
    }
  }

  if (!options.is_enabled_lines()) {
    if (out_sink->is_device_write_preferred(1)) {
      out_sink->device_write(d_list_braces.data() + 1, 1, stream);
    } else {
      out_sink->host_write(list_braces.data() + 1, 1);
    }
  }
}

void write_json(data_sink* out_sink,
                table_view const& table,
                json_writer_options const& options,
                rmm::cuda_stream_view stream)
{
  if (options.get_compression() != compression_type::NONE) {
    std::vector<char> hbuf;
    auto hbuf_sink_ptr = data_sink::create(&hbuf);
    write_json_uncompressed(hbuf_sink_ptr.get(), table, options, stream);
    stream.synchronize();
    auto comp_hbuf = cudf::io::detail::compress(
      options.get_compression(),
      host_span<uint8_t>(reinterpret_cast<uint8_t*>(hbuf.data()), hbuf.size()));
    out_sink->host_write(comp_hbuf.data(), comp_hbuf.size());
    return;
  }

  write_json_uncompressed(out_sink, table, options, stream);
}

}  // namespace cudf::io::json::detail
