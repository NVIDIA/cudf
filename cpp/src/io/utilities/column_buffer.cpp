/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file column_buffer.cpp
 * @brief cuDF-IO column_buffer class implementation
 */

#include "column_buffer.hpp"

#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>

#include <functional>
#include <sstream>

namespace cudf::io::detail {

void gather_column_buffer::allocate_strings_data(bool memset_data, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(type.id() == type_id::STRING, "allocate_strings_data called for non-string column");
  // The contents of _strings will never be directly returned to the user.
  // Due to the fact that make_strings_column copies the input data to
  // produce its outputs, _strings is actually a temporary. As a result, we
  // do not pass the provided mr to the call to
  // make_zeroed_device_uvector_async here and instead let it use the
  // default rmm memory resource.
  _strings = std::make_unique<rmm::device_uvector<string_index_pair>>(
    cudf::detail::make_zeroed_device_uvector_async<string_index_pair>(
      size, stream, cudf::get_current_device_resource_ref()));
}

std::unique_ptr<column> gather_column_buffer::make_string_column_impl(rmm::cuda_stream_view stream)
{
  // make_strings_column allocates new memory, it does not simply move
  // from the inputs, so we need to pass it the memory resource given to
  // the buffer on construction so that the memory is allocated using the
  // resource that the calling code expected.
  return make_strings_column(*_strings, stream, _mr);
}

void cudf::io::detail::inline_column_buffer::allocate_strings_data(bool memset_data,
                                                                   rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(type.id() == type_id::STRING, "allocate_strings_data called for non-string column");
  // size + 1 for final offset. _string_data will be initialized later.
  _data = create_data(data_type{type_id::INT32}, size + 1, memset_data, stream, _mr);
}

void cudf::io::detail::inline_column_buffer::create_string_data(size_t num_bytes,
                                                                bool is_large_strings_col,
                                                                rmm::cuda_stream_view stream)
{
  _is_large_strings_col = is_large_strings_col;
  _string_data          = rmm::device_buffer(num_bytes, stream, _mr);
}

namespace {

/**
 * @brief Recursively copy `name`, `user_data`, and `string_as_binary` fields of one buffer to
 * another.
 *
 * @param buff The old output buffer
 * @param new_buff The new output buffer
 */
template <class string_policy>
void copy_buffer_data(string_policy const& buff, string_policy& new_buff)
{
  new_buff.name             = buff.name;
  new_buff.user_data        = buff.user_data;
  new_buff.string_as_binary = buff.string_as_binary;
  for (auto const& child : buff.children) {
    auto& new_child = new_buff.children.emplace_back(string_policy(child.type, child.is_nullable));
    copy_buffer_data(child, new_child);
  }
}

}  // namespace

template <class string_policy>
void column_buffer_base<string_policy>::create_with_mask(size_type _size,
                                                         cudf::mask_state null_mask_state,
                                                         bool memset_data,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(_size >= 0 and _size <= std::numeric_limits<cudf::size_type>::max(),
               "Unexpected column buffer allocation size requested",
               std::overflow_error);
  size = _size;
  _mr  = mr;

  switch (type.id()) {
    case type_id::STRING:
      static_cast<string_policy*>(this)->allocate_strings_data(memset_data, stream);
      break;

    // list columns store a buffer of int32's as offsets to represent
    // their individual rows
    case type_id::LIST:
      _data = create_data(data_type{type_id::INT32}, size, memset_data, stream, _mr);
      break;

    // struct columns store no data themselves.  just validity and children.
    case type_id::STRUCT: break;

    default: _data = create_data(type, size, memset_data, stream, _mr); break;
  }
  if (is_nullable) {
    _null_mask =
      cudf::detail::create_null_mask(size, null_mask_state, rmm::cuda_stream_view(stream), _mr);
  }
}

template <class string_policy>
void column_buffer_base<string_policy>::create(size_type _size,
                                               bool memset_data,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  create_with_mask(_size, mask_state::ALL_NULL, memset_data, stream, mr);
}

template <class string_policy>
void column_buffer_base<string_policy>::create(size_type _size,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  create_with_mask(_size, mask_state::ALL_NULL, true, stream, mr);
}

template <class string_policy>
string_policy column_buffer_base<string_policy>::empty_like(string_policy const& input)
{
  auto new_buff = string_policy(input.type, input.is_nullable);
  copy_buffer_data(input, new_buff);
  return new_buff;
}

template <typename string_policy>
std::string type_to_name(column_buffer_base<string_policy> const& buffer)
{
  if (buffer.type.id() == cudf::type_id::LIST) {
    return "List<" + (type_to_name<string_policy>(buffer.children[0])) + ">";
  }

  if (buffer.type.id() == cudf::type_id::STRUCT) {
    std::ostringstream out;

    out << "Struct<";
    auto iter = cuda::counting_iterator<cudf::size_type>{0};
    std::transform(
      iter,
      iter + buffer.children.size(),
      std::ostream_iterator<std::string>(out, ","),
      [&buffer](size_type i) { return type_to_name<string_policy>(buffer.children[i]); });
    out << ">";
    return out.str();
  }

  return cudf::type_to_name(buffer.type);
}

template <class string_policy>
std::unique_ptr<column> make_column(column_buffer_base<string_policy>& buffer,
                                    column_name_info* schema_info,
                                    std::optional<reader_column_schema> const& schema,
                                    rmm::cuda_stream_view stream)
{
  std::function<std::unique_ptr<column>(column_buffer_base<string_policy> & buffer,
                                        column_name_info * schema_info,
                                        std::optional<reader_column_schema> const& schema)> const
    construct_column =
      [&construct_column, stream](
        column_buffer_base<string_policy>& buffer,
        column_name_info* schema_info,
        std::optional<reader_column_schema> const& schema) -> std::unique_ptr<column> {
    if (schema_info != nullptr) {
      schema_info->name        = buffer.name;
      schema_info->is_nullable = buffer.is_nullable;
    }
    switch (buffer.type.id()) {
      case type_id::STRING: {
        if (schema.value_or(reader_column_schema{}).is_enabled_convert_binary_to_strings() and
            not buffer.string_as_binary) {
          if (schema_info != nullptr) { schema_info->children.emplace_back("offsets"); }

          // make_strings_column allocates new memory, it does not simply move
          // from the inputs, so we need to pass it the memory resource given to
          // the buffer on construction so that the memory is allocated using the
          // resource that the calling code expected.
          return buffer.make_string_column(stream);
        } else {
          // convert to binary
          auto const string_col = buffer.make_string_column(stream);
          auto const num_rows   = string_col->size();
          auto const null_count = string_col->null_count();
          auto col_content      = string_col->release();

          // convert to uint8 column, strings are currently stored as int8
          auto data      = col_content.data.release();
          auto char_size = data->size();

          CUDF_EXPECTS(char_size < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
                       "Cannot convert strings column to lists column due to size_type limit",
                       std::overflow_error);

          auto uint8_col = std::make_unique<column>(
            data_type{type_id::UINT8}, char_size, std::move(*data), rmm::device_buffer{}, 0);

          if (schema_info != nullptr) {
            schema_info->children.emplace_back("offsets");
            schema_info->children.emplace_back("binary");
            // cuDF type will be list<UINT8>, but remember it was originally binary data
            schema_info->is_binary = true;
            if (schema.has_value() and schema->get_type_length() > 0) {
              schema_info->type_length = schema->get_type_length();
            }
          }

          return make_lists_column(
            num_rows,
            std::move(col_content.children[strings_column_view::offsets_column_index]),
            std::move(uint8_col),
            null_count,
            std::move(*col_content.null_mask));
        }
      }

      case type_id::LIST: {
        // make offsets column
        auto offsets = std::make_unique<column>(
          data_type{type_id::INT32}, buffer.size, std::move(buffer._data), rmm::device_buffer{}, 0);

        column_name_info* child_info = nullptr;
        if (schema_info != nullptr) {
          schema_info->children.emplace_back("offsets");
          schema_info->children.emplace_back("");
          child_info = &schema_info->children.back();
        }

        CUDF_EXPECTS(not schema.has_value() or schema->get_num_children() > 0,
                     "Invalid schema provided for read, expected child data for list!");
        auto const child_schema = schema.has_value()
                                    ? std::make_optional<reader_column_schema>(schema->child(0))
                                    : std::nullopt;

        // make child column
        CUDF_EXPECTS(buffer.children.size() > 0, "Encountered malformed column_buffer");
        auto child = construct_column(buffer.children[0], child_info, child_schema);

        // make the final list column (note : size is the # of offsets, so our actual # of rows is 1
        // less)
        return make_lists_column(buffer.size - 1,
                                 std::move(offsets),
                                 std::move(child),
                                 buffer._null_count,
                                 std::move(buffer._null_mask));
      } break;

      case type_id::STRUCT: {
        std::vector<std::unique_ptr<cudf::column>> output_children;
        output_children.reserve(buffer.children.size());
        for (size_t i = 0; i < buffer.children.size(); ++i) {
          column_name_info* child_info = nullptr;
          if (schema_info != nullptr) {
            schema_info->children.emplace_back("");
            child_info = &schema_info->children.back();
          }

          CUDF_EXPECTS(not schema.has_value() or schema->get_num_children() > i,
                       "Invalid schema provided for read, expected more child data for struct!");
          auto const child_schema = schema.has_value()
                                      ? std::make_optional<reader_column_schema>(schema->child(i))
                                      : std::nullopt;

          output_children.emplace_back(
            construct_column(buffer.children[i], child_info, child_schema));
        }

        return create_structs_hierarchy(buffer.size,
                                        std::move(output_children),
                                        buffer._null_count,
                                        std::move(buffer._null_mask),
                                        stream,
                                        buffer._mr);
      } break;

      default: {
        return std::make_unique<column>(buffer.type,
                                        buffer.size,
                                        std::move(buffer._data),
                                        std::move(buffer._null_mask),
                                        buffer._null_count);
      }
    }
  };

  return construct_column(buffer, schema_info, schema);
}

/**
 * @copydoc cudf::io::detail::empty_like
 */
template <class string_policy>
std::unique_ptr<column> empty_like(column_buffer_base<string_policy>& buffer,
                                   column_name_info* schema_info,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  if (schema_info != nullptr) { schema_info->name = buffer.name; }

  switch (buffer.type.id()) {
    case type_id::STRING: {
      if (buffer.string_as_binary) {
        auto offsets = cudf::make_empty_column(type_id::INT32);
        auto child   = cudf::make_empty_column(type_id::UINT8);
        if (schema_info != nullptr) {
          // Mirror the binary path in `make_column` so that the schema_info for an empty
          // list<uint8> column matches the shape produced for a non-empty one.
          schema_info->children.emplace_back("offsets");
          schema_info->children.emplace_back("binary");
          schema_info->is_binary = true;
        }
        return make_lists_column(
          0, std::move(offsets), std::move(child), 0, rmm::device_buffer{0, stream, mr});
      }
      return cudf::make_empty_column(buffer.type);
    }

    case type_id::LIST: {
      // make offsets column
      auto offsets = cudf::make_empty_column(type_id::INT32);

      column_name_info* child_info = nullptr;
      if (schema_info != nullptr) {
        schema_info->children.emplace_back("offsets");
        schema_info->children.emplace_back("");
        child_info = &schema_info->children.back();
      }

      // make child column
      CUDF_EXPECTS(buffer.children.size() > 0, "Encountered malformed column_buffer");
      auto child =
        cudf::io::detail::empty_like<string_policy>(buffer.children[0], child_info, stream, mr);

      // make the final list column
      return make_lists_column(
        0, std::move(offsets), std::move(child), 0, rmm::device_buffer{0, stream, mr});
    } break;

    case type_id::STRUCT: {
      std::vector<std::unique_ptr<cudf::column>> output_children;
      output_children.reserve(buffer.children.size());
      std::transform(buffer.children.begin(),
                     buffer.children.end(),
                     std::back_inserter(output_children),
                     [&](auto& col) {
                       column_name_info* child_info = nullptr;
                       if (schema_info != nullptr) {
                         schema_info->children.emplace_back("");
                         child_info = &schema_info->children.back();
                       }
                       return cudf::io::detail::empty_like<string_policy>(
                         col, child_info, stream, mr);
                     });

      return make_structs_column(
        0, std::move(output_children), 0, rmm::device_buffer{0, stream, mr}, stream, mr);
    } break;

    default: return cudf::make_empty_column(buffer.type);
  }
}

using pointer_type = gather_column_buffer;
using string_type  = cudf::io::detail::inline_column_buffer;

using pointer_column_buffer = column_buffer_base<pointer_type>;
using string_column_buffer  = column_buffer_base<string_type>;

template std::unique_ptr<column> make_column<string_type>(
  string_column_buffer& buffer,
  column_name_info* schema_info,
  std::optional<reader_column_schema> const& schema,
  rmm::cuda_stream_view stream);

template std::unique_ptr<column> make_column<pointer_type>(
  pointer_column_buffer& buffer,
  column_name_info* schema_info,
  std::optional<reader_column_schema> const& schema,
  rmm::cuda_stream_view stream);

template std::unique_ptr<column> empty_like<string_type>(string_column_buffer& buffer,
                                                         column_name_info* schema_info,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr);

template std::unique_ptr<column> empty_like<pointer_type>(pointer_column_buffer& buffer,
                                                          column_name_info* schema_info,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr);

template std::string type_to_name<string_type>(string_column_buffer const& buffer);
template std::string type_to_name<pointer_type>(pointer_column_buffer const& buffer);

template class column_buffer_base<pointer_type>;
template class column_buffer_base<string_type>;

}  // namespace cudf::io::detail
