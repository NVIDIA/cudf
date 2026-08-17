/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nested_json.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <optional>
#include <string>
#include <vector>

namespace cudf::io {
namespace {
bool validate_column_order(schema_element const& types)
{
  // For struct types, check if column_order size matches child_types size and all elements in
  // column_order are in child_types, in child_types, call this function recursively.
  // For list types, check if child_types size is 1 and call this function recursively.
  if (types.type.id() == type_id::STRUCT) {
    if (types.column_order.has_value()) {
      if (types.column_order.value().size() != types.child_types.size()) { return false; }
      for (auto const& column_name : types.column_order.value()) {
        auto it = types.child_types.find(column_name);
        if (it == types.child_types.end()) { return false; }
        if (it->second.type.id() == type_id::STRUCT or it->second.type.id() == type_id::LIST) {
          if (!validate_column_order(it->second)) { return false; }
        }
      }
    }
  } else if (types.type.id() == type_id::LIST) {
    if (types.child_types.size() != 1) { return false; }
    auto it = types.child_types.begin();
    if (it->second.type.id() == type_id::STRUCT or it->second.type.id() == type_id::LIST) {
      if (!validate_column_order(it->second)) { return false; }
    }
  }
  return true;
}
}  // namespace

void json_reader_options::set_dtypes(schema_element types)
{
  CUDF_EXPECTS(
    validate_column_order(types), "Column order does not match child types", std::invalid_argument);
  _dtypes = std::move(types);
}
}  // namespace cudf::io

namespace cudf::io::json::detail {
/// Created an empty column of the specified schema
struct empty_column_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, CUDF_ENABLE_IF(!cudf::is_nested<T>())>
  std::unique_ptr<column> operator()(schema_element const& schema) const
  {
    return make_empty_column(schema.type);
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  std::unique_ptr<column> operator()(schema_element const& schema) const
  {
    CUDF_EXPECTS(schema.child_types.size() == 1, "List column should have only one child");
    auto const& child_name        = schema.child_types.begin()->first;
    std::unique_ptr<column> child = cudf::type_dispatcher(
      schema.child_types.at(child_name).type, *this, schema.child_types.at(child_name));
    auto offsets = make_empty_column(data_type(type_id::INT32));
    std::vector<std::unique_ptr<column>> child_columns;
    child_columns.push_back(std::move(offsets));
    child_columns.push_back(std::move(child));
    // Do not use `cudf::make_lists_column` since we do not need to call `purge_nonempty_nulls` on
    // the child column as it does not have non-empty nulls. Look issue #17356
    return std::make_unique<column>(cudf::data_type{type_id::LIST},
                                    0,
                                    rmm::device_buffer{},
                                    rmm::device_buffer{},
                                    0,
                                    std::move(child_columns));
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  std::unique_ptr<column> operator()(schema_element const& schema) const
  {
    std::vector<std::unique_ptr<column>> child_columns;
    for (auto const& child_name : schema.column_order.value_or(std::vector<std::string>{})) {
      child_columns.push_back(cudf::type_dispatcher(
        schema.child_types.at(child_name).type, *this, schema.child_types.at(child_name)));
    }
    return make_structs_column(0, std::move(child_columns), 0, {}, stream, mr);
  }
};

std::unique_ptr<column> make_empty_column(schema_element const& schema,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(schema.type, empty_column_functor{stream, mr}, schema);
}

/// Created all null column of the specified schema
struct allnull_column_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

 private:
  [[nodiscard]] auto make_zeroed_offsets(size_type size) const
  {
    auto offsets_buff =
      cudf::detail::make_zeroed_device_uvector_async<int32_t>(size + 1, stream, mr);
    return std::make_unique<column>(std::move(offsets_buff), rmm::device_buffer{}, 0);
  }

 public:
  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  std::unique_ptr<column> operator()(schema_element const& schema, size_type size) const
  {
    return make_fixed_width_column(schema.type, size, mask_state::ALL_NULL, stream, mr);
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_dictionary<T>())>
  std::unique_ptr<column> operator()(schema_element const& schema, size_type size) const
  {
    CUDF_EXPECTS(schema.child_types.size() == 1, "Dictionary column should have only one child");
    auto const& child_name = schema.child_types.begin()->first;
    std::unique_ptr<column> child =
      make_empty_column(schema.child_types.at(child_name), stream, mr);
    return make_fixed_width_column(schema.type, size, mask_state::ALL_NULL, stream, mr);
    auto indices   = make_zeroed_offsets(size - 1);
    auto null_mask = cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr);
    return make_dictionary_column(std::move(child), std::move(indices), std::move(null_mask), size);
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  std::unique_ptr<column> operator()(schema_element const& schema, size_type size) const
  {
    auto offsets   = make_zeroed_offsets(size);
    auto null_mask = cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr);
    return make_strings_column(
      size, std::move(offsets), rmm::device_buffer{}, size, std::move(null_mask));
  }
  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  std::unique_ptr<column> operator()(schema_element const& schema, size_type size) const
  {
    CUDF_EXPECTS(schema.child_types.size() == 1, "List column should have only one child");
    auto const& child_name = schema.child_types.begin()->first;
    std::unique_ptr<column> child =
      make_empty_column(schema.child_types.at(child_name), stream, mr);
    auto offsets   = make_zeroed_offsets(size);
    auto null_mask = cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr);
    std::vector<std::unique_ptr<column>> child_columns;
    child_columns.push_back(std::move(offsets));
    child_columns.push_back(std::move(child));
    // Do not use `cudf::make_lists_column` since we do not need to call `purge_nonempty_nulls` on
    // the child column as it does not have non-empty nulls. Look issue #17356
    return std::make_unique<column>(cudf::data_type{type_id::LIST},
                                    size,
                                    rmm::device_buffer{},
                                    std::move(null_mask),
                                    size,
                                    std::move(child_columns));
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  std::unique_ptr<column> operator()(schema_element const& schema, size_type size) const
  {
    std::vector<std::unique_ptr<column>> child_columns;
    for (auto const& child_name : schema.column_order.value_or(std::vector<std::string>{})) {
      child_columns.push_back(cudf::type_dispatcher(
        schema.child_types.at(child_name).type, *this, schema.child_types.at(child_name), size));
    }
    auto null_mask = cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr);
    // Do not use `cudf::make_structs_column` since we do not need to call
    // `superimpose_and_sanitize_nulls` on the children columns. Creating the struct hierarchy is
    // sufficient.
    return create_structs_hierarchy(
      size,
      std::move(child_columns),
      size,
      cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr),
      stream,
      mr);
  }
};

std::unique_ptr<column> make_all_nulls_column(schema_element const& schema,
                                              size_type num_rows,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(schema.type, allnull_column_functor{stream, mr}, schema, num_rows);
}

column_name_info make_column_name_info(schema_element const& schema, std::string const& col_name)
{
  column_name_info info;
  info.name = col_name;
  switch (schema.type.id()) {
    case type_id::STRUCT:
      for (auto const& child_name : schema.column_order.value_or(std::vector<std::string>{})) {
        info.children.push_back(
          make_column_name_info(schema.child_types.at(child_name), child_name));
      }
      break;
    case type_id::LIST:
      info.children.emplace_back("offsets");
      for (auto const& [child_name, child_schema] : schema.child_types) {
        info.children.push_back(make_column_name_info(child_schema, child_name));
      }
      break;
    case type_id::DICTIONARY32:
      info.children.emplace_back("indices");
      for (auto const& [child_name, child_schema] : schema.child_types) {
        info.children.push_back(make_column_name_info(child_schema, child_name));
      }
      break;
    case type_id::STRING: info.children.emplace_back("offsets"); break;
    default: break;
  }
  return info;
}
}  // namespace cudf::io::json::detail
