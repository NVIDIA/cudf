/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <jit/util.hpp>
#include <rtcx/rtcx.hpp>

#include <format>

namespace cudf::jit {

inline std::string get_element_type_name(transform_input_spec const& spec, bool use_physical_type);

struct element_type_name_fn {
  template <typename T>
  std::string operator()(transform_input_spec const& spec, bool use_physical_type) const
    requires(is_fixed_width<T>() || std::same_as<T, cudf::string_view>)
  {
    auto type = data_type{spec.type};
    return type_to_name(use_physical_type ? jit::physical_type_of(type) : type);
  }

  template <typename T>
  std::string operator()(transform_input_spec const& spec, bool use_physical_type) const
    requires(std::same_as<T, cudf::dictionary32>)
  {
    return std::format(
      "cudf::dictionary_element<{}, {}>",
      get_element_type_name(spec.children.at(dictionary_indices_column_index), use_physical_type),
      get_element_type_name(spec.children.at(dictionary_keys_column_index), use_physical_type));
  }

  template <typename T>
  std::string operator()(transform_input_spec const& spec, bool) const
    requires(!is_fixed_width<T>() && !std::same_as<T, cudf::string_view> &&
             !std::same_as<T, cudf::dictionary32>)
  {
    CUDF_FAIL("Unsupported type for JIT compilation: " + type_to_name(data_type{spec.type}));
  }
};

inline std::string get_element_type_name(transform_input_spec const& spec, bool use_physical_type)
{
  return cudf::type_dispatcher(
    data_type{spec.type}, element_type_name_fn{}, spec, use_physical_type);
}

inline std::string reflect_input_element(transform_input_spec const& spec, bool use_physical_type)
{
  return get_element_type_name(spec, use_physical_type);
}

inline std::string reflect_output_element(transform_output_spec const& spec, bool use_physical_type)
{
  if (spec.type == type_id::STRING) {
    return spec.has_string_offsets ? "cuda::std::span<char>" : "cudf::string_view";
  }
  return get_element_type_name(transform_input_spec{.type = spec.type}, use_physical_type);
}

inline std::string reflect_input_value_type(transform_input_spec const& spec,
                                            bool use_physical_type)
{
  if (spec.type == type_id::DICTIONARY32) {
    return reflect_input_value_type(spec.children.at(dictionary_keys_column_index),
                                    use_physical_type);
  }
  return reflect_input_element(spec, use_physical_type);
}

inline std::string reflect_output_value_type(transform_output_spec const& spec,
                                             bool use_physical_type)
{
  return reflect_output_element(spec, use_physical_type);
}

inline std::string reflect_input_column(transform_input_spec const&)
{
  return "cudf::column_device_view_core";
}

inline std::string reflect_output_column(transform_output_spec const& spec)
{
  if (spec.type == type_id::STRING) {
    return spec.has_string_offsets ? "cudf::jit::mutable_strings_column_device_view"
                                   : "cudf::jit::mutable_vector_device_view";
  }
  return "cudf::mutable_column_device_view_core";
}

inline rtcx::binary_type as_rtcx_binary_type(lto_binary_type type)
{
  switch (type) {
    case lto_binary_type::LTO_IR: return rtcx::binary_type::LTO_IR;
    case lto_binary_type::FATBIN: return rtcx::binary_type::FATBIN;
    default:
      CUDF_FAIL(std::format("Unrecognized LTO binary type {} for LTO UDF", static_cast<int>(type)),
                std::invalid_argument);
  }
}

}  // namespace cudf::jit
