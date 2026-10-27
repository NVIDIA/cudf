/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <jit/helpers.hpp>
#include <jit/reflect.hpp>
#include <jit/util.hpp>
#include <rtcx/rtcx.hpp>

#include <concepts>
#include <format>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cudf::jit {
namespace {

std::string get_element_type_name(transform_input_spec const& spec, bool use_physical_type);

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

std::string get_element_type_name(transform_input_spec const& spec, bool use_physical_type)
{
  return cudf::type_dispatcher(
    data_type{spec.type}, element_type_name_fn{}, spec, use_physical_type);
}

std::string reflect_input_element(transform_input_spec const& spec, bool use_physical_type)
{
  return get_element_type_name(spec, use_physical_type);
}

std::string reflect_output_element(transform_output_spec const& spec, bool use_physical_type)
{
  if (spec.type == type_id::STRING) {
    return spec.has_string_offsets ? "cuda::std::span<char>" : "cudf::string_view";
  }
  return get_element_type_name(transform_input_spec{.type = spec.type}, use_physical_type);
}

std::string reflect_input_value_type(transform_input_spec const& spec, bool use_physical_type)
{
  if (spec.type == type_id::DICTIONARY32) {
    return reflect_input_value_type(spec.children.at(dictionary_keys_column_index),
                                    use_physical_type);
  }
  return reflect_input_element(spec, use_physical_type);
}

std::string reflect_output_value_type(transform_output_spec const& spec, bool use_physical_type)
{
  return reflect_output_element(spec, use_physical_type);
}

std::string reflect_input_column(transform_input_spec const&)
{
  return "cudf::column_device_view_core";
}

std::string reflect_output_column(transform_output_spec const& spec)
{
  if (spec.type == type_id::STRING) {
    return spec.has_string_offsets ? "cudf::jit::mutable_strings_column_device_view"
                                   : "cudf::jit::mutable_vector_device_view";
  }
  return "cudf::mutable_column_device_view_core";
}

transform_input_spec make_input_spec(column_view const& column, bool is_scalar)
{
  transform_input_spec result{.type = column.type().id(), .is_scalar = is_scalar};
  if (is_dictionary(column.type())) {
    for (size_type i = 0; i < column.num_children(); ++i) {
      result.children.push_back(make_input_spec(column.child(i), false));
    }
  } else if (column.type().id() == type_id::STRING &&
             column.num_children() > strings_column_view::offsets_column_index) {
    result.children.push_back(
      make_input_spec(column.child(strings_column_view::offsets_column_index), false));
  }
  return result;
}

transform_input_spec make_input_spec(transform_input const& input)
{
  return std::visit(
    [](auto& value) {
      return make_input_spec(as_column_view(value),
                             std::is_same_v<std::decay_t<decltype(value)>, scalar_column_view>);
    },
    input);
}

transform_output_spec make_output_spec(fixed_width_column const& output)
{
  return {.type = output._col->type().id()};
}

transform_output_spec make_output_spec(string_views_column const&)
{
  return {.type = type_id::STRING};
}

transform_output_spec make_output_spec(mutable_strings_column const& output)
{
  auto offsets = output._col->view().child(strings_column_view::offsets_column_index);
  return {.type               = type_id::STRING,
          .has_string_offsets = true,
          .children           = {{.type = offsets.type().id()}}};
}

}  // namespace

transform_reflection reflect(std::variant<udf_source_type, lto_binary_type> source_type,
                             std::span<transform_input_spec const> inputs,
                             std::span<transform_output_spec const> outputs,
                             std::span<int32_t const> input_table_sources,
                             std::span<int32_t const> input_table_indices,
                             std::span<int32_t const> output_table_sources,
                             std::span<int32_t const> output_table_indices)
{
  CUDF_EXPECTS(input_table_sources.empty() || input_table_sources.size() == inputs.size(),
               "Input table sources size must match the number of inputs or be empty",
               std::runtime_error);
  CUDF_EXPECTS(input_table_sources.size() == input_table_indices.size(),
               "Input table sources and indices size must match",
               std::runtime_error);
  CUDF_EXPECTS(output_table_sources.empty() || output_table_sources.size() == outputs.size(),
               "Output table sources size must match the number of outputs or be empty",
               std::runtime_error);
  CUDF_EXPECTS(output_table_sources.size() == output_table_indices.size(),
               "Output table sources and indices size must match",
               std::runtime_error);

  std::vector<std::string> in_types;
  bool use_physical_types = std::holds_alternative<lto_binary_type>(source_type);
  for (int32_t i = 0; i < std::ssize(inputs); i++) {
    auto& in       = inputs[i];
    auto column    = reflect_input_column(in);
    auto element   = reflect_input_element(in, use_physical_types);
    bool as_scalar = in.is_scalar;
    auto accessor  = rtcx::reflect_template(
      "cudf::jit::column_accessor",
      input_table_sources.empty() ? rtcx::reflect(i) : rtcx::reflect(input_table_indices[i]),
      column,
      element,
      rtcx::reflect(as_scalar),
      input_table_sources.empty() ? rtcx::reflect(0) : rtcx::reflect(input_table_sources[i]));
    in_types.push_back(accessor);
  }

  std::vector<std::string> out_types;
  for (int32_t i = 0; i < std::ssize(outputs); i++) {
    auto& out      = outputs[i];
    auto column    = reflect_output_column(out);
    auto element   = reflect_output_element(out, use_physical_types);
    bool as_scalar = false;  // never scalar
    auto accessor  = rtcx::reflect_template(
      "cudf::jit::column_accessor",
      output_table_sources.empty() ? rtcx::reflect(i) : rtcx::reflect(output_table_indices[i]),
      column,
      element,
      rtcx::reflect(as_scalar),
      output_table_sources.empty() ? rtcx::reflect(0) : rtcx::reflect(output_table_sources[i]));

    out_types.push_back(accessor);
  }

  auto ins  = rtcx::reflect_template("cudf::jit::type_list", in_types);
  auto outs = rtcx::reflect_template("cudf::jit::type_list", out_types);

  std::vector<std::string> ptx_in_types;
  std::vector<std::string> ptx_out_types;
  if (std::holds_alternative<udf_source_type>(source_type) &&
      std::get<udf_source_type>(source_type) == udf_source_type::PTX) {
    for (auto& in : inputs) {
      ptx_in_types.push_back(reflect_input_value_type(in, use_physical_types));
    }

    for (auto& out : outputs) {
      ptx_out_types.push_back(reflect_output_value_type(out, use_physical_types));
    }
  }

  return {ins, outs, ptx_in_types, ptx_out_types};
}

std::vector<transform_input_spec> make_input_specs(std::span<transform_input const> inputs)
{
  std::vector<transform_input_spec> result;
  for (auto& input : inputs) {
    result.push_back(make_input_spec(input));
  }
  return result;
}

std::vector<transform_output_spec> make_output_specs(std::span<output_column const> outputs)
{
  std::vector<transform_output_spec> result;
  for (auto& output : outputs) {
    result.push_back(std::visit([](auto& value) { return make_output_spec(value); }, output));
  }
  return result;
}

std::vector<transform_output_spec> make_output_specs(
  std::span<transform_output const> outputs,
  std::span<std::unique_ptr<column> const> string_offsets)
{
  CUDF_EXPECTS(string_offsets.empty() || string_offsets.size() == outputs.size(),
               "Number of string offsets must be empty or match the number of outputs",
               std::invalid_argument);
  std::vector<transform_output_spec> result;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto has_string_offsets = !string_offsets.empty() && string_offsets[i] != nullptr;
    transform_output_spec spec{.type               = outputs[i].type.id(),
                               .nullability        = outputs[i].nullability,
                               .has_string_offsets = has_string_offsets};
    if (has_string_offsets) { spec.children.push_back({.type = string_offsets[i]->type().id()}); }
    result.push_back(std::move(spec));
  }
  return result;
}

transform_reflection reflect(std::variant<udf_source_type, lto_binary_type> source_type,
                             std::span<transform_input const> inputs,
                             std::span<output_column const> outputs,
                             std::span<int32_t const> input_table_sources,
                             std::span<int32_t const> input_table_indices,
                             std::span<int32_t const> output_table_sources,
                             std::span<int32_t const> output_table_indices)
{
  auto input_specs  = make_input_specs(inputs);
  auto output_specs = make_output_specs(outputs);
  return reflect(source_type,
                 input_specs,
                 output_specs,
                 input_table_sources,
                 input_table_indices,
                 output_table_sources,
                 output_table_indices);
}

std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<transform_input_spec const> inputs,
                                  std::span<transform_output_spec const> outputs,
                                  bool use_physical_types)
{
  std::vector<std::string> in_types;

  for (auto const& input : inputs) {
    auto element = reflect_input_element(input, use_physical_types);
    in_types.push_back(is_null_aware ? std::format("cuda::std::optional<{}>", element) : element);
  }

  std::vector<std::string> out_types;

  for (auto const& output : outputs) {
    auto element = reflect_output_element(output, use_physical_types);
    out_types.push_back(is_null_aware ? std::format("cuda::std::optional<{}> *", element)
                                      : std::format("{} *", element));
  }

  std::vector<std::string> params;
  if (has_user_data) {
    params.emplace_back("void*");
    params.emplace_back("cudf::size_type");
  }
  params.insert(params.end(), out_types.begin(), out_types.end());
  params.insert(params.end(), in_types.begin(), in_types.end());

  auto joined =
    params.empty()
      ? ""
      : std::accumulate(
          std::next(params.begin()), params.end(), params[0], [](auto const& a, auto const& b) {
            return std::format("{}, {}", a, b);
          });

  return std::format("int({})", joined);
}

std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<transform_input const> inputs,
                                  std::span<output_column const> outputs,
                                  bool use_physical_types)
{
  auto input_specs  = make_input_specs(inputs);
  auto output_specs = make_output_specs(outputs);
  return reflect_udf_signature(
    is_null_aware, has_user_data, input_specs, output_specs, use_physical_types);
}

}  // namespace cudf::jit
