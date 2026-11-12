/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <jit/helpers.hpp>
#include <jit/reflect.hpp>
#include <jit/util.hpp>
#include <rtcx/rtcx.hpp>

#include <format>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cudf::jit {

namespace {

struct reflect_input_element_fn {
  template <typename T>
  std::string operator()(transform_input_spec const& spec, bool use_physical_type) const
    requires(cudf::concepts::fixed_width<T> || cudf::concepts::string_view<T>)
  {
    auto type = data_type{spec.type};
    return type_to_name(use_physical_type ? jit::physical_type_of(type) : type);
  }

  template <cudf::concepts::dictionary T>
  std::string operator()(transform_input_spec const& spec, bool use_physical_type) const
  {
    return std::format(
      "cudf::dictionary_element<{}, {}>",
      reflect_input_element(spec.children.at(dictionary_indices_column_index), use_physical_type),
      reflect_input_element(spec.children.at(dictionary_keys_column_index), use_physical_type));
  }

  template <typename T>
  std::string operator()(transform_input_spec const& spec, bool) const
  {
    CUDF_FAIL(std::format("Unsupported input element type for JIT input: {}",
                          type_to_name(data_type{spec.type})),
              std::invalid_argument);
  }
};

std::string reflect_output_element(transform_output_spec const& spec, bool use_physical_type);

struct reflect_output_element_fn {
  template <cudf::concepts::fixed_width T>
  std::string operator()(transform_output_spec const& spec, bool use_physical_type) const
  {
    auto type = data_type{spec.type};
    return type_to_name(use_physical_type ? jit::physical_type_of(type) : type);
  }

  template <cudf::concepts::string_view T>
  std::string operator()(transform_output_spec const& spec,
                         [[maybe_unused]] bool use_physical_type) const
  {
    return spec.has_string_offsets ? "cuda::std::span<char>"
                                   : "cudf::repr_element<cudf::string_view>";
  }

  template <typename T>
  std::string operator()(transform_output_spec const& spec, bool) const
  {
    CUDF_FAIL(std::format("Unsupported output element type for JIT output: {}",
                          type_to_name(data_type{spec.type})),
              std::invalid_argument);
  }
};

struct reflect_input_value_fn {
  template <typename T>
  std::string operator()(transform_input_spec const& spec, bool use_physical_type) const
    requires(cudf::concepts::fixed_width<T> || cudf::concepts::string_view<T>)
  {
    auto type = data_type{spec.type};
    return type_to_name(use_physical_type ? jit::physical_type_of(type) : type);
  }

  template <cudf::concepts::dictionary T>
  std::string operator()(transform_input_spec const& spec, bool use_physical_type) const
  {
    return reflect_input_element(spec.children.at(dictionary_keys_column_index), use_physical_type);
  }

  template <typename T>
  std::string operator()(transform_input_spec const& spec, bool) const
  {
    CUDF_FAIL(std::format("Unsupported input value type for JIT input: {}",
                          type_to_name(data_type{spec.type})),
              std::invalid_argument);
  }
};

struct reflect_output_value_fn {
  template <cudf::concepts::fixed_width T>
  std::string operator()(transform_output_spec const& spec, bool use_physical_type) const
  {
    auto type = data_type{spec.type};
    return type_to_name(use_physical_type ? jit::physical_type_of(type) : type);
  }

  template <cudf::concepts::string_view T>
  std::string operator()(transform_output_spec const& spec,
                         [[maybe_unused]] bool use_physical_type) const
  {
    return spec.has_string_offsets ? "cuda::std::span<char>" : "cudf::string_view";
  }

  template <typename T>
  std::string operator()(transform_output_spec const& spec, bool) const
  {
    CUDF_FAIL(std::format("Unsupported output value type for JIT output: {}",
                          type_to_name(data_type{spec.type})),
              std::invalid_argument);
  }
};

std::string reflect_output_element(transform_output_spec const& spec, bool use_physical_type)
{
  return type_dispatcher(
    data_type{spec.type}, reflect_output_element_fn{}, spec, use_physical_type);
}

std::string reflect_output_value_type(transform_output_spec const& spec, bool use_physical_type)
{
  return type_dispatcher(data_type{spec.type}, reflect_output_value_fn{}, spec, use_physical_type);
}

transform_input_spec make_input_spec(column_view const& column)
{
  return cudf::jit::make_input_spec(column, false);
}

transform_input_spec make_input_spec(scalar_column_view const& scalar)
{
  return cudf::jit::make_input_spec(scalar.as_column_view(), true);
}

transform_input_spec make_input_spec(user_data const&) { return {.is_user_data = true}; }

transform_input_spec make_input_spec(transform_input const& input)
{
  return std::visit([](auto& value) { return make_input_spec(value); }, input);
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

std::string reflect_input_element(transform_input_spec const& spec, bool use_physical_type)
{
  if (spec.is_user_data) { return "cudf::user_data"; }
  return type_dispatcher(data_type{spec.type}, reflect_input_element_fn{}, spec, use_physical_type);
}

std::string reflect_input_value_type(transform_input_spec const& spec, bool use_physical_type)
{
  if (spec.is_user_data) { return "void*"; }
  return type_dispatcher(data_type{spec.type}, reflect_input_value_fn{}, spec, use_physical_type);
}

transform_input_spec make_input_spec(column_view const& column, bool is_scalar)
{
  CUDF_EXPECTS(is_fixed_width(column.type()) || is_dictionary(column.type()) ||
                 column.type().id() == type_id::STRING,
               "Column type must be fixed-width, dictionary, or string",
               std::invalid_argument);

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

transform_reflection reflect(bool use_physical_types,
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
  for (int32_t i = 0; i < std::ssize(inputs); i++) {
    auto& in       = inputs[i];
    auto element   = reflect_input_element(in, use_physical_types);
    bool as_scalar = in.is_scalar;
    auto accessor  = rtcx::reflect_template(
      "cudf::jit::column_accessor",
      input_table_sources.empty() ? rtcx::reflect(i) : rtcx::reflect(input_table_indices[i]),
      element,
      rtcx::reflect(as_scalar),
      input_table_sources.empty() ? rtcx::reflect(0) : rtcx::reflect(input_table_sources[i]));
    in_types.push_back(accessor);
  }

  std::vector<std::string> out_types;
  for (int32_t i = 0; i < std::ssize(outputs); i++) {
    auto& out      = outputs[i];
    auto element   = reflect_output_element(out, use_physical_types);
    bool as_scalar = false;  // never scalar
    auto accessor  = rtcx::reflect_template(
      "cudf::jit::column_accessor",
      output_table_sources.empty() ? rtcx::reflect(i) : rtcx::reflect(output_table_indices[i]),
      element,
      rtcx::reflect(as_scalar),
      output_table_sources.empty() ? rtcx::reflect(0) : rtcx::reflect(output_table_sources[i]));

    out_types.push_back(accessor);
  }

  auto ins  = rtcx::reflect_template("cudf::jit::type_list", in_types);
  auto outs = rtcx::reflect_template("cudf::jit::type_list", out_types);

  return {ins, outs};
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
    CUDF_EXPECTS(outputs[i].type.id() == type_id::STRING || is_fixed_width(outputs[i].type),
                 "Output type must be either STRING or fixed-width",
                 std::invalid_argument);

    auto has_string_offsets = !string_offsets.empty() && string_offsets[i] != nullptr;
    transform_output_spec spec{.type               = outputs[i].type.id(),
                               .nullability        = outputs[i].nullability,
                               .has_string_offsets = has_string_offsets};
    if (spec.type == type_id::STRING && spec.has_string_offsets) {
      spec.children.push_back({.type = string_offsets[i]->type().id()});
    }
    result.push_back(std::move(spec));
  }
  return result;
}

transform_reflection reflect(bool use_physical_types,
                             std::span<transform_input const> inputs,
                             std::span<output_column const> outputs,
                             std::span<int32_t const> input_table_sources,
                             std::span<int32_t const> input_table_indices,
                             std::span<int32_t const> output_table_sources,
                             std::span<int32_t const> output_table_indices)
{
  auto input_specs  = make_input_specs(inputs);
  auto output_specs = make_output_specs(outputs);
  return reflect(use_physical_types,
                 input_specs,
                 output_specs,
                 input_table_sources,
                 input_table_indices,
                 output_table_sources,
                 output_table_indices);
}

std::vector<std::string> reflect_udf_parameter_types(bool is_null_aware,
                                                     std::span<transform_input_spec const> inputs,
                                                     std::span<transform_output_spec const> outputs,
                                                     bool use_physical_types,
                                                     int32_t index_count)
{
  std::vector<std::string> in_types;

  for (auto const& input : inputs) {
    auto element = reflect_input_value_type(input, use_physical_types);
    in_types.push_back(is_null_aware && !input.is_user_data
                         ? std::format("cuda::std::optional<{}>", element)
                         : element);
  }

  std::vector<std::string> out_types;

  for (auto const& output : outputs) {
    auto element = reflect_output_value_type(output, use_physical_types);
    out_types.push_back(is_null_aware ? std::format("cuda::std::optional<{}> *", element)
                                      : std::format("{} *", element));
  }

  auto params = std::move(out_types);
  params.insert(params.begin(), index_count, "cudf::size_type");
  params.insert(params.end(), in_types.begin(), in_types.end());
  return params;
}

std::string reflect_udf_signature(bool is_null_aware,
                                  std::span<transform_input_spec const> inputs,
                                  std::span<transform_output_spec const> outputs,
                                  bool use_physical_types,
                                  int32_t index_count)
{
  auto params =
    reflect_udf_parameter_types(is_null_aware, inputs, outputs, use_physical_types, index_count);
  auto joined =
    params.empty()
      ? ""
      : std::accumulate(
          std::next(params.begin()), params.end(), params[0], [](auto const& a, auto const& b) {
            return std::format("{}, {}", a, b);
          });

  return std::format("cudf::errc({})", joined);
}

std::string reflect_udf_signature(bool is_null_aware,
                                  std::span<transform_input const> inputs,
                                  std::span<output_column const> outputs,
                                  bool use_physical_types,
                                  int32_t index_count)
{
  auto input_specs  = make_input_specs(inputs);
  auto output_specs = make_output_specs(outputs);
  return reflect_udf_signature(
    is_null_aware, input_specs, output_specs, use_physical_types, index_count);
}

}  // namespace cudf::jit
