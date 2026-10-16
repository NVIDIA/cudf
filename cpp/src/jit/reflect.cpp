/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <jit/reflect.hpp>
#include <jit/util.hpp>
#include <rtcx/rtcx.hpp>

#include <concepts>
#include <format>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>

namespace cudf::jit {

namespace {

std::string get_element_type_name(transform_input_spec const& spec, bool use_physical_type);

std::string reflect_input_element(transform_input_spec const& spec, bool use_physical_type);

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
      reflect_input_element(spec.children.at(dictionary_keys_column_index), use_physical_type));
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
  if (spec.type == type_id::LIST) {
    return "cudf::list_element";
  } else {
    return get_element_type_name(spec, use_physical_type);
  }
}

std::string reflect_output_element(transform_output_spec const& spec, bool use_physical_type)
{
  if (spec.type == type_id::LIST) {
    return "cudf::mutable_list_element";
  } else if (spec.type == type_id::STRING) {
    return spec.has_offsets ? "cudf::mutable_string_view" : "cudf::string_view";
  } else {
    return get_element_type_name(transform_input_spec{.type = spec.type}, use_physical_type);
  }
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
  if (spec.type == type_id::STRING && !spec.has_offsets) {
    return "cudf::jit::mutable_vector_device_view";
  }
  return "cudf::mutable_column_device_view_core";
}

}  // namespace

transform_reflection reflect(bool is_ptx,
                             std::span<transform_input_spec const> inputs,
                             std::span<transform_output_spec const> outputs,
                             bool use_physical_types)
{
  std::vector<std::string> in_types;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto& in       = inputs[i];
    auto column    = reflect_input_column(in);
    auto element   = reflect_input_element(in, use_physical_types);
    bool as_scalar = in.is_scalar;
    auto accessor  = rtcx::reflect_template("cudf::jit::column_accessor",
                                           rtcx::reflect(i),
                                           column,
                                           element,
                                           rtcx::reflect(as_scalar),
                                           rtcx::reflect(0));
    in_types.push_back(accessor);
  }

  std::vector<std::string> out_types;
  for (size_t i = 0; i < outputs.size(); i++) {
    auto& out      = outputs[i];
    auto column    = reflect_output_column(out);
    auto element   = reflect_output_element(out, use_physical_types);
    bool as_scalar = false;  // never scalar
    auto accessor  = rtcx::reflect_template("cudf::jit::column_accessor",
                                           rtcx::reflect(i),
                                           column,
                                           element,
                                           rtcx::reflect(as_scalar),
                                           rtcx::reflect(0));

    out_types.push_back(accessor);
  }

  auto ins  = rtcx::reflect_template("cudf::jit::type_list", in_types);
  auto outs = rtcx::reflect_template("cudf::jit::type_list", out_types);

  std::vector<std::string> ptx_in_types;
  std::vector<std::string> ptx_out_types;
  if (is_ptx) {
    for (auto& in : inputs) {
      ptx_in_types.push_back(reflect_input_value_type(in, use_physical_types));
    }

    for (auto& out : outputs) {
      ptx_out_types.push_back(reflect_output_value_type(out, use_physical_types));
    }
  }

  return std::make_tuple(ins, outs, ptx_in_types, ptx_out_types);
}

namespace {

column_view as_column_view(scalar_column_view const& scalar) { return scalar.as_column_view(); }

column_view as_column_view(column_view const& column) { return column; }

transform_input_spec make_input_spec(column_view const& column, bool is_scalar)
{
  transform_input_spec result{.type = column.type().id(), .is_scalar = is_scalar};
  if (is_dictionary(column.type()) || column.type().id() == type_id::LIST) {
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

}  // namespace

std::vector<transform_input_spec> make_input_specs(std::span<transform_input const> inputs)
{
  std::vector<transform_input_spec> result;
  for (auto& input : inputs) {
    result.push_back(make_input_spec(input));
  }
  return result;
}

namespace {

transform_output_spec make_output_spec(fixed_width_column const& output)
{
  return {.type = output._col->type().id()};
}

transform_output_spec make_output_spec(string_views_column const&)
{
  return {.type = type_id::STRING};
}

transform_output_spec make_output_spec(mutable_lists_column const& output)
{
  return {
    .type        = type_id::LIST,
    .has_offsets = true,
    .children    = {{.type = type_id::INT32}, {.type = output._col->view().child(1).type().id()}}};
}

transform_output_spec make_output_spec(mutable_strings_column const& output)
{
  auto offsets = output._col->view().child(strings_column_view::offsets_column_index);
  return {
    .type = type_id::STRING, .has_offsets = true, .children = {{.type = offsets.type().id()}}};
}

}  // namespace

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
  std::span<std::unique_ptr<column> const> output_offsets)
{
  std::vector<transform_output_spec> result;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto has_offsets = !output_offsets.empty() && output_offsets[i] != nullptr;
    transform_output_spec spec{.type        = outputs[i].type.id(),
                               .nullability = outputs[i].nullability,
                               .has_offsets = has_offsets};
    if (spec.type == type_id::LIST) {
      for (auto const& child : outputs[i].children) {
        spec.children.push_back({.type = child.type.id()});
      }
    } else if (spec.type == type_id::STRING && spec.has_offsets) {
      spec.children.push_back({.type = output_offsets[i]->type().id()});
    }
    result.push_back(std::move(spec));
  }
  return result;
}

transform_reflection reflect(bool is_ptx,
                             std::span<transform_input const> inputs,
                             std::span<output_column const> outputs,
                             bool use_physical_types)
{
  auto input_specs  = make_input_specs(inputs);
  auto output_specs = make_output_specs(outputs);
  return reflect(is_ptx, input_specs, output_specs, use_physical_types);
}

std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<transform_input_spec const> inputs,
                                  std::span<transform_output_spec const> outputs,
                                  bool use_physical_types)
{
  std::vector<std::string> in_types;

  for (size_t i = 0; i < inputs.size(); i++) {
    auto element = reflect_input_element(inputs[i], use_physical_types);
    in_types.push_back(is_null_aware ? std::format("cuda::std::optional<{}>", element) : element);
  }

  std::vector<std::string> out_types;

  for (size_t i = 0; i < outputs.size(); i++) {
    auto element = reflect_output_element(outputs[i], use_physical_types);
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
